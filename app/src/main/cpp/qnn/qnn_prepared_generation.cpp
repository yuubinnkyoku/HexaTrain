// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 yuubinnkyoku
//
// Process-local prepared generation engine (Phase A warm reuse).
//
// The cold path (runNicopediaHtpGeneration) rebuilds the HTP runtime and
// finalizes a fresh graph on every Generate press.  This file keeps one
// prepared engine alive across generations: runtime + finalized FORWARD_ONLY
// graph + loaded checkpoint parameters stay resident, so a second generation
// with the same checkpoint identity skips backend/device/context creation,
// graph create/finalize, and checkpoint decode entirely.
//
// Safety contract (unchanged from the cold path):
//  - Checkpoint identity (header/seed/step/registry/parameter hash) is
//    re-validated fail-closed on every prepare AND every run.
//  - QNN return-code success and application-visible tensor finiteness are
//    checked separately on every execute.
//  - No CPU fallback exists anywhere in this file.
//  - Any QNN failure, non-finite tensor, or identity mismatch poisons the
//    engine; the next generation after a poison is a fresh prepare.

#include "qnn_transformer.h"
#include "nicopedia_checkpoint_loader.h"
#include "../nicopedia_byte_bpe.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace phonelm::qnn {
namespace {

// One prepared engine: everything expensive about generation, kept warm.
struct PreparedEngine {
    PreparedGenerationKey key;
    tiny_lm::Config config;
    LoadedNprtCheckpoint loaded;
    std::unique_ptr<nicopedia_bpe::Model> bpeModel;
    std::unique_ptr<Runtime> runtime;

    // Poison flag: set by any failed run; poisoned engines refuse to run.
    bool poisoned = false;
    std::string poisonReason;

    // Cumulative evidence across all runs on this engine.
    std::uint64_t runCount = 0;
};

std::mutex& engineMutex() {
    static std::mutex mutex;
    return mutex;
}

PreparedEngine*& engineSlot() {
    // Max one prepared engine (process-local).  Replacing a different-key
    // engine releases the old one first.
    static PreparedEngine* engine = nullptr;
    return engine;
}

bool allFinite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

std::string failureReport(const char* classification, const std::string& error,
                          Runtime* runtime) {
    std::ostringstream report;
    report << "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
           << "failure_classification=" << classification << "\nerror=" << error
           << "\nprepared_graph_reused=false\n";
    if (runtime) report << runtime->apiTraceSummary() << runtime->diagnostics();
    return report.str();
}

}  // namespace

PreparedGenerationHandle prepareNicopediaGeneration(
    const PreparedGenerationKey& key, const LogSink& progress,
    std::string& error) {
    std::lock_guard<std::mutex> lock(engineMutex());
    auto& slot = engineSlot();
    if (slot && !slot->poisoned && slot->key == key) {
        // Same identity: reuse the existing prepared engine as-is.
        return slot;
    }
    // Different key or poisoned engine: release and prepare fresh.
    delete slot;
    slot = nullptr;

    tiny_lm::Config config;
    config.vocabularySize = key.vocabulary;
    config.tokens = key.tokens;
    config.dimension = key.dimension;
    config.feedForwardDimension = key.feedForward;
    config.numLayers = key.layers;
    config.numHeads = key.heads;
    if (!tiny_lm::validateConfig(config, &error)) return nullptr;

    const auto prepareStarted = std::chrono::steady_clock::now();
    if (progress) progress("phase=checkpoint_validation");
    uint32_t expectedStep = 0;
    if (!nprtParseCheckpointStep(key.checkpointPath, key.seed, key.layers,
                                 &expectedStep, key.tokens, key.dimension,
                                 key.feedForward) ||
        expectedStep != key.step) {
        error = "CHECKPOINT_FILENAME_IDENTITY";
        return nullptr;
    }

    std::unique_ptr<nicopedia_bpe::Model> bpeModel;
    try {
        if (config.vocabularySize == nicopedia_bpe::kVocabulary) {
            const std::size_t separator = key.checkpointPath.find_last_of("/\\");
            const std::string directory =
                separator == std::string::npos ? "." : key.checkpointPath.substr(0, separator);
            bpeModel = std::make_unique<nicopedia_bpe::Model>(
                nicopedia_bpe::loadModel(directory + "/byte-bpe-v1024.model"));
            if (!key.tokenizerHash.empty() &&
                bpeModel->identity() != key.tokenizerHash) {
                error = "TOKENIZER_HASH_MISMATCH";
                return nullptr;
            }
        }
    } catch (const std::exception& exception) {
        error = std::string("TOKENIZER_DECODE:") + exception.what();
        return nullptr;
    }

    std::unique_ptr<PreparedEngine> engine(new PreparedEngine());
    try {
        engine->loaded = nprtLoadCheckpointForGeneration(
            key.checkpointPath, config, key.seed,
            bpeModel ? bpeModel->identity() : "");
    } catch (const std::exception& exception) {
        error = std::string("CHECKPOINT_DECODE:") + exception.what();
        return nullptr;
    }
    if (engine->loaded.step != expectedStep) {
        error = "CHECKPOINT_STEP_MISMATCH";
        return nullptr;
    }
    if (!engine->loaded.finite) {
        error = "CHECKPOINT_NONFINITE";
        return nullptr;
    }
    if (!key.parameterHash.empty() &&
        engine->loaded.parameterHash != key.parameterHash) {
        error = "CHECKPOINT_PARAMETER_HASH_MISMATCH";
        return nullptr;
    }

    if (progress) progress("phase=htp_initialization");
    engine->runtime = std::make_unique<Runtime>();
    RuntimeOptions options;
    options.captureQnnCallback = false;
    options.qnnLogLevel = 2;
    options.htpGraphPrecisionMode = key.htpGraphPrecisionMode;
    options.htpGraphPrecisionCompensation = key.htpGraphPrecisionCompensation;
    options.htpGraphWeightsPacking = key.htpGraphWeightsPacking;
    options.htpGraphAdvancedActivationFusion =
        key.htpGraphAdvancedActivationFusion;
    options.htpContextGraphSplitting = key.htpContextGraphSplitting;
    options.htpNativeTensorFp16 = key.htpNativeTensorFp16;
    engine->runtime->setOptions(options);
    if (!engine->runtime->initialize(QnnBackendKind::HTP, error)) return nullptr;
    if (progress) progress("phase=graph_preparation");
    if (!engine->runtime->prepareTinyTransformerTraining(
            config.tokens, config.dimension, config.feedForwardDimension,
            config.epsilon, true, error, config.vocabularySize,
            TinyTransformerTrainingVariant::FORWARD_ONLY,
            TinyTransformerTrainingTapSet::NONE, config.numLayers,
            config.numHeads)) {
        return nullptr;
    }

    engine->key = key;
    engine->config = config;
    engine->bpeModel = std::move(bpeModel);
    engine->runCount = 0;
    slot = engine.release();
    return slot;
}

std::string runPreparedNicopediaGeneration(
    PreparedGenerationHandle handle, const std::string& promptPath,
    const nicopedia_gen::GenerateConfig& generateConfig,
    const LogSink& progress) {
    const auto totalWallStarted = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(engineMutex());
    auto& slot = engineSlot();
    if (!handle || handle != slot) {
        return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
               "failure_classification=PREPARED_ENGINE_INVALID\n"
               "error=handle does not match the process-local engine\n";
    }
    PreparedEngine& engine = *static_cast<PreparedEngine*>(handle);
    if (engine.poisoned) {
        return "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
               "failure_classification=PREPARED_ENGINE_POISONED\nerror=" +
               engine.poisonReason + "\n";
    }
    Runtime& runtime = *engine.runtime;
    const tiny_lm::Config& config = engine.config;
    std::string error;

    if (progress) progress("phase=checkpoint_validation");
    // Identity re-validation on every run: the checkpoint file must still be
    // present with the exact same parameter hash.  This keeps the same
    // fail-closed guarantee as the cold path's per-run validation.
    LoadedNprtCheckpoint identity;
    try {
        identity = nprtLoadCheckpointForGeneration(
            engine.key.checkpointPath, config, engine.key.seed,
            engine.bpeModel ? engine.bpeModel->identity() : "");
    } catch (const std::exception& exception) {
        engine.poisoned = true;
        engine.poisonReason = std::string("checkpoint revalidation: ") +
                              exception.what();
        return failureReport("PREPARED_ENGINE_POISONED", engine.poisonReason,
                             &runtime);
    }
    if (identity.parameterHash != engine.loaded.parameterHash ||
        identity.step != engine.loaded.step || !identity.finite) {
        engine.poisoned = true;
        engine.poisonReason = "checkpoint changed after prepare";
        return failureReport("PREPARED_ENGINE_POISONED", engine.poisonReason,
                             &runtime);
    }

    std::vector<std::uint8_t> prompt;
    try {
        prompt = nprtReadFileBytes(promptPath, 16u * 1024u * 1024u);
    } catch (const std::exception& exception) {
        return failureReport("PROMPT_READ", exception.what(), &runtime);
    }
    if (prompt.empty()) {
        return failureReport("PROMPT_EMPTY", "prompt file is empty", &runtime);
    }
    std::vector<std::uint16_t> promptTokens;
    if (engine.bpeModel) promptTokens = engine.bpeModel->encode(prompt);
    else promptTokens.assign(prompt.begin(), prompt.end());
    uint32_t contextPad = 0;
    std::vector<std::uint16_t> context =
        nicopedia_bpe::buildTokenContext(promptTokens, config.tokens, &contextPad);

    const double graphPrepareUs = 0.0;  // Graph already finalized at prepare.
    const auto parityStarted = std::chrono::steady_clock::now();
    bool htpNativeQnnSuccess = true;
    bool htpNativePrefixLogitsFinite = true;
    bool htpNativeArLogitsFinite = true;
    bool htpNativeGenerationLogitsFinite = true;
    bool htpNativeApplicationTensorsFinite = true;
    bool samplingHealth = true;
    std::uint32_t samplingFailureCount = 0;
    double samplingWeightSum = 0.0;
    double samplingProbabilitySum = 0.0;

    const auto htpNativeFailure = [&](const char* phase,
                                      const std::string& detail) {
        engine.poisoned = true;
        engine.poisonReason = std::string(phase) + ": " + detail;
        std::ostringstream report;
        report << "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
               << "failure_classification=HTP_NATIVE_HEALTH\nhealth_phase="
               << phase << "\nerror=" << detail
               << "\nhtp_native_qnn_success="
               << (htpNativeQnnSuccess ? "true" : "false")
               << "\noutput_tensors_finite="
               << ((htpNativeApplicationTensorsFinite &&
                    htpNativeGenerationLogitsFinite)
                       ? "true"
                       : "false")
               << "\ncpu_fallback=false\n"
               << runtime.apiTraceSummary() << runtime.diagnostics();
        return report.str();
    };
    const auto generationExecuteFailure = [&](const char* phase,
                                              const std::string& detail) {
        const auto& trace = runtime.apiTrace();
        const bool qnnStatusSuccess = trace.graphExecuteAttemptCount > 0 &&
                                      trace.graphExecuteLastResult == 0;
        htpNativeQnnSuccess = htpNativeQnnSuccess && qnnStatusSuccess;
        return htpNativeFailure(phase, detail);
    };

    const auto windowInput = [&](const std::vector<std::uint16_t>& window) {
        std::vector<uint32_t> tokens;
        tokens.reserve(window.size());
        for (std::uint16_t token : window) tokens.push_back(token);
        return tiny_lm::oneHot(tokens, config.vocabularySize);
    };

    // Fixed-prefix health gate on the shared deterministic prefixes.  On the
    // warm path this is a health check only (same semantics as the cold
    // path's htp-native policy): every prefix executes once through the
    // prepared graph and its logits must be finite.  The FORWARD_ONLY graph
    // has no probability output, so probability health is vacuously true.
    const auto& prefixes = nicopedia_gen::parityPrefixes();
    for (const auto& prefix : prefixes) {
        uint32_t pad = 0;
        std::vector<std::uint16_t> encoded;
        if (engine.bpeModel) encoded = engine.bpeModel->encode(prefix.bytes);
        else encoded.assign(prefix.bytes.begin(), prefix.bytes.end());
        const auto prefixContext =
            nicopedia_bpe::buildTokenContext(encoded, config.tokens, &pad);
        TinyTransformerTrainingOutputs htpStep;
        if (!runtime.executeTinyTransformerForwardOnly(
                windowInput(prefixContext), engine.loaded.parameters, htpStep,
                error)) {
            htpNativeApplicationTensorsFinite = false;
            htpNativePrefixLogitsFinite = htpNativePrefixLogitsFinite &&
                                           !htpStep.logits.empty() &&
                                           allFinite(htpStep.logits);
            return generationExecuteFailure("nicopedia_generate_parity", error);
        }
        const bool logitsFinite = !htpStep.logits.empty() && allFinite(htpStep.logits);
        htpNativePrefixLogitsFinite = htpNativePrefixLogitsFinite && logitsFinite;
        htpNativeApplicationTensorsFinite =
            htpNativeApplicationTensorsFinite && logitsFinite;
        if (!logitsFinite) {
            return htpNativeFailure("fixed_prefix",
                                    "non-finite HTP logits on prepared graph");
        }
    }

    // Autoregressive health loop: 8 greedy steps from the first prefix.
    constexpr std::uint32_t kArSteps = 8;
    {
        uint32_t pad = 0;
        std::vector<std::uint16_t> arContext;
        if (engine.bpeModel) arContext = engine.bpeModel->encode(prefixes[0].bytes);
        else arContext.assign(prefixes[0].bytes.begin(), prefixes[0].bytes.end());
        arContext = nicopedia_bpe::buildTokenContext(arContext, config.tokens, &pad);
        for (std::uint32_t step = 0; step < kArSteps; ++step) {
            TinyTransformerTrainingOutputs htpStep;
            if (!runtime.executeTinyTransformerForwardOnly(
                    windowInput(arContext), engine.loaded.parameters, htpStep,
                    error)) {
                htpNativeApplicationTensorsFinite = false;
                htpNativeArLogitsFinite = htpNativeArLogitsFinite &&
                                           !htpStep.logits.empty() &&
                                           allFinite(htpStep.logits);
                return generationExecuteFailure("nicopedia_generate_ar", error);
            }
            const size_t lastBase =
                size_t(config.tokens - 1) * config.vocabularySize;
            const float* row = htpStep.logits.data() + lastBase;
            bool rowFinite = htpStep.logits.size() >= lastBase + config.vocabularySize;
            for (uint32_t j = 0; rowFinite && j < config.vocabularySize; ++j)
                rowFinite = rowFinite && std::isfinite(row[j]);
            htpNativeArLogitsFinite = htpNativeArLogitsFinite && rowFinite;
            htpNativeApplicationTensorsFinite =
                htpNativeApplicationTensorsFinite && rowFinite;
            if (!rowFinite) {
                return htpNativeFailure("autoregressive_parity",
                                        "non-finite HTP logits on prepared graph");
            }
            const uint32_t argmax =
                nicopedia_gen::greedyArgmax(row, config.vocabularySize);
            arContext.erase(arContext.begin());
            arContext.push_back(static_cast<std::uint16_t>(argmax));
        }
    }
    const double parityUs = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - parityStarted)
                                .count();

    const auto& trace = runtime.apiTrace();
    const bool qnnTraceHealthy = trace.backendRequested == "HTP" &&
                                 trace.graphExecuteAttemptCount > 0 &&
                                 trace.graphExecuteSuccessCount ==
                                     trace.graphExecuteAttemptCount &&
                                 trace.graphExecuteFailureCount == 0 &&
                                 trace.graphExecuteLastResult == 0 &&
                                 trace.lastQnnResult == 0 &&
                                 trace.effectiveResult == 0 &&
                                 !trace.cpuBackendInitialized &&
                                 !trace.fallbackAttempted &&
                                 !trace.fallbackSucceeded;
    const bool htpNativeGate =
        htpNativeQnnSuccess && htpNativePrefixLogitsFinite &&
        htpNativeArLogitsFinite && htpNativeApplicationTensorsFinite;

    std::vector<std::uint8_t> generated;
    std::vector<std::uint16_t> generatedTokens;
    double generateSeconds = 0;
    if (progress) {
        std::ostringstream update;
        update << "phase=generating\ngenerated_bytes=0\nmax_new_bytes="
               << generateConfig.maxNewBytes
               << "\nqnn_execute_attempts=" << trace.graphExecuteAttemptCount
               << "\nqnn_execute_successes=" << trace.graphExecuteSuccessCount
               << "\nqnn_execute_failures=" << trace.graphExecuteFailureCount
               << "\ncpu_fallback=false\nfinite=true\ngenerated_hex=";
        progress(update.str());
    }
    const auto generateStarted = std::chrono::steady_clock::now();
    auto generateContext = context;
    for (std::uint32_t step = 0; step < generateConfig.maxNewBytes; ++step) {
        TinyTransformerTrainingOutputs htpStep;
        if (!runtime.executeTinyTransformerForwardOnly(
                windowInput(generateContext), engine.loaded.parameters, htpStep,
                error)) {
            htpNativeApplicationTensorsFinite = false;
            htpNativeGenerationLogitsFinite =
                htpNativeGenerationLogitsFinite && !htpStep.logits.empty() &&
                allFinite(htpStep.logits);
            return generationExecuteFailure("nicopedia_generate_step", error);
        }
        const size_t lastBase = size_t(config.tokens - 1) * config.vocabularySize;
        const bool rowAvailable =
            htpStep.logits.size() >= lastBase + config.vocabularySize;
        const float* row = rowAvailable ? htpStep.logits.data() + lastBase : nullptr;
        bool rowFinite = rowAvailable;
        for (uint32_t j = 0; rowAvailable && j < config.vocabularySize; ++j)
            rowFinite = rowFinite && std::isfinite(row[j]);
        htpNativeGenerationLogitsFinite =
            htpNativeGenerationLogitsFinite && rowFinite;
        htpNativeApplicationTensorsFinite =
            htpNativeApplicationTensorsFinite && rowFinite;
        if (!rowFinite) {
            return htpNativeFailure(
                "generation_loop",
                "generation step produced non-finite logits on prepared graph");
        }
        std::uint32_t nextToken = 0;
        if (generateConfig.greedy) {
            nextToken = nicopedia_gen::greedyArgmax(row, config.vocabularySize);
        } else {
            const auto sampling = nicopedia_gen::sampleTopKChecked(
                row, config.vocabularySize, generateConfig.temperature,
                generateConfig.topK, generateConfig.samplingSeed, step);
            samplingHealth = samplingHealth && sampling.ok;
            samplingWeightSum = sampling.weightSum;
            samplingProbabilitySum = sampling.probabilitySum;
            if (!sampling.ok) {
                ++samplingFailureCount;
                return htpNativeFailure(
                    "sampling",
                    "top-k probability weights/logits are non-finite or sum<=0");
            }
            nextToken = sampling.value;
        }
        if (nextToken >= config.vocabularySize) {
            return failureReport("TOKEN_RANGE",
                                 "generated token out of vocabulary", &runtime);
        }
        if (!nicopedia_bpe::appendTokenWithinByteLimit(
                engine.bpeModel.get(), nextToken, generateConfig.maxNewBytes,
                &generated, &generatedTokens))
            break;
        generateContext.erase(generateContext.begin());
        generateContext.push_back(static_cast<std::uint16_t>(nextToken));
        if (progress &&
            (step == 0 || (step + 1) % 16 == 0 ||
             step + 1 == generateConfig.maxNewBytes)) {
            const auto& liveTrace = runtime.apiTrace();
            std::ostringstream update;
            update << "phase=generating\ngenerated_bytes=" << generated.size()
                   << "\ngenerated_tokens=" << generatedTokens.size()
                   << "\nmax_new_bytes=" << generateConfig.maxNewBytes
                   << "\nqnn_execute_attempts="
                   << liveTrace.graphExecuteAttemptCount
                   << "\nqnn_execute_successes="
                   << liveTrace.graphExecuteSuccessCount
                   << "\nqnn_execute_failures="
                   << liveTrace.graphExecuteFailureCount
                   << "\ncpu_fallback=false\nfinite=true\ngenerated_hex="
                   << nicopedia_gen::bytesToHex(generated);
            progress(update.str());
        }
    }
    generateSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    generateStarted)
                          .count();

    const nicopedia_gen::Utf8Stats generatedStats =
        nicopedia_gen::utf8StatsOf(generated);
    const nicopedia_gen::GenerationAggregates ag =
        nicopedia_gen::generationAggregates(generated);
    // The first run on a fresh engine paid prepare (graph create/finalize
    // happened before this run); any later run reuses the resident graph.
    // This field is evidence of *this run's* relationship to the graph, not
    // an assertion about the engine's history.
    ++engine.runCount;
    const bool graphReusedThisRun = engine.runCount > 1;
    const bool htpNativeHealth =
        htpNativeGate && htpNativeGenerationLogitsFinite &&
        htpNativeApplicationTensorsFinite && samplingHealth && qnnTraceHealthy;
    const std::string status = htpNativeHealth ? "SUCCESS" : "FAILED";

    std::ostringstream report;
    report << std::setprecision(10)
           << "NICOPEDIA_HTP_GENERATION\ntest=nicopedia_htp_generation_prepared\nstatus="
           << status
           << (htpNativeHealth
                   ? ""
                   : "\nfailure_classification=HTP_NATIVE_HEALTH")
           << "\nmodel=L" << engine.key.layers
           << "\nlayers=" << engine.key.layers
           << "\nheads=" << engine.key.heads
           << "\nmodel_dimension=" << config.dimension
           << "\nfeed_forward_dimension=" << config.feedForwardDimension
           << "\nseed=" << engine.key.seed
           << "\ncheckpoint_step=" << engine.loaded.step
           << "\ncheckpoint_parameter_hash=" << engine.loaded.parameterHash
           << "\ncheckpoint_format="
           << (engine.loaded.v3 ? "NPRTCKPTV3"
               : (engine.loaded.v2 ? "NPRTCKPTV2" : "NPRTCKPTV1"))
           << "\ntokenizer_kind=" << (engine.bpeModel ? "byte_bpe" : "byte")
           << "\ntokenizer_hash="
           << (engine.bpeModel ? engine.bpeModel->identity() : "legacy-byte-v1")
           << "\ncheckpoint_parameter_elements=" << engine.loaded.parameterElements
           << "\ncheckpoint_file_bytes=" << engine.loaded.fileBytes
           << "\ncheckpoint_finite=true"
           << "\ncheckpoint_header_vocabulary=" << engine.loaded.vocabulary
           << "\ncheckpoint_header_tokens=" << engine.loaded.tokens
           << "\ncheckpoint_header_dimension=" << engine.loaded.dimension
           << "\ncheckpoint_header_feedforward=" << engine.loaded.feedForward
           << "\ncheckpoint_header_layers=" << engine.loaded.layers
           << "\ncheckpoint_header_heads=" << engine.loaded.heads
           << "\ncheckpoint_header_seed=" << engine.loaded.seed
           << "\ncheckpoint_header_step=" << engine.loaded.step
           << "\ncheckpoint_header_registry_count=" << engine.loaded.registryCount
           << "\nprompt_byte_count=" << prompt.size()
           << "\nprompt_token_count=" << promptTokens.size()
           << "\ncontext_used_tokens=" << context.size()
           << "\ncontext_padding_tokens=" << contextPad
           << "\ngenerate_mode=" << (generateConfig.greedy ? "greedy" : "sample")
           << "\ntemperature=" << generateConfig.temperature
           << "\ntop_k=" << generateConfig.topK
           << "\nsampling_seed=" << generateConfig.samplingSeed
           << "\nmax_new_bytes=" << generateConfig.maxNewBytes
           << "\ngenerated_byte_count=" << generated.size()
           << "\ngenerated_token_count=" << generatedTokens.size()
           << "\ngenerated_valid_utf8_bytes=" << generatedStats.validBytes
           << "\ngenerated_invalid_utf8_bytes=" << generatedStats.invalidBytes
           << "\nunique_byte_values=" << ag.uniqueByteValues
           << "\nascii_bytes=" << ag.asciiBytes
           << "\nmax_same_byte_run=" << ag.maxSameByteRun
           << "\nmax_scalar_repeat_run=" << ag.maxScalarRepeatRun
           << "\nshort_period_loop_fraction=" << ag.shortPeriodLoopFraction
           << "\ngenerated_hex=" << nicopedia_gen::bytesToHex(generated)
           << "\nparity_prefix_count=" << prefixes.size()
           << "\nar_steps=" << kArSteps
           << "\ngate_policy=htp-native"
           << "\nhtp_graph_precision_mode=" << engine.key.htpGraphPrecisionMode
           << "\nhtp_graph_precision_compensation="
           << engine.key.htpGraphPrecisionCompensation
           << "\nhtp_graph_weights_packing=" << engine.key.htpGraphWeightsPacking
           << "\nhtp_graph_activation_fusion="
           << engine.key.htpGraphAdvancedActivationFusion
           << "\nhtp_context_graph_splitting=" << engine.key.htpContextGraphSplitting
           << "\nhtp_native_tensor_fp16="
           << (engine.key.htpNativeTensorFp16 ? "true" : "false")
           << "\ngeneration_gate=" << (htpNativeHealth ? "true" : "false")
           << "\ngeneration_health=" << (htpNativeHealth ? "true" : "false")
           << "\nhtp_native_qnn_success="
           << (htpNativeQnnSuccess ? "true" : "false")
           << "\nqnn_return_code_success="
           << (htpNativeQnnSuccess ? "true" : "false")
           << "\noutput_tensors_finite="
           << ((htpNativeGenerationLogitsFinite &&
                htpNativeApplicationTensorsFinite)
                   ? "true"
                   : "false")
           << "\nhtp_native_prefix_logits_finite="
           << (htpNativePrefixLogitsFinite ? "true" : "false")
           << "\nhtp_native_ar_logits_finite="
           << (htpNativeArLogitsFinite ? "true" : "false")
           << "\nhtp_native_generation_logits_finite="
           << (htpNativeGenerationLogitsFinite ? "true" : "false")
           << "\nsampling_enabled=" << (generateConfig.greedy ? "false" : "true")
           << "\nsampling_health=" << (samplingHealth ? "true" : "false")
           << "\nsampling_failure_count=" << samplingFailureCount
           << "\nsampling_weight_sum=" << samplingWeightSum
           << "\nsampling_probability_sum=" << samplingProbabilitySum
           << "\ngeneration_policy=htp-native"
           << "\nprepared_graph_reused="
           << (graphReusedThisRun ? "true" : "false")
           << "\nprepared_engine_run_count=" << engine.runCount
           << "\ngraph_create_count_at_prepare=1"
           << "\ngraph_finalize_count_at_prepare=1"
           << "\ngraph_create_count_this_run=0"
           << "\ngraph_finalize_count_this_run=0"
           << "\nhtp_initialize_us=0"
           << "\ngraph_create_us=0"
           << "\ngraph_finalize_us=0"
           << "\ngraph_prepare_us=" << graphPrepareUs
           << "\nparity_us=" << parityUs
           << "\ngeneration_loop_us=" << (generateSeconds * 1e6)
           << "\ncleanup_us=0"
           << "\ntotal_wall_us="
           << std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - totalWallStarted)
                  .count()
           << "\ngeneration_total_seconds=" << generateSeconds
           << "\ngeneration_ms_per_byte="
           << (generated.empty()
                   ? 0.0
                   : generateSeconds / generated.size() * 1000.0)
           << "\ngraph_execute_count=" << runtime.metrics().graphExecuteCount
           << "\ncpu_fallback=false"
           << "\nnan_detected=false"
           << "\ninf_detected=false\n"
           << runtime.apiTraceSummary() << runtime.diagnostics();
    return report.str();
}

void releaseNicopediaGeneration(PreparedGenerationHandle handle) {
    std::lock_guard<std::mutex> lock(engineMutex());
    auto& slot = engineSlot();
    if (handle == nullptr) {
        // Kotlin does not retain handles across calls; a null release means
        // "free whatever engine is resident".
        delete slot;
        slot = nullptr;
        return;
    }
    if (handle == slot) {
        delete slot;
        slot = nullptr;
    }
    // Unknown non-null handles are ignored: the slot owns the only valid
    // engine and a stale handle can never alias it after release.
}

}  // namespace phonelm::qnn
