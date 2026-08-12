#include "benchmark_runner.h"
#include "training_engine.h"
#include "qnn/qnn_transformer.h"

#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "PhoneLMBench";
std::atomic_bool gStopRequested{false};
std::atomic_bool gRunning{false};
// A Kotlin stop can arrive after the worker is accepted but before the
// synchronous JNI method has entered.  Keep that request separate from the
// per-run flag so the method cannot erase it with its normal reset.
std::mutex gStopStateMutex;
bool gNativeCallEntered = false;
bool gPendingStop = false;

void beginNativeCall() {
    std::lock_guard<std::mutex> lock(gStopStateMutex);
    gStopRequested.store(gPendingStop, std::memory_order_release);
    gPendingStop = false;
    gNativeCallEntered = true;
}

void logcat(const std::string& message) {
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s", message.c_str());
}

jstring toJavaString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

std::string failedReport(const std::string& error) {
    return "RESULT\n"
           "backend_requested=UNKNOWN\n"
           "backend_actual=UNINITIALIZED\n"
           "fallback_detected=false\n"
           "nan_detected=false\n"
           "status=FAILED\n"
           "error=" + error;
}

class RunningGuard {
public:
    ~RunningGuard() {
        std::lock_guard<std::mutex> lock(gStopStateMutex);
        gNativeCallEntered = false;
        gPendingStop = false;
        gRunning.store(false, std::memory_order_release);
    }
};

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeGetEnvironmentInfo(
    JNIEnv* env, jobject /* receiver */) {
    try {
        const auto report = phonelm::BenchmarkRunner::environmentReport();
        logcat(report);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = failedReport(std::string("environment probe exception: ") + exception.what());
        logcat(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = failedReport("environment probe unknown exception");
        logcat(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeGetQnnStatus(
    JNIEnv* env, jobject /* receiver */) {
    try {
        const auto report = phonelm::TrainingEngine::capabilityReport();
        logcat(report);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = std::string("qnn_status=FAILED\nerror=") + exception.what();
        logcat(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRunBenchmark(
    JNIEnv* env,
    jobject /* receiver */,
    jint backend,
    jint batchSize,
    jint dimension,
    jint steps,
    jint warmupSteps,
    jfloat learningRate,
    jlong seed,
    jobject progressCallback) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        const auto report = failedReport("a benchmark is already running");
        logcat(report);
        return toJavaString(env, report);
    }
    RunningGuard guard;
    beginNativeCall();

    jmethodID progressMethod = nullptr;
    if (progressCallback != nullptr) {
        jclass callbackClass = env->GetObjectClass(progressCallback);
        if (callbackClass != nullptr) {
            progressMethod = env->GetMethodID(
                callbackClass, "onNativeProgress", "(Ljava/lang/String;)V");
            env->DeleteLocalRef(callbackClass);
        }
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            progressMethod = nullptr;
        }
    }

    bool callbackEnabled = progressCallback != nullptr && progressMethod != nullptr;
    auto sink = [&](const std::string& message) {
        logcat(message);
        if (!callbackEnabled) {
            return;
        }
        jstring javaMessage = toJavaString(env, message);
        if (javaMessage == nullptr) {
            callbackEnabled = false;
            return;
        }
        env->CallVoidMethod(progressCallback, progressMethod, javaMessage);
        env->DeleteLocalRef(javaMessage);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            callbackEnabled = false;
            logcat("progress_callback_error=true");
        }
    };

    phonelm::TrainingConfig config;
    config.backend = static_cast<phonelm::BackendKind>(backend);
    config.batchSize = batchSize;
    config.dimension = dimension;
    config.steps = steps;
    config.warmupSteps = warmupSteps;
    config.learningRate = learningRate;
    config.seed = static_cast<std::uint64_t>(seed);

    try {
        const auto report = phonelm::BenchmarkRunner::run(config, gStopRequested, sink);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = failedReport(std::string("native exception: ") + exception.what());
        sink(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = failedReport("unknown native exception");
        sink(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRunExecutionMode(
    JNIEnv* env,
    jobject /* receiver */,
    jint executionMode,
    jint batchSize,
    jint dimension,
    jint hiddenDimension,
    jint outputDimension,
    jint steps,
    jint warmupSteps,
    jfloat learningRate,
    jlong seed,
    jint sampleCount,
    jint epochs,
    jint measuredSteps,
    jint correctnessInterval,
    jboolean benchmarkMode,
    jint seedSelectionMode,
    jint trainingStabilityMode,
    jint depthPairInitMode,
    jint checkpointSelectionMode,
    jboolean diagnosticTrajectory,
    jstring diagnosticCheckpointDir,
    jint diagnosticResumeStep,
    jint diagnosticCheckpointInterval,
    jobject progressCallback) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        const auto report = failedReport("a benchmark is already running");
        logcat(report);
        return toJavaString(env, report);
    }
    RunningGuard guard;
    beginNativeCall();

    jmethodID progressMethod = nullptr;
    if (progressCallback != nullptr) {
        jclass callbackClass = env->GetObjectClass(progressCallback);
        if (callbackClass != nullptr) {
            progressMethod = env->GetMethodID(
                callbackClass, "onNativeProgress", "(Ljava/lang/String;)V");
            env->DeleteLocalRef(callbackClass);
        }
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            progressMethod = nullptr;
        }
    }
    bool callbackEnabled = progressCallback != nullptr && progressMethod != nullptr;
    auto sink = [&](const std::string& message) {
        logcat(message);
        if (!callbackEnabled) return;
        jstring javaMessage = toJavaString(env, message);
        if (javaMessage == nullptr) {
            callbackEnabled = false;
            return;
        }
        env->CallVoidMethod(progressCallback, progressMethod, javaMessage);
        env->DeleteLocalRef(javaMessage);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            callbackEnabled = false;
            logcat("progress_callback_error=true");
        }
    };

    phonelm::TrainingConfig config;
    config.backend = phonelm::BackendKind::CPU;
    config.batchSize = batchSize;
    config.dimension = dimension;
    config.hiddenDimension = hiddenDimension;
    // The Nicopedia HTP path reads its FFN width separately so its checkpoint
    // identity does not borrow generic-MNN semantics at the call site.
    config.feedForwardDimension = hiddenDimension;
    config.outputDimension = outputDimension;
    config.steps = steps;
    config.warmupSteps = warmupSteps;
    config.learningRate = learningRate;
    config.seed = static_cast<std::uint64_t>(seed);
    config.sampleCount = sampleCount;
    config.epochs = epochs;
    config.measuredSteps = measuredSteps;
    config.correctnessInterval = correctnessInterval;
    config.benchmarkMode = benchmarkMode == JNI_TRUE;
    config.seedSelectionMode = static_cast<int>(seedSelectionMode);
    config.trainingStabilityMode = static_cast<int>(trainingStabilityMode);
    config.depthPairInitMode = static_cast<int>(depthPairInitMode);
    config.checkpointSelectionMode = static_cast<int>(checkpointSelectionMode);
    config.diagnosticTrajectory = diagnosticTrajectory == JNI_TRUE;
    config.diagnosticResumeStep = static_cast<int>(diagnosticResumeStep);
    config.diagnosticCheckpointInterval =
        static_cast<int>(diagnosticCheckpointInterval);
    if (diagnosticCheckpointDir != nullptr) {
        const char* chars = env->GetStringUTFChars(diagnosticCheckpointDir, nullptr);
        if (chars != nullptr) {
            config.diagnosticCheckpointDir = chars;
            env->ReleaseStringUTFChars(diagnosticCheckpointDir, chars);
        }
    }

    try {
        const auto report = phonelm::TrainingEngine::run(
            static_cast<phonelm::ExecutionMode>(executionMode), config, gStopRequested, sink);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = failedReport(std::string("native mode exception: ") + exception.what());
        sink(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = failedReport("unknown native mode exception");
        sink(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRunNicopediaDivergenceLocalization(
    JNIEnv* env,
    jobject /* receiver */,
    jstring checkpointPath,
    jlong seed,
    jint layers,
    jstring tapScope,
    jint diagnosticLayerIndex) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return toJavaString(env, failedReport("a benchmark is already running"));
    }
    RunningGuard guard;
    beginNativeCall();

    std::string checkpoint, scopeString;
    if (checkpointPath) {
        const char* chars = env->GetStringUTFChars(checkpointPath, nullptr);
        if (chars) { checkpoint = chars; env->ReleaseStringUTFChars(checkpointPath, chars); }
    }
    if (tapScope) {
        const char* chars = env->GetStringUTFChars(tapScope, nullptr);
        if (chars) { scopeString = chars; env->ReleaseStringUTFChars(tapScope, chars); }
    }
    if (scopeString.empty()) scopeString = "NONE";
    if (checkpoint.empty()) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
            "failure_classification=CHECKPOINT_UNAVAILABLE\n"
            "error=checkpoint_path_required\n");
    }

    phonelm::tiny_lm::Config config;
    config.vocabularySize = 256;
    config.tokens = 32;
    config.dimension = 16;
    config.feedForwardDimension = 32;
    config.numLayers = static_cast<uint32_t>(layers > 0 ? layers : 6);
    config.numHeads = 2;
    phonelm::nicopedia_gen::GenerateConfig generateConfig;
    generateConfig.samplingSeed = static_cast<std::uint64_t>(seed);
    generateConfig.diagnosticTapScope = scopeString;
    generateConfig.diagnosticLayerIndex =
        diagnosticLayerIndex >= 0 ? static_cast<uint32_t>(diagnosticLayerIndex)
                                  : 0xffffffffu;

    auto sink = [&](const std::string& message) { logcat(message); };
    try {
        const auto report = phonelm::qnn::runNicopediaHtpDivergenceLocalization(
            config, checkpoint, generateConfig, sink);
        logcat(report);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = std::string("NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\nerror=") +
                            exception.what();
        logcat(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = std::string("NICOPEDIA_HTP_DIVERGENCE_LOCALIZATION\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\n"
                                        "error=unknown");
        logcat(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRunNicopediaGenerate(
    JNIEnv* env,
    jobject /* receiver */,
    jstring checkpointPath,
    jstring promptPath,
    jlong seed,
    jint layers,
    jint tokens,
    jint dimension,
    jint feedForwardDimension,
    jint maxNewBytes,
    jstring generateMode,
    jfloat temperature,
    jint topK,
    jlong samplingSeed,
    jstring gatePolicy,
    jint htpGraphPrecisionMode,
    jint htpGraphPrecisionCompensation,
    jint htpGraphWeightsPacking,
    jint htpGraphAdvancedActivationFusion,
    jint htpContextGraphSplitting,
    jboolean htpNativeTensorFp16) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return toJavaString(env, failedReport("a benchmark is already running"));
    }
    RunningGuard guard;
    beginNativeCall();

    std::string checkpoint, prompt, modeString, policyString;
    if (checkpointPath) {
        const char* chars = env->GetStringUTFChars(checkpointPath, nullptr);
        if (chars) { checkpoint = chars; env->ReleaseStringUTFChars(checkpointPath, chars); }
    }
    if (promptPath) {
        const char* chars = env->GetStringUTFChars(promptPath, nullptr);
        if (chars) { prompt = chars; env->ReleaseStringUTFChars(promptPath, chars); }
    }
    if (generateMode) {
        const char* chars = env->GetStringUTFChars(generateMode, nullptr);
        if (chars) { modeString = chars; env->ReleaseStringUTFChars(generateMode, chars); }
    }
    if (gatePolicy) {
        const char* chars = env->GetStringUTFChars(gatePolicy, nullptr);
        if (chars) { policyString = chars; env->ReleaseStringUTFChars(gatePolicy, chars); }
    }
    if (policyString.empty()) policyString = "legacy";
    if (checkpoint.empty()) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
            "failure_classification=CHECKPOINT_UNAVAILABLE\n"
            "error=checkpoint_path_required\n");
    }
    if (prompt.empty()) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
            "failure_classification=PROMPT_EMPTY\n"
            "error=prompt_path_required\n");
    }
    if (htpContextGraphSplitting < 0 || htpContextGraphSplitting > 2) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
            "failure_classification=APP_CONFIGURATION_VALIDATION\n"
            "error=phonelm.htp_context_graph_splitting must be 0, 1 or 2\n");
    }

    // 256 is the cache context range max; out-of-range falls back to the
    // legacy T=32.
    if (tokens < 1 || tokens > 256) tokens = 32;
    phonelm::tiny_lm::Config config;
    config.vocabularySize = 256;
    config.tokens = static_cast<uint32_t>(tokens);
    if (dimension < 2 || dimension > 256 || dimension % 2 != 0 ||
        feedForwardDimension < 2 || feedForwardDimension > 1024) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
            "failure_classification=APP_CONFIGURATION_VALIDATION\n"
            "error=dimension/feed_forward_dimension out of supported range\n");
    }
    config.dimension = static_cast<uint32_t>(dimension);
    config.feedForwardDimension = static_cast<uint32_t>(feedForwardDimension);
    config.numLayers = static_cast<uint32_t>(layers > 0 ? layers : 6);
    config.numHeads = 2;
    phonelm::TrainingConfig trainingConfig;
    trainingConfig.seed = static_cast<std::uint64_t>(seed);
    trainingConfig.epochs = static_cast<int>(config.numLayers);
    trainingConfig.measuredSteps = 2;
    trainingConfig.steps = maxNewBytes > 0 ? maxNewBytes : 64;
    phonelm::nicopedia_gen::GenerateConfig generateConfig;
    generateConfig.maxNewBytes =
        static_cast<uint32_t>(maxNewBytes > 0 ? maxNewBytes : 64);
    generateConfig.greedy = modeString != "sample";
    generateConfig.temperature = temperature;
    generateConfig.topK = static_cast<uint32_t>(topK);
    generateConfig.samplingSeed = static_cast<std::uint64_t>(samplingSeed);
    generateConfig.gatePolicy = policyString;
    generateConfig.htpGraphPrecisionMode =
        htpGraphPrecisionMode > 0 ? static_cast<uint32_t>(htpGraphPrecisionMode)
                                  : 0;
    generateConfig.htpGraphPrecisionCompensation =
        htpGraphPrecisionCompensation > 0
            ? static_cast<uint32_t>(htpGraphPrecisionCompensation)
            : 0;
    generateConfig.htpGraphWeightsPacking =
        htpGraphWeightsPacking > 0
            ? static_cast<uint32_t>(htpGraphWeightsPacking)
            : 0;
    generateConfig.htpGraphAdvancedActivationFusion =
        htpGraphAdvancedActivationFusion > 0
            ? static_cast<uint32_t>(htpGraphAdvancedActivationFusion)
            : 0;
    generateConfig.htpContextGraphSplitting =
        static_cast<uint32_t>(htpContextGraphSplitting);
    generateConfig.htpNativeTensorFp16 = htpNativeTensorFp16 == JNI_TRUE;

    auto sink = [&](const std::string& message) { logcat(message); };
    try {
        const auto report = phonelm::qnn::runNicopediaHtpGeneration(
            config, checkpoint, prompt, trainingConfig, generateConfig, sink);
        logcat(report);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = std::string("NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\nerror=") +
                            exception.what();
        logcat(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = std::string("NICOPEDIA_HTP_GENERATION\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\n"
                                        "error=unknown");
        logcat(report);
        return toJavaString(env, report);
    }
}

// HTP-native held-out evaluation: loads the fixed checkpoint step from the
// app-private directory (NPRTCKPTV1 or NPRTCKPTV2) together with
// validation.bin / development.bin caches and teacher-forces them through
// the HTP forward graph, using the new NICOPEDIA_EVAL execution mode.
extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRunNicopediaEvaluate(
    JNIEnv* env,
    jobject /* receiver */,
    jstring checkpointDir,
    jlong seed,
    jint layers,
    jint heads,
    jint tokens,
    jint dimension,
    jint feedForwardDimension,
    jint checkpointStep,
    jint validationChunks,
    jint developmentChunks) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return toJavaString(env, failedReport("a benchmark is already running"));
    }
    RunningGuard guard;
    beginNativeCall();

    std::string dir;
    if (checkpointDir) {
        const char* chars = env->GetStringUTFChars(checkpointDir, nullptr);
        if (chars) { dir = chars; env->ReleaseStringUTFChars(checkpointDir, chars); }
    }
    if (dir.empty()) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_EVAL\nstatus=FAILED\n"
            "failure_classification=APP_CONFIGURATION_VALIDATION\n"
            "error=checkpoint_dir_required\n");
    }
    if (dimension < 2 || dimension > 256 || dimension % 2 != 0 ||
        feedForwardDimension < 2 || feedForwardDimension > 1024) {
        return toJavaString(
            env,
            "NICOPEDIA_HTP_EVAL\nstatus=FAILED\n"
            "failure_classification=APP_CONFIGURATION_VALIDATION\n"
            "error=dimension/feed_forward_dimension out of supported range\n");
    }
    // 256 is the cache context range max; out-of-range falls back to the
    // legacy T=32.
    if (tokens < 1 || tokens > 256) tokens = 32;
    phonelm::TrainingConfig config;
    config.seed = static_cast<std::uint64_t>(seed);
    config.epochs = layers > 0 ? layers : 6;
    config.measuredSteps = heads > 0 ? heads : 2;
    config.dimension = dimension;
    config.feedForwardDimension = feedForwardDimension;
    config.sampleCount = tokens;
    config.diagnosticResumeStep = checkpointStep;
    config.steps = validationChunks > 0 ? validationChunks : 8192;
    config.batchSize = developmentChunks > 0 ? developmentChunks : 16384;
    config.diagnosticCheckpointDir = dir;
    auto sink = [&](const std::string& message) { logcat(message); };
    try {
        const auto report = phonelm::TrainingEngine::run(
            phonelm::ExecutionMode::QNN_HTP_TINY_LANGUAGE_MODEL_NICOPEDIA_EVAL,
            config, gStopRequested, sink);
        logcat(report);
        return toJavaString(env, report);
    } catch (const std::exception& exception) {
        const auto report = std::string("NICOPEDIA_HTP_EVAL\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\nerror=") +
                            exception.what();
        logcat(report);
        return toJavaString(env, report);
    } catch (...) {
        const auto report = std::string("NICOPEDIA_HTP_EVAL\nstatus=FAILED\n"
                                        "failure_classification=JNI_EXCEPTION\n"
                                        "error=unknown");
        logcat(report);
        return toJavaString(env, report);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeReplayFirstNonfiniteCheckpoint(
    JNIEnv* env,
    jobject /* receiver */,
    jbyteArray payload,
    jint repeatCount,
    jint tapSet) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return toJavaString(env, failedReport("a benchmark is already running"));
    }
    RunningGuard guard;
    if (payload == nullptr) {
        return toJavaString(
            env,
            "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
            "failure_classification=CHECKPOINT_UNAVAILABLE\n"
            "error=null_checkpoint_payload\n");
    }
    const jsize length = env->GetArrayLength(payload);
    std::vector<std::uint8_t> bytes(static_cast<size_t>(length));
    if (length > 0) {
        env->GetByteArrayRegion(
            payload, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            return toJavaString(
                env,
                "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
                "failure_classification=CHECKPOINT_DECODE\n"
                "error=jni_payload_copy_failed\n");
        }
    }
    if (tapSet < static_cast<jint>(
                     phonelm::qnn::TinyTransformerTrainingTapSet::NONE) ||
        tapSet > static_cast<jint>(
                     phonelm::qnn::TinyTransformerTrainingTapSet::ALL_INTERNAL)) {
        return toJavaString(
            env,
            "FIRST_NONFINITE_REPLAY\nstatus=FAILED\n"
            "failure_classification=APP_CONFIGURATION_VALIDATION\n"
            "error=invalid_tap_set\n");
    }
    try {
        return toJavaString(
            env,
            phonelm::qnn::replayFirstNonfiniteCheckpoint(
                bytes, static_cast<std::uint32_t>(repeatCount),
                static_cast<phonelm::qnn::TinyTransformerTrainingTapSet>(tapSet)));
    } catch (const std::exception& exception) {
        return toJavaString(
            env,
            failedReport(std::string("checkpoint replay exception: ") +
                         exception.what()));
    } catch (...) {
        return toJavaString(
            env, failedReport("unknown checkpoint replay exception"));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeRequestStop(
    JNIEnv* /* env */, jobject /* receiver */) {
    {
        std::lock_guard<std::mutex> lock(gStopStateMutex);
        if (gNativeCallEntered) {
            gStopRequested.store(true, std::memory_order_release);
        } else {
            // Preserve a stop that races the JNI entry.  The Android backend
            // explicitly clears this flag when it cancels before entering JNI.
            gPendingStop = true;
        }
    }
    logcat("stop_requested=true");
}

extern "C" JNIEXPORT void JNICALL
Java_com_yuubinnkyoku_phonelm_NativeBridge_nativeClearStop(
    JNIEnv* /* env */, jobject /* receiver */) {
    std::lock_guard<std::mutex> lock(gStopStateMutex);
    if (!gRunning.load(std::memory_order_acquire) && !gNativeCallEntered) {
        gPendingStop = false;
        gStopRequested.store(false, std::memory_order_release);
    }
}

JNIEXPORT jint JNI_OnLoad(JavaVM* /* vm */, void* /* reserved */) {
    logcat("jni_on_load=true");
    return JNI_VERSION_1_6;
}
