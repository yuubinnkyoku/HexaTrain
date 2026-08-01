#include "benchmark_runner.h"
#include "training_engine.h"
#include "qnn/qnn_transformer.h"

#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "PhoneLMBench";
std::atomic_bool gStopRequested{false};
std::atomic_bool gRunning{false};

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
    gStopRequested.store(false, std::memory_order_release);

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
    jboolean diagnosticTrajectory,
    jstring diagnosticCheckpointDir,
    jobject progressCallback) {
    bool expected = false;
    if (!gRunning.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        const auto report = failedReport("a benchmark is already running");
        logcat(report);
        return toJavaString(env, report);
    }
    RunningGuard guard;
    gStopRequested.store(false, std::memory_order_release);

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
    config.diagnosticTrajectory = diagnosticTrajectory == JNI_TRUE;
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
    gStopRequested.store(true, std::memory_order_release);
    logcat("stop_requested=true");
}

JNIEXPORT jint JNI_OnLoad(JavaVM* /* vm */, void* /* reserved */) {
    logcat("jni_on_load=true");
    return JNI_VERSION_1_6;
}
