import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val phoneLmEnableQnn = providers.gradleProperty("phonelm.enableQnn").orElse("false")
val qairtSdkRoot = providers.gradleProperty("qairt.sdkRoot").orElse("")
val expectedQairtBuildId = providers.gradleProperty("qairt.expectedBuildId")
val qairtPolicyText = rootProject.file("scripts/qairt_version.ps1").readText()
fun pinnedQairtValue(name: String): String =
    Regex("""(?m)^\${'$'}$name\s*=\s*'([^']+)'\s*$""")
        .find(qairtPolicyText)?.groupValues?.get(1)
        ?: error("Missing $name in scripts/qairt_version.ps1")
val pinnedQairtSdkRoot = pinnedQairtValue("PhoneLmQairtSdkRoot")
val pinnedQairtBuildId = pinnedQairtValue("PhoneLmQairtBuildId")
val androidNdkVersion = "26.2.11394342"
val htpArchitecture = "V81"

fun File.sha256(): String = inputStream().use { input ->
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        digest.update(buffer, 0, count)
    }
    digest.digest().joinToString("") { "%02x".format(it) }
}

data class QairtMetadata(
    val version: String,
    val buildId: String,
    val qnnApiVersion: String,
    val skelSha256: String,
)

fun inspectQairt(): QairtMetadata {
    val expectedBuildId = expectedQairtBuildId.orNull?.takeIf { it.isNotBlank() }
        ?: error("QNN build requires explicit -Pqairt.expectedBuildId")
    require(expectedBuildId == pinnedQairtBuildId) {
        "QAIRT Build ID differs from scripts/qairt_version.ps1"
    }
    val expectedVersion = expectedBuildId.substringBeforeLast('.')
    val root = file(qairtSdkRoot.get())
    require(root.canonicalFile == file(pinnedQairtSdkRoot).canonicalFile) {
        "QAIRT SDK root differs from scripts/qairt_version.ps1; fallback is forbidden"
    }
    require(root.isDirectory) { "QAIRT SDK root does not exist: $root" }
    val yaml = root.resolve("sdk.yaml").readText()
    val header = root.resolve("include/QNN/QnnSdkBuildId.h").readText()
    val version = Regex("(?m)^version:\\s*(\\S+)").find(yaml)!!.groupValues[1]
    val build = Regex("(?m)^build_id:\\s*(\\S+)").find(yaml)!!.groupValues[1]
    val headerId = Regex("QNN_SDK_BUILD_ID\\s+\"v([^\"]+)\"").find(header)!!.groupValues[1]
    val yamlId = "$version.$build"
    require(headerId == yamlId) { "Mixed QAIRT distribution: sdk.yaml=$yamlId, header=$headerId" }
    require(headerId == expectedBuildId) {
        "Unsupported QAIRT distribution: expected=$expectedBuildId, actual=$headerId"
    }
    require(version == expectedVersion) {
        "Unsupported QAIRT version: expected=$expectedVersion, actual=$version"
    }
    require(Regex("(?m)^android-ndk:\\s*r26c\\s*$").containsMatchIn(yaml)) {
        "QAIRT 2.48 metadata does not declare the expected Android NDK r26c"
    }
    listOf("QnnCommon.h", "QnnInterface.h", "QnnSdkBuildId.h", "QnnBackend.h",
        "QnnDevice.h", "QnnContext.h", "QnnGraph.h", "HTP/QnnHtpDevice.h").forEach {
        require(root.resolve("include/QNN/$it").isFile) { "Incomplete QAIRT headers: $it" }
    }
    listOf("libQnnSystem.so", "libQnnCpu.so", "libQnnHtp.so", "libQnnHtpPrepare.so",
        "libQnnHtpV81Stub.so").forEach {
        require(root.resolve("lib/aarch64-android/$it").isFile) { "Incomplete QAIRT distribution: $it" }
    }
    val skel = root.resolve("lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so")
    require(skel.isFile) { "Incomplete QAIRT distribution: libQnnHtpV81Skel.so" }
    val common = root.resolve("include/QNN/QnnCommon.h").readText()
    fun macro(name: String) = Regex("(?m)^#define\\s+$name\\s+(\\d+)")
        .find(common)?.groupValues?.get(1) ?: error("Missing $name")
    val api = listOf("QNN_API_VERSION_MAJOR", "QNN_API_VERSION_MINOR", "QNN_API_VERSION_PATCH")
        .joinToString(".") { macro(it) }
    logger.lifecycle("PhoneLM QAIRT SDK root: ${root.absolutePath}")
    logger.lifecycle("PhoneLM QAIRT version: $version")
    logger.lifecycle("PhoneLM QAIRT build ID: $headerId")
    logger.lifecycle("PhoneLM QNN API version: $api")
    logger.lifecycle("PhoneLM Android NDK version: $androidNdkVersion (QAIRT requirement: r26c)")
    logger.lifecycle("PhoneLM target ABI: arm64-v8a")
    logger.lifecycle("PhoneLM HTP architecture: $htpArchitecture")
    return QairtMetadata(version, headerId, api, skel.sha256())
}
val selectedQairt = if (phoneLmEnableQnn.get().toBoolean()) inspectQairt() else null
val selectedQairtBuildId = selectedQairt?.buildId ?: "DISABLED"
val qnnJniDir = layout.buildDirectory.dir("generated/qnnJni/arm64-v8a")
val qnnDspAssetDir = layout.buildDirectory.dir("generated/qnnDspAssets/qnn")
val stageQnnDspAsset by tasks.registering(Sync::class) {
    onlyIf { phoneLmEnableQnn.get().toBoolean() }
    from(provider { file("${qairtSdkRoot.get()}/lib/hexagon-v81/unsigned") }) {
        include("libQnnHtpV81Skel.so")
    }
    into(qnnDspAssetDir)
    doLast {
        val metadata = selectedQairt ?: return@doLast
        qnnDspAssetDir.get().file("qairt.properties").asFile.writeText(
            "version=${metadata.version}\n" +
                "buildId=${metadata.buildId}\n" +
                "qnnApiVersion=${metadata.qnnApiVersion}\n" +
                "htpArchitecture=$htpArchitecture\n" +
                "skelSha256=${metadata.skelSha256}\n",
        )
    }
}
val stageQnnJni by tasks.registering(Sync::class) {
    onlyIf { phoneLmEnableQnn.get().toBoolean() }
    from(provider { file("${qairtSdkRoot.get()}/lib/aarch64-android") }) {
        include("libQnnSystem.so", "libQnnCpu.so", "libQnnHtp.so", "libQnnHtpPrepare.so", "libQnnHtpV81Stub.so")
    }
    into(qnnJniDir)
}

android {
    namespace = "com.yuubinnkyoku.phonelm"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }
    // Match the Android NDK declared by the selected QAIRT distribution.
    ndkVersion = androidNdkVersion

    buildFeatures {
        buildConfig = true
        compose = true
    }

    defaultConfig {
        applicationId = "com.yuubinnkyoku.phonelm"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("boolean", "PHONELM_QNN_ENABLED", phoneLmEnableQnn.get())
        buildConfigField("String", "QAIRT_BUILD_ID", "\"$selectedQairtBuildId\"")
        buildConfigField("String", "HTP_ARCHITECTURE", "\"$htpArchitecture\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
            debugSymbolLevel = "FULL"
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DPHONELM_ENABLE_QNN=${phoneLmEnableQnn.get()}",
                    "-DQAIRT_SDK_ROOT=${qairtSdkRoot.get()}",
                    "-DPHONELM_EXPECTED_QAIRT_BUILD_ID=$selectedQairtBuildId",
                )
                targets += listOf("phonelm_native")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            isJniDebuggable = true
        }
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += setOf("**/libMNN.so", "**/libphonelm_native.so")
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }

    // SDK binaries remain outside Git. A QNN-enabled local build packages the
    // installed SDK's Android libraries so dlopen can resolve the selected
    // backend and the device-specific HTP transport at runtime.
    if (phoneLmEnableQnn.get().toBoolean()) {
        require(qairtSdkRoot.get().isNotBlank()) { "qairt.sdkRoot is required when QNN is enabled" }
        sourceSets.getByName("main").jniLibs.srcDir(layout.buildDirectory.dir("generated/qnnJni"))
        sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("generated/qnnDspAssets"))
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("JniLibFolders") }
    .configureEach { dependsOn(stageQnnJni) }
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(stageQnnDspAsset) }

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2025.12.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3:1.4.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
