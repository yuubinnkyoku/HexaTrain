package com.yuubinnkyoku.phonelm

import android.content.Context
import android.system.Os
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/** Process-local QNN setup shared by the UI and headless instrumentation. */
object QnnEnvironment {
    @Synchronized
    fun prepare(context: Context) {
        if (!BuildConfig.PHONELM_QNN_ENABLED) return
        val assets = context.assets
        val assetName = "qnn/libQnnHtpV81Skel.so"
        check(assets.list("qnn")?.contains("libQnnHtpV81Skel.so") == true) {
            "QNN HTP initialization blocked: V81 Skel asset is missing"
        }
        val metadata = assets.open("qnn/qairt.properties").bufferedReader().useLines { lines ->
            lines.mapNotNull { line -> line.indexOf('=').takeIf { it > 0 }?.let { line.substring(0, it) to line.substring(it + 1) } }.toMap()
        }
        val buildId = checkNotNull(metadata["buildId"]) { "QNN asset metadata has no build ID" }
        val expectedHash = checkNotNull(metadata["skelSha256"]) { "QNN asset metadata has no Skel hash" }
        check(buildId == BuildConfig.QAIRT_BUILD_ID) { "QNN asset build ID does not match the app" }
        val root = File(context.filesDir, "qnn-dsp")
        val directory = File(root, buildId)
        check(directory.canonicalPath.startsWith(root.canonicalPath + File.separator)) { "QNN DSP directory escaped app-private root" }
        check(directory.mkdirs() || directory.isDirectory) { "Cannot create QNN DSP directory" }
        val target = File(directory, "libQnnHtpV81Skel.so")
        var actualHash = if (target.isFile) sha256(target) else "MISSING"
        var action = "reused"
        if (!actualHash.equals(expectedHash, ignoreCase = true)) {
            val temporary = File.createTempFile("libQnnHtpV81Skel.so.", ".part", directory)
            try {
                assets.open(assetName).use { input -> FileOutputStream(temporary).use { output -> input.copyTo(output); output.fd.sync() } }
                check(sha256(temporary).equals(expectedHash, ignoreCase = true)) { "QNN HTP Skel asset SHA-256 mismatch" }
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                action = "replaced"
            } finally { temporary.delete() }
            actualHash = sha256(target)
        }
        check(actualHash.equals(expectedHash, ignoreCase = true)) { "QNN HTP deployed Skel SHA-256 mismatch" }
        val entries = mutableListOf(directory.absolutePath)
        Os.getenv("ADSP_LIBRARY_PATH")?.split(';')?.filterTo(entries) { it.isNotBlank() }
        entries += listOf("/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/system/lib/rfsa/adsp")
        val libraryPath = entries.map(String::trim).filter(String::isNotBlank).distinct().joinToString(";")
        Os.setenv("ADSP_LIBRARY_PATH", libraryPath, true)
        Os.setenv("PHONELM_QNN_SKEL_DIR", directory.absolutePath, true)
        Os.setenv("PHONELM_QNN_SKEL_EXPECTED_SHA256", expectedHash, true)
        Os.setenv("PHONELM_QNN_SKEL_ACTUAL_SHA256", actualHash, true)
        Os.setenv("PHONELM_QNN_SKEL_ACTION", action, true)
        check(Os.getenv("ADSP_LIBRARY_PATH") == libraryPath) { "Failed to set process-local ADSP_LIBRARY_PATH" }
        Log.i("PhoneLMQnn", "qairt_build_id=$buildId qnn_skel_action=$action")
    }

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256"); val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        digest.digest().joinToString("") { "%02x".format(it) }
    }
}
