package com.yuubinnkyoku.phonelm

import android.content.Context
import java.io.File
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.util.UUID

data class HeadlessStatus(
    val runId: String,
    val suite: String,
    val status: String,
    val phase: String,
    val currentTest: String,
    val completedTests: Int,
    val totalTests: Int,
    val result: String = "",
    val failureCode: String = "",
    val reportRelativePath: String = "",
    val startTime: Long = System.currentTimeMillis(),
    val lastHeartbeat: Long = System.currentTimeMillis(),
    val pid: Int = android.os.Process.myPid(),
)

class HeadlessTestState private constructor(private val root: File) {
    private val statusFile = File(root, "status.json")
    private val lockFile = File(root, "single-flight.lock")

    fun acquire(): LockResult {
        root.mkdirs()
        val channel = FileChannel.open(lockFile.toPath(), StandardOpenOption.CREATE, StandardOpenOption.WRITE)
        val lock = try { channel.tryLock() } catch (_: Exception) { null }
        if (lock == null) {
            channel.close()
            return LockResult(null, readExisting())
        }
        return LockResult(Lease(channel, lock), null)
    }

    @Synchronized
    fun write(status: HeadlessStatus) {
        root.mkdirs()
        val text = """{"schema_version":1,"run_id":"${escape(status.runId)}","suite":"${escape(status.suite)}","status":"${escape(status.status)}","pid":${status.pid},"start_time":${status.startTime},"last_heartbeat":${status.lastHeartbeat},"current_phase":"${escape(status.phase)}","current_test":"${escape(status.currentTest)}","completed_tests":${status.completedTests},"total_tests":${status.totalTests},"result":"${escape(status.result)}","failure_code":"${escape(status.failureCode)}","report_relative_path":"${escape(status.reportRelativePath)}"}"""
        val encoded = text.toByteArray(Charsets.UTF_8)
        require(encoded.size <= 4096) { "headless status exceeds 4096-byte bound" }
        // A fixed-size record makes polling robust and leaves no partial JSON visible after a crash.
        val bounded = ByteArray(4096) { ' '.code.toByte() }
        encoded.copyInto(bounded)
        val temporary = File(root, "status-${UUID.randomUUID()}.tmp")
        java.io.FileOutputStream(temporary).use { output ->
            output.write(bounded)
            output.flush()
            output.fd.sync()
        }
        atomicReplace(temporary, statusFile)
    }

    fun writeReport(runId: String, report: String): String {
        val reports = File(root, "reports").also { it.mkdirs() }
        val file = File(reports, "$runId-result.txt")
        val temporary = File(reports, "$runId-result.tmp")
        java.io.FileOutputStream(temporary).use { output ->
            output.write(report.toByteArray(Charsets.UTF_8))
            output.flush()
            output.fd.sync()
        }
        atomicReplace(temporary, file)
        return "headless/reports/${file.name}"
    }

    private fun atomicReplace(source: File, target: File) {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun readExisting(): String = runCatching { statusFile.readText(Charsets.UTF_8).trim() }.getOrDefault("{}")
    private fun escape(value: String) = value.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    class Lease(private val channel: FileChannel, private val lock: FileLock) : AutoCloseable {
        override fun close() { runCatching { lock.release() }; runCatching { channel.close() } }
    }
    data class LockResult(val lease: Lease?, val existingStatus: String?)

    companion object { fun forContext(context: Context) = HeadlessTestState(File(context.filesDir, "headless")) }
}
