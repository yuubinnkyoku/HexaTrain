package com.yuubinnkyoku.phonelm

/** Progress emitted at the JNI boundary.  Native code owns the measurements; UI code never
 * derives progress by inspecting QNN buffers or execution state. */
sealed interface RunProgress {
    data class Started(val kind: String, val totalSteps: Long?) : RunProgress
    data class PhaseChanged(val phase: String) : RunProgress
    data class Step(
        val completed: Long,
        val total: Long?,
        val loss: Double? = null,
    ) : RunProgress
    data class Completed(val metric: String?) : RunProgress
    data class Failed(val reason: String) : RunProgress
    data object Cancelled : RunProgress
}

internal object NativeProgressParser {
    fun parse(message: String): RunProgress? {
        val values = message.lineSequence()
            .mapNotNull { line -> line.substringBefore('=', "").takeIf { it.isNotEmpty() }?.let { it to line.substringAfter('=') } }
            .toMap()
        val step = values["step"]?.toLongOrNull()
        val warmup = values["warmup_step"]?.toLongOrNull()
        if (step != null || warmup != null) {
            return RunProgress.Step(
                completed = step ?: warmup!!,
                total = values["steps"]?.toLongOrNull(),
                loss = values["loss"]?.toDoubleOrNull(),
            )
        }
        if (message.startsWith("RESULT") || (message.contains('\n') && "status" in values)) {
            return when (values["status"]) {
                "SUCCESS" -> RunProgress.Completed(values["final_loss"]?.let { "loss $it" })
                "CANCELLED" -> RunProgress.Cancelled
                else -> RunProgress.Failed(values["error"].orEmpty())
            }
        }
        return null
    }
}

internal class ProgressUpdateThrottle(private val minimumIntervalMs: Long = 1_000L) {
    private var lastAtMs = Long.MIN_VALUE
    private var lastPercent = -1
    private var lastPhase: String? = null

    fun shouldPost(event: RunProgress, nowMs: Long): Boolean = when (event) {
        is RunProgress.Started, is RunProgress.Completed, is RunProgress.Failed, RunProgress.Cancelled -> true
        is RunProgress.PhaseChanged -> event.phase != lastPhase || update(nowMs)
        is RunProgress.Step -> {
            val percent = event.total?.takeIf { it > 0 }?.let { (event.completed * 100 / it).toInt() }
            val changed = percent != null && percent > lastPercent
            if (changed) lastPercent = percent
            changed || update(nowMs)
        }
    }.also {
        if (it) {
            lastAtMs = nowMs
            if (event is RunProgress.PhaseChanged) lastPhase = event.phase
        }
    }

    private fun update(nowMs: Long) = nowMs - lastAtMs >= minimumIntervalMs
}
