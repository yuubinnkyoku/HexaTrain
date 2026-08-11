package com.yuubinnkyoku.phonelm

import java.util.EnumMap

/** Injectable monotonic clock so timing attribution is testable without Android APIs. */
fun interface TrainingClock {
    fun elapsedRealtimeMs(): Long
}

data class CpuProcessMetrics(
    val processCpuTimeMs: Long,
    val threadCount: Int? = null,
) {
    init { require(processCpuTimeMs >= 0) { "processCpuTimeMs must be non-negative" } }
}

/** Platform adapter boundary; an unavailable reading must be represented by null, never invented. */
fun interface CpuProcessMetricSource {
    fun read(): CpuProcessMetrics?
}

object UnavailableCpuProcessMetricSource : CpuProcessMetricSource {
    override fun read(): CpuProcessMetrics? = null
}

data class HtpActivityWindow(
    val startedAtMs: Long,
    val endedAtMs: Long,
    val executeDurationMs: Double? = null,
    val executeCount: Long? = null,
) {
    init {
        require(endedAtMs >= startedAtMs) { "HTP activity cannot end before it starts" }
        require(executeDurationMs == null || executeDurationMs.isFinite()) {
            "HTP execute duration must be finite"
        }
        require(executeDurationMs == null || executeDurationMs >= 0.0) {
            "HTP execute duration must be non-negative"
        }
        require(executeCount == null || executeCount >= 0) {
            "HTP execute count must be non-negative"
        }
    }

    val durationMs: Long get() = endedAtMs - startedAtMs

    /** This is an observation-window ratio, not hardware/NPU utilization. */
    val activityPercent: Double?
        get() = executeDurationMs?.let { duration ->
            if (durationMs <= 0L) null else (duration / durationMs * 100.0).coerceIn(0.0, 100.0)
        }
}

enum class TrainingOperationPhase { FORWARD, BACKWARD, ADAM, HOST }

enum class TimingBackend { HTP, CPU, UNAVAILABLE }

/** One native-reported phase sample. Missing values stay unavailable. */
data class PhaseTiming(
    val backend: TimingBackend,
    val qnnExecuteMs: Double? = null,
    val qnnExecuteCount: Long = 0L,
    val hostMs: Double? = null,
) {
    init {
        require(qnnExecuteMs == null || qnnExecuteMs.isFinite()) { "qnnExecuteMs must be finite" }
        require(qnnExecuteMs == null || qnnExecuteMs >= 0.0) { "qnnExecuteMs must be non-negative" }
        require(qnnExecuteCount >= 0L) { "qnnExecuteCount must be non-negative" }
        require(hostMs == null || hostMs.isFinite()) { "hostMs must be finite" }
        require(hostMs == null || hostMs >= 0.0) { "hostMs must be non-negative" }
        if (backend != TimingBackend.HTP) require(qnnExecuteMs == null && qnnExecuteCount == 0L) {
            "QNN timing requires an HTP backend"
        }
    }
}

data class TrainingTimingSample(
    val forward: PhaseTiming? = null,
    val backward: PhaseTiming? = null,
    val adam: PhaseTiming? = null,
    val host: PhaseTiming? = null,
) {
    fun entries(): Map<TrainingOperationPhase, PhaseTiming> = buildMap {
        forward?.let { put(TrainingOperationPhase.FORWARD, it) }
        backward?.let { put(TrainingOperationPhase.BACKWARD, it) }
        adam?.let { put(TrainingOperationPhase.ADAM, it) }
        host?.let { put(TrainingOperationPhase.HOST, it) }
    }

    /** Remove unverified HTP labels while preserving explicitly reported CPU/host work. */
    fun withoutHtpEvidence(): TrainingTimingSample = TrainingTimingSample(
        forward = forward.withoutHtpEvidence(),
        backward = backward.withoutHtpEvidence(),
        adam = adam.withoutHtpEvidence(),
        host = host.withoutHtpEvidence(),
    )

    private fun PhaseTiming?.withoutHtpEvidence(): PhaseTiming? = when {
        this == null -> null
        backend != TimingBackend.HTP -> this
        else -> PhaseTiming(TimingBackend.UNAVAILABLE)
    }
}

/** Running aggregate independent of UI refresh frequency. */
data class TrainingTimingAggregate(
    val current: TrainingTimingSample?,
    val average: TrainingTimingSample?,
    val cumulative: TrainingTimingSample?,
    val sampleCount: Long,
    val htpExecuteTimeMs: Double?,
    val htpExecuteCount: Long?,
    val cpuHostTimeMs: Double?,
    val checkpointIoMs: Double?,
) {
    init {
        require(sampleCount >= 0L) { "sampleCount must be non-negative" }
        listOf(htpExecuteTimeMs, cpuHostTimeMs, checkpointIoMs).forEach {
            require(it == null || it.isFinite() && it >= 0.0) { "aggregate time must be finite and non-negative" }
        }
        require(htpExecuteCount == null || htpExecuteCount >= 0L) { "htpExecuteCount must be non-negative" }
    }
}

/** Small, allocation-light accumulator used by a TrainingSession worker. */
class TimingAccumulator {
    private val sums = EnumMap<TrainingOperationPhase, PhaseTimingTotals>(TrainingOperationPhase::class.java)
    private val counts = EnumMap<TrainingOperationPhase, Long>(TrainingOperationPhase::class.java)
    private val htpCounts = EnumMap<TrainingOperationPhase, Long>(TrainingOperationPhase::class.java)
    private val cpuCounts = EnumMap<TrainingOperationPhase, Long>(TrainingOperationPhase::class.java)
    private var current: TrainingTimingSample? = null
    private var sampleCount = 0L
    private var htpExecuteTimeMs = 0.0
    private var htpExecuteCount = 0L
    private var cpuHostTimeMs = 0.0
    private var checkpointIoMs = 0.0
    private var hasHtpTime = false
    private var hasHtpCount = false
    private var hasCpuHostTime = false
    private var hasCheckpointIo = false

    fun add(sample: TrainingTimingSample?, checkpointIoMs: Double? = null) {
        if (sample != null) {
            current = sample
            sampleCount += 1L
            sample.entries().forEach { (phase, timing) ->
                val totals = sums.getOrPut(phase) { PhaseTimingTotals() }
                counts[phase] = (counts[phase] ?: 0L) + 1L
                when (timing.backend) {
                    TimingBackend.HTP -> htpCounts[phase] = (htpCounts[phase] ?: 0L) + 1L
                    TimingBackend.CPU -> cpuCounts[phase] = (cpuCounts[phase] ?: 0L) + 1L
                    TimingBackend.UNAVAILABLE -> Unit
                }
                timing.qnnExecuteMs?.let {
                    totals.qnnExecuteMs += it
                    htpExecuteTimeMs += it
                    hasHtpTime = true
                }
                totals.qnnExecuteCount += timing.qnnExecuteCount
                htpExecuteCount += timing.qnnExecuteCount
                if (timing.qnnExecuteCount > 0L) hasHtpCount = true
                timing.hostMs?.let {
                    totals.hostMs += it
                    if (phase == TrainingOperationPhase.HOST) {
                        cpuHostTimeMs += it
                        hasCpuHostTime = true
                    }
                }
            }
        }
        checkpointIoMs?.let {
            require(it.isFinite() && it >= 0.0) { "checkpoint I/O time must be finite and non-negative" }
            this.checkpointIoMs += it
            hasCheckpointIo = true
        }
    }

    fun snapshot(): TrainingTimingAggregate {
        fun build(average: Boolean): TrainingTimingSample? {
            if (sums.isEmpty()) return null
            val values = sums.mapValues { (phase, totals) ->
                val divisor = if (average) (counts[phase] ?: 1L).toDouble() else 1.0
                val htpSamples = htpCounts[phase] ?: 0L
                val cpuSamples = cpuCounts[phase] ?: 0L
                val backend = if (htpSamples > 0L && cpuSamples == 0L) {
                    TimingBackend.HTP
                } else if (cpuSamples > 0L && htpSamples == 0L) {
                    TimingBackend.CPU
                } else {
                    TimingBackend.UNAVAILABLE
                }
                PhaseTiming(
                    backend = backend,
                    qnnExecuteMs = totals.qnnExecuteMs.takeIf { backend == TimingBackend.HTP && it > 0.0 }?.div(divisor),
                    qnnExecuteCount = if (backend == TimingBackend.HTP && average) {
                        (totals.qnnExecuteCount.toDouble() / divisor).toLong()
                    } else if (backend == TimingBackend.HTP) {
                        totals.qnnExecuteCount
                    } else {
                        0L
                    },
                    hostMs = totals.hostMs.takeIf { it > 0.0 }?.div(divisor),
                )
            }
            return TrainingTimingSample(
                forward = values[TrainingOperationPhase.FORWARD],
                backward = values[TrainingOperationPhase.BACKWARD],
                adam = values[TrainingOperationPhase.ADAM],
                host = values[TrainingOperationPhase.HOST],
            )
        }
        return TrainingTimingAggregate(
            current = current,
            average = build(average = true),
            cumulative = build(average = false),
            sampleCount = sampleCount,
            htpExecuteTimeMs = htpExecuteTimeMs.takeIf { hasHtpTime },
            htpExecuteCount = htpExecuteCount.takeIf { hasHtpCount },
            cpuHostTimeMs = cpuHostTimeMs.takeIf { hasCpuHostTime },
            checkpointIoMs = checkpointIoMs.takeIf { hasCheckpointIo },
        )
    }

    private data class PhaseTimingTotals(
        var qnnExecuteMs: Double = 0.0,
        var qnnExecuteCount: Long = 0L,
        var hostMs: Double = 0.0,
    )
}

/** Structured timing record. HTP time is only reported when a backend explicitly supplies its window. */
data class TrainingTiming(
    val startedAtMs: Long,
    val endedAtMs: Long,
    val htpActivity: HtpActivityWindow? = null,
    val cpuAtStart: CpuProcessMetrics? = null,
    val cpuAtEnd: CpuProcessMetrics? = null,
    val currentStepMs: Long? = null,
    val averageStepMs: Double? = null,
    val cumulativeHtpActivityMs: Long? = null,
    val htpExecuteCount: Long? = null,
    val checkpointIoMs: Long? = null,
    val aggregate: TrainingTimingAggregate? = null,
) {
    init {
        require(endedAtMs >= startedAtMs) { "training cannot end before it starts" }
        require(currentStepMs == null || currentStepMs >= 0) { "currentStepMs must be non-negative" }
        require(averageStepMs == null || averageStepMs.isFinite() && averageStepMs >= 0) { "averageStepMs must be finite and non-negative" }
        require(cumulativeHtpActivityMs == null || cumulativeHtpActivityMs >= 0) { "cumulativeHtpActivityMs must be non-negative" }
        require(htpExecuteCount == null || htpExecuteCount >= 0) { "htpExecuteCount must be non-negative" }
        require(checkpointIoMs == null || checkpointIoMs >= 0) { "checkpointIoMs must be non-negative" }
    }
    val elapsedMs: Long get() = endedAtMs - startedAtMs
    val cpuProcessDeltaMs: Long?
        get() = if (cpuAtStart != null && cpuAtEnd != null && cpuAtEnd.processCpuTimeMs >= cpuAtStart.processCpuTimeMs) {
            cpuAtEnd.processCpuTimeMs - cpuAtStart.processCpuTimeMs
        } else null

    /** Process CPU time divided by wall time; null when either measurement is unavailable. */
    val cpuProcessPercent: Double?
        get() = cpuProcessDeltaMs?.let { delta ->
            if (elapsedMs <= 0L) null else (delta.toDouble() / elapsedMs * 100.0).coerceAtLeast(0.0)
        }
}
