package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingTimingTest {
    @Test fun runningAverageAndCumulativeAreIndependentOfUiRefresh() {
        val accumulator = TimingAccumulator()
        accumulator.add(
            TrainingTimingSample(
                forward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 10.0, qnnExecuteCount = 1),
                host = PhaseTiming(TimingBackend.CPU, hostMs = 2.0),
            ),
        )
        accumulator.add(
            TrainingTimingSample(
                forward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 30.0, qnnExecuteCount = 2),
                host = PhaseTiming(TimingBackend.CPU, hostMs = 4.0),
            ),
        )

        val result = accumulator.snapshot()
        assertEquals(2L, result.sampleCount)
        assertEquals(20.0, result.average?.forward?.qnnExecuteMs ?: -1.0, 0.0)
        assertEquals(40.0, result.cumulative?.forward?.qnnExecuteMs ?: -1.0, 0.0)
        assertEquals(3L, result.htpExecuteCount)
        assertEquals(40.0, result.htpExecuteTimeMs ?: -1.0, 0.0)
        assertEquals(6.0, result.cpuHostTimeMs ?: -1.0, 0.0)
    }

    @Test fun htpActivityIsObservationWindowRatioNotUtilization() {
        val window = HtpActivityWindow(100, 1_100, executeDurationMs = 250.0, executeCount = 4)
        assertEquals(25.0, window.activityPercent ?: -1.0, 0.0)
        assertEquals(4L, window.executeCount)
    }

    @Test fun unavailableNativeTimingDoesNotInventValues() {
        val timing = TrainingTiming(0, 100)
        assertNull(timing.htpActivity)
        assertNull(timing.cpuProcessPercent)
        assertTrue(timing.aggregate == null)
    }

    @Test fun mixedHtpAndCpuPhaseIsUnavailableWithoutInvalidQnnFields() {
        val accumulator = TimingAccumulator()
        accumulator.add(TrainingTimingSample(forward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 4.0, qnnExecuteCount = 1)))
        accumulator.add(TrainingTimingSample(forward = PhaseTiming(TimingBackend.CPU, hostMs = 3.0)))
        val phase = accumulator.snapshot().average?.forward
        assertEquals(TimingBackend.UNAVAILABLE, phase?.backend)
        assertNull(phase?.qnnExecuteMs)
        assertEquals(0L, phase?.qnnExecuteCount)
    }

    @Test fun throttledFusedSamplesUseNativeStepWeight() {
        val accumulator = TimingAccumulator()
        accumulator.add(
            TrainingTimingSample(
                fusedForwardBackward = PhaseTiming(
                    TimingBackend.HTP,
                    qnnExecuteMs = 4.0,
                    qnnExecuteCount = 2,
                ),
            ),
            sampleWeight = 4,
        )
        val snapshot = accumulator.snapshot()
        assertEquals(4L, snapshot.sampleCount)
        assertEquals(16.0, snapshot.cumulative?.fusedForwardBackward?.qnnExecuteMs ?: -1.0, 1e-6)
        assertEquals(4.0, snapshot.average?.fusedForwardBackward?.qnnExecuteMs ?: -1.0, 1e-6)
        assertEquals(8L, snapshot.htpExecuteCount)
    }

    @Test fun missingCurrentSampleDoesNotReusePreviousTiming() {
        val accumulator = TimingAccumulator()
        accumulator.add(
            TrainingTimingSample(
                fusedForwardBackward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 4.0, qnnExecuteCount = 1),
            ),
        )
        accumulator.add(null)
        val snapshot = accumulator.snapshot()
        assertNull(snapshot.current)
        assertEquals(4.0, snapshot.average?.fusedForwardBackward?.qnnExecuteMs ?: -1.0, 1e-6)
        assertEquals(4.0, snapshot.cumulative?.fusedForwardBackward?.qnnExecuteMs ?: -1.0, 1e-6)
    }

    @Test fun unavailableEvidenceMakesMixedAggregateFailClosed() {
        val accumulator = TimingAccumulator()
        accumulator.add(
            TrainingTimingSample(
                fusedForwardBackward = PhaseTiming(TimingBackend.HTP, qnnExecuteMs = 4.0, qnnExecuteCount = 1),
            ),
        )
        accumulator.add(
            TrainingTimingSample(
                fusedForwardBackward = PhaseTiming(TimingBackend.UNAVAILABLE),
            ),
        )
        val snapshot = accumulator.snapshot()
        assertEquals(TimingBackend.UNAVAILABLE, snapshot.current?.fusedForwardBackward?.backend)
        assertEquals(TimingBackend.UNAVAILABLE, snapshot.average?.fusedForwardBackward?.backend)
        assertNull(snapshot.average?.fusedForwardBackward?.qnnExecuteMs)
        assertEquals(TimingBackend.UNAVAILABLE, snapshot.cumulative?.fusedForwardBackward?.backend)
    }
}
