package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeTrainingFieldsParserTest {
    @Test
    fun parsesStructuredProgressWithoutUsingDisplayText() {
        val fields = NativeTrainingFieldsParser.parse(
            "phase=training\nstep=32\nsteps=8000\nloss=2.75\n" +
                "timing_sample_steps=31\nfused_forward_backward_qnn_us=3100\n",
        )
        assertEquals("training", fields.string("phase"))
        assertEquals(32, fields.int("step"))
        assertEquals(8000, fields.int("steps"))
        assertEquals(2.75, fields.double("loss")!!, 1e-6)
        assertEquals(31L, fields.long("timing_sample_steps"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun conflictingDuplicateIsRejected() {
        NativeTrainingFieldsParser.parse("status=SUCCESS\nstatus=FAILED\n")
    }

    @Test
    fun evidenceRequiresAllIndependentGates() {
        assertTrue(
            TrainingRuntimeEvidence(true, true, false, backend = "HTP").isAuthoritativelyHtp,
        )
        assertFalse(
            TrainingRuntimeEvidence(true, true, false, backend = "CPU").isAuthoritativelyHtp,
        )
        assertFalse(
            TrainingRuntimeEvidence(true, true, false, backend = "HTP", error = "trace").isAuthoritativelyHtp,
        )
        assertFalse(
            TrainingRuntimeEvidence(true, true, false, backend = null).isAuthoritativelyHtp,
        )
    }

    @Test
    fun hostScratchReplayIsIgnoredRatherThanReportedAsHtpPhase() {
        assertEquals(null, classifyNativeTrainingPhase("cpu_replay"))
        assertEquals(TrainingPhase.TRAINING, classifyNativeTrainingPhase("training"))
    }
}
