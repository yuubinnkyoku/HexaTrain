package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test

class RunProgressTest {
    @Test fun parsesNativeStepWithoutInventingTotal() {
        val event = NativeProgressParser.parse("step=12\nloss=0.125\nbackend_actual=CPU") as RunProgress.Step
        assertEquals(12, event.completed)
        assertEquals(0.125, event.loss!!, 0.0)
        assertEquals(null, event.total)
    }

    @Test fun parsesQnnTerminalReportWithModeSpecificHeader() {
        val event = NativeProgressParser.parse(
            "QNN_LINEAR_TRAINING_RESULT\nstatus=SUCCESS\nfinal_loss=0.125",
        ) as RunProgress.Completed
        assertEquals("loss 0.125", event.metric)
    }

    @Test fun parsesTinyLmSeedStepLossProgress() {
        val event = NativeProgressParser.parse(
            "phase=training\nseed=3\nseeds=5\nstep=200\nsteps=320\nloss=0.125",
        ) as RunProgress.Step
        assertEquals(200, event.completed)
        assertEquals(320L, event.total)
        assertEquals(0.125, event.loss!!, 0.0)
        assertEquals("training", event.phase)
        assertEquals(3L, event.seed)
        assertEquals(5L, event.seeds)
    }

    @Test fun parsesNativePhaseWithoutStep() {
        val event = NativeProgressParser.parse("phase=evaluation") as RunProgress.PhaseChanged
        assertEquals("evaluation", event.phase)
    }

    @Test fun throttlePostsPhaseAndOnePercentChangesButNotEveryStep() {
        val throttle = ProgressUpdateThrottle(minimumIntervalMs = 1_000)
        assertTrue(throttle.shouldPost(RunProgress.Started("学習", 100), 0))
        assertTrue(throttle.shouldPost(RunProgress.Step(1, 100), 10))
        assertFalse(throttle.shouldPost(RunProgress.Step(1, 100), 20))
        assertTrue(throttle.shouldPost(RunProgress.PhaseChanged("評価"), 30))
        assertTrue(throttle.shouldPost(RunProgress.Completed("loss 0.1"), 31))
    }
}
