package com.yuubinnkyoku.phonelm

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.CopyOnWriteArrayList
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkViewModelTest {
    @Test
    fun forwardsQnnTerminalReportExactlyOnceWhenCallbackMatchesReturnValue() {
        val completed = CountDownLatch(1)
        val events = CopyOnWriteArrayList<RunProgress>()
        val report = "QNN_LINEAR_TRAINING_RESULT\nstatus=SUCCESS\nfinal_loss=0.125"
        val engine = object : BenchmarkEngine {
            override fun environmentReport() = "mnn_version=test"
            override fun run(config: BenchmarkConfig, progress: (String) -> Unit) = report
            override fun runMode(
                mode: ExecutionMode,
                config: BenchmarkConfig,
                progress: (String) -> Unit,
            ): String {
                progress(report)
                return report
            }
            override fun requestStop() = Unit
        }
        val notifications = object : RunNotificationSink {
            override fun onRunStarted(kind: String, totalSteps: Long?) = Unit
            override fun onProgress(progress: RunProgress) {
                events += progress
                if (progress is RunProgress.Completed) completed.countDown()
            }
        }
        val viewModel = BenchmarkViewModel(
            engine = engine,
            runNotifications = notifications,
            uiDispatcher = UiDispatcher { it() },
        )

        assertTrue(viewModel.startMode(ExecutionMode.QNN_HTP_FORWARD_DW, BenchmarkConfig.small()))
        assertTrue(completed.await(2, TimeUnit.SECONDS))
        assertEquals(1, events.count { it is RunProgress.Completed })

        viewModel.close()
    }

    @Test
    fun preventsDoubleStartAndForwardsStop() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val stopped = AtomicBoolean(false)
        val engine = object : BenchmarkEngine {
            override fun environmentReport() = "mnn_version=test"

            override fun run(config: BenchmarkConfig, progress: (String) -> Unit): String {
                entered.countDown()
                release.await(5, TimeUnit.SECONDS)
                return "RESULT\nstatus=CANCELLED\nerror=stopped"
            }

            override fun requestStop() {
                stopped.set(true)
                release.countDown()
            }
        }
        val viewModel = BenchmarkViewModel(engine = engine, uiDispatcher = UiDispatcher { it() })

        assertTrue(viewModel.start(BenchmarkConfig.small()))
        assertTrue(entered.await(2, TimeUnit.SECONDS))
        assertFalse(viewModel.start(BenchmarkConfig.small()))
        assertTrue(viewModel.requestStop())
        assertTrue(stopped.get())

        viewModel.close()
    }
}
