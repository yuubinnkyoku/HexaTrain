package com.yuubinnkyoku.phonelm

import java.util.ArrayDeque
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

    @Test
    fun coalescesAndBoundsDisplayedProgressWhileRetainingTheFinalReport() {
        val engineFinished = CountDownLatch(1)
        val report = "RESULT\nstatus=SUCCESS\nfinal_loss=0.125"
        val engine = object : BenchmarkEngine {
            override fun environmentReport() = "mnn_version=test"

            override fun run(config: BenchmarkConfig, progress: (String) -> Unit): String {
                repeat(4_000) { step ->
                    progress("phase=train step=$step loss=0.125000 diagnostic=${"x".repeat(64)}")
                }
                engineFinished.countDown()
                return report
            }

            override fun requestStop() = Unit
        }
        val dispatcher = QueuedUiDispatcher()
        var finalState = BenchmarkUiState()
        val viewModel = BenchmarkViewModel(engine = engine, uiDispatcher = dispatcher)
        viewModel.setListener { state -> finalState = state }

        assertTrue(viewModel.start(BenchmarkConfig.small()))
        assertTrue(engineFinished.await(5, TimeUnit.SECONDS))
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        while (viewModel.snapshot().running && System.nanoTime() < deadline) {
            Thread.yield()
        }
        assertFalse(viewModel.snapshot().running)
        assertEquals(1, dispatcher.pendingCount())
        assertEquals(1, dispatcher.maximumPendingCount)

        dispatcher.drain()
        assertFalse(finalState.running)
        assertTrue(finalState.output.length <= MAX_UI_OUTPUT_CHARS)
        assertTrue(finalState.output.startsWith("[earlier output truncated]\n"))
        assertTrue(finalState.output.endsWith("$report\n"))
        assertEquals("SUCCESS", finalState.lastResult?.status)

        viewModel.close()
    }

    private class QueuedUiDispatcher : UiDispatcher {
        private val blocks = ArrayDeque<() -> Unit>()
        var maximumPendingCount = 0
            private set

        override fun dispatch(block: () -> Unit) {
            synchronized(blocks) {
                blocks.addLast(block)
                maximumPendingCount = maxOf(maximumPendingCount, blocks.size)
            }
        }

        fun pendingCount(): Int = synchronized(blocks) { blocks.size }

        fun drain() {
            while (true) {
                val block = synchronized(blocks) {
                    if (blocks.isEmpty()) null else blocks.removeFirst()
                } ?: return
                block()
            }
        }
    }
}
