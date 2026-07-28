package com.yuubinnkyoku.phonelm

import android.app.Notification
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LiveUpdateNotificationInstrumentedTest {
    private lateinit var context: Context
    private lateinit var manager: NotificationManager

    @Before
    fun setUp() {
        assumeTrue(Build.VERSION.SDK_INT >= Build.VERSION_CODES.BAKLAVA)
        context = InstrumentationRegistry.getInstrumentation().targetContext
        manager = context.getSystemService(NotificationManager::class.java)
        cancelAndWait()
    }

    @After
    fun tearDown() {
        cancelAndWait()
    }

    @Test
    fun android16FallbackCoversProgressAndTerminalLifecycle() {
        context.getSharedPreferences("phonelm_live_update", Context.MODE_PRIVATE).edit()
            .putBoolean("10001", true).commit()
        val controller = LiveUpdateNotificationController(context)
        controller.onRunStarted("学習", null)
        assertRunningNotification(indeterminate = true, expectedText = "初期化中")

        controller.onProgress(RunProgress.PhaseChanged("学習"))
        assertRunningNotification(indeterminate = true, expectedText = "学習")
        val oldRunDeleteIntent = singleNotification().deleteIntent
        assertNotNull(oldRunDeleteIntent)

        cancelAndWait()
        controller.onRunStarted("学習", 100)
        oldRunDeleteIntent!!.send()
        controller.onProgress(RunProgress.Step(10, 100, loss = 0.5))
        assertRunningNotification(indeterminate = false, expectedText = "step 10 / 100")

        controller.onProgress(RunProgress.Step(50, 100, loss = 0.25))
        assertRunningNotification(indeterminate = false, expectedText = "step 50 / 100")

        controller.onProgress(RunProgress.Completed("loss 0.25"))
        val completed = singleNotification { it.flags and Notification.FLAG_ONGOING_EVENT == 0 }
        assertFalse(completed.flags and Notification.FLAG_ONGOING_EVENT != 0)
        assertTrue(completed.extras.getCharSequence(Notification.EXTRA_TITLE).toString().contains("完了"))

        cancelAndWait()
        val dismissedController = LiveUpdateNotificationController(context)
        dismissedController.onRunStarted("推論", 1)
        val currentRunDeleteIntent = singleNotification().deleteIntent
        assertNotNull(currentRunDeleteIntent)
        currentRunDeleteIntent!!.send()
        cancelAndWait()
        dismissedController.onProgress(RunProgress.Failed("must stay dismissed"))
        Thread.sleep(200)
        assertTrue(manager.activeNotifications.none { it.packageName == context.packageName })

        val failedController = LiveUpdateNotificationController(context)
        failedController.onRunStarted("評価", 1)
        failedController.onProgress(RunProgress.Failed("safe failure"))
        assertFalse(singleNotification { it.flags and Notification.FLAG_ONGOING_EVENT == 0 }
            .flags and Notification.FLAG_ONGOING_EVENT != 0)

        cancelAndWait()
        val cancelledController = LiveUpdateNotificationController(context)
        cancelledController.onRunStarted("ベンチマーク", 1)
        cancelledController.onProgress(RunProgress.Cancelled)
        assertFalse(singleNotification { it.flags and Notification.FLAG_ONGOING_EVENT == 0 }
            .flags and Notification.FLAG_ONGOING_EVENT != 0)
    }

    private fun assertRunningNotification(indeterminate: Boolean, expectedText: String) {
        val notification = singleNotification {
            it.extras.getCharSequence(Notification.EXTRA_TEXT).toString().contains(expectedText)
        }
        assertTrue(notification.flags and Notification.FLAG_ONGOING_EVENT != 0)
        assertNotNull(notification.extras.getCharSequence(Notification.EXTRA_TITLE))
        assertTrue(notification.extras.getCharSequence(Notification.EXTRA_TEXT).toString().contains(expectedText))
        assertEquals(Notification.ProgressStyle::class.java.name, notification.extras.getString(Notification.EXTRA_TEMPLATE))
        assertEquals(indeterminate, notification.extras.getBoolean(Notification.EXTRA_PROGRESS_INDETERMINATE))
        assertFalse(notification.flags and Notification.FLAG_GROUP_SUMMARY != 0)
        assertFalse(notification.extras.getBoolean(Notification.EXTRA_COLORIZED))
        assertFalse(notification.extras.getBoolean("android.requestPromotedOngoing"))
        assertNull(notification.contentView)
        assertNull(notification.bigContentView)
        assertNull(notification.headsUpContentView)
        assertTrue(notification.flags and Notification.FLAG_ONLY_ALERT_ONCE != 0)
    }

    private fun singleNotification(predicate: (Notification) -> Boolean = { true }): Notification {
        var notifications = manager.activeNotifications
            .filter { it.packageName == context.packageName }
        val deadline = System.nanoTime() + 2_000_000_000L
        while ((notifications.size != 1 || !predicate(notifications.single().notification)) &&
            System.nanoTime() < deadline
        ) {
            Thread.sleep(25)
            notifications = manager.activeNotifications
                .filter { it.packageName == context.packageName }
        }
        assertEquals(1, notifications.size)
        return notifications.single().notification.also { assertTrue(predicate(it)) }
    }

    private fun cancelAndWait() {
        manager.cancelAll()
        val deadline = System.nanoTime() + 2_000_000_000L
        while (manager.activeNotifications.any { it.packageName == context.packageName } &&
            System.nanoTime() < deadline
        ) {
            Thread.sleep(25)
        }
        assertTrue(manager.activeNotifications.none { it.packageName == context.packageName })
        Thread.sleep(1_100)
    }
}
