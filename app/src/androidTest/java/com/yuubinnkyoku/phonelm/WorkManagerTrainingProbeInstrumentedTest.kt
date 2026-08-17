package com.yuubinnkyoku.phonelm

import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ExistingWorkPolicy
import androidx.work.WorkManager
import androidx.work.testing.WorkManagerTestInitHelper
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WorkManagerTrainingProbeInstrumentedTest {
    private lateinit var context: Context

    @Before fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        WorkManagerTestInitHelper.initializeTestWorkManager(context)
    }

    @After fun tearDown() {
        WorkManager.getInstance(context).cancelAllWork()
    }

    @Test fun foregroundInfoUsesSpecialUseAndOngoingCancelNotification() {
        val info = WorkManagerTrainingProbeWorker.createForegroundInfo(
            context,
            java.util.UUID.randomUUID(),
            2,
            45,
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE, info.foregroundServiceType)
        }
        assertTrue(info.notification.flags and android.app.Notification.FLAG_ONGOING_EVENT != 0)
        assertEquals("HexaTrain background work probe",
            info.notification.extras.getCharSequence(android.app.Notification.EXTRA_TITLE).toString())
        assertEquals(1, info.notification.actions.size)
        assertEquals("Cancel", info.notification.actions.single().title.toString())
    }

    @Test fun uniqueKeepDoesNotReplaceFirstRequest() {
        val manager = WorkManager.getInstance(context)
        val first = WorkManagerTrainingProbeWorker.request()
        val second = WorkManagerTrainingProbeWorker.request()
        manager.enqueueUniqueWork(
            WorkManagerTrainingProbeWorker.UNIQUE_WORK_NAME,
            ExistingWorkPolicy.KEEP,
            first,
        ).result.get()
        manager.enqueueUniqueWork(
            WorkManagerTrainingProbeWorker.UNIQUE_WORK_NAME,
            ExistingWorkPolicy.KEEP,
            second,
        ).result.get()
        val infos = manager.getWorkInfosForUniqueWork(WorkManagerTrainingProbeWorker.UNIQUE_WORK_NAME).get()
        assertTrue(infos.any { it.id == first.id })
        assertFalse(infos.any { it.id == second.id })
    }

    @Test fun productionForegroundInfoUsesSpecialUseAndStopAction() {
        val info = StandaloneTrainingWorker.createForegroundInfo(
            context,
            java.util.UUID.randomUUID(),
            "TRAINING",
            2,
            10,
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE, info.foregroundServiceType)
        }
        assertTrue(info.notification.flags and android.app.Notification.FLAG_ONGOING_EVENT != 0)
        assertEquals("HexaTrain training",
            info.notification.extras.getCharSequence(android.app.Notification.EXTRA_TITLE).toString())
        assertEquals("Stop", info.notification.actions.single().title.toString())
    }
}
