package com.yuubinnkyoku.phonelm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay

/** Isolated long-running WorkManager probe. It never enters training, JNI, QNN, or network code. */
class WorkManagerTrainingProbeWorker(
    appContext: Context,
    params: WorkerParameters,
) : CoroutineWorker(appContext, params) {
    override suspend fun doWork(): Result {
        val durationSeconds = inputData.getInt(KEY_DURATION_SECONDS, DEFAULT_DURATION_SECONDS)
            .coerceIn(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
        setForeground(createForegroundInfo(applicationContext, id, 0, durationSeconds))
        recordDiagnostic("RUNNING", 0, stopReasonOrUnknown())
        var checksum = 0L
        try {
            for (second in 1..durationSeconds) {
                // Deliberately small and bounded CPU work: no QNN, training, network, or file transfer.
                repeat(20_000) { index -> checksum = (checksum * 31L + index + second) xor (checksum ushr 7) }
                setProgress(progressData(second, durationSeconds, checksum))
                if (second == 1 || second == durationSeconds || second % NOTIFICATION_INTERVAL_SECONDS == 0) {
                    setForeground(createForegroundInfo(applicationContext, id, second, durationSeconds))
                    recordDiagnostic("RUNNING", second, stopReasonOrUnknown())
                }
                if (second < durationSeconds) delay(1_000L)
            }
        } catch (cancelled: CancellationException) {
            recordDiagnostic("STOPPED", -1, stopReasonOrUnknown())
            Log.i(TAG, "probe cancelled reason=${stopReasonOrUnknown()} workId=$id")
            throw cancelled
        }
        recordDiagnostic("SUCCEEDED", durationSeconds, stopReasonOrUnknown())
        return Result.success(progressData(durationSeconds, durationSeconds, checksum))
    }

    private fun stopReasonOrUnknown(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) stopReason else STOP_REASON_UNAVAILABLE

    private fun recordDiagnostic(state: String, elapsedSeconds: Int, stopReason: Int) {
        applicationContext.getSharedPreferences(DIAGNOSTIC_PREFS, Context.MODE_PRIVATE).edit()
            .putString(KEY_DIAGNOSTIC_WORK_ID, id.toString())
            .putString(KEY_DIAGNOSTIC_STATE, state)
            .putInt(KEY_ELAPSED_SECONDS, elapsedSeconds)
            .putInt(KEY_STOP_REASON, stopReason)
            .putLong(KEY_DIAGNOSTIC_AT_MS, System.currentTimeMillis())
            .apply()
        Log.i(TAG, "probe state=$state elapsed=$elapsedSeconds stopReason=$stopReason workId=$id")
    }

    companion object {
        const val UNIQUE_WORK_NAME = "phonelm-workmanager-training-probe"
        const val TAG = "PhoneLMWorkProbe"
        const val DIAGNOSTIC_PREFS = "phonelm_workmanager_probe"
        const val KEY_DURATION_SECONDS = "duration_seconds"
        const val KEY_ELAPSED_SECONDS = "elapsed_seconds"
        const val KEY_TOTAL_SECONDS = "total_seconds"
        const val KEY_CHECKSUM = "checksum"
        const val KEY_STOP_REASON = "stop_reason"
        const val KEY_DIAGNOSTIC_WORK_ID = "work_id"
        const val KEY_DIAGNOSTIC_STATE = "state"
        const val KEY_DIAGNOSTIC_AT_MS = "diagnostic_at_ms"
        const val DEFAULT_DURATION_SECONDS = 45
        const val MIN_DURATION_SECONDS = 30
        const val MAX_DURATION_SECONDS = 300
        const val NOTIFICATION_ID = 0x504d31
        private const val NOTIFICATION_INTERVAL_SECONDS = 5
        private const val CHANNEL_ID = "phonelm_workmanager_probe"
        private const val STOP_REASON_UNAVAILABLE = -1

        fun request(durationSeconds: Int = DEFAULT_DURATION_SECONDS): OneTimeWorkRequest =
            OneTimeWorkRequestBuilder<WorkManagerTrainingProbeWorker>()
                .setInputData(workDataOf(KEY_DURATION_SECONDS to durationSeconds))
                .addTag(UNIQUE_WORK_NAME)
                .build()

        fun enqueue(context: Context, durationSeconds: Int = DEFAULT_DURATION_SECONDS): OneTimeWorkRequest {
            val request = request(durationSeconds)
            WorkManager.getInstance(context).enqueueUniqueWork(
                UNIQUE_WORK_NAME,
                ExistingWorkPolicy.KEEP,
                request,
            )
            return request
        }

        fun cancel(context: Context) {
            WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_WORK_NAME)
        }

        fun progressData(elapsedSeconds: Int, totalSeconds: Int, checksum: Long): Data = workDataOf(
            KEY_ELAPSED_SECONDS to elapsedSeconds,
            KEY_TOTAL_SECONDS to totalSeconds,
            KEY_CHECKSUM to checksum,
        )

        fun createForegroundInfo(
            context: Context,
            workId: java.util.UUID,
            elapsedSeconds: Int,
            totalSeconds: Int,
        ): ForegroundInfo {
            val manager = context.getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "HexaTrain work probe", NotificationManager.IMPORTANCE_LOW),
            )
            val launchIntent = context.packageManager.getLaunchIntentForPackage(context.packageName)
            val contentIntent = launchIntent?.let {
                PendingIntent.getActivity(
                    context,
                    workId.hashCode(),
                    it,
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                )
            }
            val cancelIntent = WorkManager.getInstance(context).createCancelPendingIntent(workId)
            val notification = Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher)
                .setContentTitle("HexaTrain background work probe")
                .setContentText("$elapsedSeconds / $totalSeconds seconds")
                .setOnlyAlertOnce(true)
                .setOngoing(true)
                .setProgress(totalSeconds, elapsedSeconds, false)
                .setContentIntent(contentIntent)
                .addAction(Notification.Action.Builder(null, "Cancel", cancelIntent).build())
                .build()
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                ForegroundInfo(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
                )
            } else {
                ForegroundInfo(NOTIFICATION_ID, notification)
            }
        }
    }
}
