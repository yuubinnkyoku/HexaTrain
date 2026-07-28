package com.yuubinnkyoku.phonelm

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import java.util.concurrent.atomic.AtomicInteger

interface RunNotificationSink {
    fun onRunStarted(kind: String, totalSteps: Long?)
    fun onProgress(progress: RunProgress)
}

object NoOpRunNotificationSink : RunNotificationSink {
    override fun onRunStarted(kind: String, totalSteps: Long?) = Unit
    override fun onProgress(progress: RunProgress) = Unit
}

/** Android-only endpoint for user-initiated runs. It never runs in instrumentation's direct JNI path. */
class LiveUpdateNotificationController(context: Context) : RunNotificationSink {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(NotificationManager::class.java)
    private val sequence = AtomicInteger(10_000)
    private var current: ActiveRun? = null

    override fun onRunStarted(kind: String, totalSteps: Long?) {
        val run = ActiveRun(sequence.incrementAndGet(), kind, totalSteps, System.currentTimeMillis())
        appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(run.id.toString()).apply()
        current = run
        ensureChannel()
        post(run, RunProgress.Started(kind, totalSteps), force = true)
    }

    override fun onProgress(progress: RunProgress) {
        val run = current ?: return
        if (isDismissed(run.id)) return
        if (!run.throttle.shouldPost(progress, System.currentTimeMillis())) return
        val finished = progress is RunProgress.Completed || progress is RunProgress.Failed || progress is RunProgress.Cancelled
        post(run, progress, force = finished)
        if (finished) {
            current = null
        }
    }

    fun openPromotionSettings(): Boolean {
        if (!supportsQpr2Promotion()) return false
        return runCatching {
            appContext.startActivity(Intent(Settings.ACTION_APP_NOTIFICATION_PROMOTION_SETTINGS).apply {
                data = Uri.parse("package:${appContext.packageName}")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            })
            true
        }.getOrDefault(false)
    }

    private fun post(run: ActiveRun, event: RunProgress, force: Boolean) {
        if (!notificationsAllowed()) return
        val finished = event is RunProgress.Completed || event is RunProgress.Failed || event is RunProgress.Cancelled
        val snapshot = run.apply(event)
        val builder = Notification.Builder(appContext, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher)
            .setContentTitle(if (finished) "PhoneLM ${snapshot.kind} ${snapshot.terminalTitle}" else "PhoneLMを実行中")
            .setContentText(snapshot.contentText())
            .setSubText(snapshot.detailText())
            .setOnlyAlertOnce(true)
            .setOngoing(!finished)
            .setAutoCancel(finished)
            .setColorized(false)
            .setContentIntent(appContext.packageManager.getLaunchIntentForPackage(appContext.packageName)?.let {
                PendingIntent.getActivity(appContext, run.id, it, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
            })
        if (!finished) {
            builder.setDeleteIntent(PendingIntent.getBroadcast(appContext, run.id,
                Intent(appContext, LiveUpdateDismissReceiver::class.java).putExtra(EXTRA_RUN_ID, run.id),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT))
        }
        if (Build.VERSION.SDK_INT >= 36) {
            builder.setStyle(Notification.ProgressStyle()
                .setProgress(snapshot.percent ?: 0)
                .setProgressIndeterminate(snapshot.percent == null))
            if (!finished && supportsQpr2Promotion() && manager.canPostPromotedNotifications()) {
                builder.setRequestPromotedOngoing(true)
            }
            builder.setShortCriticalText(snapshot.chipText())
        } else {
            builder.setProgress(100, snapshot.percent ?: 0, snapshot.percent == null)
        }
        val notification = builder.build()
        // This is diagnostic only: OEM policy may still deny promotion after posting.
        run.promotable = Build.VERSION.SDK_INT >= 36 && notification.hasPromotableCharacteristics()
        manager.notify(run.id, notification)
    }

    private fun notificationsAllowed() = Build.VERSION.SDK_INT < 33 ||
        appContext.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

    private fun supportsQpr2Promotion() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.BAKLAVA &&
        Build.VERSION.SDK_INT_FULL >= Build.VERSION_CODES_FULL.BAKLAVA_1

    private fun ensureChannel() {
        manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "PhoneLM runs", NotificationManager.IMPORTANCE_LOW))
    }

    private fun isDismissed(id: Int) = appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("$id", false)

    private data class ActiveRun(
        val id: Int, val kind: String, val total: Long?, val startedAtMs: Long,
        val throttle: ProgressUpdateThrottle = ProgressUpdateThrottle(), var phase: String = "初期化中",
        var completed: Long = 0, var loss: Double? = null, var terminalTitle: String = "完了", var promotable: Boolean = false,
    ) {
        fun apply(event: RunProgress): ActiveRun = apply {
            when (event) {
                is RunProgress.PhaseChanged -> phase = event.phase
                is RunProgress.Step -> { completed = event.completed; loss = event.loss ?: loss }
                is RunProgress.Completed -> { terminalTitle = "完了"; phase = "完了"; loss = event.metric?.removePrefix("loss ")?.toDoubleOrNull() ?: loss }
                is RunProgress.Failed -> { terminalTitle = "失敗"; phase = "失敗" }
                RunProgress.Cancelled -> { terminalTitle = "キャンセル済み"; phase = "キャンセル済み" }
                is RunProgress.Started -> Unit
            }
        }
        val percent: Int? get() = total?.takeIf { it > 0 }?.let { (completed * 100 / it).toInt().coerceIn(0, 100) }
        fun contentText() = "$phase — " + (total?.let { "step $completed / $it" } ?: "進捗を確認中")
        fun detailText(): String {
            val elapsed = ((System.currentTimeMillis() - startedAtMs) / 1_000).coerceAtLeast(0)
            return listOfNotNull(loss?.let { "loss %.4f".format(java.util.Locale.ROOT, it) }, "経過 ${elapsed / 60}分${elapsed % 60}秒").joinToString("・")
        }
        fun chipText() = percent?.let { "$it%" } ?: phase.take(7)
    }

    companion object {
        private const val CHANNEL_ID = "phonelm_runs"
        private const val PREFS = "phonelm_live_update"
        const val EXTRA_RUN_ID = "run_id"
    }
}

class LiveUpdateDismissReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        context.getSharedPreferences("phonelm_live_update", Context.MODE_PRIVATE).edit()
            .putBoolean(intent.getIntExtra(LiveUpdateNotificationController.EXTRA_RUN_ID, -1).toString(), true).apply()
    }
}
