package com.yuubinnkyoku.phonelm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder

/** Keeps user-started long runs out of the cached-empty process state. */
class LiveUpdateForegroundService : Service() {
    private var activeNotificationId: Int = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_FINISH) {
            val notificationId = intent.getIntExtra(EXTRA_NOTIFICATION_ID, 0)
            val showTerminalNotification = intent.getBooleanExtra(EXTRA_SHOW_TERMINAL, true)
            // A terminal command from an older run must not stop a newer run's
            // foreground service. Controllers use process-wide unique ids.
            if (notificationId <= 0 || notificationId != activeNotificationId) {
                return START_NOT_STICKY
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            activeNotificationId = 0
            if (notificationId > 0 && showTerminalNotification) {
                val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
                val contentIntent = launchIntent?.let {
                    PendingIntent.getActivity(
                        this,
                        notificationId,
                        it,
                        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                    )
                }
                val notification = Notification.Builder(this, CHANNEL_ID)
                    .setSmallIcon(R.drawable.ic_launcher)
                    .setContentTitle(intent.getStringExtra(EXTRA_TITLE) ?: "PhoneLM")
                    .setContentText(intent.getStringExtra(EXTRA_TEXT) ?: "完了")
                    .setSubText(intent.getStringExtra(EXTRA_SUBTEXT))
                    .setOnlyAlertOnce(true)
                    .setAutoCancel(true)
                    .setContentIntent(contentIntent)
                    .build()
                getSystemService(NotificationManager::class.java).notify(notificationId, notification)
            }
            stopSelf()
            return START_NOT_STICKY
        }
        val notificationId = intent?.getIntExtra(EXTRA_NOTIFICATION_ID, 0) ?: 0
        if (notificationId <= 0) {
            stopSelf()
            return START_NOT_STICKY
        }
        activeNotificationId = notificationId
        val kind = intent?.getStringExtra(EXTRA_KIND) ?: "run"
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "PhoneLM runs", NotificationManager.IMPORTANCE_LOW),
        )
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val contentIntent = launchIntent?.let {
            PendingIntent.getActivity(
                this,
                notificationId,
                it,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
        }
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher)
            .setContentTitle("PhoneLMを実行中")
            .setContentText("$kind — 初期化中")
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentIntent(contentIntent)
            .build()
        startForeground(notificationId, notification)
        return START_NOT_STICKY
    }

    companion object {
        private const val ACTION_START = "com.yuubinnkyoku.phonelm.LIVE_UPDATE_START"
        private const val ACTION_FINISH = "com.yuubinnkyoku.phonelm.LIVE_UPDATE_FINISH"
        private const val EXTRA_NOTIFICATION_ID = "notification_id"
        private const val EXTRA_KIND = "kind"
        private const val EXTRA_TITLE = "title"
        private const val EXTRA_TEXT = "text"
        private const val EXTRA_SUBTEXT = "subtext"
        private const val EXTRA_SHOW_TERMINAL = "show_terminal_notification"
        private const val CHANNEL_ID = "phonelm_runs"

        fun start(context: Context, notificationId: Int, kind: String) {
            context.startForegroundService(
                Intent(context, LiveUpdateForegroundService::class.java)
                    .setAction(ACTION_START)
                    .putExtra(EXTRA_NOTIFICATION_ID, notificationId)
                    .putExtra(EXTRA_KIND, kind),
            )
        }

        fun finish(
            context: Context,
            notificationId: Int,
            title: String,
            text: String,
            subtext: String,
            showTerminalNotification: Boolean = true,
        ) {
            context.startService(
                Intent(context, LiveUpdateForegroundService::class.java)
                    .setAction(ACTION_FINISH)
                    .putExtra(EXTRA_NOTIFICATION_ID, notificationId)
                    .putExtra(EXTRA_TITLE, title)
                    .putExtra(EXTRA_TEXT, text)
                    .putExtra(EXTRA_SUBTEXT, subtext)
                    .putExtra(EXTRA_SHOW_TERMINAL, showTerminalNotification),
            )
        }
    }
}
