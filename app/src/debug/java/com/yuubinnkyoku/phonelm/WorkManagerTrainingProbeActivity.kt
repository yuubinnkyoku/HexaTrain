package com.yuubinnkyoku.phonelm

import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.activity.ComponentActivity

/** Debug-only manual surface for the isolated long-running WorkManager device probe. */
class WorkManagerTrainingProbeActivity : ComponentActivity() {
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { text = "Probe idle" }
        val start = Button(this).apply {
            text = "Start 45-second WorkManager probe"
            setOnClickListener {
                WorkManagerTrainingProbeWorker.enqueue(applicationContext)
                render()
            }
        }
        val cancel = Button(this).apply {
            text = "Cancel probe"
            setOnClickListener { WorkManagerTrainingProbeWorker.cancel(applicationContext) }
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
            addView(start)
            addView(cancel)
            addView(status)
        })
        when (intent.getStringExtra(EXTRA_COMMAND)) {
            COMMAND_START -> {
                WorkManagerTrainingProbeWorker.enqueue(
                    applicationContext,
                    intent.getIntExtra(EXTRA_DURATION_SECONDS, WorkManagerTrainingProbeWorker.DEFAULT_DURATION_SECONDS),
                )
                finish()
                return
            }
            COMMAND_CANCEL -> {
                WorkManagerTrainingProbeWorker.cancel(applicationContext)
                finish()
                return
            }
            COMMAND_RECREATE -> {
                observeAcrossRecreation(savedInstanceState != null)
                return
            }
        }
        WorkManager.getInstance(this)
            .getWorkInfosForUniqueWorkLiveData(WorkManagerTrainingProbeWorker.UNIQUE_WORK_NAME)
            .observe(this) { render(it) }
    }

    private fun observeAcrossRecreation(recreated: Boolean) {
        WorkManager.getInstance(this)
            .getWorkInfosForUniqueWorkLiveData(WorkManagerTrainingProbeWorker.UNIQUE_WORK_NAME)
            .observe(this) { infos ->
                val info = infos.maxByOrNull { it.runAttemptCount } ?: return@observe
                val prefs = getSharedPreferences(
                    WorkManagerTrainingProbeWorker.DIAGNOSTIC_PREFS,
                    MODE_PRIVATE,
                )
                if (!recreated) {
                    prefs.edit()
                        .putString(KEY_RECREATE_WORK_ID_BEFORE, info.id.toString())
                        .putString(KEY_RECREATE_STATE_BEFORE, info.state.name)
                        .apply()
                    recreate()
                } else {
                    prefs.edit()
                        .putString(KEY_RECREATE_WORK_ID_AFTER, info.id.toString())
                        .putString(KEY_RECREATE_STATE_AFTER, info.state.name)
                        .putBoolean(KEY_ACTIVITY_RECREATED, true)
                        .apply()
                    finish()
                }
            }
    }

    private fun render(infos: List<WorkInfo>? = null) {
        val info = infos?.maxByOrNull { it.runAttemptCount }
        val progress = info?.progress
        status.text = if (info == null) {
            "Probe enqueue requested"
        } else {
            buildString {
                append("id=${info.id}\nstate=${info.state}\nattempt=${info.runAttemptCount}")
                append("\nprogress=${progress?.getInt(WorkManagerTrainingProbeWorker.KEY_ELAPSED_SECONDS, 0)}")
                append(" / ${progress?.getInt(WorkManagerTrainingProbeWorker.KEY_TOTAL_SECONDS, 0)} seconds")
                if (info.state.isFinished) append("\nterminal=${info.state}")
            }
        }
    }

    companion object {
        const val EXTRA_COMMAND = "phonelm.probe.command"
        const val EXTRA_DURATION_SECONDS = "phonelm.probe.duration_seconds"
        const val COMMAND_START = "start"
        const val COMMAND_CANCEL = "cancel"
        const val COMMAND_RECREATE = "recreate"
        const val KEY_RECREATE_WORK_ID_BEFORE = "recreate_work_id_before"
        const val KEY_RECREATE_WORK_ID_AFTER = "recreate_work_id_after"
        const val KEY_RECREATE_STATE_BEFORE = "recreate_state_before"
        const val KEY_RECREATE_STATE_AFTER = "recreate_state_after"
        const val KEY_ACTIVITY_RECREATED = "activity_recreated"
    }
}
