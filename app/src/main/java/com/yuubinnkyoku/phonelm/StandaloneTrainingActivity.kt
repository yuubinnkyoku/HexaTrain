package com.yuubinnkyoku.phonelm

import android.app.Activity
import android.app.AlertDialog
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * UI-only entry point for the standalone HTP training session.  Session ownership is
 * application-scoped so an Activity recreation reattaches to the in-progress session
 * instead of creating a second worker.
 */
class StandaloneTrainingActivity : Activity() {
    private lateinit var repository: StandaloneTrainingRepository
    private var subscription: AutoCloseable? = null
    private val subscriptionLock = Any()
    @Volatile private var activityStarted = false
    private var subscriptionGeneration = 0L
    private val ioExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-training-ui-io").apply { isDaemon = true }
    }

    private lateinit var overview: TextView
    private lateinit var timing: TextView
    private lateinit var activity: TextView
    private lateinit var checkpoint: TextView
    private lateinit var error: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var datasetUri: TextView
    private lateinit var selectDataset: Button
    private lateinit var start: Button
    private lateinit var stop: Button
    private lateinit var pause: Button
    private lateinit var resume: Button
    private lateinit var startOver: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingStart = savedInstanceState?.getBoolean(KEY_PENDING_START, false) ?: false
        setContentView(R.layout.activity_standalone_training)

        repository = StandaloneTrainingRepositoryRegistry.get(applicationContext)
        overview = findViewById(R.id.trainingOverviewText)
        timing = findViewById(R.id.trainingTimingText)
        activity = findViewById(R.id.trainingActivityText)
        checkpoint = findViewById(R.id.trainingCheckpointText)
        error = findViewById(R.id.trainingErrorText)
        progressBar = findViewById(R.id.trainingProgressBar)
        datasetUri = findViewById(R.id.trainingDatasetUriText)
        selectDataset = findViewById(R.id.selectTrainingDatasetButton)
        start = findViewById(R.id.startTrainingButton)
        stop = findViewById(R.id.stopTrainingButton)
        pause = findViewById(R.id.pauseTrainingButton)
        resume = findViewById(R.id.resumeTrainingButton)
        startOver = findViewById(R.id.startOverTrainingButton)

        selectDataset.setOnClickListener(::openDatasetPicker)
        start.setOnClickListener { startTrainingAfterNotificationCheck() }
        stop.setOnClickListener { ioExecutor.execute { repository.stop() } }
        pause.setOnClickListener { ioExecutor.execute { repository.pause() } }
        resume.setOnClickListener { ioExecutor.execute { repository.resume() } }
        startOver.setOnClickListener { confirmStartOver() }
    }

    override fun onStart() {
        super.onStart()
        val generation = synchronized(subscriptionLock) {
            activityStarted = true
            ++subscriptionGeneration
        }
        ioExecutor.execute {
            val candidate = repository.subscribe(::render)
            val keep = synchronized(subscriptionLock) {
                if (activityStarted && generation == subscriptionGeneration) {
                    subscription = candidate
                    true
                } else false
            }
            if (!keep) candidate.close()
        }
    }

    override fun onStop() {
        val old = synchronized(subscriptionLock) {
            activityStarted = false
            ++subscriptionGeneration
            subscription.also { subscription = null }
        }
        old?.close()
        super.onStop()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_OPEN_DATASET || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        val readFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
        if (readFlags == 0) {
            toast(getString(R.string.training_dataset_access_failed))
            return
        }
        ioExecutor.execute {
            try {
                // ACTION_OPEN_DOCUMENT providers should grant persistable read access.
                contentResolver.takePersistableUriPermission(uri, readFlags)
                val selected = repository.selectDataset(TrainingDataset(uri.toString(), displayName(uri)))
                runOnUiThread {
                    if (!selected) toast(getString(R.string.training_dataset_access_failed))
                }
            } catch (_: SecurityException) {
                runOnUiThread { toast(getString(R.string.training_dataset_access_failed)) }
            } catch (_: Throwable) {
                runOnUiThread { toast(getString(R.string.training_dataset_access_failed)) }
            }
        }
    }

    private fun openDatasetPicker(@Suppress("UNUSED_PARAMETER") button: android.view.View) {
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "*/*"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            },
            REQUEST_OPEN_DATASET,
        )
    }

    private fun confirmStartOver() {
        AlertDialog.Builder(this)
            .setTitle(R.string.training_start_over_title)
            .setMessage(R.string.training_start_over_message)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.training_start_over_confirm) { _, _ ->
                ioExecutor.execute {
                    val started = repository.startOver()
                    if (!started) runOnUiThread { toast("Training is still active or the dataset is unavailable") }
                }
            }
            .show()
    }

    private fun render(state: TrainingUiState) {
        if (isFinishing || isDestroyed) return
        runOnUiThread {
            overview.text = "Training overview: ${state.overview}"
            timing.text = timingText(state)
            activity.text = activityText(state)
            checkpoint.text = checkpointText(state)
            progressBar.progress = state.progress?.let { (it.fraction * 100.0f).toInt() } ?: 0
            datasetUri.text = state.datasetDisplayName?.let { name ->
                "$name\n${state.datasetUri}"
            } ?: state.datasetUri
                ?: getString(R.string.training_dataset_not_selected)
            error.text = state.message.orEmpty()
            start.isEnabled = state.canStart
            stop.isEnabled = state.canStop
            pause.isEnabled = state.canPause
            resume.isEnabled = state.canResume
            startOver.isEnabled = state.phase !in activePhases && state.phase != TrainingPhase.IDLE
            selectDataset.isEnabled = state.phase !in activePhases
        }
    }

    private fun startTrainingAfterNotificationCheck() {
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
            pendingStart = true
            return
        }
        ioExecutor.execute {
            if (!repository.start()) runOnUiThread { toast("Training could not be started; check dataset and native HTP availability") }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_NOTIFICATIONS || !pendingStart) return
        pendingStart = false
        // Notification permission is not training evidence; a denial leaves the
        // run UI usable but the foreground notification may be hidden by Android.
        ioExecutor.execute {
            if (!repository.start()) runOnUiThread { toast("Training could not be started; check dataset and native HTP availability") }
        }
    }

    private fun displayName(uri: Uri): String? = runCatching {
        contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (!cursor.moveToFirst()) return@use null
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index < 0) null else cursor.getString(index)
        }
    }.getOrNull()

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(KEY_PENDING_START, pendingStart)
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        synchronized(subscriptionLock) { activityStarted = false }
        ioExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

    private fun timingText(state: TrainingUiState): String = state.timingText

    private fun activityText(state: TrainingUiState): String = state.activityText

    private fun checkpointText(state: TrainingUiState): String =
        "CHECKPOINT\n${state.checkpointText}" +
            state.lastCheckpoint?.let { "\nfinite=${it.finite}" }.orEmpty()

    private companion object {
        const val REQUEST_OPEN_DATASET = 121
        const val REQUEST_NOTIFICATIONS = 122
        const val KEY_PENDING_START = "standalone_training_pending_start"
        val activePhases = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
            TrainingPhase.PAUSED,
        )
    }

    private var pendingStart = false
}
