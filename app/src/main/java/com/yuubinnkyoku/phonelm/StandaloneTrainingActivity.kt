package com.yuubinnkyoku.phonelm

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.yuubinnkyoku.phonelm.ui.training.TrainingDashboardApp
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Compose entry point for the application-scoped standalone training session.
 * The repository, including its foreground-service lifecycle, deliberately
 * remains outside this Activity so recreation only reconnects the UI.
 */
class StandaloneTrainingActivity : ComponentActivity() {
    private lateinit var repository: StandaloneTrainingRepository
    private lateinit var generationSession: GenerationSession
    private var subscription: AutoCloseable? = null
    private var generationSubscription: AutoCloseable? = null
    private val subscriptionLock = Any()
    @Volatile private var activityStarted = false
    private var subscriptionGeneration = 0L
    private val ioExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PhoneLM-training-ui-io").apply { isDaemon = true }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val uiDispatchLock = Any()
    private var pendingUiUpdate: PendingUiUpdate? = null
    @Volatile private var lastDeliveredState: TrainingUiState? = null
    private var uiUpdateScheduled = false
    private var composeState by mutableStateOf<TrainingUiState?>(null)
    private var generationComposeState by mutableStateOf(GenerationUiState())

    private val deliverUiUpdate = Runnable {
        val update = synchronized(uiDispatchLock) {
            val next = pendingUiUpdate ?: return@Runnable
            pendingUiUpdate = null
            uiUpdateScheduled = false
            next
        }
        if (isCurrentGeneration(update.generation)) {
            lastDeliveredState = update.state
            composeState = update.state
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingStart = savedInstanceState?.getBoolean(KEY_PENDING_START, false) ?: false
        repository = StandaloneTrainingRepositoryRegistry.get(applicationContext)
        generationSession = GenerationSessionRegistry.get(applicationContext)
        setContent {
            TrainingDashboardApp(
                state = composeState,
                generationState = generationComposeState,
                onSelectDataset = ::openDatasetPicker,
                onStart = ::startTrainingAfterNotificationCheck,
                onStop = { ioExecutor.execute { repository.stop() } },
                onPause = { ioExecutor.execute { repository.pause() } },
                onResume = { ioExecutor.execute { repository.resume() } },
                onStartOver = ::confirmStartOver,
                onGenerationPromptChange = generationSession::updatePrompt,
                onGenerationModeChange = generationSession::updateMode,
                onGenerationTemperatureChange = generationSession::updateTemperature,
                onGenerationTopKChange = generationSession::updateTopK,
                onGenerationSamplingSeedChange = generationSession::updateSamplingSeed,
                onGenerationMaxNewBytesChange = generationSession::updateMaxNewBytes,
                onGenerate = {
                    generationSession.generate(composeState?.phase in generationBlockingTrainingPhases)
                },
            )
        }
    }

    override fun onStart() {
        super.onStart()
        val generation = synchronized(subscriptionLock) {
            activityStarted = true
            ++subscriptionGeneration
            if (pendingStart) pendingStartGeneration = subscriptionGeneration
            subscriptionGeneration
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
        val generationCandidate = generationSession.subscribe { next ->
            // Keep controlled TextFields synchronous with onValueChange on the
            // main thread. Posting even main-thread edits can briefly feed an
            // old String back to the IME and commit/reset an active Japanese
            // composition. Native completion still crosses to main safely.
            if (Looper.myLooper() == Looper.getMainLooper()) {
                if (isCurrentGeneration(generation)) generationComposeState = next
            } else {
                mainHandler.post {
                    if (isCurrentGeneration(generation)) generationComposeState = next
                }
            }
        }
        synchronized(subscriptionLock) {
            if (activityStarted && generation == subscriptionGeneration) {
                generationSubscription = generationCandidate
            } else {
                generationCandidate.close()
            }
        }
    }

    override fun onStop() {
        val old = synchronized(subscriptionLock) {
            activityStarted = false
            ++subscriptionGeneration
            pendingStartGeneration = null
            subscription.also { subscription = null }
        }
        generationSubscription?.close()
        generationSubscription = null
        synchronized(uiDispatchLock) {
            mainHandler.removeCallbacks(deliverUiUpdate)
            uiUpdateScheduled = false
            pendingUiUpdate = null
        }
        old?.close()
        super.onStop()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_OPEN_DATASET || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        if (data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION == 0) {
            toast(getString(R.string.training_dataset_access_failed))
            return
        }
        val callbackGeneration = synchronized(subscriptionLock) {
            if (!activityStarted) return
            subscriptionGeneration
        }
        val selectionToken = repository.nextDatasetSelectionToken()
        ioExecutor.execute {
            try {
                if (!isCurrentGeneration(callbackGeneration)) return@execute
                val selected = repository.selectDataset(
                    TrainingDataset(uri.toString(), displayName(uri)),
                    selectionToken,
                )
                if (!selected) postIfCurrent(callbackGeneration) {
                    toast(getString(R.string.training_dataset_access_failed))
                }
            } catch (_: Throwable) {
                postIfCurrent(callbackGeneration) { toast(getString(R.string.training_dataset_access_failed)) }
            }
        }
    }

    private fun openDatasetPicker() {
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
                    if (!repository.startOver()) postIfCurrentCurrent { toast(START_FAILURE_MESSAGE) }
                }
            }
            .show()
    }

    /** Coalesce worker-thread callbacks; phase/checkpoint/terminal transitions bypass the 125 ms window. */
    private fun render(state: TrainingUiState) {
        val generation = synchronized(subscriptionLock) {
            if (!activityStarted) return
            subscriptionGeneration
        }
        if (isFinishing || isDestroyed) return
        val immediate = state.phase != lastDeliveredState?.phase ||
            state.phase in terminalPhases ||
            state.lastCheckpoint != lastDeliveredState?.lastCheckpoint
        if (state.lastCheckpoint != lastDeliveredState?.lastCheckpoint) generationSession.refreshCheckpoint()
        synchronized(uiDispatchLock) {
            pendingUiUpdate = PendingUiUpdate(state, generation)
            if (immediate) {
                mainHandler.removeCallbacks(deliverUiUpdate)
                uiUpdateScheduled = true
                mainHandler.post(deliverUiUpdate)
            } else if (!uiUpdateScheduled) {
                uiUpdateScheduled = true
                mainHandler.postDelayed(deliverUiUpdate, UI_COALESCE_MS)
            }
        }
    }

    private fun isCurrentGeneration(generation: Long): Boolean =
        synchronized(subscriptionLock) { activityStarted && subscriptionGeneration == generation } &&
            !isFinishing && !isDestroyed

    private fun startTrainingAfterNotificationCheck() {
        val generation = synchronized(subscriptionLock) {
            if (!activityStarted) return
            subscriptionGeneration
        }
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            synchronized(subscriptionLock) {
                if (!activityStarted || subscriptionGeneration != generation) return
                pendingStartGeneration = generation
            }
            pendingStart = true
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
            return
        }
        startOnIo(generation)
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_NOTIFICATIONS || !pendingStart) return
        val generation = synchronized(subscriptionLock) {
            if (!activityStarted || pendingStartGeneration != subscriptionGeneration) return
            pendingStart = false
            pendingStartGeneration = null
            subscriptionGeneration
        }
        // A denial may hide the notification; it is not training evidence and does not block a user-run session.
        startOnIo(generation)
    }

    private fun startOnIo(generation: Long) = ioExecutor.execute {
        if (!isCurrentGeneration(generation)) return@execute
        if (generationSession.snapshot().execution is GenerationState.Running) {
            postIfCurrent(generation) { toast("Generation is already running") }
            return@execute
        }
        // repository.start invokes the lifecycle's foreground-service start before it starts the worker.
        if (!repository.start()) postIfCurrent(generation) { toast(START_FAILURE_MESSAGE) }
    }

    private fun displayName(uri: Uri): String? = runCatching {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            val nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (!cursor.moveToFirst() || nameColumn < 0) null else cursor.getString(nameColumn)
        }
    }.getOrNull()

    private fun postIfCurrent(generation: Long, block: () -> Unit) {
        mainHandler.post { if (isCurrentGeneration(generation)) block() }
    }

    private fun postIfCurrentCurrent(block: () -> Unit) {
        val generation = synchronized(subscriptionLock) { subscriptionGeneration }
        postIfCurrent(generation, block)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(KEY_PENDING_START, pendingStart)
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        synchronized(subscriptionLock) {
            activityStarted = false
            pendingStartGeneration = null
        }
        synchronized(uiDispatchLock) { mainHandler.removeCallbacks(deliverUiUpdate) }
        ioExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

    private companion object {
        const val REQUEST_OPEN_DATASET = 121
        const val REQUEST_NOTIFICATIONS = 122
        const val KEY_PENDING_START = "standalone_training_pending_start"
        const val UI_COALESCE_MS = 125L
        const val START_FAILURE_MESSAGE = "Training could not be started; check dataset and native HTP availability"
        val terminalPhases = setOf(TrainingPhase.COMPLETED, TrainingPhase.ERROR, TrainingPhase.INTERRUPTED)
        val generationBlockingTrainingPhases = setOf(
            TrainingPhase.PREPARING,
            TrainingPhase.INITIALIZING_HTP,
            TrainingPhase.TRAINING,
            TrainingPhase.SAVING_CHECKPOINT,
            TrainingPhase.PAUSED,
        )
    }

    private data class PendingUiUpdate(val state: TrainingUiState, val generation: Long)
    private var pendingStart = false
    private var pendingStartGeneration: Long? = null
}
