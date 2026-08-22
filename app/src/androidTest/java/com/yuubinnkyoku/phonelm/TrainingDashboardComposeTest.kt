package com.yuubinnkyoku.phonelm

import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasAnyDescendant
import androidx.compose.ui.test.hasScrollToIndexAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.v2.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToIndex
import com.yuubinnkyoku.phonelm.ui.training.TrainingDashboardApp
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertEquals

class TrainingDashboardComposeTest {
    @get:Rule val compose = createComposeRule()

    private val trainingDestination = hasClickAction() and hasText("Training")
    private val generationDestination = hasClickAction() and hasText("Generation")

    @Test fun trainingOverviewKeepsMonitoringAndToolbarActionsVisible() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState(),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithContentDescription("Training overview, no scrolling").assertIsDisplayed()
        compose.onNodeWithText("HTP activity").assertIsDisplayed()
        compose.onNodeWithText("Loss trend").assertIsDisplayed()
        compose.onNodeWithText("Stop").assertIsDisplayed()
        compose.onNodeWithText("Pause").assertIsDisplayed()
        compose.onNodeWithText("Details").assertIsDisplayed()
    }

    @Test fun largeFontKeepsRequiredMonitoringAndActionsReachable() {
        compose.setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale = 2f)) {
                TrainingDashboardApp(
                    state = trainingState(),
                    onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
                )
            }
        }

        compose.onNodeWithContentDescription("Training overview, no scrolling").assertIsDisplayed()
        compose.onNodeWithContentDescription(
            "HTP activity is QNN execute wall time divided by its observation window; it is not hardware utilization.",
        ).assertIsDisplayed()
        compose.onNodeWithText("Loss trend").assertIsDisplayed()
        compose.onNodeWithText("Stop").assertIsDisplayed()
        compose.onNodeWithText("Pause").assertIsDisplayed()
        compose.onNodeWithText("Info").assertIsDisplayed()
    }

    @Test fun freshIdleWithoutTargetDoesNotShowStepTargetUnavailableError() {
        compose.setContent {
            TrainingDashboardApp(
                state = TrainingUiState(
                    phase = TrainingPhase.IDLE,
                    progress = null,
                    overview = "Idle",
                    overviewText = "Idle",
                    message = null,
                    timing = null,
                    timingText = "Unavailable",
                    htpActivity = null,
                    activityText = "HTP activity is an observation ratio, not hardware utilization.",
                    lastCheckpoint = null,
                    checkpointText = "Unavailable",
                    datasetUri = null,
                    datasetDisplayName = null,
                    canStart = false,
                    canStop = false,
                    canPause = false,
                    canResume = false,
                    modelConfig = TrainingModelConfig.NICOPEDIA_L19,
                    dashboard = TrainingDashboardSnapshot(),
                ),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithText("Step target unavailable").assertDoesNotExist()
        compose.onNodeWithText("Ready").assertIsDisplayed()
        compose.onNodeWithText("Select").assertIsDisplayed()
    }

    @Test fun runningWithValidTargetShowsStepsAndNotReady() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState(),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithText("16 / 8000 steps").assertIsDisplayed()
        compose.onNodeWithText("Step target unavailable").assertDoesNotExist()
    }

    @Test fun errorWithoutTargetShowsRepositoryMessage() {
        compose.setContent {
            TrainingDashboardApp(
                state = TrainingUiState(
                    phase = TrainingPhase.ERROR,
                    progress = null,
                    overview = "Error",
                    overviewText = "Error",
                    message = "HTP training backend is unavailable: no JNI backend has been configured",
                    timing = null,
                    timingText = "Unavailable",
                    htpActivity = null,
                    activityText = "HTP activity is an observation ratio, not hardware utilization.",
                    lastCheckpoint = null,
                    checkpointText = "Unavailable",
                    datasetUri = "content://test/dataset",
                    datasetDisplayName = "dataset.txt",
                    canStart = false,
                    canStop = false,
                    canPause = false,
                    canResume = false,
                    modelConfig = TrainingModelConfig.NICOPEDIA_L19,
                    dashboard = TrainingDashboardSnapshot(),
                ),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithText("HTP training backend is unavailable: no JNI backend has been configured")
            .assertIsDisplayed()
        compose.onNodeWithText("Step target unavailable").assertDoesNotExist()
    }

    @Test fun initialDestinationIsTrainingAndGenerationIsOneTapAway() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(phase = TrainingPhase.IDLE),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithContentDescription("Training overview, no scrolling").assertIsDisplayed()
        compose.onNodeWithContentDescription("Top-level navigation").assertIsDisplayed()
        compose.onNode(trainingDestination).assertIsSelected()
        compose.onNode(generationDestination).assertIsNotSelected()
        compose.onNode(generationDestination).performClick()
        compose.onNode(generationDestination).assertIsSelected()
        compose.onNodeWithContentDescription("Generation screen").assertIsDisplayed()
        compose.onNodeWithText("Select a compatible checkpoint to resolve the model identity").assertIsDisplayed()
    }

    @Test fun idleModelSettingsApplyCatalogSelection() {
        var applied: TrainingModelConfig? = null
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(
                    phase = TrainingPhase.IDLE,
                    canEditModelConfig = true,
                    selectedModelConfig = TrainingModelConfig.NICOPEDIA_L19,
                ),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
                onModelConfigSelected = { applied = it },
            )
        }

        compose.onNodeWithText("Details").performClick()
        compose.onNodeWithText("Model settings").performClick()
        compose.onNodeWithText("V1024").performClick()
        compose.onNodeWithText("D64").performClick()
        compose.onNodeWithText("FFN64").performClick()
        compose.onNode(
            hasScrollToIndexAction() and hasAnyDescendant(hasText("Apply model settings")),
        ).performScrollToIndex(4)
        compose.onNodeWithText("602,880 parameters").assertExists().assertIsDisplayed()
        compose.onNodeWithText("Head dim 32").assertExists()
        compose.onNodeWithText("Apply model settings").performClick()
        compose.runOnIdle {
            assertEquals(ModelConfigurationCatalog.config(1024, 64, 64), applied)
        }
    }

    @Test fun generationDestinationReturnsToTraining() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(phase = TrainingPhase.IDLE),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNode(generationDestination).performClick()
        compose.onNode(trainingDestination).performClick()
        compose.onNodeWithContentDescription("Training overview, no scrolling").assertIsDisplayed()
    }

    @Test fun runningGenerationStateSurvivesDestinationSwitches() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(phase = TrainingPhase.IDLE),
                generationState = GenerationUiState(
                    execution = GenerationState.Running(
                        GenerationProgress(
                            phase = GenerationPhase.GENERATING,
                            generatedBytes = 37,
                            maxNewBytes = 64,
                            elapsedMs = 420,
                            displayText = "途中",
                        ),
                    ),
                ),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNode(generationDestination).performClick()
        compose.onAllNodesWithText("37 / 64 bytes").onFirst().assertIsDisplayed()
        compose.onNodeWithText("途中").assertExists().performScrollTo().assertIsDisplayed()
        compose.onNode(trainingDestination).performClick()
        compose.onNode(generationDestination).performClick()
        compose.onAllNodesWithText("37 / 64 bytes").onFirst().assertIsDisplayed()
        compose.onNodeWithText("420 ms").assertExists()
        compose.onNodeWithText("途中").assertExists().performScrollTo().assertIsDisplayed()
    }

    @Test fun completedGenerationResultSurvivesDestinationSwitches() {
        val bytes = "answer".toByteArray()
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(phase = TrainingPhase.IDLE),
                generationState = GenerationUiState(
                    execution = GenerationState.Success(
                        GenerationResult(bytes, "answer", bytes.size, 42, "HTP", "hash", false, true, true, "report"),
                    ),
                ),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNode(generationDestination).performClick()
        compose.onNodeWithText("answer").assertExists().performScrollTo().assertIsDisplayed()
        compose.onNode(trainingDestination).performClick()
        compose.onNode(generationDestination).performClick()
        compose.onNodeWithText("answer").assertExists().performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("6 bytes · 42 ms · backend=HTP")
            .assertExists().performScrollTo().assertIsDisplayed()
    }

    @Test fun trainingDetailsDoesNotContainGenerationEntry() {
        compose.setContent {
            TrainingDashboardApp(
                state = trainingState().copy(phase = TrainingPhase.IDLE),
                onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
            )
        }

        compose.onNodeWithText("Details").performClick()
        compose.onNodeWithText("Training details").assertIsDisplayed()
        compose.onNodeWithText("Open Generation").assertDoesNotExist()
    }

    private fun trainingState() = TrainingUiState(
        phase = TrainingPhase.TRAINING,
        progress = TrainingProgress(16, 8_000, 1.25f),
        overview = "Training",
        overviewText = "Training",
        message = null,
        timing = null,
        timingText = "Unavailable",
        htpActivity = null,
        activityText = "QNN execute wall / observation window; not utilization.",
        lastCheckpoint = null,
        checkpointText = "Unavailable",
        datasetUri = "content://test/dataset",
        datasetDisplayName = "dataset.txt",
        canStart = false,
        canStop = true,
        canPause = true,
        canResume = false,
        modelConfig = TrainingModelConfig.NICOPEDIA_L19,
        dashboard = TrainingDashboardSnapshot(
            lossHistory = listOf(TrainingLossHistoryEntry(8, 1.5f), TrainingLossHistoryEntry(16, 1.25f)),
            activityHistory = listOf(TrainingActivityHistoryEntry(16, null, null)),
        ),
    )
}
