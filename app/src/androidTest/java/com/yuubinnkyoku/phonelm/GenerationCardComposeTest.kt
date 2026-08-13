package com.yuubinnkyoku.phonelm

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.v2.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.yuubinnkyoku.phonelm.ui.training.GenerationScreen
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertEquals

class GenerationCardComposeTest {
    @get:Rule val compose = createComposeRule()

    @Test fun emptyPromptAndNoCheckpointDisableGenerate() {
        compose.setContent {
            MaterialTheme { GenerationScreen(GenerationUiState(), trainingActive = false) }
        }
        compose.onNodeWithText("No usable checkpoint").assertIsDisplayed()
        compose.onNodeWithText("Generate on HTP").assertIsNotEnabled()
    }

    @Test fun sampleControlsAreVisibleAndValidRequestEnablesGenerate() {
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    GenerationUiState(
                        prompt = "prompt",
                        mode = GenerationMode.SAMPLE,
                        checkpoints = listOf(checkpoint(8_000)),
                        selectedCheckpointPath = checkpoint(8_000).path,
                        checkpointMessage = null,
                    ),
                    trainingActive = false,
                )
            }
        }
        compose.onNodeWithText("Temperature").assertIsDisplayed()
        compose.onNodeWithText("TopK").assertIsDisplayed()
        compose.onNodeWithText("SamplingSeed").assertIsDisplayed()
        compose.onNodeWithText("Generate on HTP").assertIsEnabled()
    }

    @Test fun checkpointDropdownPrioritizesCompatibleAndKeepsInvalidDiagnosticSeparate() {
        var selected: String? = null
        val latest = checkpoint(8_000)
        val older = checkpoint(7_500)
        val invalid = checkpoint(8_250, finite = false)
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    GenerationUiState(
                        prompt = "prompt",
                        checkpoints = listOf(latest, older, invalid),
                        selectedCheckpointPath = latest.path,
                        checkpointMessage = null,
                    ),
                    trainingActive = false,
                    onCheckpointSelected = { selected = it; true },
                )
            }
        }
        compose.onNodeWithText(latest.label).performClick()
        compose.onNodeWithText(older.label).assertIsDisplayed().performClick()
        assertEquals(older.path, selected)

        compose.onNodeWithText(invalid.label).assertDoesNotExist()
        compose.onNodeWithText("Show incompatible checkpoints").performClick()
        compose.onNodeWithText(invalid.label).assertIsDisplayed()
    }

    @Test fun generatingProgressAndLiveByteCountRender() {
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    GenerationUiState(
                        execution = GenerationState.Running(
                            GenerationProgress(
                                phase = GenerationPhase.GENERATING,
                                generatedBytes = 37,
                                maxNewBytes = 64,
                                elapsedMs = 420,
                                qnnExecuteAttempts = 45,
                                qnnExecuteSuccesses = 45,
                                displayText = "途中",
                            ),
                        ),
                    ),
                    trainingActive = false,
                )
            }
        }
        compose.onNodeWithText("Generating on HTP").assertIsDisplayed()
        compose.onNodeWithText("37 / 64 bytes").assertIsDisplayed()
        compose.onNodeWithText("45 / 45").assertIsDisplayed()
        compose.onNodeWithText("途中").assertIsDisplayed()
        compose.onNodeWithText("Generate on HTP").assertIsNotEnabled()
    }

    @Test fun successAndFailureResultsRender() {
        val bytes = "answer".toByteArray()
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    GenerationUiState(
                        checkpoints = listOf(checkpoint(8_000)),
                        selectedCheckpointPath = checkpoint(8_000).path,
                        checkpointMessage = null,
                        execution = GenerationState.Success(
                            GenerationResult(bytes, "answer", bytes.size, 42, "HTP", "hash", false, true, true, "report"),
                        ),
                    ),
                    trainingActive = false,
                )
            }
        }
        compose.onNodeWithText("answer").assertIsDisplayed()
        compose.onNodeWithText("6 bytes · 42 ms · backend=HTP").assertIsDisplayed()

        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    GenerationUiState(execution = GenerationState.Failed("QNN init failed")),
                    trainingActive = false,
                )
            }
        }
        compose.onNodeWithText("Generation failed").assertIsDisplayed()
        compose.onNodeWithText("QNN init failed").assertIsDisplayed()
    }

    @Test fun historyListRendersAndRowsOpenById() {
        var selected: String? = null
        val item = historyItem()
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    generationState = GenerationUiState(
                        history = GenerationHistoryUiState(items = listOf(item)),
                    ),
                    trainingActive = false,
                    onHistorySelected = { selected = it; true },
                )
            }
        }

        compose.onNodeWithText("History").performClick()
        compose.onNodeWithText("\"saved prompt\"").assertIsDisplayed().performClick()
        assertEquals("history-1", selected)
        compose.onNodeWithText("Sample · T0.8 · K32").assertIsDisplayed()
        compose.onNodeWithText("step8000 · D32/FFN32").assertIsDisplayed()
        compose.onNodeWithText("64 bytes · 21300 ms · HTP").assertIsDisplayed()
        compose.onNodeWithText("Success").assertIsDisplayed()
    }

    @Test fun historyDetailRendersIdentityRuntimeAndActions() {
        val item = historyItem()
        var used: String? = null
        var deleted: String? = null
        compose.setContent {
            MaterialTheme {
                GenerationScreen(
                    generationState = GenerationUiState(
                        history = GenerationHistoryUiState(items = listOf(item), selectedId = item.record.id),
                    ),
                    trainingActive = false,
                    onUseHistoryAgain = { used = it; true },
                    onDeleteHistory = { deleted = it },
                )
            }
        }

        compose.onNodeWithText("History").performClick()
        compose.onNodeWithText("saved prompt").assertIsDisplayed()
        compose.onNodeWithText("saved output", substring = true).assertIsDisplayed()
        compose.onNodeWithText("fnv1a64:5d1d51359d00d17a").assertIsDisplayed()
        compose.onNodeWithText("QNN attempts").assertIsDisplayed()
        compose.onNodeWithText("QNN successes").assertIsDisplayed()
        compose.onNodeWithText("Use settings again").performClick()
        assertEquals("history-1", used)
        compose.onNodeWithText("History").performClick()
        compose.onNodeWithText("Delete this history entry").performClick()
        assertEquals("history-1", deleted)
    }

    private fun checkpoint(step: Int, finite: Boolean = true) = GenerationCheckpoint(
        path = "checkpoint-$step",
        step = step,
        seed = 1,
        tokens = 32,
        dimension = 32,
        feedForwardDimension = 32,
        layers = 19,
        heads = 2,
        finite = finite,
        parameterHash = "fnv1a64:5d1d51359d00d17a",
        modifiedAtMs = step.toLong(),
        fileSizeBytes = 100,
        compatibility = GenerationCheckpointCompatibility.COMPATIBLE,
        formatValid = true,
    )

    private fun historyItem(): GenerationHistoryItem {
        val output = "saved output".padEnd(64, '!')
        val record = GenerationHistoryRecord(
            id = "history-1",
            createdAtMs = 1,
            promptBytes = "saved prompt".toByteArray(),
            mode = GenerationMode.SAMPLE,
            temperature = 0.8f,
            topK = 32,
            samplingSeed = 7,
            maxNewBytes = 64,
            checkpointStep = 8_000,
            checkpointParameterHash = "fnv1a64:5d1d51359d00d17a",
            vocabulary = 256,
            tokens = 32,
            dimension = 32,
            feedForwardDimension = 32,
            layers = 19,
            heads = 2,
            generatedBytes = output.toByteArray(),
            elapsedMs = 21_300,
            backend = "HTP",
            qnnExecuteAttempts = 45,
            qnnExecuteSuccesses = 45,
            qnnExecuteFailures = 0,
            cpuFallback = false,
            finite = true,
            status = GenerationHistoryStatus.SUCCESS,
        )
        return GenerationHistoryItem(record, "saved prompt", output)
    }
}
