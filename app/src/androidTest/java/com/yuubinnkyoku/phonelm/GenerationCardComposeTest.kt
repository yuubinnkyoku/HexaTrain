package com.yuubinnkyoku.phonelm

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.v2.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import com.yuubinnkyoku.phonelm.ui.training.GenerationCard
import org.junit.Rule
import org.junit.Test

class GenerationCardComposeTest {
    @get:Rule val compose = createComposeRule()

    @Test fun emptyPromptAndNoCheckpointDisableGenerate() {
        compose.setContent {
            MaterialTheme { GenerationCard(GenerationUiState(), trainingActive = false) }
        }
        compose.onNodeWithText("No trained checkpoint").assertIsDisplayed()
        compose.onNodeWithText("Generate on HTP").assertIsNotEnabled()
    }

    @Test fun sampleControlsAreVisibleAndValidRequestEnablesGenerate() {
        compose.setContent {
            MaterialTheme {
                GenerationCard(
                    GenerationUiState(
                        prompt = "prompt",
                        mode = GenerationMode.SAMPLE,
                        checkpoint = GenerationCheckpointStatus.Available(checkpoint()),
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

    @Test fun successAndFailureResultsRender() {
        val bytes = "answer".toByteArray()
        compose.setContent {
            MaterialTheme {
                GenerationCard(
                    GenerationUiState(
                        checkpoint = GenerationCheckpointStatus.Available(checkpoint()),
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
                GenerationCard(
                    GenerationUiState(execution = GenerationState.Failed("QNN init failed")),
                    trainingActive = false,
                )
            }
        }
        compose.onNodeWithText("Generation failed").assertIsDisplayed()
        compose.onNodeWithText("QNN init failed").assertIsDisplayed()
    }

    private fun checkpoint() = TrainingCheckpointMetadata(
        "checkpoint", 8_000, TrainingModelConfig.NICOPEDIA_L19,
        "NPRTCKPTV2", 2, true, 1L,
    )
}
