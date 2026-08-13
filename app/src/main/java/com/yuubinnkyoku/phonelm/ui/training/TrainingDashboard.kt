package com.yuubinnkyoku.phonelm.ui.training

import android.os.Build
import androidx.activity.compose.BackHandler
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingToolbarDefaults
import androidx.compose.material3.HorizontalFloatingToolbar
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearWavyProgressIndicator
import androidx.compose.material3.MaterialExpressiveTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.MotionScheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.TextButton
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.RocketLaunch
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yuubinnkyoku.phonelm.PhaseTiming
import com.yuubinnkyoku.phonelm.GenerationMode
import com.yuubinnkyoku.phonelm.GenerationHistoryItem
import com.yuubinnkyoku.phonelm.GenerationHistoryStatus
import com.yuubinnkyoku.phonelm.GenerationHistoryUiState
import com.yuubinnkyoku.phonelm.GenerationPhase
import com.yuubinnkyoku.phonelm.GenerationProgress
import com.yuubinnkyoku.phonelm.GenerationState
import com.yuubinnkyoku.phonelm.GenerationUiState
import com.yuubinnkyoku.phonelm.TimingBackend
import com.yuubinnkyoku.phonelm.TrainingDashboardEvent
import com.yuubinnkyoku.phonelm.TrainingDashboardEventType
import com.yuubinnkyoku.phonelm.TrainingDashboardSnapshot
import com.yuubinnkyoku.phonelm.TrainingActivityHistoryEntry
import com.yuubinnkyoku.phonelm.TrainingLossHistoryEntry
import com.yuubinnkyoku.phonelm.TrainingModelConfig
import com.yuubinnkyoku.phonelm.TrainingOperationPhase
import com.yuubinnkyoku.phonelm.TrainingPhase
import com.yuubinnkyoku.phonelm.TrainingProgress
import com.yuubinnkyoku.phonelm.TrainingRuntimeEvidence
import com.yuubinnkyoku.phonelm.TrainingTiming
import com.yuubinnkyoku.phonelm.TrainingUiState
import java.util.Locale
import java.text.DateFormat
import java.util.Date

private enum class TopLevelDestination { TRAINING, GENERATION }
private enum class GenerationView { GENERATE, HISTORY }

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun TrainingDashboardApp(
    state: TrainingUiState?,
    generationState: GenerationUiState = GenerationUiState(),
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStartOver: () -> Unit,
    onGenerationPromptChange: (String) -> Unit = {},
    onGenerationModeChange: (GenerationMode) -> Unit = {},
    onGenerationTemperatureChange: (String) -> Unit = {},
    onGenerationTopKChange: (String) -> Unit = {},
    onGenerationSamplingSeedChange: (String) -> Unit = {},
    onGenerationMaxNewBytesChange: (String) -> Unit = {},
    onGenerationCheckpointSelected: (String) -> Boolean = { false },
    onGenerate: () -> Unit = {},
    onGenerationHistorySelected: (String) -> Boolean = { false },
    onGenerationHistoryDetailClosed: () -> Unit = {},
    onDeleteGenerationHistory: (String) -> Unit = {},
    onClearGenerationHistory: () -> Unit = {},
    onUseGenerationHistoryAgain: (String) -> Boolean = { false },
) {
    var destination by remember { mutableStateOf(TopLevelDestination.TRAINING) }
    BackHandler(enabled = destination == TopLevelDestination.GENERATION) {
        destination = TopLevelDestination.TRAINING
    }
    val context = LocalContext.current
    val base = phoneLmDarkScheme()
    val colors = if (Build.VERSION.SDK_INT >= 31) dynamicDarkColorScheme(context).copy(
        primary = base.primary,
        onPrimary = base.onPrimary,
        primaryContainer = base.primaryContainer,
        onPrimaryContainer = base.onPrimaryContainer,
        secondary = base.secondary,
        onSecondary = base.onSecondary,
        secondaryContainer = base.secondaryContainer,
        onSecondaryContainer = base.onSecondaryContainer,
        tertiary = base.tertiary,
        onTertiary = base.onTertiary,
        error = base.error,
        background = base.background,
        surface = base.surface,
        surfaceVariant = base.surfaceVariant,
    ) else base
    MaterialExpressiveTheme(
        colorScheme = colors,
        motionScheme = MotionScheme.expressive(),
        typography = MaterialTheme.typography.copy(
            displaySmall = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Black),
            headlineSmall = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
        ),
        shapes = Shapes(
            small = RoundedCornerShape(12.dp),
            medium = RoundedCornerShape(18.dp),
            large = RoundedCornerShape(28.dp),
            extraLarge = RoundedCornerShape(36.dp),
        ),
    ) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(Modifier.fillMaxSize()) {
                TopLevelDestinationBar(
                    selected = destination,
                    onSelected = { destination = it },
                )
                Box(Modifier.fillMaxWidth().weight(1f)) {
                    when (destination) {
                        TopLevelDestination.TRAINING -> if (state == null) {
                            TrainingLoading()
                        } else {
                            TrainingDashboard(
                                state = state,
                                onSelectDataset = onSelectDataset,
                                onStart = onStart,
                                onStop = onStop,
                                onPause = onPause,
                                onResume = onResume,
                                onStartOver = onStartOver,
                                generationRunning = generationState.execution is GenerationState.Running,
                            )
                        }
                        TopLevelDestination.GENERATION -> GenerationScreen(
                            generationState = generationState,
                            trainingActive = state?.phase in activePhases,
                            onPromptChange = onGenerationPromptChange,
                            onModeChange = onGenerationModeChange,
                            onTemperatureChange = onGenerationTemperatureChange,
                            onTopKChange = onGenerationTopKChange,
                            onSamplingSeedChange = onGenerationSamplingSeedChange,
                            onMaxNewBytesChange = onGenerationMaxNewBytesChange,
                            onCheckpointSelected = onGenerationCheckpointSelected,
                            onGenerate = onGenerate,
                            onHistorySelected = onGenerationHistorySelected,
                            onHistoryDetailClosed = onGenerationHistoryDetailClosed,
                            onDeleteHistory = onDeleteGenerationHistory,
                            onClearHistory = onClearGenerationHistory,
                            onUseHistoryAgain = onUseGenerationHistoryAgain,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TopLevelDestinationBar(
    selected: TopLevelDestination,
    onSelected: (TopLevelDestination) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth().statusBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .semantics { contentDescription = "Top-level navigation" },
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TopLevelDestinationButton(
            label = "Training",
            icon = Icons.Default.ModelTraining,
            selected = selected == TopLevelDestination.TRAINING,
            onClick = { onSelected(TopLevelDestination.TRAINING) },
            modifier = Modifier.weight(1f),
        )
        TopLevelDestinationButton(
            label = "Generation",
            icon = Icons.Default.AutoAwesome,
            selected = selected == TopLevelDestination.GENERATION,
            onClick = { onSelected(TopLevelDestination.GENERATION) },
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun TopLevelDestinationButton(
    label: String,
    icon: ImageVector,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier,
) {
    if (selected) {
        Button(onClick = onClick, modifier = modifier) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
            Text(label)
        }
    } else {
        OutlinedButton(onClick = onClick, modifier = modifier) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
            Text(label)
        }
    }
}

private fun phoneLmDarkScheme(): ColorScheme = darkColorScheme(
    primary = Color(0xFF9BEAFF), onPrimary = Color(0xFF003640),
    primaryContainer = Color(0xFF103C46), onPrimaryContainer = Color(0xFFC4F3FF),
    secondary = Color(0xFFE3B7FF), onSecondary = Color(0xFF3A1454),
    secondaryContainer = Color(0xFF263D2B), onSecondaryContainer = Color(0xFFB9F2BF),
    tertiary = Color(0xFFFFD28A), onTertiary = Color(0xFF432C00),
    surface = Color(0xFF111318), surfaceVariant = Color(0xFF20232B), background = Color(0xFF090B10),
    onSurface = Color(0xFFE3E5ED), onSurfaceVariant = Color(0xFFC3C6D1), error = Color(0xFFFFB4AB),
)

private val TelemetryFontFamily = FontFamily.Monospace

/** Semantic roles keep the console palette meaningful while still following Material colors. */
private data class TrainingSemanticColors(
    val qnn: Color,
    val cpu: Color,
    val loss: Color,
    val checkpoint: Color,
    val warning: Color,
    val unavailable: Color,
)

@Composable
private fun trainingSemanticColors() = TrainingSemanticColors(
    qnn = MaterialTheme.colorScheme.primary,
    cpu = MaterialTheme.colorScheme.tertiary,
    loss = MaterialTheme.colorScheme.secondary,
    checkpoint = MaterialTheme.colorScheme.onSecondaryContainer,
    warning = MaterialTheme.colorScheme.error,
    unavailable = MaterialTheme.colorScheme.onSurfaceVariant,
)

@Composable
private fun TrainingLoading() = Box(Modifier.fillMaxSize().padding(24.dp)) {
    Text("Connecting to training session…")
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun TrainingDashboard(
    state: TrainingUiState,
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStartOver: () -> Unit,
    generationRunning: Boolean = false,
) {
    var detailsVisible by remember { mutableStateOf(false) }
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            Box(
                modifier = Modifier.fillMaxWidth().navigationBarsPadding()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                CompactActionDock(
                    state,
                    onSelectDataset,
                    onStart,
                    onStop,
                    onPause,
                    onResume,
                    onStartOver,
                    onDetails = { detailsVisible = true },
                    generationRunning = generationRunning,
                )
            }
        },
    ) { scaffoldPadding ->
        BoxWithConstraints(
            Modifier.fillMaxSize().padding(scaffoldPadding)
                .semantics { contentDescription = "Training overview, no scrolling" },
        ) {
            val density = trainingOverviewDensity(
                heightDp = maxHeight.value.toInt(),
                fontScale = LocalDensity.current.fontScale,
            )
            TrainingOverview(state, density)
        }
    }
    if (detailsVisible) {
        ModalBottomSheet(onDismissRequest = { detailsVisible = false }) {
            TrainingDetails(state, onSelectDataset, onStartOver)
        }
    }
}

/** Fixed portrait monitoring surface. Deliberately contains no scroll container. */
@Composable
private fun TrainingOverview(state: TrainingUiState, density: TrainingOverviewDensity) {
    val gap = density.sectionGapDp.dp
    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = density.outerPaddingDp.dp, vertical = gap),
        verticalArrangement = Arrangement.spacedBy(gap),
    ) {
        CompactTrainingHero(state, density)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(gap)) {
            ActivityMetric(
                "HTP activity",
                currentHtpObservationRatio(state.dashboard.activityHistory)?.let(::percent) ?: "—",
                "wall/obs ≠ utilization",
                Modifier.weight(1f),
                density,
                "HTP activity is QNN execute wall time divided by its observation window; it is not hardware utilization.",
            )
            ActivityMetric("CPU activity", currentProcessCpuPercent(state.dashboard.activityHistory)?.let(::percent) ?: "—", "process CPU", Modifier.weight(1f), density)
        }
        CompactPerformanceGrid(state, density)
        if (density != TrainingOverviewDensity.ACCESSIBLE) CompactTelemetryStrip(state, density)
        CompactLossCard(
            state.dashboard,
            Modifier.fillMaxWidth().weight(1f).defaultMinSize(minHeight = density.graphMinimumHeightDp.dp),
            density,
        )
        LatestStatusRow(state, density)
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun CompactTrainingHero(state: TrainingUiState, density: TrainingOverviewDensity) {
    val semantic = trainingSemanticColors()
    val target = when (state.phase) {
        TrainingPhase.ERROR -> MaterialTheme.colorScheme.errorContainer
        TrainingPhase.TRAINING -> MaterialTheme.colorScheme.primaryContainer
        TrainingPhase.COMPLETED -> MaterialTheme.colorScheme.secondaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    val container by animateColorAsState(target, label = "training phase card")
    val progress = state.progress
    DenseCard(container, density) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(
                    compactPhaseLabel(state.phase, density),
                    style = if (density in compactDensityTiers) MaterialTheme.typography.labelLarge else MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    stepTargetLine(state.phase, state.progress, state.message)
                        ?: if (state.phase == TrainingPhase.IDLE) "Ready" else "—",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontFamily = TelemetryFontFamily,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Text(
                progress?.let { String.format(Locale.ROOT, "%.1f%%", it.fraction * 100f) } ?: "—",
                fontSize = density.heroPercentSp.sp,
                fontWeight = FontWeight.Black,
                fontFamily = TelemetryFontFamily,
                color = when (state.phase) {
                    TrainingPhase.ERROR -> semantic.warning
                    TrainingPhase.COMPLETED -> semantic.checkpoint
                    else -> semantic.qnn
                },
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        val fraction by animateFloatAsState(progress?.fraction ?: 0f, label = "training progress")
        LinearWavyProgressIndicator(
            progress = { fraction },
            modifier = Modifier.fillMaxWidth().height(8.dp),
            color = semantic.qnn,
            trackColor = MaterialTheme.colorScheme.surface,
            amplitude = { if (state.phase == TrainingPhase.TRAINING) 0.35f else 0f },
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            InlineMetric("Loss", progress?.loss?.let(::loss) ?: "—", Modifier.weight(1f))
            InlineMetric("ETA", state.dashboard.etaMs?.let(::duration) ?: "—", Modifier.weight(1f), end = true)
        }
    }
}

@Composable
private fun ActivityMetric(
    label: String,
    value: String,
    qualifier: String,
    modifier: Modifier,
    density: TrainingOverviewDensity,
    semanticDescription: String? = null,
) = DenseCard(
    MaterialTheme.colorScheme.surfaceVariant,
    density,
    if (semanticDescription == null) modifier else modifier.semantics { contentDescription = semanticDescription },
) {
        val semantic = trainingSemanticColors()
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text(
            value,
            style = if (density == TrainingOverviewDensity.ACCESSIBLE) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Black,
            fontFamily = TelemetryFontFamily,
            color = if (label.startsWith("HTP")) semantic.qnn else semantic.cpu,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        Text(qualifier, style = MaterialTheme.typography.labelSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }

@Composable
private fun CompactPerformanceGrid(state: TrainingUiState, density: TrainingOverviewDensity) {
    val current = state.timing?.aggregate?.current
    val semantic = trainingSemanticColors()
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            PerformanceCell(if (density in compactDensityTiers) "F+B" else "Fwd + Bwd", phaseMs(current?.entries()?.get(TrainingOperationPhase.FUSED_FORWARD_BACKWARD)), Modifier.weight(1f), semantic.qnn, density)
            PerformanceCell("Adam", phaseMs(current?.entries()?.get(TrainingOperationPhase.ADAM)), Modifier.weight(1f), semantic.loss, density)
            PerformanceCell("Step", state.dashboard.currentStepWallTimeMs?.let(::duration) ?: "—", Modifier.weight(1f), semantic.cpu, density)
        }
    }
}

@Composable
private fun PerformanceCell(label: String, value: String, modifier: Modifier, valueColor: Color, density: TrainingOverviewDensity) = Column(modifier) {
    Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
    Text(value, style = if (density == TrainingOverviewDensity.ACCESSIBLE) MaterialTheme.typography.labelLarge else MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, fontFamily = TelemetryFontFamily, color = valueColor, maxLines = 1, overflow = TextOverflow.Ellipsis)
}

@Composable
private fun CompactTelemetryStrip(state: TrainingUiState, density: TrainingOverviewDensity) {
    val semantic = trainingSemanticColors()
    val memory = state.dashboard.currentMemoryBytes?.let(::bytes) ?: "—"
    val elapsed = state.timing?.elapsedMs?.let(::duration) ?: "—"
    val average = state.dashboard.averageStepWallTimeMs?.let(::duration) ?: "—"
    val checkpoint = state.lastCheckpoint?.createdAtMs?.let { createdAt ->
        checkpointAgeMs(System.currentTimeMillis(), createdAt)?.let(::duration)
    } ?: "—"
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CompactTelemetryItem("PSS", memory, semantic.cpu, Modifier.weight(1f))
            CompactTelemetryItem("ELAPSED", elapsed, MaterialTheme.colorScheme.onSurface, Modifier.weight(1f))
            CompactTelemetryItem("AVG STEP", average, semantic.loss, Modifier.weight(1f))
            if (density != TrainingOverviewDensity.COMPACT) {
                CompactTelemetryItem("CKPT AGE", checkpoint, semantic.checkpoint, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun CompactTelemetryItem(label: String, value: String, valueColor: Color, modifier: Modifier) = Column(
    modifier,
    horizontalAlignment = Alignment.Start,
) {
    Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
    Text(value, style = MaterialTheme.typography.labelMedium, fontFamily = TelemetryFontFamily, color = valueColor, maxLines = 1, overflow = TextOverflow.Ellipsis)
}

@Composable
private fun CompactLossCard(snapshot: TrainingDashboardSnapshot, modifier: Modifier, density: TrainingOverviewDensity) =
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density, modifier, fillHeight = true) {
        val values = snapshot.lossHistory
        val summary = summarizeLoss(values)
        val semantic = trainingSemanticColors()
        val gridColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.18f)
        val historyLineColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.55f)
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Loss trend", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f), color = semantic.loss)
            Text(
                summary.latest?.let { "${loss(it.loss)} · step ${it.step}" } ?: "No samples",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = TelemetryFontFamily,
                color = if (summary.latest == null) semantic.unavailable else semantic.loss,
            )
        }
        if (density != TrainingOverviewDensity.ACCESSIBLE) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("min ${summary.minimum?.let(::loss) ?: "—"}", style = MaterialTheme.typography.labelSmall, fontFamily = TelemetryFontFamily, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text("max ${summary.maximum?.let(::loss) ?: "—"}", style = MaterialTheme.typography.labelSmall, fontFamily = TelemetryFontFamily, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text("n=${values.size}", style = MaterialTheme.typography.labelSmall, fontFamily = TelemetryFontFamily, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Canvas(
            Modifier.fillMaxWidth().weight(1f).semantics {
                contentDescription = summary.latest?.let {
                    "Loss history with ${values.size} observed samples; latest ${loss(it.loss)} at step ${it.step}"
                } ?: "Loss history with no observed samples"
            },
        ) {
            if (values.isEmpty()) return@Canvas
            val min = values.minOf { it.loss }
            val max = values.maxOf { it.loss }
            val range = max - min
            fun point(index: Int) = Offset(
                if (values.lastIndex == 0) size.width / 2f else index.toFloat() / values.lastIndex * size.width,
                if (range > 0f) size.height - ((values[index].loss - min) / range * size.height) else size.height / 2f,
            )
            for (row in 1..3) {
                val y = size.height * row / 4f
                drawLine(gridColor, Offset(0f, y), Offset(size.width, y), 1f)
            }
            for (column in 1..3) {
                val x = size.width * column / 4f
                drawLine(gridColor, Offset(x, 0f), Offset(x, size.height), 1f)
            }
            if (values.size == 1) {
                drawCircle(semantic.loss, radius = 5f, center = point(0))
                return@Canvas
            }
            for (index in 1..values.lastIndex) drawLine(historyLineColor, point(index - 1), point(index), 2f, StrokeCap.Round)
            val recentStart = (values.size - minOf(values.size, 48)).coerceAtLeast(1)
            for (index in recentStart..values.lastIndex) drawLine(semantic.loss, point(index - 1), point(index), 4f, StrokeCap.Round)
            drawCircle(semantic.loss, radius = 5f, center = point(values.lastIndex))
        }
    }

@Composable
private fun LatestStatusRow(state: TrainingUiState, density: TrainingOverviewDensity) =
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density) {
        val semantic = trainingSemanticColors()
        val event = latestImportantEvent(state.dashboard.eventTimeline)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Column(Modifier.weight(0.42f)) {
                Text("Checkpoint", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(
                    state.lastCheckpoint?.completedStep?.let { "Step $it" } ?: "—",
                    fontWeight = FontWeight.Bold,
                    fontFamily = TelemetryFontFamily,
                    color = if (state.lastCheckpoint == null) semantic.unavailable else semantic.checkpoint,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Column(Modifier.weight(0.58f)) {
                Text("Important event", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(
                    event?.let { eventDescription(it).first } ?: "—",
                    fontWeight = FontWeight.Bold,
                    color = eventColor(event, semantic),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                event?.step?.takeIf { density != TrainingOverviewDensity.ACCESSIBLE }?.let { step ->
                    Text("step $step", style = MaterialTheme.typography.labelSmall, fontFamily = TelemetryFontFamily, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }

private fun eventColor(event: TrainingDashboardEvent?, semantic: TrainingSemanticColors): Color = when (event?.type) {
    TrainingDashboardEventType.ERROR,
    TrainingDashboardEventType.QNN_RETURN,
    TrainingDashboardEventType.TENSOR_FINITE,
    TrainingDashboardEventType.CPU_FALLBACK,
    -> semantic.warning
    TrainingDashboardEventType.CHECKPOINT -> semantic.checkpoint
    TrainingDashboardEventType.RESUME -> semantic.qnn
    null, TrainingDashboardEventType.PHASE -> semantic.unavailable
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun CompactActionDock(
    state: TrainingUiState,
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStartOver: () -> Unit,
    onDetails: () -> Unit,
    generationRunning: Boolean,
) {
    val primary = trainingToolbarPrimaryAction(
        phase = state.phase,
        canStart = state.canStart,
        canStop = state.canStop,
        canPause = state.canPause,
        canResume = state.canResume,
    ).toToolbarAction()
    val fontScale = LocalDensity.current.fontScale
    val compactPadding = if (fontScale >= 1.2f) {
        PaddingValues(horizontal = 8.dp, vertical = 4.dp)
    } else {
        PaddingValues(horizontal = 10.dp, vertical = 6.dp)
    }
    val semantic = trainingSemanticColors()
    val secondarySelect = state.phase !in activePhases && primary != ToolbarAction.SELECT
    HorizontalFloatingToolbar(
        expanded = true,
        modifier = Modifier.fillMaxWidth(),
        colors = FloatingToolbarDefaults.standardFloatingToolbarColors(),
        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
        shape = RoundedCornerShape(26.dp),
        leadingContent = {
            if (state.canStop) {
                ToolbarActionButton(ToolbarAction.STOP, onStop, primary = false, compactPadding, semantic, fontScale >= 1.2f)
            } else if (secondarySelect) {
                ToolbarActionButton(ToolbarAction.SELECT, onSelectDataset, primary = false, compactPadding, semantic, fontScale >= 1.2f)
            }
        },
        trailingContent = {
            if (primary != ToolbarAction.DETAILS) {
                ToolbarActionButton(ToolbarAction.DETAILS, onDetails, primary = false, compactPadding, semantic, fontScale >= 1.2f)
            }
        },
    ) {
        ToolbarActionButton(
            action = primary,
            onClick = when (primary) {
                ToolbarAction.PAUSE -> onPause
                ToolbarAction.RESUME -> onResume
                ToolbarAction.START -> onStart
                ToolbarAction.START_OVER -> onStartOver
                ToolbarAction.SELECT -> onSelectDataset
                ToolbarAction.DETAILS -> onDetails
                ToolbarAction.STOP -> onStop
            },
            primary = true,
            compactPadding = compactPadding,
            semantic = semantic,
            compactLabel = fontScale >= 1.2f,
            enabled = !generationRunning || primary !in setOf(ToolbarAction.START, ToolbarAction.START_OVER),
        )
    }
}

private enum class ToolbarAction {
    STOP,
    PAUSE,
    RESUME,
    START,
    START_OVER,
    SELECT,
    DETAILS,
}

private fun TrainingToolbarPrimaryAction.toToolbarAction() = when (this) {
    TrainingToolbarPrimaryAction.PAUSE -> ToolbarAction.PAUSE
    TrainingToolbarPrimaryAction.RESUME -> ToolbarAction.RESUME
    TrainingToolbarPrimaryAction.START -> ToolbarAction.START
    TrainingToolbarPrimaryAction.START_OVER -> ToolbarAction.START_OVER
    TrainingToolbarPrimaryAction.SELECT -> ToolbarAction.SELECT
    TrainingToolbarPrimaryAction.DETAILS -> ToolbarAction.DETAILS
}

@Composable
private fun ToolbarActionButton(
    action: ToolbarAction,
    onClick: () -> Unit,
    primary: Boolean,
    compactPadding: PaddingValues,
    semantic: TrainingSemanticColors,
    compactLabel: Boolean,
    enabled: Boolean = true,
) {
    val icon = when (action) {
        ToolbarAction.STOP -> Icons.Default.Stop
        ToolbarAction.PAUSE -> Icons.Default.Pause
        ToolbarAction.RESUME -> Icons.Default.PlayArrow
        ToolbarAction.START -> Icons.Default.PlayArrow
        ToolbarAction.START_OVER -> Icons.Default.Refresh
        ToolbarAction.SELECT -> Icons.Default.FolderOpen
        ToolbarAction.DETAILS -> Icons.Default.Info
    }
    val label = when (action) {
        ToolbarAction.STOP -> "Stop"
        ToolbarAction.PAUSE -> "Pause"
        ToolbarAction.RESUME -> "Resume"
        ToolbarAction.START -> "Start"
        ToolbarAction.START_OVER -> if (compactLabel) "Restart" else "Start over"
        ToolbarAction.SELECT -> "Select"
        ToolbarAction.DETAILS -> if (compactLabel) "Info" else "Details"
    }
    val contentColor = when (action) {
        ToolbarAction.STOP -> semantic.warning
        ToolbarAction.SELECT, ToolbarAction.DETAILS -> MaterialTheme.colorScheme.onSurface
        else -> MaterialTheme.colorScheme.onPrimaryContainer
    }
    val buttonContent: @Composable () -> Unit = {
        Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        Text(label, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
    if (primary) {
        Button(
            onClick = onClick,
            enabled = enabled,
            contentPadding = compactPadding,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                contentColor = contentColor,
            ),
        ) {
            buttonContent()
        }
    } else {
        TextButton(
            onClick = onClick,
            enabled = enabled,
            contentPadding = compactPadding,
            colors = ButtonDefaults.textButtonColors(contentColor = contentColor),
        ) {
            buttonContent()
        }
    }
}

@Composable
private fun TrainingDetails(
    state: TrainingUiState,
    onSelectDataset: () -> Unit,
    onStartOver: () -> Unit,
) {
    LazyColumn(
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Text("Training details", style = MaterialTheme.typography.headlineSmall) }
        if (state.phase in terminalPhases) item { TrainingSummaryCard(state) }
        item { ModelConfigCard(state.modelConfig, state.datasetDisplayName, state.datasetUri) }
        item { DetailedPerformanceCard(state) }
        item { TrainingEventTimeline(state.dashboard.eventTimeline) }
        item { DiagnosticCard(state) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onSelectDataset, enabled = state.phase !in activePhases, modifier = Modifier.weight(1f)) { Text("Select dataset") }
                OutlinedButton(
                    onClick = onStartOver,
                    enabled = state.phase !in activePhases && state.phase != TrainingPhase.IDLE,
                    modifier = Modifier.weight(1f),
                ) { Text("Start over") }
            }
        }
    }
}

@Composable
internal fun GenerationScreen(
    generationState: GenerationUiState,
    trainingActive: Boolean,
    onPromptChange: (String) -> Unit = {},
    onModeChange: (GenerationMode) -> Unit = {},
    onTemperatureChange: (String) -> Unit = {},
    onTopKChange: (String) -> Unit = {},
    onSamplingSeedChange: (String) -> Unit = {},
    onMaxNewBytesChange: (String) -> Unit = {},
    onCheckpointSelected: (String) -> Boolean = { false },
    onGenerate: () -> Unit = {},
    onHistorySelected: (String) -> Boolean = { false },
    onHistoryDetailClosed: () -> Unit = {},
    onDeleteHistory: (String) -> Unit = {},
    onClearHistory: () -> Unit = {},
    onUseHistoryAgain: (String) -> Boolean = { false },
) {
    var view by remember { mutableStateOf(GenerationView.GENERATE) }
    val running = generationState.execution is GenerationState.Running
    var checkpointMenuExpanded by remember { mutableStateOf(false) }
    var showIncompatible by remember { mutableStateOf(false) }
    val compatible = generationState.checkpoints.filter { it.usable }
    val incompatible = generationState.checkpoints.filterNot { it.usable }
    BackHandler(enabled = view == GenerationView.HISTORY) {
        if (generationState.history.selected != null) onHistoryDetailClosed()
        else view = GenerationView.GENERATE
    }
    Column(Modifier.fillMaxSize().semantics { contentDescription = "Generation screen" }) {
        if (view == GenerationView.GENERATE) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Generation", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                OutlinedButton(onClick = { view = GenerationView.HISTORY }) {
                    Icon(Icons.Default.History, contentDescription = null, modifier = Modifier.size(18.dp))
                    Text("History")
                }
            }
    LazyColumn(
        modifier = Modifier.fillMaxWidth().weight(1f),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            DetailCard("Model identity") {
                SelectionContainer {
                    Text("D32 / FFN32 / L19 / H2 / T32", fontFamily = TelemetryFontFamily)
                }
            }
        }
        item {
            DetailCard("Generate on HTP") {
    Text("Checkpoint", style = MaterialTheme.typography.labelLarge)
    Box(Modifier.fillMaxWidth()) {
        OutlinedButton(
            onClick = { checkpointMenuExpanded = true },
            enabled = !running && compatible.isNotEmpty(),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                generationState.selectedCheckpoint?.label
                    ?: generationState.checkpointMessage
                    ?: "Select checkpoint",
                modifier = Modifier.weight(1f),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Icon(Icons.Default.KeyboardArrowDown, contentDescription = "Open checkpoint selector")
        }
        DropdownMenu(
            expanded = checkpointMenuExpanded,
            onDismissRequest = { checkpointMenuExpanded = false },
        ) {
            DropdownMenuItem(text = { Text("Compatible") }, enabled = false, onClick = {})
            compatible.forEach { checkpoint ->
                DropdownMenuItem(
                    text = { Text(checkpoint.label) },
                    onClick = {
                        if (onCheckpointSelected(checkpoint.path)) checkpointMenuExpanded = false
                    },
                )
            }
        }
    }
    generationState.selectedCheckpoint?.let { checkpoint ->
        SelectionContainer {
            Text(
                "step ${checkpoint.step}\nD${checkpoint.dimension} / FFN${checkpoint.feedForwardDimension} / " +
                    "L${checkpoint.layers} / H${checkpoint.heads} / T${checkpoint.tokens}\nfinite",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
    generationState.checkpointMessage?.let {
        SelectionContainer { Text(it, color = MaterialTheme.colorScheme.error) }
    }
    generationState.checkpointWarning?.let {
        SelectionContainer { Text(it, color = MaterialTheme.colorScheme.tertiary) }
    }
    if (incompatible.isNotEmpty()) {
        TextButton(onClick = { showIncompatible = !showIncompatible }, enabled = !running) {
            Icon(
                if (showIncompatible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
            Text(if (showIncompatible) "Hide incompatible checkpoints" else "Show incompatible checkpoints")
        }
        if (showIncompatible) {
            Card(Modifier.fillMaxWidth()) {
                SelectionContainer {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Incompatible / diagnostic only", fontWeight = FontWeight.Bold)
                        incompatible.forEach { Text(it.label, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                    }
                }
            }
        }
    }
    OutlinedTextField(
        value = generationState.prompt,
        onValueChange = onPromptChange,
        enabled = !running,
        label = { Text("Prompt") },
        minLines = 3,
        modifier = Modifier.fillMaxWidth(),
    )
    Row(verticalAlignment = Alignment.CenterVertically) {
        RadioButton(
            selected = generationState.mode == GenerationMode.GREEDY,
            onClick = { onModeChange(GenerationMode.GREEDY) },
            enabled = !running,
        )
        Text("Greedy")
        RadioButton(
            selected = generationState.mode == GenerationMode.SAMPLE,
            onClick = { onModeChange(GenerationMode.SAMPLE) },
            enabled = !running,
        )
        Text("Sample")
    }
    if (generationState.mode == GenerationMode.SAMPLE) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = generationState.temperatureText,
                onValueChange = onTemperatureChange,
                enabled = !running,
                label = { Text("Temperature") },
                singleLine = true,
                modifier = Modifier.weight(1f),
            )
            OutlinedTextField(
                value = generationState.topKText,
                onValueChange = onTopKChange,
                enabled = !running,
                label = { Text("TopK") },
                singleLine = true,
                modifier = Modifier.weight(1f),
            )
        }
        OutlinedTextField(
            value = generationState.samplingSeedText,
            onValueChange = onSamplingSeedChange,
            enabled = !running,
            label = { Text("SamplingSeed") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
    }
    OutlinedTextField(
        value = generationState.maxNewBytesText,
        onValueChange = onMaxNewBytesChange,
        enabled = !running,
        label = { Text("Max new bytes") },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
    Button(
        onClick = onGenerate,
        enabled = generationState.canGenerate && !trainingActive,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Icon(Icons.Default.RocketLaunch, contentDescription = null, modifier = Modifier.size(18.dp))
        Text("Generate on HTP")
    }
    if (trainingActive) SelectionContainer { Text("Generation is unavailable while training is active") }
    when (val execution = generationState.execution) {
        GenerationState.Idle -> Unit
        is GenerationState.Running -> GenerationProgressCard(execution.progress)
        is GenerationState.Success -> {
            GenerationOutputCard(
                title = "Generated output",
                text = execution.result.displayText,
                supportingText = "${execution.result.byteCount} bytes · ${execution.result.elapsedMs} ms · " +
                    "backend=${execution.result.backend}",
            )
        }
        is GenerationState.Failed -> Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
            modifier = Modifier.fillMaxWidth(),
        ) {
            SelectionContainer {
                Column(Modifier.padding(12.dp)) {
                    Text("Generation failed", fontWeight = FontWeight.Bold)
                    Text(execution.message)
                }
            }
        }
    }
            }
        }
    }
        } else {
            GenerationHistoryView(
                history = generationState.history,
                onBackToGenerate = { view = GenerationView.GENERATE },
                onSelected = onHistorySelected,
                onCloseDetail = onHistoryDetailClosed,
                onDelete = onDeleteHistory,
                onClear = onClearHistory,
                onUseAgain = { id ->
                    if (onUseHistoryAgain(id)) view = GenerationView.GENERATE
                },
            )
        }
    }
}

@Composable
private fun GenerationHistoryView(
    history: GenerationHistoryUiState,
    onBackToGenerate: () -> Unit,
    onSelected: (String) -> Boolean,
    onCloseDetail: () -> Unit,
    onDelete: (String) -> Unit,
    onClear: () -> Unit,
    onUseAgain: (String) -> Unit,
) {
    var confirmClear by remember { mutableStateOf(false) }
    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear generation history?") },
            text = { Text("This deletes history records only. Checkpoints are not affected.") },
            confirmButton = {
                Button(onClick = {
                    confirmClear = false
                    onClear()
                }) {
                    Icon(Icons.Default.DeleteSweep, contentDescription = null, modifier = Modifier.size(18.dp))
                    Text("Clear history")
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) {
                    Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(18.dp))
                    Text("Cancel")
                }
            },
        )
    }
    val selected = history.selected
    if (selected != null) {
        GenerationHistoryDetail(
            item = selected,
            onBack = onCloseDetail,
            onDelete = {
                onDelete(selected.record.id)
                onCloseDetail()
            },
            onUseAgain = { onUseAgain(selected.record.id) },
        )
        return
    }
    LazyColumn(
        modifier = Modifier.fillMaxSize().semantics { contentDescription = "Generation history" },
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onBackToGenerate, modifier = Modifier.weight(1f)) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null, modifier = Modifier.size(18.dp))
                    Text("Back to Generation")
                }
                TextButton(onClick = { confirmClear = true }, enabled = history.items.isNotEmpty()) {
                    Icon(Icons.Default.DeleteSweep, contentDescription = null, modifier = Modifier.size(18.dp))
                    Text("Clear history")
                }
            }
        }
        item { Text("Generation history", style = MaterialTheme.typography.headlineSmall) }
        history.message?.let { message -> item { Text(message, color = MaterialTheme.colorScheme.error) } }
        if (history.loading) item { CircularProgressIndicator(Modifier.size(24.dp)) }
        if (!history.loading && history.items.isEmpty()) item { Text("No generation history") }
        items(history.items, key = { it.record.id }) { item ->
            Card(onClick = { onSelected(item.record.id) }, modifier = Modifier.fillMaxWidth()) {
                SelectionContainer {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(formatHistoryTimestamp(item.record.createdAtMs), fontWeight = FontWeight.Bold)
                        Text(quotedPreview(item.promptText), maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(
                            if (item.record.mode == GenerationMode.SAMPLE) {
                                "Sample · T${item.record.temperature} · K${item.record.topK}"
                            } else "Greedy",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            "step${item.record.checkpointStep} · D${item.record.dimension}/FFN${item.record.feedForwardDimension}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(quotedPreview(item.outputText), maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(
                            "${item.record.generatedBytes.size} bytes · ${item.record.elapsedMs} ms · ${item.record.backend}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            if (item.record.status == GenerationHistoryStatus.SUCCESS) "Success" else "Failed",
                            color = if (item.record.status == GenerationHistoryStatus.SUCCESS) {
                                MaterialTheme.colorScheme.primary
                            } else MaterialTheme.colorScheme.error,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun GenerationHistoryDetail(
    item: GenerationHistoryItem,
    onBack: () -> Unit,
    onDelete: () -> Unit,
    onUseAgain: () -> Unit,
) {
    val record = item.record
    LazyColumn(
        modifier = Modifier.fillMaxSize().semantics { contentDescription = "Generation history detail" },
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            TextButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null, modifier = Modifier.size(18.dp))
                Text("Back to history")
            }
        }
        item {
            DetailCard("Prompt") {
                SelectionContainer { Text(item.promptText, fontFamily = TelemetryFontFamily) }
            }
        }
        item {
            GenerationOutputCard(
                title = "Generated output",
                text = item.outputText,
                supportingText = "${record.generatedBytes.size} bytes · ${record.elapsedMs} ms · ${record.backend}",
            )
        }
        item {
            DetailCard("Generation settings") {
                SelectionContainer {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricRow("Mode", record.mode.name.lowercase().replaceFirstChar(Char::uppercase))
                        MetricRow("Temperature", record.temperature.toString())
                        MetricRow("TopK", record.topK.toString())
                        MetricRow("SamplingSeed", record.samplingSeed.toString())
                        MetricRow("Max new bytes", record.maxNewBytes.toString())
                    }
                }
            }
        }
        item {
            DetailCard("Checkpoint") {
                SelectionContainer {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricRow("Step", record.checkpointStep.toString())
                        MetricRow(
                            "Model",
                            "V${record.vocabulary} / T${record.tokens} / D${record.dimension} / " +
                                "FFN${record.feedForwardDimension} / L${record.layers} / H${record.heads}",
                        )
                        MetricRow("Parameter hash", record.checkpointParameterHash)
                    }
                }
            }
        }
        item {
            DetailCard("Runtime") {
                SelectionContainer {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricRow("Backend", record.backend)
                        MetricRow("Elapsed", "${record.elapsedMs} ms")
                        MetricRow("Generated bytes", record.generatedBytes.size.toString())
                        MetricRow("QNN attempts", record.qnnExecuteAttempts.toString())
                        MetricRow("QNN successes", record.qnnExecuteSuccesses.toString())
                        MetricRow("QNN failures", record.qnnExecuteFailures.toString())
                        MetricRow("CPU fallback", if (record.cpuFallback) "YES" else "NO")
                        MetricRow("Finite", if (record.finite) "YES" else "NO")
                        MetricRow("Status", record.status.name)
                        record.failureMessage?.let { MetricRow("Failure reason", it) }
                    }
                }
            }
        }
        item {
            Button(onClick = onUseAgain, modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Default.Replay, contentDescription = null, modifier = Modifier.size(18.dp))
                Text("Use settings again")
            }
        }
        item {
            OutlinedButton(onClick = onDelete, modifier = Modifier.fillMaxWidth()) {
                Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                Text("Delete this history entry")
            }
        }
    }
}

private fun quotedPreview(value: String): String = "\"${value.replace("\n", " ").take(80)}\""

private fun formatHistoryTimestamp(createdAtMs: Long): String =
    DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(createdAtMs))

@Composable
private fun GenerationProgressCard(progress: GenerationProgress) {
    val phase = when (progress.phase) {
        GenerationPhase.PREPARING -> "Preparing"
        GenerationPhase.CHECKPOINT_VALIDATION -> "Checkpoint validation"
        GenerationPhase.HTP_INITIALIZATION -> "HTP initialization"
        GenerationPhase.GRAPH_PREPARATION -> "Graph preparation"
        GenerationPhase.GENERATING -> "Generating"
        GenerationPhase.COMPLETED -> "Completed"
        GenerationPhase.FAILED -> "Failed"
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SelectionContainer {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (progress.phase == GenerationPhase.GENERATING) "Generating on HTP" else phase,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                    )
                    Text("${progress.generatedBytes} / ${progress.maxNewBytes} bytes", style = MaterialTheme.typography.headlineSmall)
                }
            }
            val fraction = (progress.generatedBytes.toFloat() / progress.maxNewBytes.coerceAtLeast(1)).coerceIn(0f, 1f)
            LinearWavyProgressIndicator(
                progress = { fraction },
                modifier = Modifier.fillMaxWidth().height(8.dp),
                amplitude = { if (progress.phase == GenerationPhase.GENERATING) 0.35f else 0f },
            )
            if (progress.displayText.isNotEmpty()) {
                GenerationOutputCard(
                    title = "Live output",
                    text = progress.displayText,
                    supportingText = "${progress.generatedBytes} / ${progress.maxNewBytes} bytes",
                    live = true,
                )
            }
            SelectionContainer {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    MetricRow("Phase", phase)
                    MetricRow("Elapsed", "${progress.elapsedMs} ms")
                    MetricRow("Backend", "HTP")
                    MetricRow("QNN", "${progress.qnnExecuteSuccesses} / ${progress.qnnExecuteAttempts}")
                    if (progress.qnnExecuteFailures > 0) MetricRow("QNN failures", progress.qnnExecuteFailures.toString())
                    MetricRow("CPU fallback", if (progress.cpuFallback) "YES" else "NO")
                    MetricRow("Finite", if (progress.finite) "YES" else "NO")
                }
            }
        }
    }
}

@Composable
private fun GenerationOutputCard(
    title: String,
    text: String,
    supportingText: String,
    live: Boolean = false,
) = Card(
    colors = CardDefaults.cardColors(
        containerColor = MaterialTheme.colorScheme.primaryContainer,
        contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
    ),
    shape = RoundedCornerShape(24.dp),
    modifier = Modifier.fillMaxWidth().defaultMinSize(minHeight = 132.dp)
        .semantics { contentDescription = title },
) {
    SelectionContainer {
        Column(
            Modifier.fillMaxWidth().padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(20.dp))
                Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                if (live) Text("LIVE", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            }
            Text(
                text.ifEmpty { "No output bytes" },
                style = MaterialTheme.typography.bodyLarge.copy(fontSize = 18.sp, lineHeight = 26.sp),
                fontFamily = TelemetryFontFamily,
            )
            Text(
                supportingText,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
            )
        }
    }
}

@Composable
private fun DetailedPerformanceCard(state: TrainingUiState) = DetailCard("Performance totals") {
    val aggregate = state.timing?.aggregate
    listOf(
        "Fwd+Backward fused" to TrainingOperationPhase.FUSED_FORWARD_BACKWARD,
        "Adam" to TrainingOperationPhase.ADAM,
        "Host CPU" to TrainingOperationPhase.HOST,
    ).forEach { (name, phase) ->
        val current = aggregate?.current?.entries()?.get(phase)
        val average = aggregate?.average?.entries()?.get(phase)
        val cumulative = aggregate?.cumulative?.entries()?.get(phase)
        MetricRow(name, phaseMs(current), "average ${phaseMs(average)} · cumulative ${phaseMs(cumulative)}")
    }
    MetricRow("Elapsed", state.timing?.elapsedMs?.let(::duration) ?: "—")
    MetricRow("Process PSS", state.dashboard.currentMemoryBytes?.let(::bytes) ?: "—", "peak ${state.dashboard.peakMemoryBytes?.let(::bytes) ?: "—"}")
}

@Composable
private fun TrainingEventTimeline(events: List<TrainingDashboardEvent>) = DetailCard("Event history") {
    if (events.isEmpty()) Text("No training events observed yet.")
    val visibleEvents = events.takeLast(50)
    if (events.size > visibleEvents.size) Text("Showing latest ${visibleEvents.size} of ${events.size} events.")
    visibleEvents.asReversed().forEach { event ->
        val (title, detail) = eventDescription(event)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, fontWeight = FontWeight.Bold)
            event.step?.let { Text("step $it", fontWeight = FontWeight.Bold, fontFamily = TelemetryFontFamily) }
        }
        detail?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun TrainingSummaryCard(state: TrainingUiState) = DetailCard("Training summary") {
    MetricRow("Final loss", state.progress?.loss?.let(::loss) ?: "—")
    MetricRow("Elapsed", state.timing?.elapsedMs?.let(::duration) ?: "—")
    MetricRow("Average step", state.dashboard.averageStepWallTimeMs?.let(::duration) ?: "—")
    MetricRow("Checkpoints", state.dashboard.checkpointCount.toString())
    val lastHtp = state.dashboard.activityHistory.lastOrNull { it.htpObservationRatioPercent != null }?.htpObservationRatioPercent
    MetricRow("Last observed HTP ratio", lastHtp?.let(::percent) ?: "—")
    MetricRow("Peak process PSS", state.dashboard.peakMemoryBytes?.let(::bytes) ?: "—")
}

@Composable
private fun DiagnosticCard(state: TrainingUiState) = DetailCard("Raw diagnostics") {
    Text("Phase: ${state.phase}")
    Text("Repository message: ${state.message ?: "Unavailable"}")
    Text("Overview: ${state.overviewText}")
    Text("Timing: ${state.timingText}")
    Text("Activity: ${state.activityText}")
    EvidenceText(state.dashboard.runtimeEvidence)
    Text("Checkpoint: ${state.checkpointText}")
}

@Composable
private fun EvidenceText(evidence: TrainingRuntimeEvidence?) {
    Text("QNN return code success: ${evidence?.qnnReturnCodeSuccess ?: "Unavailable"}")
    Text("Output tensors finite: ${evidence?.outputTensorsFinite ?: "Unavailable"}")
    Text("CPU fallback: ${evidence?.cpuFallback ?: "Unavailable"}")
    Text("Backend: ${evidence?.backend ?: "Unavailable"}")
    evidence?.error?.let { Text("Runtime error: $it") }
}

@Composable
private fun ModelConfigCard(config: TrainingModelConfig, name: String?, uri: String?) = DetailCard("Dataset & model") {
    Text(name ?: "No document selected", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
    Text(uri ?: "Select a SAF document to enable training.", style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
    Text("L${config.layers} · H${config.heads} · T${config.tokens} · D${config.dimension}")
    Text("FFN${config.feedForwardDimension} · V${config.vocabularySize} · B${config.batchSize} · LR ${config.learningRate}")
}

@Composable
private fun DenseCard(
    containerColor: Color,
    density: TrainingOverviewDensity,
    modifier: Modifier = Modifier,
    fillHeight: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) = Card(
    colors = CardDefaults.cardColors(containerColor = containerColor),
    shape = RoundedCornerShape(if (density in compactDensityTiers) 16.dp else 20.dp),
    modifier = modifier.fillMaxWidth(),
) {
    Column(
        (if (fillHeight) Modifier.fillMaxSize() else Modifier.fillMaxWidth()).padding(density.cardPaddingDp.dp),
        verticalArrangement = Arrangement.spacedBy(if (density in compactDensityTiers) 3.dp else 5.dp),
        content = content,
    )
}

@Composable
private fun DetailCard(title: String, content: @Composable ColumnScope.() -> Unit) = Card(
    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    shape = RoundedCornerShape(24.dp),
    modifier = Modifier.fillMaxWidth(),
) {
    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        content()
    }
}

@Composable
private fun InlineMetric(label: String, value: String, modifier: Modifier, end: Boolean = false) = Row(
    modifier,
    horizontalArrangement = if (end) Arrangement.End else Arrangement.Start,
) {
    Text("$label ", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Text(value, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, fontFamily = TelemetryFontFamily, maxLines = 1)
}

@Composable
private fun MetricRow(label: String, value: String, detail: String? = null) = Row(
    Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp),
) {
    Text(label, modifier = Modifier.weight(0.45f), color = MaterialTheme.colorScheme.onSurfaceVariant)
    Column(modifier = Modifier.weight(0.55f), horizontalAlignment = Alignment.End) {
        Text(value, fontWeight = FontWeight.Bold, fontFamily = TelemetryFontFamily, textAlign = TextAlign.End)
        detail?.let { Text(it, style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.End) }
    }
}

private fun phaseLabel(phase: TrainingPhase) = phase.name.replace('_', ' ').lowercase().replaceFirstChar { it.titlecase() }
private fun compactPhaseLabel(phase: TrainingPhase, density: TrainingOverviewDensity) = when {
    density != TrainingOverviewDensity.COMPACT -> phaseLabel(phase)
    phase == TrainingPhase.INITIALIZING_HTP -> "Initializing"
    phase == TrainingPhase.SAVING_CHECKPOINT -> "Saving checkpoint"
    else -> phaseLabel(phase)
}
private fun loss(value: Float) = String.format(Locale.ROOT, "%.6f", value)
private fun percent(value: Double) = String.format(Locale.ROOT, "%.1f%%", value)
private fun duration(value: Long) = if (value < 1_000) "${value} ms" else String.format(Locale.ROOT, "%.2f s", value / 1_000.0)
private fun duration(value: Double) = if (value < 1_000) String.format(Locale.ROOT, "%.1f ms", value) else String.format(Locale.ROOT, "%.2f s", value / 1_000.0)
private fun bytes(value: Long) = String.format(Locale.ROOT, "%.1f MiB", value / 1_048_576.0)
private fun phaseMs(timing: PhaseTiming?): String = when {
    timing == null || timing.backend == TimingBackend.UNAVAILABLE -> "—"
    timing.qnnExecuteMs != null -> duration(timing.qnnExecuteMs)
    timing.hostMs != null -> duration(timing.hostMs)
    else -> "—"
}
private fun eventDescription(event: TrainingDashboardEvent): Pair<String, String?> = when (event.type) {
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.PHASE -> "Training state changed" to event.message?.lowercase()?.replace('_', ' ')
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.CHECKPOINT -> "Checkpoint saved" to null
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.RESUME -> "Training resumed" to null
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.ERROR -> "Training error" to event.message
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.QNN_RETURN -> "QNN return code" to if (event.message.equals("true", ignoreCase = true)) "success" else "failure"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.TENSOR_FINITE -> "Tensor finite check" to if (event.message.equals("true", ignoreCase = true)) "finite" else "non-finite"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.CPU_FALLBACK -> "CPU fallback" to if (event.message.equals("true", ignoreCase = true)) "observed" else "not observed"
}
private val activePhases = setOf(
    TrainingPhase.PREPARING, TrainingPhase.INITIALIZING_HTP, TrainingPhase.TRAINING,
    TrainingPhase.SAVING_CHECKPOINT, TrainingPhase.PAUSED,
)
private val compactDensityTiers = setOf(TrainingOverviewDensity.ACCESSIBLE, TrainingOverviewDensity.COMPACT)
private val terminalPhases = setOf(TrainingPhase.COMPLETED, TrainingPhase.ERROR, TrainingPhase.INTERRUPTED)

@Preview(name = "Portrait 360 × 800", widthDp = 360, heightDp = 800, showBackground = true)
@Preview(name = "Portrait 412 × 915", widthDp = 412, heightDp = 915, showBackground = true)
@Preview(name = "Portrait large font", widthDp = 360, heightDp = 800, fontScale = 2f, showBackground = true)
@Composable
private fun TrainingOverviewPreview() {
    TrainingDashboardApp(
        state = previewTrainingState(),
        onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
    )
}

@Preview(name = "Paused state", widthDp = 360, heightDp = 800, showBackground = true)
@Composable
private fun TrainingPausedPreview() = PreviewState(previewTrainingState(TrainingPhase.PAUSED))

@Preview(name = "Ready state", widthDp = 360, heightDp = 800, showBackground = true)
@Composable
private fun TrainingReadyPreview() = PreviewState(previewTrainingState(TrainingPhase.IDLE))

@Preview(name = "Complete state", widthDp = 360, heightDp = 800, showBackground = true)
@Composable
private fun TrainingCompletePreview() = PreviewState(previewTrainingState(TrainingPhase.COMPLETED))

@Preview(name = "Missing telemetry", widthDp = 360, heightDp = 800, showBackground = true)
@Composable
private fun TrainingMissingTelemetryPreview() = PreviewState(previewTrainingState(TrainingPhase.TRAINING, telemetryMissing = true))

@Preview(name = "Warning state", widthDp = 360, heightDp = 800, showBackground = true)
@Composable
private fun TrainingWarningPreview() = PreviewState(previewTrainingState(TrainingPhase.ERROR, warning = true))

@Composable
private fun PreviewState(state: TrainingUiState) {
    TrainingDashboardApp(
        state = state,
        onSelectDataset = {}, onStart = {}, onStop = {}, onPause = {}, onResume = {}, onStartOver = {},
    )
}

private fun previewTrainingState(
    phase: TrainingPhase = TrainingPhase.TRAINING,
    telemetryMissing: Boolean = false,
    warning: Boolean = false,
) = TrainingUiState(
    phase = phase,
    progress = TrainingProgress(3_248, 8_000, 0.842731f),
    overview = "Training",
    overviewText = "Training",
    message = null,
    timing = if (telemetryMissing) null else TrainingTiming(0L, 3_258_000L, currentStepMs = 388L, averageStepMs = 398.9),
    timingText = "Unavailable",
    htpActivity = null,
    activityText = "HTP activity is an observation ratio, not hardware utilization.",
    lastCheckpoint = null,
    checkpointText = "step 3,000",
    datasetUri = null,
    datasetDisplayName = "nicopedia.txt",
    canStart = phase == TrainingPhase.IDLE || phase == TrainingPhase.COMPLETED,
    canStop = phase == TrainingPhase.TRAINING || phase == TrainingPhase.SAVING_CHECKPOINT,
    canPause = phase == TrainingPhase.TRAINING,
    canResume = phase == TrainingPhase.PAUSED,
    modelConfig = TrainingModelConfig.NICOPEDIA_L19,
    dashboard = TrainingDashboardSnapshot(
        lossHistory = listOf(1.12f, 1.03f, 0.98f, 0.91f, 0.88f, 0.84f).mapIndexed { index, value ->
            TrainingLossHistoryEntry(3_208 + index * 8, value)
        },
        activityHistory = listOf(
            TrainingActivityHistoryEntry(
                3_248,
                htpObservationRatioPercent = if (telemetryMissing) null else 67.4,
                processCpuPercent = if (telemetryMissing) null else 18.2,
                processMemoryBytes = if (telemetryMissing) null else 284L * 1_048_576L,
            ),
        ),
        eventTimeline = if (warning) {
            listOf(TrainingDashboardEvent(TrainingDashboardEventType.ERROR, 3_248, "QNN output unavailable"))
        } else {
            listOf(TrainingDashboardEvent(TrainingDashboardEventType.CHECKPOINT, 3_000, "checkpoint saved"))
        },
        etaMs = 418_000,
        averageStepWallTimeMs = if (telemetryMissing) null else 398.9,
        currentStepWallTimeMs = if (telemetryMissing) null else 388L,
        currentMemoryBytes = if (telemetryMissing) null else 284L * 1_048_576L,
    ),
)
