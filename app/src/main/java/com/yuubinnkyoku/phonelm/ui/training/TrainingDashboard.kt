package com.yuubinnkyoku.phonelm.ui.training

import android.os.Build
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yuubinnkyoku.phonelm.PhaseTiming
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
import com.yuubinnkyoku.phonelm.TrainingUiState
import java.util.Locale

@Composable
fun TrainingDashboardApp(
    state: TrainingUiState?,
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStartOver: () -> Unit,
) {
    val context = LocalContext.current
    val base = phoneLmDarkScheme()
    val colors = if (Build.VERSION.SDK_INT >= 31) dynamicDarkColorScheme(context).copy(
        background = base.background,
        surface = base.surface,
        surfaceVariant = base.surfaceVariant,
    ) else base
    MaterialTheme(
        colorScheme = colors,
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
            if (state == null) TrainingLoading() else TrainingDashboard(
                state, onSelectDataset, onStart, onStop, onPause, onResume, onStartOver,
            )
        }
    }
}

private fun phoneLmDarkScheme(): ColorScheme = darkColorScheme(
    primary = Color(0xFFB7C4FF), onPrimary = Color(0xFF08164D), secondary = Color(0xFFBCE9DD),
    surface = Color(0xFF111318), surfaceVariant = Color(0xFF20232B), background = Color(0xFF090B10),
    onSurface = Color(0xFFE3E5ED), onSurfaceVariant = Color(0xFFC3C6D1), error = Color(0xFFFFB4AB),
)

@Composable
private fun TrainingLoading() = Box(Modifier.fillMaxSize().padding(24.dp)) {
    Text("Connecting to training session…")
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingDashboard(
    state: TrainingUiState,
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStartOver: () -> Unit,
) {
    var detailsVisible by remember { mutableStateOf(false) }
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            Surface(
                modifier = Modifier.fillMaxWidth().navigationBarsPadding(),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.98f),
            ) {
                CompactActionDock(
                    state, onSelectDataset, onStart, onStop, onPause, onResume,
                    onDetails = { detailsVisible = true },
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
            ActivityMetric("HTP activity", currentHtpObservationRatio(state.dashboard.activityHistory)?.let(::percent) ?: "—", "QNN wall / observation · not utilization", Modifier.weight(1f), density)
            ActivityMetric("CPU activity", currentProcessCpuPercent(state.dashboard.activityHistory)?.let(::percent) ?: "—", "process CPU", Modifier.weight(1f), density)
        }
        CompactPerformanceGrid(state, density)
        CompactLossCard(
            state.dashboard,
            Modifier.fillMaxWidth().weight(1f).defaultMinSize(minHeight = density.graphMinimumHeightDp.dp),
            density,
        )
        LatestStatusRow(state, density)
    }
}

@Composable
private fun CompactTrainingHero(state: TrainingUiState, density: TrainingOverviewDensity) {
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
                    style = if (density == TrainingOverviewDensity.COMPACT) MaterialTheme.typography.labelLarge else MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    progress?.let { "${it.completedSteps} / ${it.totalSteps} steps" } ?: "Step target unavailable",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Text(
                progress?.let { String.format(Locale.ROOT, "%.1f%%", it.fraction * 100f) } ?: "—",
                fontSize = density.heroPercentSp.sp,
                fontWeight = FontWeight.Black,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        val fraction by animateFloatAsState(progress?.fraction ?: 0f, label = "training progress")
        Box(
            Modifier.fillMaxWidth().height(7.dp).clip(RoundedCornerShape(99.dp))
                .background(MaterialTheme.colorScheme.surface)
                .semantics { progressBarRangeInfo = ProgressBarRangeInfo(fraction, 0f..1f) },
        ) {
            Box(Modifier.fillMaxWidth(fraction).fillMaxHeight().background(MaterialTheme.colorScheme.primary))
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            InlineMetric("Loss", progress?.loss?.let(::loss) ?: "—", Modifier.weight(1f))
            InlineMetric("ETA", state.dashboard.etaMs?.let(::duration) ?: "—", Modifier.weight(1f), end = true)
        }
    }
}

@Composable
private fun ActivityMetric(label: String, value: String, qualifier: String, modifier: Modifier, density: TrainingOverviewDensity) =
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density, modifier) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text(value, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Black, maxLines = 1, overflow = TextOverflow.Ellipsis)
        Text(qualifier, style = MaterialTheme.typography.labelSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }

@Composable
private fun CompactPerformanceGrid(state: TrainingUiState, density: TrainingOverviewDensity) {
    val current = state.timing?.aggregate?.current
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            PerformanceCell(if (density == TrainingOverviewDensity.COMPACT) "F+B" else "Fwd + Bwd", phaseMs(current?.entries()?.get(TrainingOperationPhase.FUSED_FORWARD_BACKWARD)), Modifier.weight(1f))
            PerformanceCell("Adam", phaseMs(current?.entries()?.get(TrainingOperationPhase.ADAM)), Modifier.weight(1f))
            PerformanceCell("Step", state.dashboard.currentStepWallTimeMs?.let(::duration) ?: "—", Modifier.weight(1f))
        }
    }
}

@Composable
private fun PerformanceCell(label: String, value: String, modifier: Modifier) = Column(modifier) {
    Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
    Text(value, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
}

@Composable
private fun CompactLossCard(snapshot: TrainingDashboardSnapshot, modifier: Modifier, density: TrainingOverviewDensity) =
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density, modifier, fillHeight = true) {
        val values = snapshot.lossHistory
        val primary = MaterialTheme.colorScheme.primary
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Loss trend", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Text(values.lastOrNull()?.let { "${loss(it.loss)} · step ${it.step}" } ?: "No samples", style = MaterialTheme.typography.labelSmall)
        }
        Canvas(
            Modifier.fillMaxWidth().weight(1f).semantics {
                contentDescription = "Compact loss history with ${values.size} observed samples"
            },
        ) {
            if (values.size < 2) return@Canvas
            val min = values.minOf { it.loss }
            val max = values.maxOf { it.loss }
            val range = (max - min).takeIf { it > 0f } ?: 1f
            fun point(index: Int) = Offset(
                index.toFloat() / values.lastIndex * size.width,
                size.height - ((values[index].loss - min) / range * size.height),
            )
            for (index in 1..values.lastIndex) drawLine(Color(0xFF68708A), point(index - 1), point(index), 2f, StrokeCap.Round)
            val recentStart = (values.size - minOf(values.size, 48)).coerceAtLeast(1)
            for (index in recentStart..values.lastIndex) drawLine(primary, point(index - 1), point(index), 4f, StrokeCap.Round)
        }
    }

@Composable
private fun LatestStatusRow(state: TrainingUiState, density: TrainingOverviewDensity) =
    DenseCard(MaterialTheme.colorScheme.surfaceVariant, density) {
        val event = state.dashboard.eventTimeline.lastOrNull()
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Column(Modifier.weight(0.42f)) {
                Text("Checkpoint", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(state.lastCheckpoint?.completedStep?.let { "Step $it" } ?: "—", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Column(Modifier.weight(0.58f)) {
                Text("Latest event", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(event?.let { eventDescription(it).first } ?: "No events", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }

@Composable
private fun CompactActionDock(
    state: TrainingUiState,
    onSelectDataset: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onDetails: () -> Unit,
) {
    val compactContentPadding = if (LocalDensity.current.fontScale >= 1.2f) {
        PaddingValues(horizontal = 8.dp, vertical = 6.dp)
    } else {
        ButtonDefaults.ContentPadding
    }
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        when {
            state.canStop -> Button(onClick = onStop, modifier = Modifier.weight(1.25f), contentPadding = compactContentPadding) { Text("Stop", maxLines = 1, overflow = TextOverflow.Ellipsis) }
            state.canResume -> Button(onClick = onResume, modifier = Modifier.weight(1.25f), contentPadding = compactContentPadding) { Text("Resume", maxLines = 1, overflow = TextOverflow.Ellipsis) }
            state.canStart -> Button(onClick = onStart, modifier = Modifier.weight(1.25f), contentPadding = compactContentPadding) { Text("Start", maxLines = 1, overflow = TextOverflow.Ellipsis) }
            else -> Button(onClick = onSelectDataset, enabled = state.phase !in activePhases, modifier = Modifier.weight(1.25f), contentPadding = compactContentPadding) { Text("Select", maxLines = 1, overflow = TextOverflow.Ellipsis) }
        }
        OutlinedButton(onClick = onPause, enabled = state.canPause, modifier = Modifier.weight(1f), contentPadding = compactContentPadding) { Text("Pause", maxLines = 1, overflow = TextOverflow.Ellipsis) }
        OutlinedButton(onClick = onDetails, modifier = Modifier.weight(1f), contentPadding = compactContentPadding) { Text("Details", maxLines = 1, overflow = TextOverflow.Ellipsis) }
    }
}

@Composable
private fun TrainingDetails(state: TrainingUiState, onSelectDataset: () -> Unit, onStartOver: () -> Unit) {
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
        Text("$title${event.step?.let { " · step $it" }.orEmpty()}", fontWeight = FontWeight.Bold)
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
    shape = RoundedCornerShape(if (density == TrainingOverviewDensity.COMPACT) 16.dp else 20.dp),
    modifier = modifier.fillMaxWidth(),
) {
    Column(
        (if (fillHeight) Modifier.fillMaxSize() else Modifier.fillMaxWidth()).padding(density.cardPaddingDp.dp),
        verticalArrangement = Arrangement.spacedBy(if (density == TrainingOverviewDensity.COMPACT) 3.dp else 5.dp),
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
    Text(value, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, maxLines = 1)
}

@Composable
private fun MetricRow(label: String, value: String, detail: String? = null) = Row(
    Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp),
) {
    Text(label, modifier = Modifier.weight(0.45f), color = MaterialTheme.colorScheme.onSurfaceVariant)
    Column(modifier = Modifier.weight(0.55f), horizontalAlignment = Alignment.End) {
        Text(value, fontWeight = FontWeight.Bold, textAlign = TextAlign.End)
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
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.QNN_RETURN -> "QNN return code" to if (event.message == "true") "success" else "failure"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.TENSOR_FINITE -> "Tensor finite check" to if (event.message == "true") "finite" else "non-finite"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.CPU_FALLBACK -> "CPU fallback" to if (event.message == "true") "observed" else "not observed"
}
private val activePhases = setOf(
    TrainingPhase.PREPARING, TrainingPhase.INITIALIZING_HTP, TrainingPhase.TRAINING,
    TrainingPhase.SAVING_CHECKPOINT, TrainingPhase.PAUSED,
)
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

private fun previewTrainingState() = TrainingUiState(
    phase = TrainingPhase.TRAINING,
    progress = TrainingProgress(3_248, 8_000, 0.842731f),
    overview = "Training",
    overviewText = "Training",
    message = null,
    timing = null,
    timingText = "Unavailable",
    htpActivity = null,
    activityText = "HTP activity is an observation ratio, not hardware utilization.",
    lastCheckpoint = null,
    checkpointText = "step 3,000",
    datasetUri = null,
    datasetDisplayName = "nicopedia.txt",
    canStart = false,
    canStop = true,
    canPause = false,
    canResume = false,
    modelConfig = TrainingModelConfig.NICOPEDIA_L19,
    dashboard = TrainingDashboardSnapshot(
        lossHistory = listOf(1.12f, 1.03f, 0.98f, 0.91f, 0.88f, 0.84f).mapIndexed { index, value ->
            TrainingLossHistoryEntry(3_208 + index * 8, value)
        },
        activityHistory = listOf(TrainingActivityHistoryEntry(3_248, 67.4, 18.2)),
        eventTimeline = listOf(TrainingDashboardEvent(TrainingDashboardEventType.CHECKPOINT, 3_000, "checkpoint saved")),
        etaMs = 418_000,
    ),
)
