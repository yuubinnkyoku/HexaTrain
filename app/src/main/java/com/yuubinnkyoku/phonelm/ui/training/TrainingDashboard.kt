package com.yuubinnkyoku.phonelm.ui.training

import android.os.Build
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yuubinnkyoku.phonelm.PhaseTiming
import com.yuubinnkyoku.phonelm.TimingBackend
import com.yuubinnkyoku.phonelm.TrainingDashboardEvent
import com.yuubinnkyoku.phonelm.TrainingDashboardSnapshot
import com.yuubinnkyoku.phonelm.TrainingModelConfig
import com.yuubinnkyoku.phonelm.TrainingOperationPhase
import com.yuubinnkyoku.phonelm.TrainingPhase
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
        background = base.background, surface = base.surface, surfaceVariant = base.surfaceVariant,
    ) else base
    MaterialTheme(colorScheme = colors, typography = MaterialTheme.typography.copy(
        displaySmall = MaterialTheme.typography.displaySmall.copy(fontSize = 34.sp, fontWeight = FontWeight.Black),
        headlineSmall = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
    ), shapes = Shapes(
        small = RoundedCornerShape(12.dp), medium = RoundedCornerShape(18.dp),
        large = RoundedCornerShape(28.dp), extraLarge = RoundedCornerShape(36.dp),
    )) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            if (state == null) TrainingLoading() else TrainingDashboard(state, onSelectDataset, onStart, onStop, onPause, onResume, onStartOver)
        }
    }
}

private fun phoneLmDarkScheme(): ColorScheme = darkColorScheme(
    primary = Color(0xFFB7C4FF), onPrimary = Color(0xFF08164D), secondary = Color(0xFFBCE9DD),
    surface = Color(0xFF111318), surfaceVariant = Color(0xFF20232B), background = Color(0xFF090B10),
    onSurface = Color(0xFFE3E5ED), onSurfaceVariant = Color(0xFFC3C6D1), error = Color(0xFFFFB4AB),
)

@Composable private fun TrainingLoading() = Box(Modifier.fillMaxSize().padding(24.dp)) { Text("Connecting to training session…") }

@Composable
fun TrainingDashboard(state: TrainingUiState, onSelectDataset: () -> Unit, onStart: () -> Unit, onStop: () -> Unit, onPause: () -> Unit, onResume: () -> Unit, onStartOver: () -> Unit) {
    Scaffold(
      containerColor = MaterialTheme.colorScheme.background,
      bottomBar = {
        Surface(
          modifier = Modifier.fillMaxWidth().navigationBarsPadding(),
          color = MaterialTheme.colorScheme.surface.copy(alpha = 0.97f),
        ) {
          TrainingActionDock(state, onSelectDataset, onStart, onStop, onPause, onResume, onStartOver)
        }
      },
    ) { scaffoldPadding ->
      LazyColumn(
        modifier = Modifier.fillMaxSize().semantics { contentDescription = "Standalone training dashboard" },
        contentPadding = PaddingValues(
          start = 16.dp,
          top = scaffoldPadding.calculateTopPadding() + 20.dp,
          end = 16.dp,
          bottom = scaffoldPadding.calculateBottomPadding() + 16.dp,
        ),
        verticalArrangement = Arrangement.spacedBy(14.dp),
      ) {
        item { TrainingHero(state) }
        item { ComputeActivityCard(state) }
        item { TrainingPerformanceGrid(state) }
        item { LossHistoryChart(state.dashboard) }
        item { ModelConfigCard(state.modelConfig, state.datasetDisplayName, state.datasetUri) }
        item { TrainingEventTimeline(state.dashboard.eventTimeline) }
        if (state.phase in terminalPhases) item { TrainingSummaryCard(state) }
        item { DiagnosticDetails(state) }
      }
    }
}

@Composable fun TrainingHero(state: TrainingUiState) {
    val target = when (state.phase) {
        TrainingPhase.ERROR -> MaterialTheme.colorScheme.errorContainer
        TrainingPhase.TRAINING -> MaterialTheme.colorScheme.primaryContainer
        TrainingPhase.COMPLETED -> MaterialTheme.colorScheme.secondaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    val container by animateColorAsState(target, label = "training phase card")
    DashboardCard("Training", container) {
    Text(state.phase.name.replace('_', ' '), style = MaterialTheme.typography.displaySmall)
    val progress = state.progress
    Text(
        progress?.let { String.format(Locale.ROOT, "%.1f%%", it.fraction * 100f) } ?: "—",
        style = MaterialTheme.typography.displayLarge.copy(fontSize = 64.sp, fontWeight = FontWeight.Black),
    )
    Text(progress?.let { "Step ${it.completedSteps} / ${it.totalSteps}" } ?: "Step target unavailable", style = MaterialTheme.typography.titleMedium)
    MetricRow("Current loss", progress?.loss?.let(::loss) ?: "—", "Δ ${state.dashboard.lossDelta?.let(::signedLoss) ?: "—"}")
    MetricRow("Elapsed", state.timing?.elapsedMs?.let(::duration) ?: "—", "ETA ${state.dashboard.etaMs?.let(::duration) ?: "—"}")
    MetricRow("Checkpoint", state.lastCheckpoint?.completedStep?.let { "step $it" } ?: "—", "${state.dashboard.checkpointCount} saved")
    val fraction by animateFloatAsState(progress?.fraction ?: 0f, label = "training progress")
    Box(
      Modifier.fillMaxWidth().height(9.dp).clip(RoundedCornerShape(99.dp))
        .background(MaterialTheme.colorScheme.surfaceVariant)
        .semantics { progressBarRangeInfo = ProgressBarRangeInfo(fraction, 0f..1f) },
    ) {
        Box(Modifier.fillMaxWidth(fraction).height(9.dp).background(MaterialTheme.colorScheme.primary))
    }
    }
}

@Composable fun ComputeActivityCard(state: TrainingUiState) = DashboardCard("Compute activity") {
    val latest = state.dashboard.activityHistory.lastOrNull()
    Text("HTP observation ratio (not hardware utilization)", style = MaterialTheme.typography.labelLarge)
    MetricRow("HTP observation ratio", latest?.htpObservationRatioPercent?.let(::percent) ?: "—")
    MetricRow("Process CPU", latest?.processCpuPercent?.let(::percent) ?: "—")
    MetricRow("Process PSS memory", state.dashboard.currentMemoryBytes?.let(::bytes) ?: "—", "peak ${state.dashboard.peakMemoryBytes?.let(::bytes) ?: "—"}")
    ActivitySparkline(state.dashboard.activityHistory.map { it.htpObservationRatioPercent to it.processCpuPercent })
    Text("Live observations: ${state.dashboard.activityHistory.size}", style = MaterialTheme.typography.bodySmall)
}

@Composable private fun ActivitySparkline(values: List<Pair<Double?, Double?>>) {
    val htp = MaterialTheme.colorScheme.primary
    val cpu = MaterialTheme.colorScheme.secondary
    val chartMaximum = maxOf(100.0, values.mapNotNull { it.second }.maxOrNull() ?: 100.0)
    Text("Live trace  ·  HTP observation", color = htp, style = MaterialTheme.typography.labelSmall)
    Text("Process CPU", color = cpu, style = MaterialTheme.typography.labelSmall)
    Text(
        "Shared scale 0–${String.format(Locale.ROOT, "%.0f", chartMaximum)}% (process CPU may exceed 100%)",
        style = MaterialTheme.typography.bodySmall,
    )
    Canvas(Modifier.fillMaxWidth().height(78.dp).semantics { contentDescription = "HTP observation ratio and process CPU live history" }) {
        if (values.size < 2) return@Canvas
        fun point(index: Int, value: Double) = Offset(
            index.toFloat() / values.lastIndex * size.width,
            size.height - (value.coerceIn(0.0, chartMaximum).toFloat() / chartMaximum.toFloat() * size.height),
        )
        fun trace(selector: (Pair<Double?, Double?>) -> Double?, color: Color) {
            var previous: Pair<Int, Double>? = null
            values.forEachIndexed { index, pair ->
                val value = selector(pair)
                if (value == null) { previous = null } else {
                    previous?.let { (oldIndex, oldValue) -> drawLine(color, point(oldIndex, oldValue), point(index, value), 3f, StrokeCap.Round) }
                    previous = index to value
                }
            }
        }
        trace({ it.first }, htp)
        trace({ it.second }, cpu)
    }
}

@Composable fun TrainingPerformanceGrid(state: TrainingUiState) = DashboardCard("Step performance") {
    Text("Current sample / average per step / cumulative. Measured timings only; unavailable values are not estimated.", style = MaterialTheme.typography.bodySmall)
    val aggregate = state.timing?.aggregate
    listOf(
        "Fwd+Backward fused" to TrainingOperationPhase.FUSED_FORWARD_BACKWARD,
        "Adam" to TrainingOperationPhase.ADAM,
        "Host CPU" to TrainingOperationPhase.HOST,
    ).forEach { (name, phase) ->
        val current = aggregate?.current?.entries()?.get(phase)
        val average = aggregate?.average?.entries()?.get(phase)
        val cumulative = aggregate?.cumulative?.entries()?.get(phase)
        Text(name, fontWeight = FontWeight.Bold)
        MetricRow("current sample", phaseMs(current), "avg / step ${phaseMs(average)} · cumulative ${phaseMs(cumulative)}")
    }
    MetricRow("Current step", state.dashboard.currentStepWallTimeMs?.let(::duration) ?: "—", "average ${state.dashboard.averageStepWallTimeMs?.let(::duration) ?: "—"}")
}

@Composable fun LossHistoryChart(snapshot: TrainingDashboardSnapshot) = DashboardCard("Loss history") {
    val values = snapshot.lossHistory
    val primary = MaterialTheme.colorScheme.primary
    Text("All observed samples · recent segment is the brighter trace", style = MaterialTheme.typography.bodySmall)
    Canvas(Modifier.fillMaxWidth().height(170.dp).semantics { contentDescription = "Loss history chart with ${values.size} observed samples" }) {
        if (values.size < 2) return@Canvas
        val min = values.minOf { it.loss }; val max = values.maxOf { it.loss }; val range = (max - min).takeIf { it > 0f } ?: 1f
        fun point(index: Int) = Offset(index.toFloat() / (values.lastIndex) * size.width, size.height - ((values[index].loss - min) / range * size.height))
        for (index in 1..values.lastIndex) drawLine(Color(0xFF68708A), point(index - 1), point(index), 2f, StrokeCap.Round)
        val start = (values.size - minOf(values.size, 64)).coerceAtLeast(1)
        for (index in start..values.lastIndex) drawLine(primary, point(index - 1), point(index), 4f, StrokeCap.Round)
    }
    MetricRow("Observed range", values.firstOrNull()?.let { "step ${it.step}" } ?: "—", values.lastOrNull()?.let { "latest ${loss(it.loss)}" } ?: "—")
}

@Composable fun TrainingEventTimeline(events: List<TrainingDashboardEvent>) = DashboardCard("Event timeline") {
    if (events.isEmpty()) Text("No training events observed yet.")
    events.takeLast(12).asReversed().forEach { event ->
        val (title, detail) = eventDescription(event)
        Text("$title${event.step?.let { " · step $it" }.orEmpty()}", fontWeight = FontWeight.Bold)
        detail?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable fun TrainingSummaryCard(state: TrainingUiState) = DashboardCard("Training summary") {
    MetricRow("Final loss", state.progress?.loss?.let(::loss) ?: "—")
    MetricRow("Elapsed", state.timing?.elapsedMs?.let(::duration) ?: "—", "run wall / observed step ${state.dashboard.averageStepWallTimeMs?.let(::duration) ?: "—"}")
    MetricRow("Checkpoints", state.dashboard.checkpointCount.toString(), "last observed HTP ratio ${state.dashboard.activityHistory.lastOrNull()?.htpObservationRatioPercent?.let(::percent) ?: "—"}")
    MetricRow("Peak process PSS", state.dashboard.peakMemoryBytes?.let(::bytes) ?: "—")
}

@Composable fun DiagnosticDetails(state: TrainingUiState) {
    var expanded by remember { mutableStateOf(false) }
    DashboardCard("Diagnostics") {
        OutlinedButton(onClick = { expanded = !expanded }, modifier = Modifier.semantics { contentDescription = "Toggle diagnostic details" }) { Text(if (expanded) "Hide details" else "Show details") }
        if (expanded) {
            Text("Phase: ${state.phase}")
            Text("Repository message: ${state.message ?: "Unavailable"}")
            Text("Overview: ${state.overviewText}")
            Text("Timing: ${state.timingText}")
            Text("Activity: ${state.activityText}")
            EvidenceText(state.dashboard.runtimeEvidence)
            Text("Checkpoint: ${state.checkpointText}")
        }
    }
}

@Composable private fun EvidenceText(evidence: TrainingRuntimeEvidence?) {
    Text("QNN return code success: ${evidence?.qnnReturnCodeSuccess ?: "Unavailable"}")
    Text("Output tensors finite: ${evidence?.outputTensorsFinite ?: "Unavailable"}")
    Text("CPU fallback: ${evidence?.cpuFallback ?: "Unavailable"}")
    Text("Backend: ${evidence?.backend ?: "Unavailable"}")
    evidence?.error?.let { Text("Runtime error: $it") }
}

@Composable fun ModelConfigCard(config: TrainingModelConfig, name: String?, uri: String?) = DashboardCard("Dataset & model") {
    Text(name ?: "No document selected", fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
    Text(uri ?: "Select a SAF document to enable training.", style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
    Text("L${config.layers} · H${config.heads} · T${config.tokens} · D${config.dimension}")
    Text("FFN${config.feedForwardDimension} · V${config.vocabularySize} · B${config.batchSize} · LR ${config.learningRate}")
}

@Composable fun TrainingActionDock(state: TrainingUiState, onSelectDataset: () -> Unit, onStart: () -> Unit, onStop: () -> Unit, onPause: () -> Unit, onResume: () -> Unit, onStartOver: () -> Unit) = DashboardCard("Actions") {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(onClick = onSelectDataset, enabled = state.phase !in activePhases, modifier = Modifier.weight(1f)) { Text("Select") }
        Button(onClick = onStart, enabled = state.canStart, modifier = Modifier.weight(1f)) { Text("Start") }
        OutlinedButton(onClick = onStop, enabled = state.canStop, modifier = Modifier.weight(1f)) { Text("Stop") }
    }
    Spacer(Modifier.height(8.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(onClick = onPause, enabled = state.canPause, modifier = Modifier.weight(1f)) { Text("Pause") }
        Button(onClick = onResume, enabled = state.canResume, modifier = Modifier.weight(1f)) { Text("Resume") }
        OutlinedButton(onClick = onStartOver, enabled = state.phase !in activePhases && state.phase != TrainingPhase.IDLE, modifier = Modifier.weight(1f)) { Text("Start over") }
    }
}

@Composable private fun DashboardCard(title: String, containerColor: Color = MaterialTheme.colorScheme.surfaceVariant, content: @Composable ColumnScope.() -> Unit) = Card(colors = CardDefaults.cardColors(containerColor = containerColor), shape = RoundedCornerShape(24.dp), modifier = Modifier.fillMaxWidth()) {
    Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp), content = { Text(title, style = MaterialTheme.typography.headlineSmall); content() })
}
@Composable private fun MetricRow(label: String, value: String, detail: String? = null) = Row(
    Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp),
) {
    Text(label, modifier = Modifier.weight(0.45f), color = MaterialTheme.colorScheme.onSurfaceVariant)
    Column(modifier = Modifier.weight(0.55f), horizontalAlignment = androidx.compose.ui.Alignment.End) {
        Text(value, fontWeight = FontWeight.Bold, textAlign = TextAlign.End)
        detail?.let { Text(it, style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.End) }
    }
}
private fun loss(value: Float) = String.format(Locale.ROOT, "%.6f", value)
private fun signedLoss(value: Float) = String.format(Locale.ROOT, "%+.6f", value)
private fun percent(value: Double) = String.format(Locale.ROOT, "%.1f%%", value)
private fun duration(value: Long) = if (value < 1_000) "${value} ms" else String.format(Locale.ROOT, "%.2f s", value / 1_000.0)
private fun duration(value: Double) = if (value < 1_000) String.format(Locale.ROOT, "%.1f ms", value) else String.format(Locale.ROOT, "%.2f s", value / 1_000.0)
private fun bytes(value: Long) = String.format(Locale.ROOT, "%.1f MiB", value / 1_048_576.0)
private fun eventDescription(event: TrainingDashboardEvent): Pair<String, String?> = when (event.type) {
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.PHASE ->
        "Training state changed" to event.message?.lowercase()?.replace('_', ' ')
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.CHECKPOINT -> "Checkpoint saved" to null
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.RESUME -> "Training resumed" to null
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.ERROR -> "Training error" to event.message
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.QNN_RETURN ->
        "QNN return code" to if (event.message == "true") "success" else "failure"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.TENSOR_FINITE ->
        "Tensor finite check" to if (event.message == "true") "finite" else "non-finite"
    com.yuubinnkyoku.phonelm.TrainingDashboardEventType.CPU_FALLBACK ->
        "CPU fallback" to if (event.message == "true") "observed" else "not observed"
}
private fun phaseMs(timing: PhaseTiming?): String = when { timing == null || timing.backend == TimingBackend.UNAVAILABLE -> "—"; timing.qnnExecuteMs != null -> duration(timing.qnnExecuteMs); timing.hostMs != null -> duration(timing.hostMs); else -> "—" }
private val activePhases = setOf(TrainingPhase.PREPARING, TrainingPhase.INITIALIZING_HTP, TrainingPhase.TRAINING, TrainingPhase.SAVING_CHECKPOINT, TrainingPhase.PAUSED)
private val terminalPhases = setOf(TrainingPhase.COMPLETED, TrainingPhase.ERROR, TrainingPhase.INTERRUPTED)
