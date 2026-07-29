[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\docs\results\qnn-htp-root-cause-2026-07'),
    [ValidateRange(1, 1024)][int]$NumLayers = 1,
    [ValidateRange(1, 1024)][int]$NumHeads = 1,
    [ValidateRange(1, 65536)][int]$Tokens = 8,
    [ValidateRange(1, 65536)][int]$EmbeddingDim = 16
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Fail([string]$Message) { throw "qnn tiny LM graph map: $Message" }
function Assert-True([bool]$Condition, [string]$Message) { if (-not $Condition) { Fail $Message } }
function Join-Names([string[]]$Names) { return ($Names -join ';') }
if (($EmbeddingDim % $NumHeads) -ne 0) { Fail 'EmbeddingDim must be divisible by NumHeads' }
$HeadDim = $EmbeddingDim / $NumHeads

# This is intentionally a source-layout exporter, rather than a QNN runtime
# inspector: QNN only exposes the finalized graph through opaque handles.  It
# must consequently reject a requested topology until the runtime source has a
# matching indexed builder/name contract; it never invents a larger graph map.
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$approvedOutputDirectory = [IO.Path]::GetFullPath((Join-Path $repositoryRoot 'docs\results\qnn-htp-root-cause-2026-07'))
if ([IO.Path]::GetFullPath($OutputDirectory) -ne $approvedOutputDirectory) {
    Fail 'OutputDirectory must be the approved public root-cause results directory'
}
$enumSource = Join-Path $repositoryRoot 'app\src\main\cpp\qnn\qnn_runtime_qairt.cpp'
$graphSource = Join-Path $repositoryRoot 'app\src\main\cpp\qnn\qnn_runtime_transformer_training.inc'
Assert-True (Test-Path -LiteralPath $enumSource) 'TensorIndex source is missing'
Assert-True (Test-Path -LiteralPath $graphSource) 'training graph source is missing'
$enumText = Get-Content -LiteralPath $enumSource -Raw
$graphText = Get-Content -LiteralPath $graphSource -Raw
Assert-True $enumText.Contains('struct TinyTransformerTrainingGraph') 'TensorIndex owner is missing'
Assert-True ($graphText -match 'g\.names\[i\]\s*=\s*"layer_00_tensor_"') `
    'runtime tensor naming must retain the layer_00 indexed contract'
Assert-True ($graphText -match 'g\.names\[id\]\s*=\s*std::string\("layer_00_"\)') `
    'runtime parameter naming must retain the layer_00 indexed contract'
$hasGeneralizedBuilder = $graphText.Contains('buildTransformerLayer(') -and
    -not $graphText.Contains('tiny training multi-layer/multi-head graph builder unavailable')
if (($NumLayers -ne 1 -or $NumHeads -ne 1) -and -not $hasGeneralizedBuilder) {
    Fail 'runtime source does not yet provide the requested indexed multi-layer/multi-head builder'
}
foreach ($marker in @('auto make =', 'auto many =', 'auto tapType =', 'TransformerTrainingLayerNormBuilder', 'layerNorm.forward(', 'layerNorm.backward(')) {
    Assert-True $graphText.Contains($marker) "source layout marker is missing: $marker"
}
Assert-True ($graphText -match '(?s)G::SOFTMAX_DOT\},\s*QNN_TENSOR_TYPE_NATIVE,\s*g\.rowDims') `
    'SOFTMAX_DOT must be declared in the tokens x 1 rowDims group'
Assert-True ($graphText -match 'g\.rowDims\[0\]\s*=\s*tokens\s*;\s*g\.rowDims\[1\]\s*=\s*1\s*;') `
    'rowDims source contract must be tokens x 1'
Assert-True ($enumText -match 'uint32_t\s+lastAxisData\[1\]\{1\}') `
    'LAST_AXIS source contract must select axis 1'
Assert-True ($graphText -match '(?s)reduce\("tt_smd".*?G::SOFTMAX_PRODUCT,\s*G::LAST_AXIS,\s*true,\s*G::SOFTMAX_DOT\)') `
    'tt_smd must ReduceSum the last axis with keep_dims=true into SOFTMAX_DOT'
$enumMatch = [regex]::Match($enumText, 'enum TensorIndex\s*\{(?<body>.*?)\bTENSOR_COUNT\b', [Text.RegularExpressions.RegexOptions]::Singleline)
Assert-True $enumMatch.Success 'could not parse TinyTransformerTrainingGraph::TensorIndex'
$tensorIds = @([regex]::Matches($enumMatch.Groups['body'].Value, '\b[A-Z][A-Z0-9_]*\b') | ForEach-Object Value)
Assert-True ($tensorIds.Count -eq 153) "expected 153 TensorIndex entries, got $($tensorIds.Count)"
Assert-True (($tensorIds | Select-Object -Unique).Count -eq $tensorIds.Count) 'TensorIndex contains duplicate entries'

# The language-model configuration used by the fixed-state suites.
$shape = @{}
function Set-Shape([string[]]$Ids, [string]$Value) {
    foreach ($id in $Ids) {
        Assert-True (-not $shape.ContainsKey($id)) "duplicate shape declaration for $id"
        $shape[$id] = $Value
    }
}
Set-Shape @('X','CENTERED1','CENTERED_S1','SQUARE1','XHAT1','XHAT_G1','LN1','Q','K','V','CONTEXT','PROJECTED','RESIDUAL1','CENTERED2','CENTERED_S2','SQUARE2','XHAT2','XHAT_G2','LN2','FF2','OUTPUT','ERROR','SQUARED_ERROR','DOUTPUT','DLN2','DXHAT2','DXHAT_XHAT2','DY_XHAT2','D_TIMES_DXHAT2','FIRST_DIFF2','XHAT_TIMES_SUM2','BRACKET2','DRESIDUAL1_LN','DRESIDUAL1','DCONTEXT','DV','DQ_RAW','DK_RAW','DQ','DK','DLN1_Q','DLN1_K','DLN1_V','DLN1_QK','DLN1','DXHAT1','DXHAT_XHAT1','DY_XHAT1','D_TIMES_DXHAT1','FIRST_DIFF1','XHAT_TIMES_SUM1','BRACKET1','DINPUT_NORM','DINPUT') '8x16'
Set-Shape @('TARGET','LOGITS','LM_PROBABILITIES','LM_DIFFERENCE','DLOGITS') '8x32'
Set-Shape @('ONE_HOT') '8x32'
Set-Shape @('EMBEDDING','DEMBEDDING','S_DEMBEDDING','N_EMBEDDING') '32x16'
Set-Shape @('POSITION','EMBEDDED_TOKEN') '8x16'
Set-Shape @('OUTPUT_PROJECTION','DOUTPUT_PROJECTION','S_DOUTPUT_PROJECTION','N_OUTPUT_PROJECTION') '16x32'
Set-Shape @('G1','B1','G2','B2','DG1','DB1','DG2','DB2','S_DG1','N_G1','S_DB1','N_B1','S_DG2','N_G2','S_DB2','N_B2') '16'
Set-Shape @('WQ','WK','WV','WO','DWQ','DWK','DWV','DWO','S_DWQ','N_WQ','S_DWK','N_WK','S_DWV','N_WV','S_DWO','N_WO') '16x16'
Set-Shape @('W1','DW1','S_DW1','N_W1') '16x32'
Set-Shape @('W2','DW2','S_DW2','N_W2') '32x16'
Set-Shape @('MASK','SCORES','SCALED_SCORES','MASKED_SCORES','PROBABILITIES','DPROBABILITIES','SOFTMAX_PRODUCT','SOFTMAX_CENTERED','DSCORES') '8x8'
Set-Shape @('FF1','RELU','DRELU','RELU_MASK','DFF1','ZERO_FF') '8x32'
Set-Shape @('MEAN1','VAR1','VAR_EPS1','INV_STD_S1','INV_STD1','MEAN2','VAR2','VAR_EPS2','INV_STD_S2','INV_STD2','SUM_DXHAT2','SUM_DXHAT_XHAT2','INV_STD_OVER_D2','SUM_DXHAT1','SUM_DXHAT_XHAT1','INV_STD_OVER_D1','SOFTMAX_DOT') '8x1'
Set-Shape @('LR','SCALE','EPS_SCALED','CENTER_SCALE','DIMENSION','INV_DIMENSION','GRAD_SCALE','LOSS') '1x1'
Set-Shape @('LAST_AXIS','ROW_AXIS') '1'
Set-Shape @('ALL_AXES') '2'
$missingShape = @($tensorIds | Where-Object { -not $shape.ContainsKey($_) })
Assert-True ($shape.Count -eq $tensorIds.Count) "shape table has $($shape.Count), expected $($tensorIds.Count); missing=$($missingShape -join ',')"

$logicalName = @{
    X='embedded_input'; TARGET='target'; G1='gamma1'; B1='beta1'; G2='gamma2'; B2='beta2';
    WQ='wq'; WK='wk'; WV='wv'; WO='wo'; W1='w1'; W2='w2'; LR='learning_rate';
    DG1='gradient_gamma1'; DB1='gradient_beta1'; DG2='gradient_gamma2'; DB2='gradient_beta2';
    DWQ='gradient_wq'; DWK='gradient_wk'; DWV='gradient_wv'; DWO='gradient_wo'; DW1='gradient_w1'; DW2='gradient_w2';
    DINPUT='embedding_input_gradient'; DEMBEDDING='token_embedding_gradient';
    DOUTPUT_PROJECTION='output_projection_gradient'; N_EMBEDDING='next_token_embedding';
    N_OUTPUT_PROJECTION='next_output_projection'; DOUTPUT='doutput'; DLOGITS='dlogits';
    LM_PROBABILITIES='lm_probabilities'; LM_DIFFERENCE='lm_difference'; ONE_HOT='one_hot';
}
function Logical-Name([string]$Id) {
    if ($logicalName.ContainsKey($Id)) { return $logicalName[$Id] }
    return $Id.ToLowerInvariant()
}
function Logical-Role([string]$Id) {
    $roles = @{
        X='token embedding plus positional encoding'; TARGET='fixed target distribution'; ONE_HOT='fixed token one-hot input'; EMBEDDING='token embedding parameter'; POSITION='static positional encoding'; EMBEDDED_TOKEN='token embedding lookup'; OUTPUT_PROJECTION='output projection parameter';
        G1='LayerNorm1 gamma parameter'; B1='LayerNorm1 beta parameter'; G2='LayerNorm2 gamma parameter'; B2='LayerNorm2 beta parameter'; LR='learning-rate scalar'; MASK='static causal attention mask'; SCALE='attention scale'; LAST_AXIS='last-axis reduction indices'; ROW_AXIS='row-axis reduction indices'; ALL_AXES='all-axis reduction indices'; EPS_SCALED='scaled LayerNorm epsilon'; CENTER_SCALE='LayerNorm centering scale'; DIMENSION='model dimension scalar'; INV_DIMENSION='inverse model dimension scalar'; GRAD_SCALE='loss-gradient scale'; ZERO_FF='static FFN zero tensor';
        MEAN1='LayerNorm1 forward mean'; VAR1='LayerNorm1 forward variance'; INV_STD1='LayerNorm1 inverse standard deviation'; XHAT1='LayerNorm1 normalized input'; LN1='LayerNorm1 output'; Q='query projection'; K='key projection'; V='value projection'; SCORES='attention scores'; PROBABILITIES='attention probabilities'; CONTEXT='attention context'; RESIDUAL1='post-attention residual';
        MEAN2='LayerNorm2 forward mean'; VAR2='LayerNorm2 forward variance'; INV_STD2='LayerNorm2 inverse standard deviation'; XHAT2='LayerNorm2 normalized input'; LN2='LayerNorm2 output'; FF1='FFN W1 output'; RELU='FFN activation'; FF2='FFN W2 output'; OUTPUT='transformer output';
        DOUTPUT='upstream transformer-output gradient'; DW2='FFN W2 gradient'; DRELU='FFN activation gradient'; RELU_MASK='FFN activation mask'; DFF1='FFN W1-output gradient'; DW1='FFN W1 gradient'; DLN2='LayerNorm2 upstream gradient'; DRESIDUAL1='residual1 gradient'; DCONTEXT='attention context gradient'; DPROBABILITIES='attention probability gradient'; DV='value projection gradient'; DSCORES='attention score gradient'; DQ='query projection gradient'; DK='key projection gradient'; DWQ='query weight gradient'; DWK='key weight gradient'; DWV='value weight gradient'; DWO='attention output weight gradient'; DLN1='LayerNorm1 upstream gradient'; DG1='LayerNorm1 gamma gradient'; DB1='LayerNorm1 beta gradient'; DG2='LayerNorm2 gamma gradient'; DB2='LayerNorm2 beta gradient'; DINPUT_NORM='LayerNorm1-derived input gradient'; DINPUT='embedding-input gradient after residual addition'; DEMBEDDING='token embedding gradient'; DOUTPUT_PROJECTION='output projection gradient';
    }
    if ($roles.ContainsKey($Id)) { return $roles[$Id] }
    if ($Id.StartsWith('N_')) { return 'next parameter value' }
    if ($Id.StartsWith('S_')) { return 'learning-rate-scaled gradient' }
    if ($Id.StartsWith('D')) { return 'backward intermediate gradient' }
    return ($Id.ToLowerInvariant() -replace '_',' ') + ' intermediate tensor'
}

$staticIds = @('MASK','SCALE','LAST_AXIS','ROW_AXIS','ALL_AXES','EPS_SCALED','CENTER_SCALE','DIMENSION','INV_DIMENSION','GRAD_SCALE','ZERO_FF','POSITION')
$appWriteIds = @('TARGET','ONE_HOT','EMBEDDING','OUTPUT_PROJECTION','G1','B1','WQ','WK','WV','WO','G2','B2','W1','W2')
$diagnosticReadIds = @('X','OUTPUT','DOUTPUT','DG1','DB1','DG2','DB2','DWQ','DWK','DWV','DWO','DW1','DW2','LM_PROBABILITIES','DLOGITS','DOUTPUT_PROJECTION')
$fullReadIds = @('LOGITS','DINPUT','N_G1','N_B1','N_WQ','N_WK','N_WV','N_WO','N_G2','N_B2','N_W1','N_W2','N_EMBEDDING','N_OUTPUT_PROJECTION')
$backwardRegionTapIds = @('DRESIDUAL1','DSCORES','DLN1','DINPUT_NORM')
$layerNorm1TapIds = @('DLN1','DXHAT1','SUM_DXHAT1','DXHAT_XHAT1','SUM_DXHAT_XHAT1','DINPUT_NORM')
function Variant-Membership([string]$Id) {
    if ($Id -eq 'DEMBEDDING') { return 'FULL|STOP_AFTER_DEMBEDDING' }
    if ($Id -in @('S_DG1','N_G1','S_DB1','N_B1','S_DWQ','N_WQ','S_DWK','N_WK','S_DWV','N_WV','S_DWO','N_WO','S_DG2','N_G2','S_DB2','N_B2','S_DW1','N_W1','S_DW2','N_W2','S_DEMBEDDING','N_EMBEDDING','S_DOUTPUT_PROJECTION','N_OUTPUT_PROJECTION')) { return 'FULL' }
    return 'FULL|STOP_AFTER_DINPUT|STOP_AFTER_DEMBEDDING'
}
function Tensor-Exposure([string]$Id) {
    if ($Id -in $appWriteIds) { return 'APP_WRITE (all variants)' }
    if ($Id -eq 'LR') { return 'FULL:APP_WRITE; STOP*:NATIVE' }
    if ($Id -in $staticIds) { return 'STATIC (all variants)' }
    if ($Id -eq 'LOGITS') { return 'APP_READ (all variants)' }
    if ($Id -eq 'DINPUT') { return 'FULL:APP_READ when diagnostics; STOP*:APP_READ' }
    if ($Id -eq 'DEMBEDDING') { return 'FULL:APP_READ when diagnostics; STOP_AFTER_DEMBEDDING:APP_READ; STOP_AFTER_DINPUT:NATIVE' }
    if ($Id -in $backwardRegionTapIds -and $Id -in $layerNorm1TapIds) { return 'BACKWARD_REGIONS|LAYERNORM1 tap:APP_READ; otherwise NATIVE' }
    if ($Id -eq 'DSCORES') { return 'DSCORES_ONLY|DPROB_DSCORES|BACKWARD_REGIONS tap:APP_READ; otherwise NATIVE' }
    if ($Id -eq 'DPROBABILITIES') { return 'DPROB_DSCORES tap:APP_READ; otherwise NATIVE' }
    if ($Id -in $backwardRegionTapIds) { return 'BACKWARD_REGIONS tap:APP_READ; otherwise NATIVE' }
    if ($Id -in $layerNorm1TapIds) { return 'LAYERNORM1 tap:APP_READ; otherwise NATIVE' }
    if ($Id -in $diagnosticReadIds) { return 'APP_READ when diagnostics; otherwise NATIVE' }
    if ($Id -in $fullReadIds) { return 'FULL:APP_READ; STOP*:NATIVE' }
    return 'NATIVE (all variants)'
}
function Tensor-Type([string]$Id) {
    $exposure = Tensor-Exposure $Id
    if ($exposure.StartsWith('APP_WRITE')) { return 'APP_WRITE' }
    if ($exposure.StartsWith('STATIC')) { return 'STATIC' }
    if ($exposure -match 'APP_READ') { return 'variant-dependent APP_READ/NATIVE' }
    if ($Id -eq 'LR') { return 'variant-dependent APP_WRITE/NATIVE' }
    return 'NATIVE'
}
function Tensor-Io([string]$Id) {
    $exposure = Tensor-Exposure $Id
    if ($exposure -match '^APP_WRITE') { return 'graph_input' }
    if ($Id -eq 'LR') { return 'graph_input in FULL; internal otherwise' }
    if ($exposure -match 'APP_READ') { return 'graph_output when exposed; internal otherwise' }
    if ($exposure -match '^STATIC') { return 'static_internal' }
    return 'internal'
}
function Tensor-DType([string]$Id) { if ($Id -in @('LAST_AXIS','ROW_AXIS','ALL_AXES')) { return 'UINT32' }; if ($Id -eq 'RELU_MASK') { return 'BOOL8' }; return 'FLOAT32' }
function Tensor-Rank([string]$Id) { return (($shape[$Id] -split 'x').Count) }

$nodes = [Collections.Generic.List[object]]::new()
function Add-Node([string]$Name, [string]$Op, [string[]]$Inputs, [string]$Output, [string]$Variants = 'FULL|STOP_AFTER_DINPUT|STOP_AFTER_DEMBEDDING') {
    $nodes.Add([pscustomobject]@{ name=$Name; op=$Op; inputs=$Inputs; output=$Output; variants=$Variants })
}
function Add-Lnf([string]$Prefix, [string]$X, [string]$Gamma, [string]$Beta, [string[]]$Outputs) {
    $spec = @(
        @('mean','QNN_OP_REDUCE_MEAN',@($X,'LAST_AXIS')),
        @('center','QNN_OP_ELEMENT_WISE_SUBTRACT',@($X,$Outputs[0])),
        @('center_scale','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[1],'CENTER_SCALE')),
        @('square','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[2],$Outputs[2])),
        @('var','QNN_OP_REDUCE_MEAN',@($Outputs[3],'LAST_AXIS')),
        @('eps','QNN_OP_ELEMENT_WISE_ADD',@($Outputs[4],'EPS_SCALED')),
        @('rsqrt','QNN_OP_ELEMENT_WISE_RSQRT',@($Outputs[5])),
        @('unscale','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[6],'CENTER_SCALE')),
        @('xhat','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[1],$Outputs[7])),
        @('gamma','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[8],$Gamma)),
        @('beta','QNN_OP_ELEMENT_WISE_ADD',@($Outputs[9],$Beta))
    )
    for ($i=0; $i -lt $spec.Count; ++$i) { Add-Node "$Prefix`_$($spec[$i][0])" $spec[$i][1] $spec[$i][2] $Outputs[$i] }
}
function Add-Lnb([string]$Prefix, [string]$Dy, [string]$Gamma, [string]$Xhat, [string]$Inv, [string[]]$Outputs) {
    $spec = @(
        @('dxhat','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Dy,$Gamma),$Outputs[0]),
        @('sum','QNN_OP_REDUCE_SUM',@($Outputs[0],'LAST_AXIS'),$Outputs[1]),
        @('prod','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[0],$Xhat),$Outputs[2]),
        @('sumprod','QNN_OP_REDUCE_SUM',@($Outputs[2],'LAST_AXIS'),$Outputs[4]),
        @('ddx','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[0],'DIMENSION'),$Outputs[5]),
        @('first','QNN_OP_ELEMENT_WISE_SUBTRACT',@($Outputs[5],$Outputs[1]),$Outputs[6]),
        @('xsum','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Xhat,$Outputs[4]),$Outputs[7]),
        @('bracket','QNN_OP_ELEMENT_WISE_SUBTRACT',@($Outputs[6],$Outputs[7]),$Outputs[8]),
        @('invd','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Inv,'INV_DIMENSION'),$Outputs[9]),
        @('dx','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Outputs[8],$Outputs[9]),$Outputs[10]),
        @('dyx','QNN_OP_ELEMENT_WISE_MULTIPLY',@($Dy,$Xhat),$Outputs[3]),
        @('dg','QNN_OP_REDUCE_SUM',@($Outputs[3],'ROW_AXIS'),$Outputs[11]),
        @('db','QNN_OP_REDUCE_SUM',@($Dy,'ROW_AXIS'),$Outputs[12])
    )
    foreach ($entry in $spec) { Add-Node "$Prefix`_$($entry[0])" $entry[1] $entry[2] $entry[3] }
}

Add-Node 'lm_embedding' 'QNN_OP_MAT_MUL' @('ONE_HOT','EMBEDDING') 'EMBEDDED_TOKEN'
Add-Node 'lm_position' 'QNN_OP_ELEMENT_WISE_ADD' @('EMBEDDED_TOKEN','POSITION') 'X'
Add-Lnf 'tt_ln1' 'X' 'G1' 'B1' @('MEAN1','CENTERED1','CENTERED_S1','SQUARE1','VAR1','VAR_EPS1','INV_STD_S1','INV_STD1','XHAT1','XHAT_G1','LN1')
foreach ($n in @(
    @('tt_q','QNN_OP_MAT_MUL',@('LN1','WQ'),'Q'), @('tt_k','QNN_OP_MAT_MUL',@('LN1','WK'),'K'), @('tt_v','QNN_OP_MAT_MUL',@('LN1','WV'),'V'), @('tt_scores','QNN_OP_MAT_MUL',@('Q','K'),'SCORES'), @('tt_scale_scores','QNN_OP_ELEMENT_WISE_MULTIPLY',@('SCORES','SCALE'),'SCALED_SCORES'), @('tt_mask_scores','QNN_OP_ELEMENT_WISE_ADD',@('SCALED_SCORES','MASK'),'MASKED_SCORES'), @('tt_softmax','QNN_OP_SOFTMAX',@('MASKED_SCORES'),'PROBABILITIES'), @('tt_context','QNN_OP_MAT_MUL',@('PROBABILITIES','V'),'CONTEXT'), @('tt_project','QNN_OP_MAT_MUL',@('CONTEXT','WO'),'PROJECTED'), @('tt_res1','QNN_OP_ELEMENT_WISE_ADD',@('X','PROJECTED'),'RESIDUAL1')
)) { Add-Node $n[0] $n[1] $n[2] $n[3] }
Add-Lnf 'tt_ln2' 'RESIDUAL1' 'G2' 'B2' @('MEAN2','CENTERED2','CENTERED_S2','SQUARE2','VAR2','VAR_EPS2','INV_STD_S2','INV_STD2','XHAT2','XHAT_G2','LN2')
foreach ($n in @(
    @('tt_ff1','QNN_OP_MAT_MUL',@('LN2','W1'),'FF1'), @('tt_relu','QNN_OP_RELU',@('FF1'),'RELU'), @('tt_ff2','QNN_OP_MAT_MUL',@('RELU','W2'),'FF2'), @('tt_out','QNN_OP_ELEMENT_WISE_ADD',@('RESIDUAL1','FF2'),'OUTPUT'), @('lm_logits','QNN_OP_MAT_MUL',@('OUTPUT','OUTPUT_PROJECTION'),'LOGITS'), @('lm_softmax','QNN_OP_SOFTMAX',@('LOGITS'),'LM_PROBABILITIES'), @('lm_difference','QNN_OP_ELEMENT_WISE_SUBTRACT',@('LM_PROBABILITIES','TARGET'),'LM_DIFFERENCE'), @('lm_dlogits','QNN_OP_ELEMENT_WISE_MULTIPLY',@('LM_DIFFERENCE','GRAD_SCALE'),'DLOGITS'), @('lm_doutput','QNN_OP_MAT_MUL',@('DLOGITS','OUTPUT_PROJECTION'),'DOUTPUT'), @('lm_doutput_projection','QNN_OP_MAT_MUL',@('OUTPUT','DLOGITS'),'DOUTPUT_PROJECTION'), @('tt_dw2','QNN_OP_MAT_MUL',@('RELU','DOUTPUT'),'DW2'), @('tt_drelu','QNN_OP_MAT_MUL',@('DOUTPUT','W2'),'DRELU'), @('tt_relu_mask','QNN_OP_ELEMENT_WISE_GREATER',@('FF1','ZERO_FF'),'RELU_MASK'), @('tt_relu_bwd','QNN_OP_ELEMENT_WISE_SELECT',@('RELU_MASK','DRELU','ZERO_FF'),'DFF1'), @('tt_dw1','QNN_OP_MAT_MUL',@('LN2','DFF1'),'DW1'), @('tt_dln2','QNN_OP_MAT_MUL',@('DFF1','W1'),'DLN2')
)) { Add-Node $n[0] $n[1] $n[2] $n[3] }
Add-Lnb 'tt_lnb2' 'DLN2' 'G2' 'XHAT2' 'INV_STD2' @('DXHAT2','SUM_DXHAT2','DXHAT_XHAT2','DY_XHAT2','SUM_DXHAT_XHAT2','D_TIMES_DXHAT2','FIRST_DIFF2','XHAT_TIMES_SUM2','BRACKET2','INV_STD_OVER_D2','DRESIDUAL1_LN','DG2','DB2')
foreach ($n in @(
    @('tt_dres1','QNN_OP_ELEMENT_WISE_ADD',@('DOUTPUT','DRESIDUAL1_LN'),'DRESIDUAL1'), @('tt_dwo','QNN_OP_MAT_MUL',@('CONTEXT','DRESIDUAL1'),'DWO'), @('tt_dcontext','QNN_OP_MAT_MUL',@('DRESIDUAL1','WO'),'DCONTEXT'), @('tt_dp','QNN_OP_MAT_MUL',@('DCONTEXT','V'),'DPROBABILITIES'), @('tt_dv','QNN_OP_MAT_MUL',@('PROBABILITIES','DCONTEXT'),'DV'), @('tt_smp','QNN_OP_ELEMENT_WISE_MULTIPLY',@('DPROBABILITIES','PROBABILITIES'),'SOFTMAX_PRODUCT'), @('tt_smd','QNN_OP_REDUCE_SUM',@('SOFTMAX_PRODUCT','LAST_AXIS'),'SOFTMAX_DOT'), @('tt_smc','QNN_OP_ELEMENT_WISE_SUBTRACT',@('DPROBABILITIES','SOFTMAX_DOT'),'SOFTMAX_CENTERED'), @('tt_ds','QNN_OP_ELEMENT_WISE_MULTIPLY',@('PROBABILITIES','SOFTMAX_CENTERED'),'DSCORES'), @('tt_dqr','QNN_OP_MAT_MUL',@('DSCORES','K'),'DQ_RAW'), @('tt_dkr','QNN_OP_MAT_MUL',@('DSCORES','Q'),'DK_RAW'), @('tt_dq','QNN_OP_ELEMENT_WISE_MULTIPLY',@('DQ_RAW','SCALE'),'DQ'), @('tt_dk','QNN_OP_ELEMENT_WISE_MULTIPLY',@('DK_RAW','SCALE'),'DK'), @('tt_dwq','QNN_OP_MAT_MUL',@('LN1','DQ'),'DWQ'), @('tt_dwk','QNN_OP_MAT_MUL',@('LN1','DK'),'DWK'), @('tt_dwv','QNN_OP_MAT_MUL',@('LN1','DV'),'DWV'), @('tt_dlq','QNN_OP_MAT_MUL',@('DQ','WQ'),'DLN1_Q'), @('tt_dlk','QNN_OP_MAT_MUL',@('DK','WK'),'DLN1_K'), @('tt_dlv','QNN_OP_MAT_MUL',@('DV','WV'),'DLN1_V'), @('tt_dlqk','QNN_OP_ELEMENT_WISE_ADD',@('DLN1_Q','DLN1_K'),'DLN1_QK'), @('tt_dln1','QNN_OP_ELEMENT_WISE_ADD',@('DLN1_QK','DLN1_V'),'DLN1')
)) { Add-Node $n[0] $n[1] $n[2] $n[3] }
Add-Lnb 'tt_lnb1' 'DLN1' 'G1' 'XHAT1' 'INV_STD1' @('DXHAT1','SUM_DXHAT1','DXHAT_XHAT1','DY_XHAT1','SUM_DXHAT_XHAT1','D_TIMES_DXHAT1','FIRST_DIFF1','XHAT_TIMES_SUM1','BRACKET1','INV_STD_OVER_D1','DINPUT_NORM','DG1','DB1')
Add-Node 'lm_dinput' 'QNN_OP_ELEMENT_WISE_ADD' @('DRESIDUAL1','DINPUT_NORM') 'DINPUT'
Add-Node 'lm_dembedding' 'QNN_OP_MAT_MUL' @('ONE_HOT','DINPUT') 'DEMBEDDING' 'FULL|STOP_AFTER_DEMBEDDING'
foreach ($u in @(
    @('tt_g1','G1','DG1','S_DG1','N_G1'), @('tt_b1','B1','DB1','S_DB1','N_B1'), @('tt_wq','WQ','DWQ','S_DWQ','N_WQ'), @('tt_wk','WK','DWK','S_DWK','N_WK'), @('tt_wv','WV','DWV','S_DWV','N_WV'), @('tt_wo','WO','DWO','S_DWO','N_WO'), @('tt_g2','G2','DG2','S_DG2','N_G2'), @('tt_b2','B2','DB2','S_DB2','N_B2'), @('tt_w1','W1','DW1','S_DW1','N_W1'), @('tt_w2','W2','DW2','S_DW2','N_W2'), @('lm_embedding','EMBEDDING','DEMBEDDING','S_DEMBEDDING','N_EMBEDDING'), @('lm_output_projection','OUTPUT_PROJECTION','DOUTPUT_PROJECTION','S_DOUTPUT_PROJECTION','N_OUTPUT_PROJECTION')
)) {
    Add-Node "$($u[0])_scale" 'QNN_OP_ELEMENT_WISE_MULTIPLY' @($u[2],'LR') $u[3] 'FULL'
    Add-Node "$($u[0])_next" 'QNN_OP_ELEMENT_WISE_SUBTRACT' @($u[1],$u[3]) $u[4] 'FULL'
}
Assert-True ($nodes.Count -eq 123) "expected 123 language-model source nodes, got $($nodes.Count)"
Assert-True (($nodes.name | Select-Object -Unique).Count -eq $nodes.Count) 'node table contains duplicate names'
foreach ($node in $nodes) {
    foreach ($id in @($node.inputs + $node.output)) { Assert-True ($tensorIds -contains $id) "node $($node.name) references unrecognized tensor $id" }
}
foreach ($marker in @('layerNorm.forward("tt_ln1"', 'layerNorm.forward("tt_ln2"', 'layerNorm.backward("tt_lnb1"', 'layerNorm.backward("tt_lnb2"', 'lm_dembedding', 'auto upd =')) { Assert-True $graphText.Contains($marker) "node-builder marker is missing: $marker" }

$producer = @{}
$consumers = @{}
foreach ($id in $tensorIds) { $consumers[$id] = [Collections.Generic.List[string]]::new() }
for ($i=0; $i -lt $nodes.Count; ++$i) {
    $node = $nodes[$i]
    Assert-True (-not $producer.ContainsKey($node.output)) "tensor $($node.output) has multiple producers"
    $producer[$node.output] = $node.name
    foreach ($input in $node.inputs) { $consumers[$input].Add($node.name) }
}

$nodeRows = for ($i=0; $i -lt $nodes.Count; ++$i) {
    $node = $nodes[$i]
    [pscustomobject][ordered]@{
        creation_index = $i
        name = $node.name
        op = $node.op
        inputs = Join-Names @($node.inputs | ForEach-Object { Logical-Name $_ })
        outputs = Logical-Name $node.output
        rank = Tensor-Rank $node.output
        dims = $shape[$node.output]
        dtype = Tensor-DType $node.output
        tensor_type = Tensor-Type $node.output
        exposure = Tensor-Exposure $node.output
        variant_membership = $node.variants
    }
}
$tensorRows = for ($i=0; $i -lt $tensorIds.Count; ++$i) {
    $id = $tensorIds[$i]
    $producerName = if ($producer.ContainsKey($id)) { $producer[$id] } elseif (($id -in $appWriteIds) -or ($id -in $staticIds) -or $id -eq 'LR') { 'source_tensor' } else { 'not_produced_in_language_model' }
    [pscustomobject][ordered]@{
        creation_index = $i
        name = Logical-Name $id
        producer = $producerName
        consumers = Join-Names @($consumers[$id])
        logical_role = Logical-Role $id
        shape = $shape[$id]
        dtype = Tensor-DType $id
        memory_type = Tensor-Type $id
        io_classification = Tensor-Io $id
        exposure = Tensor-Exposure $id
        variant_membership = Variant-Membership $id
    }
}
Assert-True ($nodeRows.Count -eq 123) 'node CSV count assertion failed'
Assert-True ($tensorRows.Count -eq 153) 'tensor CSV count assertion failed'

$resolvedOutput = $approvedOutputDirectory
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$nodePath = Join-Path $resolvedOutput 'node-map.csv'
$tensorPath = Join-Path $resolvedOutput 'tensor-map.csv'
$nodeRows | Export-Csv -LiteralPath $nodePath -NoTypeInformation -Encoding utf8
$tensorRows | Export-Csv -LiteralPath $tensorPath -NoTypeInformation -Encoding utf8

# This topology table is emitted only after the source consistency checks above.
# For the legacy graph it records its actual layer_00 naming contract.  A
# multi-layer/head request is rejected unless the runtime itself exposes the
# indexed builder contract, preventing a synthetic map from being mistaken for
# an executable graph.
$topologyRows = [Collections.Generic.List[object]]::new()
for ($layer = 0; $layer -lt $NumLayers; ++$layer) {
    $prefix = 'layer_{0:D2}' -f $layer
    $topologyRows.Add([pscustomobject][ordered]@{ layer=$layer; head=''; tensor="$prefix`_input"; shape="$Tokens`x$EmbeddingDim"; role='layer input / residual input' })
    foreach ($name in @('norm1_output','q','k','v','attention_concat','attention_output','residual1','norm2_output','ffn_output','output','input_gradient')) {
        $topologyRows.Add([pscustomobject][ordered]@{ layer=$layer; head=''; tensor="$prefix`_$name"; shape="$Tokens`x$EmbeddingDim"; role='layer-scoped transformer tensor' })
    }
    foreach ($head in 0..($NumHeads - 1)) {
        $headPrefix = "$prefix`_head_{0:D2}" -f $head
        foreach ($name in @('q','k','v','context')) {
            $topologyRows.Add([pscustomobject][ordered]@{ layer=$layer; head=$head; tensor="$headPrefix`_$name"; shape="$Tokens`x$HeadDim"; role='head-scoped projection/context tensor' })
        }
        foreach ($name in @('scores','probabilities')) {
            $topologyRows.Add([pscustomobject][ordered]@{ layer=$layer; head=$head; tensor="$headPrefix`_$name"; shape="$Tokens`x$Tokens"; role='head-scoped attention tensor' })
        }
    }
}
$topologyPath = Join-Path $resolvedOutput 'transformer-topology-map.csv'
$topologyRows | Export-Csv -LiteralPath $topologyPath -NoTypeInformation -Encoding utf8
Assert-True ($topologyRows.Count -eq ($NumLayers * (12 + 6 * $NumHeads))) 'topology row count assertion failed'
Write-Output "node_map_rows=$($nodeRows.Count)"
Write-Output "tensor_map_rows=$($tensorRows.Count)"
Write-Output "topology_map_rows=$($topologyRows.Count)"
Write-Output "shape=B1_T$Tokens`_V32_D$EmbeddingDim; layers=$NumLayers; heads=$NumHeads; head_dim=$HeadDim; diagnostic_outputs=true; creation_index=zero_based_source_order"
