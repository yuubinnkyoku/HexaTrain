# PhoneLM

QAIRT 2.48のQNN HTP上で、2層ReLU MLPの学習stepに必要な数値演算を明示的なgraphとして実行するAndroid実験です。

現在は、forward、MSE loss、出力勾配、linear backward、ReLU backward、SGD weight updateを1つのHTP graphへ統合し、従来のforward + backwardの`2 execute/step`から`1 execute/step`へ削減しています。

```text
Inputs: X, Y, W1_current, W2_current, learning_rate

Z1   = X W1_current
H    = ReLU(Z1)
P    = H W2_current
E    = P - Y
loss = mean(E²)
dP   = E × 2/(B×O)

dW2  = Hᵀ dP
dH   = dP W2_currentᵀ
mask = H > 0
dZ1  = Select(mask, dH, 0)
dW1  = Xᵀ dZ1

W1_next = W1_current - learning_rate × dW1
W2_next = W2_current - learning_rate × dW2

Outputs: loss, W1_next, W2_next
```

QNNの自動微分は使用していません。各forward/backward/optimizer式をQNN opで明示的に構築しています。

## 現在の到達点

対象実機:

- nubia Z80 Ultra / NX741J
- SM8850
- HTP V81
- Android 16 / API 36

検証済みQAIRT構成:

| 項目 | バージョン |
|---|---|
| QAIRT SDK | `2.48.40.260702` |
| QAIRT Build ID | `2.48.40.260702151143` |
| QNN Core API | `2.37.0` |
| HTP Backend API | `5.48.0` |
| Android NDK | `26.2.11394342` / r26c |
| CMake | `3.22.1` |
| ABI | `arm64-v8a` |
| compileSdk / targetSdk | 35 / 35 |
| minSdk | 26 |

成立している処理:

- QNN HTP backend/device/context作成
- V81 Stub/Skelロード、Skel再利用、破損復旧
- FP32 MatMul、transpose input parameter
- APP_WRITE tensorのruntime更新
- HTP forwardとReLU forward
- `dW = XᵀdP`
- `dX = dP Wᵀ`
- ReLU backward `Select(H > 0, dH, 0)`
- 2層ReLU MLPの複数step学習
- fused linear/ReLU backward graph
- MSE loss、dP、SGDを含むfull training-step graph
- graphをrunごとに1回だけcreate/finalizeし、全stepで再利用
- app-owned weight bufferのping-pong handoff
- correctness graphとbenchmark graphの分離
- CPU fallbackなしの実機回帰

CPUには次が残ります。

- mini-batch X/Yの供給
- `graphExecute`の呼び出しとループ制御
- step間のweight buffer handoff
- loss/weight outputの取得
- correctness用CPU参照
- ログとレポート生成

したがって、「学習stepの数値演算をHTPで実行した」と表現します。「NPUだけで学習した」「CPUを完全に使用していない」「QNNが自動微分した」とは表現しません。

## Full training-step graph

graph名は`phonelm_mlp_training_step`です。16 nodeで構成されます。

| 段階 | QNN op |
|---|---|
| Z1、P、dW2、dH、dW1 | `MatMul` |
| H | `Relu` |
| E、W1_next、W2_next | `ElementWiseSubtract` |
| E²、dP、scaled gradient | `ElementWiseMultiply` |
| loss | `ReduceMean` |
| mask | `ElementWiseGreater` |
| dZ1 | `ElementWiseSelect` |

主要tensor:

| tensor | shape | datatype / type |
|---|---|---|
| X | `[B,I]` | FP32 APP_WRITE |
| Y | `[B,O]` | FP32 APP_WRITE |
| W1_current | `[I,H]` | FP32 APP_WRITE |
| W2_current | `[H,O]` | FP32 APP_WRITE |
| learning_rate | `[1,1]` | FP32 APP_WRITE |
| loss | `[1,1]` | FP32 APP_READ |
| W1_next | `[I,H]` | FP32 APP_READ |
| W2_next | `[H,O]` | FP32 APP_READ |
| mask | `[B,H]` | BOOL_8 |
| ReduceMean axes | `[2] = {0,1}` | UINT32 STATIC |
| grad scale | `[1,1] = 2/(B×O)` | FP32 STATIC |

`ReduceMean`はaxes `{0,1}`、`keep_dims=true`で、全`B×O`要素のmeanを返します。

同じstepの`dH`は必ず更新前の`W2_current`を参照します。`W2_next`から`dH`への依存はありません。

### Weight handoff

W1/W2はそれぞれ2つのapp-owned bufferを使用します。

```text
step 0: Aをinput、Bをoutput
step 1: Bをinput、Aをoutput
step 2: Aをinput、Bをoutput
```

`graphExecute`完了後にvector ownerをswapします。同一execute内のinput/outputは別bufferで、未定義のin-place aliasは使いません。weight memcpyは`0 byte/step`です。

## 実機correctness結果

### Op micro test

shape `[2,3]`でCPU参照と比較しました。

| test | max absolute error |
|---|---:|
| MSE loss | `6.4969e-6` |
| dP | `1.9532e-4` |
| SGD W_next | `2.1487e-4` |

### Small full-step

`B=2, I=4, H=5, O=3, 10 steps`:

| 項目 | 結果 |
|---|---:|
| loss | `0.0724977 → 0.0591197` |
| prediction max error | `4.2701e-4` |
| loss absolute error | `7.5996e-5` |
| dP max error | `2.0247e-4` |
| dW2 max error | `2.3669e-4` |
| dH max error | `3.9686e-4` |
| mask mismatch | `0` |
| dZ1 max error | `3.9686e-4` |
| dW1 max error | `1.6469e-4` |
| W1_next max error | `6.0940e-4` |
| W2_next max error | `4.6563e-4` |

### 5 seed training

`sample_count=512, B=8, I=128, H=128, O=64, epochs=10, learning_rate=0.5`:

| seed | CPU all | 既存fused | HTP full-step |
|---:|---:|---:|---:|
| 20260710 | 0.0473557 | 0.0473546 | 0.0473549 |
| 20260711 | 0.0497513 | 0.0497498 | 0.0497481 |
| 20260712 | 0.0492008 | 0.0492016 | 0.0492117 |
| 20260713 | 0.0494006 | 0.0494020 | 0.0494110 |
| 20260714 | 0.0496901 | 0.0496893 | 0.0496987 |

全seedでloss低下、CPU fallbackなし、NaN/Infなし、HTP execute失敗なしでした。

CPU/full-stepの最大差:

- final loss: `1.0953e-5`
- final prediction: `1.0306e-2`
- W1: `1.0136e-2`
- W2: `6.7582e-3`

W1/W2 trajectoryはstep 0、1、2、5、10、20、50、100、finalでchecksum、L2 norm、CPU/HTP最大差を保存します。

## 性能

Debug APK、scalar C++ CPU参照、DVFS/thermal制御なしで各shapeを3 run測定した値です。端末CPUの最適性能を表すものではありません。

| B/I/H/O | 既存fused full-step | 新full-step | speedup | break-even |
|---|---:|---:|---:|---:|
| 2/4/5/3 | 7413.6 µs | 3917.8 µs | 1.89× | 6 steps |
| 8/128/128/64 | 11060.7 µs | 4848.0 µs | 2.28× | 1 step |
| 8/256/256/128 | 19779.1 µs | 8036.2 µs | 2.46× | 1 step |
| 32/256/256/128 | 19716.6 µs | 7512.0 µs | 2.62× | 1 step |

基本shapeのtraining graph execute中央値は約`3957 µs`、weight buffer swapは約`0.26 µs`です。測定時温度は38–39°C、thermal status 0でした。

## ビルド

必要なもの:

- Windows PowerShell
- JDK 17
- Android SDK Platform 35
- Android NDK `26.2.11394342`
- CMake `3.22.1`
- QAIRT `2.48.40.260702151143`（SDK rootを`QAIRT_SDK_ROOT`へ設定）
- MNN 3.5.0 source tree（CPU/MNN既存モード用）

MNN取得:

```powershell
.\scripts\fetch_mnn.ps1
```

QNN有効Debug APK:

```powershell
$env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$QairtSdkRoot = $env:QAIRT_SDK_ROOT

.\gradlew.bat :app:clean :app:assembleDebug `
  '-Pphonelm.enableQnn=true' `
  "-Pqairt.sdkRoot=$QairtSdkRoot" `
  '-Pqairt.expectedBuildId=2.48.40.260702151143' `
  --no-daemon
```

生成物:

```text
app/build/outputs/apk/debug/app-debug.apk
```

QNN無効buildでは`qnn_runtime_stub.cpp`が使用されます。

## QAIRT runtimeの扱い

QAIRTライブラリ、Stub、SkelはGitへ追加しません。QNN有効build時にローカルSDKからbuild directoryへstageし、APKへ組み込みます。

build時に次を検査します。

- `sdk.yaml`と`QnnSdkBuildId.h`のBuild ID一致
- QAIRT `2.48.40.260702151143`との一致
- NDK r26c宣言
- QNN header/runtimeの存在
- V81 Stub/unsigned Skelの存在

実行時はSkelをアプリ専用領域`files/qnn-dsp/<Build ID>/`へSHA-256検証付きで展開し、プロセス限定`ADSP_LIBRARY_PATH`からロードします。root化、SELinux変更、システム領域変更は不要です。

QAIRT配布物の再配布可否は別途ライセンス確認が必要です。ローカルでビルドできることは、公開APKへ再配布可能であることを意味しません。

## 実行モード

主な現行モード:

| mode | HTP | CPU |
|---|---|---|
| `QNN_HTP_DX_CHECK` | dX | 参照比較 |
| `QNN_HTP_RELU_BACKWARD_CHECK` | ReLU backward | 参照比較 |
| `QNN_HTP_MLP_HTP_LINEAR_BACKWARD` | forward、dW2、dH、dW1 | loss、dP、ReLU backward、SGD |
| `QNN_HTP_MLP_FUSED_BACKWARD` | forward、dW2、dH、ReLU backward、dW1 | loss、dP、SGD |
| `QNN_HTP_MSE_CHECK` | MSE、dP、SGD micro graph | 参照比較 |
| `QNN_HTP_SGD_CHECK` | MSE、dP、SGD micro graph | 参照比較 |
| `QNN_HTP_MLP_FULL_STEP` | forward、loss、dP、backward、SGD | graph制御、handoff、参照比較 |
| `QNN_HTP_MLP_FULL_STEP_BENCHMARK` | full numerical step | graph制御、handoff、最終確認 |

CPU、MNN CPU/OpenCL/Vulkan、linear regression、device probeなどの既存モードも保持されています。完全な一覧は[BenchmarkConfig.kt](app/src/main/java/com/yuubinnkyoku/phonelm/BenchmarkConfig.kt)と[training_engine.h](app/src/main/cpp/training_engine.h)を参照してください。

## 自動試験

Full training-step一括試験:

```powershell
.\scripts\run_qnn_htp_full_step_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Repetitions 3 `
  -Seeds 20260710,20260711,20260712,20260713,20260714 `
  -RunPerformance
```

処理順:

```text
SDK/Build ID検査
→ clean build
→ APK監査
→ install -r
→ MSE/dP/SGD micro test
→ small full-step correctness
→ 5 seed学習
→ fused/full-step性能比較
→ break-even推定
→ 既存device/training/HTP dW/MLP/fused回帰
→ CSV/JSON/Markdown生成
```

runnerは次を保証します。

- online ADB targetが1台でなければ停止
- ADB endpointをレポートへ保存しない
- 45°C thermal guard
- CPU fallback、NaN/Inf、結果欠落で失敗
- 成功済みrunから再開可能
- APK、ログ、CSV、JSON、レポートをGitへ追加しない

個別回帰:

```powershell
.\scripts\run_qnn_device_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143'

.\scripts\run_qnn_training_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143' `
  -BatchSize 8 -InputDim 128 -Steps 100 -LearningRate 5

.\scripts\run_qnn_htp_backward_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Repetitions 1 -Seeds 20260710

.\scripts\run_qnn_htp_mlp_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Repetitions 1 -Seeds 20260710

.\scripts\run_qnn_htp_fused_backward_tests.ps1 `
  -QairtSdkRoot $env:QAIRT_SDK_ROOT `
  -ExpectedBuildId '2.48.40.260702151143' `
  -Repetitions 1 -Seeds 20260710
```

## ディレクトリ

```text
app/src/main/cpp/
  cpu_reference_training.cpp
  training_engine.cpp
  qnn/
    qnn_runtime_qairt.cpp
    qnn_runtime_mlp.inc
    qnn_runtime_full_step.inc
    qnn_mlp_training.cpp

app/src/main/java/com/yuubinnkyoku/phonelm/
  BenchmarkConfig.kt
  BenchmarkViewModel.kt
  MainActivity.kt

scripts/
  audit_qnn_apk.ps1
  run_qnn_device_tests.ps1
  run_qnn_training_tests.ps1
  run_qnn_htp_backward_tests.ps1
  run_qnn_htp_mlp_tests.ps1
  run_qnn_htp_fused_backward_tests.ps1
  run_qnn_htp_full_step_tests.ps1
```

## 詳細文書

- [Experimental results](docs/results/README.md)
- [QNN HTP full training step](docs/qnn-htp-full-training-step.md)
- [QNN HTP fused backward](docs/qnn-htp-fused-backward.md)
- [QNN HTP 2-layer MLP](docs/qnn-htp-mlp-training.md)
- [QNN HTP dW](docs/qnn-htp-backward-dw.md)
- [QNN HTP linear training](docs/qnn-htp-linear-training.md)
- [QAIRT 2.48 HTP runtime](docs/qnn-htp-qairt-2.48.md)
- [Device create analysis](docs/qairt-2.48.40-device-create-analysis.md)
- [旧README](docs/archive/README-before-htp-full-step.md)

## 制約

- 対象は現時点でarm64-v8a / V81です。
- QAIRT 2.47との混在は禁止しています。
- Debug APKとscalar CPU参照による性能値です。
- HTP/CPU/GPU周波数は固定していません。
- thermal throttlingは無効化していません。
- bias、異なるoptimizer、任意shapeの動的graphは今回の対象外です。
- `B=32, I=512, H=512, O=256`の任意拡大性能測定は未実施です。

## README履歴

full training-step統合前のREADMEは[docs/archive/README-before-htp-full-step.md](docs/archive/README-before-htp-full-step.md)へ保存しています。

## License

PhoneLM original source code and documentation are licensed under the [Apache License 2.0](LICENSE) unless otherwise noted.

Qualcomm QAIRT/QNN components and other third-party materials are not covered by this license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency and redistribution boundaries.
