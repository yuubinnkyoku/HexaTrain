# HTP/HVX FP32 boundary feasibility

開始source: `ed5f372f3182ba0302c17122987e4b510dd83d23`。
branch: `investigation/htp-hvx-fp32-boundary-feasibility`。
現在は診断実装段階。実機結果はまだない。60k/200kを含む学習updatesは実行しない。

## 事前登録した判断基準

- Host/DSP: 256 finite FP32 raw patterns、3独立process、各6往復。
  全要素bit mismatch0、DSP独立fixture照合0、input mutation0、complete write、return0を別々に検査。
  poisonは各期待wordのbit complement。DSPはinteger memcpyのみで全要素を返す。
  DSP受信全wordを独立した整数metadata channelでも返し、Host送信bitと全要素照合する。
- QNN: FLOAT32 APP_WRITE→単一Reshape→APP_READと、Add(x,0)、Multiply(x,1)、Subtract(x,offset)を分ける。
  Reshapeもbackend loweringを経由するため、単純なメモリ共有または純粋なtransportとは仮定しない。
- D64既存20 recipes、full upstream1、double referenceとanchored FP32 Host oracleを再利用。
  input/centered/inverse/xhat/dY/dXhat/dX/dGamma/dBetaを検査する。
  Host oracleはDSP演算の代用ではなく、再投入する高精度fixtureの生成にのみ使う。
- dX/dGamma/dBetaはdoubleとの差≤0.02。xhatは従来ordinary0.002、登録済み低分散0.06を維持。
  input/centered/inverse/dY/dXhatの再投入は独立boundary diagnosticとしてFP32値との差0.002も報告。
  transfer bit-exactとconsumer accuracyを混同しない。
- HTPから取得済みの値をDSPへ渡すB3と、Host oracle fixtureをDSPで無演算echoしてHTPへ戻すB4を区別。
  B4はDSP LayerNormのPASSではない。
- 単独micrographの必要精度がFAILならfull model分割、DSP LayerNorm kernel、性能promotionへ進まない。
  別のcustom opや拡張FP32演算方式まで不可能とは断定しない。

## Reuse audit

既存`hexatrain_hvx_probe.run`のfloat sequence ABI、QAIC Stub/Skel生成、rpcmem、cdsp unsigned PD、
V81 buildを再利用。未使用mode8をdiagnostic専用で追加し、IDLは変更しない。
production Muonのmode5/7、W8 workers、optimizer publication、checkpoint/resumeは変更しない。
新しいRPC framework、CPU fallback、SDK変更はない。

production `nicopedia_hvx_muon.cpp`はpersistent Session/mutex、RPC全期間のbuffer ownership、
finite validationとpoisonを持つ。DSPはscratch allocationとworker completion後に結果をpublishする。
診断は独立Sessionで128-byte alignmentをHost/DSP両側で確認する。
raw copy試験にはFP32演算、cast、FMA、fast-mathの算術効果を含めない。
既存buildは`-mv81 -mhvx -mhvx-length=128B -O2`で、明示fast-mathは追加しない。

過去HTP MuonのNS5 relativeL2≈0.052は同じAPI成功/finiteとaccuracyの分離を要求するが、
MatMulとLNの誤差起源を同一とは断定しない。
FLOAT32/compensation、weights packing、activation fusionの否定的結果、context splittingの5010を再利用し、再探索しない。
旧LN backwardの0.005 upstreamとscale64→8→1はfull-upstream D64の証明ではない。
fixed-state reproducibility/bisectionではgraph構成・順序に依存した変動があったため、
今回のdynamic LN tap/native最終五出力もbitwise比較する。

## 実験結果とdecision

未実施。結果取得後に本節を更新する。Transport/kernel/integrationは段階別に判定する。
Host20GiB、Device12GiB、QAIRT `2.48.40.260702151143`、Hexagon SDK6.6.0.0、V81を維持。
6031は`UNRESOLVED / DORMANT / WATCH`として別incident。
private evidenceはignored `build/reports/htp-hvx-boundary/` とheadless reportに保存。
raw tensors、device identifiers、checkpoint、run IDs、APK/SDK binariesはcommitしない。
