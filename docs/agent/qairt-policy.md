# QAIRT固定方針

PhoneLMのQNN有効build、APK監査、実機試験では `scripts/qairt_version.ps1` を設定正本とする。

- SDK root: `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`
- Expected Build ID: `2.48.40.260702151143`

公開exporter内の同じBuild IDは過去evidence/manifestの固定anchorであり、runtime設定のdefaultではない。履歴を正本へ動的追従させない。

rootとBuild IDは操作ごとに明示し、正本と一致しなければ操作開始前にfail closedとする。2.47や別version、自動探索結果、既存APK、build cacheへのfallbackは禁止する。SDKが不完全でもインストール、移動、version変更を自動で行わない。

## requiredとoptional

core requiredは `QnnInterface.h`、`QnnTypes.h`、`QnnSdkBuildId.h` と、PhoneLMが直接packagingする次のarm64/V81 fileである。

- `libQnnSystem.so`
- `libQnnCpu.so`
- `libQnnHtp.so`
- `libQnnHtpPrepare.so`
- `libQnnHtpV81Stub.so`
- unsigned `libQnnHtpV81Skel.so`

`qnn-net-run`、platform validator、converter、samples、examplesなどはinventory上optionalである。optionalだけの不足でQNN buildやrunnerを停止しない。

## `check_qairt.ps1` の分類

| 分類 | exit | 扱い |
| --- | ---: | --- |
| `QAIRT_SDK_ROOT_UNAVAILABLE` / `QAIRT_SDK_ROOT_MISMATCH` | 2 | fatal。別rootへfallbackしない |
| `QAIRT_INVENTORY_INCOMPLETE` | 3 | advisory。coreが揃う場合だけ非fatal |
| `QAIRT_BUILD_ID_MISMATCH` | 4 | fatal |
| `QAIRT_CORE_INCOMPLETE` | 5 | fatal |
| inventory complete | 0 | inventory確認済み。QNN実行可否の正式gateではない |

引数なし自動探索はinstalled SDKの読み取り専用inventory調査だけに使える。QNN有効操作では必ず明示rootとBuild IDを渡す。`-SelfTest` がtemp内の偽SDKを使う場合だけ固定root制約の対象外とする。

## 正式gateとAPK audit

実行可否の正式gateは次である。

1. `qairt_version.ps1` と明示引数の一致
2. `check_qairt.ps1` のBuild ID一致とcore complete
3. 固定引数付き `verify_local.ps1 -WithQairt` のQNN build
4. `audit_qnn_apk.ps1` のarm64 ABI、runtime/backend、V81 Stub/SkelのSDK hash一致
5. APK内の期待Build ID、host path不在、2.47文字列不在

exit 3のoptional inventory advisoryだけでは停止しない。exit 2/4/5、build失敗、APK audit失敗では停止する。
