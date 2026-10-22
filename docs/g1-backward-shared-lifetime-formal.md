# G1 Shared Backend/Device Lifetime — Five-role Pilot and Blocked Formal Closure

最終分類は **BLOCKED_FORMAL_COHORT（0/3）**。唯一の新ARM processは全5 rolesを完走し、独立照合でも全58,944 consumer pairs、全observer、全数値/word gate、旧5 legacy digestsがPASSした。Backend/Device共有とrequired teardownも実測どおり成立した。

一方、process exit0後に大容量containerのADB回収がtimeoutした。さらに最終timestamp監査で、2秒の待機設定に対し実際のtop監視間隔が約4秒だったことを確認した。したがって結論を **FIVE_ROLE_NUMERICS_AND_LIFETIME_PASS** に限定し、**SHARED_LIFETIME_FIVE_ROLE_PILOT_VERIFIEDの最終protocol qualification、正式3-process、performance、fresh device regressionの認定は与えない**。追加Device arithmeticは停止した。

[Allow-list public summary](results/g1-backward-shared-lifetime-formal-summary.json)。過去の[2-role診断](g1-backward-shared-lifetime-diagnostic.md)と[旧native completion調査](g1-backward-repair-native-completion.md)は変更していない。

## A. Provenance / Frozen identities

正しいresearch cloneとGit common directory、開始時clean tree、HEAD `88b2e8f1237edd230568a1947a6e31984408bb97`を確認した。専用branchは `investigation/g1-backward-shared-lifetime-formal`。基準diagnostic branchと既存research refsは保持した。ローカル絶対path、device/app identifiers、APK hashesは公開しない。

| Required ref | Full SHA | Relationship |
| --- | --- | --- |
| `49d91b6` | `49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2` | ancestor PASS |
| `c6c30c1` | `c6c30c18713084e002ce5485b8f28d67d26faad2` | ancestor PASS |
| `dffbc50` | `dffbc50efaec04f59d16bf126f0631d355f99ae7` | ancestor PASS |
| `30d0cb3` | `30d0cb3bc5eb9e632ac869a122438b3fab64ae06` | ancestor PASS |
| `fad9d19` | `fad9d19a669032aa6d3e29651eb547920ca76506` | ancestor PASS |
| `6c4058a` | `6c4058af820151f89996065380b12778fbd965bc` | ancestor PASS |
| `88b2e8f` | `88b2e8f1237edd230568a1947a6e31984408bb97` | ancestor PASS |

| Identity | SHA256 / commit |
| --- | --- |
| Original arithmetic compiled source | `bb6aaceb42de505b29689a362ce8293687e7ab89` |
| Original repair DSP, reused byte-for-byte | `658bd9005d5ce807f460a7c05d54eb997cb1d5a0f025468c96d9fd19fdc498b5` |
| Original repair Host, reused byte-for-byte | `0aea5fe57625b236226b3cfb55f70dbdad0afda16cae98e1efd2420c4fc4e843` |
| Original ARM, unchanged | `23616e58144eb23ac91f0071f6591359f77f821a0f3bd20715f6388ad2e6cc05` |
| Previous 2-role ARM, unchanged | `2ba5640f5842f9477ad165bae4d49e3a59d19135b60751ff101de508f46ce94d` |
| New five-role ARM | `6ffeec218c2532e6b54c433dbe963f9210ed1beb6c3da4234001a67b7b03126f` |
| New ARM compiled source | `9ceb0324c2a8d82396d597feb0ac46cd1e5ac3c9` |
| Frozen build manifest | `c37ef5139de465a09eed555fdb6563a6d85689f4a6d182cd23fff19c614c9a28` |
| Exact original→new probe diff | `179f13e919ace3fb2d4306c02535774afb211ffad63076f252d50159bbbd3e48` |
| Pre-device independent plan | `5d10b659db0744c22be3fca27bd020ba929d5eff` |
| Append-only Host analysis/recovery source | `87bc0d07bc2a39d610bdd6cd99e61a29625d6230` |

固定環境はQAIRT2.48.40.260702151143、QNN API2.37.0、HTP V81、Hexagon SDK6.6.0.0、Tools19.0.07、NDK26.2.11394342、UCRT64 g++15.2.0。compiler hashes、SDK headers、source/binary/fixture identitiesをprivate manifestで固定した。SDK fallback、APK置換、DSP rebuildはしていない。

## B. Original exit139 reproduction

旧per-role Backend/Device方式は2回、Softmax全9,824 pairs PASSとteardown後のdValue最初のfinalizeでexit139を起こした。SIGSEGV / SEGV_MAPERR、`libQnnHtpPrepare.so + 0x3007bac`、main-thread17-frame stackが一致した。最後の正常APIはgraphAddNode / 0、最後のprepare観測は8MiB VTCM cache取得成功だった。該当probeでは6031/6033を観測していない。

旧46 snapshotsで認識可能な別NPU experimentがない状況でも再現した証拠を保持する。完全な競合排除や根本原因証明とはしない。`EXIT139_REPRODUCED`、`FAULT_SITE_MATCHED`、`BACKEND_REINITIALIZATION_HYPOTHESIS_SUPPORTED`、`ROOT_CAUSE_NOT_PROVEN`を維持する。

## C. Shared-lifetime 2-role intervention

基準88b2e8fの既存独立診断はBackend/Device各1/1、Context2/2、8 graphsで、Softmax9,824とbounded dValue200 pairsがPASSし、dValue first finalize return0、actual exit0、Prepareのrole間reload0だった。これは歴史的な2-role成功であり、今回のfull pilotやformal3へ算入しない。

## D. Full candidate source delta

[新plan](../host_tests/fp32_custom/backward_shared_lifetime_formal_plan.json)を実機前にcommitした。旧 `backward_repair_cohort_plan.json` とそのper-role backend_lifetime clauseは歴史的計画のまま保持した。

新ownerは実証済み2-role ownerのnamespaceとrole上限2→5だけを変えた。五role entry、失敗時の後続execute即時停止、API/finalize/setup tracing、登録済みperformance entryを加えた。performance entryは一度も実行していない。既存workerの算術・tensor layout・observer loops・gatesは失敗停止行以外exact-sourceで照合した。独立Python数値bodyもexactである。

Custom Softmax/dValue、Forward P producer、graph mathematical topology、FP32 ordering/rounding、fixtures/gradients、oracles、accuracy thresholds、production builder、Custom Op/Muon ABI、checkpoint codecは変更していない。Softmaxは `P × (dP − rowSum(P × dP))`、dValueは `Pᵀ × dContext`。Custom Backward算術のordered FP32、no FMA contraction / hidden double / rescaleを維持した。DSP/Hostはfrozen bytesのcopyであり、新kernelはない。CPUはfixture/参照/検証を担当し、これをNPU-only trainingとは呼ばない。

## E. QNN API / resource ownership

固定SDKのQnnContext_createは有効なBackend/Deviceを受け取り、Contextがgraphs/operators/tensorsを保持する。QnnContext_freeはそのContextと関連graph/tensorを解放する。QnnDevice_freeは未解放ContextがあればASSOCIATED_TO_CONTEXT errorを定義する。package登録はBackendに属し、重複登録を拒否する契約である。従って、親を保持してContextを順次生成・解放する方式はこの固定API契約に適合する。Host mockだけを合法性の証明にはしていない。

Backend/Deviceのみprocess scopeで共有し、Context/Graph/Tensor/descriptorは各role独立、非重複。Context free→Device free→Backend freeの順序を守り、free成否不明なら親解放やretryを行わない。packageはBackend当たりforward8登録とrepair Host/DSP各1回。既知のQnn_LogHandle_t leakは意図的に維持し、loggerはBackend freeまで有効だった。logger修復済み、全allocation解放済みとは主張しない。

## F. Host lifecycle and fault injection

正常5-role/20-graphでBackend/Device1/1、Context5/5、package10件、descriptor/handle faults0を確認した。logger/backend/device/package create失敗、各roleのContext create/finalize/execute/free失敗、graph create/addNode、device/backend free、各role early exit、repeated cleanup、stale Context/Graphを検証した。graphCreate/addNode/finalize/executeの実std::exit2経路もatexit cleanupを確認した。

既存ordered FP32/double reference、arithmetic/ABI/preparation rejection、shape/ownership/native fan-out、Forward P関連Host契約をfixed SDKでfresh PASSした。新ARM build前のdedicated Host gatesもPASSした。Hostの成功はvendor runtime内部の欠陥不存在を意味しない。

| Fresh gate | Result |
| --- | --- |
| Dedicated lifecycle/fault injection | PASS、正常5 roles/20 graphs、4実exit cleanup経路 |
| Exact source / frozen DSP/Host | PASS |
| Host profile | exit0、232.4s、PASS2/FAIL0/SKIP0 |
| Qnn profile | exit0、257.5s、PASS4/FAIL0/SKIP0、APK audit PASS |
| Frozen-candidate Fast | exit0、49.5s、outer PASS1/FAIL0/SKIP0 |
| Host analysis Fast | exit0、49.6sおよび47.8s |
| Error scope controls | timer/numerical3、実error8 controls PASS |
| Recovered receipt rejection | 9 cases PASS |
| Exporter rejection | old49/18/30、新33、最終recovered suiteは末尾gate参照 |

すべてPowerShell7で実行した。optional QAIRT samples不足のexit3 advisoryはcore/Build ID確認を通したうえで記録し、SDK変更や弱いgateへの代替はしていない。Project Formal profileは実行していない。

## G. Five-role full pilot

fresh preflightはsame physical device、Awake/known top、active trainingなし、approved installed APK、固定SDK/package/Skel、staged hashes、storage、critical protection、protected assetsをPASSした。自動wake/unlock、foreground takeover、force-stop、root、rebootは行っていない。

唯一の新QNN probe processは `standalone_softmax → standalone_dvalue → native_softmax → native_dvalue → dual` を全件実行した。actual OS exit0、required teardown PASS、process elapsed141.5153525s。これはcompletion観測であり、正式performance値ではない。

| Role | Consumer pairs | Softmax total maxAbs | dValue total maxAbs | Arithmetic / observer word mismatch |
| --- | ---: | ---: | ---: | ---: |
| standalone_softmax | 9824 | 6.3353929016685129e-05 | — | 0 / 0 |
| standalone_dvalue | 9824 | — | 1.6803001869902801e-05 | 0 / 0 |
| native_softmax | 9824 | 0.00010078856647055545 | — | 0 / 0 |
| native_dvalue | 9824 | — | 1.3850631340517339e-05 | 0 / 0 |
| dual | 19648 | 0.00010078856647055545 | 1.3850631340517339e-05 | 0 / 0 |

各single roleは4,912 graphExecute、dualも4,912、全体24,560。4 observersの各consumerにCore2,200 + Holdout256。dualは両consumer分19,648で、全体58,944 consumer pairs。未実行範囲をPASSとして補っていない。

**Protocol qualificationはBLOCKED。** 実行中のtop観測は全てknown、Awake観測も正常、takeover/認識可能な競合/critical eventは未観測だった。しかしruntime loop34件のtop間隔はmin3.250952s、median4.029617s、max5.244854sであり、指定のstrict2秒周期を満たしていない。37件の全top記録にはpre/postも含む。コードのWaitForExit(2000)を維持したことと、実測2秒周期は同義ではない。

## H. Native P lifetime / dual consumer

Native3 rolesは実Forward-produced NATIVE Pを使い、Host Pを再注入していない。各headに単一producer、dual両consumerが同じP tensor IDを参照する従来graphを維持した。Host graph ownership audit、実received words、immutability/finite/guards/poison/full-write checksと全case出力を照合した。

Forward P maxAbs≤1e-6、row-sum≤2e-6、P-induced Softmax bound≤2.5e-4、dValue bound≤0.001、最終Softmax≤0.002/dValue≤0.01の全gateは不変でPASS。NoReceipt/NoTapは不可視Pの直接読出しを主張せず、observed anchorをP referenceに用い、当該条件自身の実出力を独立ideal-P oracleとword contractで評価した。

## I. Three independent formal processes

**0/3、NOT_EXECUTED。** 新pilotをformalへ数えていない。process exit後のtransport failureでmission stopを保持し、formal1/2/3、performance、fresh device regressionを一切開始しなかった。復元/Host分析の成功はDevice停止を解除しない。

## J. Legacy / supplemental numerical digests

5つのlegacy digestは全て旧anchorと一致した。supplementalは従来のG1_NATIVE_COMPLETION_ROLE_V1 domain、role、ordered outputs、canonical mathematical contractsで算出し、receipt timingやdevice/process metadataを除外した。表は**一つのpilot**の結果であり、三独立process agreementではない。

| Role | Legacy digest | Supplemental role-contract digest |
| --- | --- | --- |
| standalone_softmax | `fc0db14324e12cd11dec234405f0e3a767c1787f2bc08b6be65702040b2f1267` | `238145690a71d80fdc9789c314d2329a8ffb80839cef600c81d5384fff3f68bb` |
| standalone_dvalue | `64248fcdfaf2534c37f40bc961987d0a627f6d4c457b3ba7cc553de2ac316e41` | `76beb6cafddde393b35a776bbb0ebe00fcd480c172465591e2a0124f530cdb60` |
| native_softmax | `edcaa214b0833b00c77d8c410d9e32894eef7e6aaecbce0cfb9307933bfaa16d` | `fd7f800e8fe12ea146e0d96da3701c2ac0ae532b06c6eb8c88ec08fe8c6420ab` |
| native_dvalue | `e1238888e75695d08cbd5d267851c49025a3b3578fb394dd59b58bf15ed5ee1a` | `11ca0e02e1801659a58615da3e00d6ea42f4c1bfa6a1899a2d834379a612b2fb` |
| dual | `58eed431aaccb1db918355314cbd0b0dbd5fd83955d41736a9d16ec0ace1eda0` | `1138dc23db96e2c6ca0918439d5514be6cd2354d9b81ee4afd06e404cb3b8c97` |

## K. Observer independence

Observed+Tap / Observed+NoTap / NoReceipt+Tap / NoReceipt+NoTapの全Core/Holdoutでaccuracy PASS、ordered-FP32 words mismatch0、final observer word mismatch0。Observed側のPASSから不可視側のPASSを推定した結果ではない。formal observer-free / Saved-P lifetime certificationは未付与。

## L. Backend / Prepare lifetime events

| Resource / event | Actual |
| --- | ---: |
| Backend create/free | 1/1 |
| Device create/free | 1/1 |
| Context create/free | 5/5 |
| Graph create / successful finalize | 20/20 |
| Graph execute | 24,560 |
| Prepare load/unload total | 1/1 |
| Inter-role Backend/Device recreate | 0 |
| Inter-role Prepare reload | 0 |

全4境界で前Context free、次Context create、first finalize正常returnを確認した。最重要のfirst standalone dValue finalizeはactual return0。Prepare unloadはfinal Backend free開始後で、role間では発生していない。

## M. Crash/runtime watch, transport and protocol audit

このprocessにSIGSEGV/exit139は未観測、全QNN returnは0。現行faulting library/offset/stackはnull。旧 `+0x3007bac` / 17 framesのfailure記録を変更していない。logcat crash/main/system snapshots、QNN stderr、DropBox、32 tombstone copiesをread-onlyで保存した。logcat streamは103 bytes、当該processのowned records0であり、完全な連続streamとは主張しない。

当初のHost lexical scannerは `6031.667µs` 等のapi_usまでerror tokenとして数えた。6031.*4行と6033.*3行は全て計時だった。元rawとfrozen scripts/manifestを保持し、[独立Host追補](../host_tests/fp32_custom/backward_shared_lifetime_formal_analysis_addendum.json)とtyped record分類を追加した。実api return行・stderr/runtime diagnostics・unknown JSONは除外せず、実6031/6033を拒否する8 controlsをPASSした。旧実6031/DMA6033の原因を解明/撤回したものではない。

process exit0後、container gzipのADB pullが180秒上限でtimeoutし、partial archiveを保持してmission stopした。5 role raw/gzipは回収済みだった。同じphysical identityのread-only SHA metadataだけを追加取得し、5 role filesと各gzip/decompression hashを照合した。Host上で別名containerをlossless再構成し、元device containerのSHAと一致した（1,401,659,004 bytes）。新しいDSP executeは0、probe再試行は0。

error-path partial readbackは既存role pathを再利用しようとした。最初のrole gzipは再copyされ、その後raw .binのexclusive creationがFileExistsでoverwriteを拒否した。現在の各raw/gzipはsame-device原本にhash一致するが、この処理を完全なappend-only回収workflowの成功とは扱わない。元 .bin、interrupted container gzip、旧歴史的raw/tombstonesは保持した。original runnerのidentity-privateを捏造せず、明示的なHost-only oracle viewで不変の数値assessorを再使用した。

復元直後のprivate中間評価には数値/resource成功を広く表すpilot labelがある。これを消さず、最終protocol auditで適用範囲を限定し、公開summaryはpilot protocol qualification **BLOCKED** に訂正した。未commitの初期public draftもprivateに保持した。typed numerical成功をformal成功へ昇格していない。

共有lifetimeは今回のfull five-roleでクラッシュ回避に有効だったという介入証拠が得られた。**BACKEND_REINITIALIZATION_HYPOTHESIS_STRENGTHENED、ROOT_CAUSE_NOT_PROVEN**。BackendとDeviceのどちらか、package/Prepare global state、logger allocation pattern、heap/timing依存性は分離できていない。単一processで未発生だったことはvendor defectの不存在を証明しない。

## N. Performance

**NOT_MEASURED。** formal3 entryを満たさず、registered first+31 steady samples、Forward/single/dual比較、median/p10/p90を実行していない。公開figuresはnull。process-global / Context別setup hooksは実装したが、pilotのapi_usを正式Backward costへ昇格しない。過去17.8msをpure Forwardとせず、19 layers×8 microbatchesのtraining wallへ外挿しない。

## O. Fresh HVX / Device / W8 regressions

formal後に予定したS2 LN1、Mixed QKV double Q/K、HVX V、Double QK、Custom Scale/Mask/Softmax P、独立Forward P joint、HVX FFN2/QKV、QK/backward Scale、LN2/custom backward controls、Device smoke、Muon W8 114/114は**全てNOT_EXECUTED**。pilot内Native P joint gateのPASSと別のfresh regressionを区別する。historical PASSはfreshとして流用していない。prepared v2 runner/performance entryもDeviceでは未実行。

## P. Protected-state / negative-control audit

旧333-file ledger、protected refs/assets/fixtures、旧frozen bundles、前2-role153 private filesのbyte hashesを照合した。重複する集合の件数を加算しない。40k checkpoint、100k baseline、200k UNSTABLE、Step41,647/42k、dataset/tokenizer/cache/order、旧exit139 raw/tombstones、five-role pilots、optimizer/Muon ABI、checkpoint codecを保全した。protected checkpointはhash読取のみでdecode/writeしていない。

C0 Host P9 expected FAIL、old cumulative Device P36 FAIL、standard Scale1 FAIL、Mask176 predicate FAIL、Softmax224 local FAIL、standard backward Softmax maxAbs0.30440565/dValue0.03556465 FAILを歴史的negative controlsとして保持した。35/40-node Tap incidentはREPRODUCED / CAUSE UNRESOLVED、旧6031/DMA6033も未解決。今回の共有方式でこれらを修復済みにしていない。

## Q. Remaining Attention Backward frontier / next research

直近は、deadlineを守るtop monitorと大容量rawのbounded/chunked回収、既存archiveの再利用を避けるerror handlingを独立計画で検証すること。その後に新たな許可範囲で同一frozen ARM候補のformal3を検討する。今回の停止を解除したり、自動retryする作業は行っていない。

数学上はPV/Gate/Outputからの実dP/dContext、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregationが残る。frozen gradientsを用いた今回の結果は、これらの実生成経路やFull Attention Backwardの認定を含まない。

## R. Training decision

**40k→60k / 200k training NO-GO。** training、production backend切替、optimizer/LR変更、保護checkpointのdecode/write、push/PR/mergeは行っていない。Full Attention Backward / Full Attention / Full1-layer / Full19-layer / TRAINING_READYは未認定。最終deliverableは数値/resource介入の正の証拠と、protocol/transportの正確なBLOCKED closeoutである。

## Final questions — all 50

| # | Question | Answer |
| --- | --- | --- |
| 1 | 正しいclone / 88b2e8fから開始したか | Yes。research clone/common directory/clean HEADを確認。全7 ancestry PASS。 |
| 2 | Original DSP/Host/probe不変か | Yes。旧5 bundle、旧2-role bundle、compiled sourceとbinary hashesを再照合。 |
| 3 | New five-role ARM SHA256 | `6ffeec218c2532e6b54c433dbe963f9210ed1beb6c3da4234001a67b7b03126f`。 |
| 4 | Two-roleからのdiff | owner namespace/role上限2→5、five-role entry、失敗時即時停止、finalize/setup観測、登録済みperformance entry。算術workerは失敗停止行以外exact。 |
| 5 | Math / graph / fixture / observer不変か | Yes。DSP/Host bytes同一、旧ordered FP32/double oracle・graph builder・fixture/gates不変。 |
| 6 | Original formal plan保存と別改訂 | Yes。旧plan SHA保護。新planを5d10b65で実機前commit。 |
| 7 | Five-role Host lifetime tests | PASS。正常5-role、各role失敗/early exit、stale/repeated cleanup、4実exit経路。 |
| 8 | Backend create/free | 実測1/1。 |
| 9 | Device create/free | 実測1/1。 |
| 10 | Context create/free | 実測5/5。 |
| 11 | Role間Prepare reload | 0。総load/unload各1、unloadはfinal backend free内。 |
| 12 | All20 graphs構築・解放 | 20 create/finalize return0。5 Context freeで関連graph/tensorを解放。 |
| 13 | Full five-role pilot完走 | Runtimeは全件完走、actual exit0。独立数値/resource PASS。ただし最終protocol qualificationはtransfer/cadenceでBLOCKED。 |
| 14 | Standalone Softmax maxAbs/digest | maxAbs 6.3353929016685129e-05。digestはJ表、旧anchor一致。 |
| 15 | Standalone dValue maxAbs/digest | maxAbs 1.6803001869902801e-05。digestはJ表、旧anchor一致。 |
| 16 | Native Softmax total maxAbs | 0.00010078856647055545。 |
| 17 | Native dValue total maxAbs | 1.3850631340517339e-05。 |
| 18 | Dual両total maxAbs | Softmax 0.00010078856647055545、dValue 1.3850631340517339e-05。 |
| 19 | 全4 observer accuracy | PASS、全Core/Holdoutを独立再計算。 |
| 20 | Arithmetic word mismatch | 0。 |
| 21 | Observer final word mismatch | 0。 |
| 22 | Same-P NATIVE dual ownership | pilot内PASS。producer/ID/fan-out graph audit、received P words、実出力を照合。formal未認定。 |
| 23 | P lifetime/immutability | pilot内PASS。Context間tensor持越しなし、immutable input/guards/poison checks PASS。 |
| 24 | NoReceipt/NoTap自身 | 両consumerの実出力でaccuracy/word PASS。不可視Pの直接読出しとは主張しない。 |
| 25 | Pilot actual exit/teardown | actual exit0、required Context→Device→Backend teardown PASS。既知Logger leak除外。回収workflowは失敗。 |
| 26 | Formal3独立process | 未実行、0/3。 |
| 27 | 各formal role coverage | 未実行。pilotは全5-role完全coverage。 |
| 28 | Legacy digest agreement | pilot5 rolesは旧anchorと全一致。formal3-process agreement未実行。 |
| 29 | Supplemental digest agreement | pilot5 role-contract digestsを算出。formal3-process比較未実行。 |
| 30 | Exit139再発 | この唯一のpilotでは未観測。actual exit0。 |
| 31 | API6031/6033 | 実API/errorは未観測。api_usの6031.*が4行、6033.*が3行あり、計時とerrorを分離。 |
| 32 | Awake/top safety | 全観測はAwake/known、takeover0。ただしruntime top間隔median4.029617s、strict2s未達でprotocol qualification BLOCKED。 |
| 33 | NPU競合 | 37 process snapshotsで認識可能な別workloadは未観測。完全排除とはしない。 |
| 34 | Backward追加cost | 未測定。formal3未成立でperformanceを開始していない。 |
| 35 | Forward/single/dual median/p10/p90 | すべてnull / NOT_MEASURED。pilotの診断api_usを正式性能値に昇格しない。 |
| 36 | Setup/teardown分離 | process/Context別timerを実装・診断記録。正式な比較測定は未実行。 |
| 37 | Fresh HVX FFN2/QKV | NOT_EXECUTED。 |
| 38 | Fresh Device smoke | NOT_EXECUTED。 |
| 39 | Fresh W8 114/114 | NOT_EXECUTED。historical PASS流用なし。 |
| 40 | Fresh Forward P joint | pilot内native Pの独立joint gateはPASS。別のrequired fresh device regressionはNOT_EXECUTED。 |
| 41 | Fast/Host/Qnn | fresh PASS。実装/compiled candidateにHost232.4s、Qnn257.5s、Fast49.5s。Host分析修正の専用tests/FastもPASS。最終gateは末尾追記。 |
| 42 | Old negative controls | 全保全。新しくexpected FAILをPASSへ書換えていない。 |
| 43 | 35/40 Tap / 6031の扱い | 不変。TapはREPRODUCED / CAUSE UNRESOLVED、旧6031/DMA6033も原因未解決。今回の計時誤検出とは別問題。 |
| 44 | Protected checkpoint/refs/evidence | 再照合PASS。保護checkpointはhash読取のみ。旧333 ledgerと前2-role153 private filesを別々に照合、件数を加算しない。 |
| 45 | Shared lifetime正式candidate認定 | 不可。数値/resource pilotは有望だがstrict protocol qualification BLOCKED、formal0/3。 |
| 46 | 根本原因証明か支持仮説か | 支持仮説強化、ROOT_CAUSE_NOT_PROVEN。Backend/Device/Prepare/package stateやheap/timingの寄与は分離できていない。 |
| 47 | Saved-P Backward正式closure | 不可。BLOCKED_FORMAL_COHORT / 0/3。performance/fresh regressionも未完了。 |
| 48 | 残るfrontier | 直近はdeadlineを守るmonitorと大容量readbackの独立計画。その後formal3。数学は実dP/dContext生成、dQ/dK、Head Scatter、projection gradients、LN1 dY aggregation。 |
| 49 | Full1-layer / full19-layer | ともにNOT_CERTIFIED。Full Attention/Full Attention Backward/TRAINING_READYも未認定。 |
| 50 | 40k→60k / 200k training | NO-GO維持。今回trainingは実行していない。 |


## Final validation / closeout

最終FastはPowerShell7でactual exit0、47.6s、outer PASS1/FAIL0/SKIP0。recovered public exporter拒否45 cases、runtime error分類のtimer/numerical3 + actual error8 controls、recovery receipt拒否9 casesがPASSした。旧exporter49/18/30 casesと従来の新formal33 casesも保持した。publicの最終出力は現行allow-list exporterによる再導出と一致した。Exporter PASSは投影の正しさであり、pilot protocol/formal certificationのPASSではない。

最終保護/公開監査はPASS。旧tracked pathの変更0、旧333 ledgerと前2-role153 private filesは個別にbyte照合し、全protected refs/frozen binaries/old rawを保持した。全A〜R節と50回答、public schema/privacyを検査した。私的raw、tombstone、fault address、device/app identifiers、APK hashes、SDK binaries、absolute local paths、endpointsはcommit対象に含めていない。

**BLOCKED_FORMAL_COHORT = 0/3。追加Device arithmetic停止、no retry。Backward performance / fresh Device regressions NOT_EXECUTED。Full Attention / Full1-layer / Full19-layer未認定、40k→60k / 200k training NO-GO。**
