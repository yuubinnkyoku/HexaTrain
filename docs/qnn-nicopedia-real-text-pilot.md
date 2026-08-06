# ニコニコ大百科実テキスト seed 安定性 pilot

## 結論

2024-01末表示分を収録したニコニコ大百科データ 2024-11-25版を、記事単位の
決定的split、TRAIN限定tokenizer選択、private cache、validation選択へ接続した。
この小規模pilotでは、人工データで見られた大きな性能上のseed不安定性は再現しなかった。
seed 1/2/4のvalidation NLL範囲はL6で0.0204、L19で0.0376、development
NLL範囲はそれぞれ0.0209、0.0360だった。6 runはすべてfiniteで、通常のcausal
Attentionを保ったまま学習できた。

一方、greedy free-running生成のseed差は残った。3 seedのbyte単位argmax一致率は
L6で0.602、L19で0.379、最初の分岐までの平均長は6.44 byteと2.13 byteだった。
従って今回もっとも支持された説明は、held-out性能の大きな学習失敗ではなく、
小さな次token順位差がfree-runningで増幅されることである。L19はdevelopment NLLを
わずかに改善したが生成分岐は早く、深さがこの予算で安定化要因だったとはいえない。

人工データの「均質TRAINがmixed-prefix挙動を未拘束にする」という因果機構は、
自然なcontext多様性を持つ今回のTRAINでは再現できなかった。paired-prefix評価では
遠方prefix置換への感度を観測したものの、保持した8 byteだけでtargetの意味的不変性を
保証できないため探索的証拠に限定する。人工データの結論を実テキストへそのまま
一般化する根拠には使わない。

## 利用条件とデータ保護

[NIIのデータセット説明](https://www.nii.ac.jp/dsc/idr/nico/)と
[個別利用規約](https://www.nii.ac.jp/dsc/idr/service/documents/service-policy-indv.html)、
[ニコニコ大百科データ利用規約](https://www.nii.ac.jp/dsc/idr/nico/documents/dwango-pedia-policy.html)
を確認した。研究利用、出所表示、データセットを第三者へ再配布しないこと、個人・組織の
特定や必要以上のデータ開示を避けることを境界とした。利用は非商用の研究目的に限定し、
成果公開時の通知・成果情報提出を含む規約上の手続きは利用者が履行する必要がある。
再配布可能とは扱わない。

本研究では、国立情報学研究所のIDRデータセット提供サービスにより株式会社ドワンゴから
提供を受けた「ニコニコ大百科データ」を利用した。

元データはread-onlyで扱い、変更、移動、削除、Git登録、外部送信を行っていない。
公開成果物には記事本文、タイトル、識別子、token列、生成文、tokenizer語彙・model、
checkpoint、parameter、logit、hidden state、Attention、ローカルpathを含めない。
公開対象は件数、byte数、token数、hash、設定、aggregate指標だけである。

## データ形式と前処理

- UTF-8、MySQL CSV形式。comma delimiter、double quote、backslash escape、本文内LFを許容する。
- header/bodyは各291,055 recordで一対一、parse error、重複識別子、対応欠落は0だった。
- bodyはstreamingで処理し、全recordを一括してRAMへ保持しない。
- NFKC、改行のLF統一、LF/TAB以外のC0制御文字除去、HTML entity decodeを行う。
- script/styleを除外し、block境界を改行として保つHTML text抽出を行う。URL、日本語句読点、
  改行、英数字、記号は一律削除しない。
- cleaning後96 byte未満を除外し、1 MiBを上限とする。
- normalized textのSHA-256完全重複をsplit前に除外する。近重複除去は計算予算と誤除外を
  避けるため導入していない。

総record 291,055件のうち、空またはmarkup-only 247件、短すぎる本文2,163件、
完全重複428件を除き、288,217記事を利用可能とした。raw本文は2,390,169,831 byte、
cleaning後は1,204,368,332 byteだった。元コーパスの完走scanはinventoryとprepareの2回で、
以後はhashで保護したprivate cacheを再利用した。

## split、subset、tokenizer

記事identityのstable SHA-256で、TRAIN 90%、validation 5%、development 4%、
final test 1%へ分けた。件数は259,420 / 14,312 / 11,554 / 2,931である。
split決定後にcontext windowへchunk化した。final testの記事も全体cleaning・dedupe・split
集計までは処理したが、token cacheを作らず、tokenizer学習、checkpoint選択、model評価には
使用していない。同一normalized textはsplitをまたがない。

subsetはsplit内のstable hash順で記事単位に選んだ。pilot TRAINは1,995記事、
8,418,135 cleaned UTF-8 byte、8,385,120 target tokenである。validationとdevelopmentは
それぞれ500記事・2,278,752 target token、508記事・2,094,144 target tokenをcacheした。

結果を見る前にTRAINだけでUTF-8 byteと上位Unicode code pointの2候補を比較した。
byte方式はV=256、未知token率0、round-trip可能、平均2.531 token/文字だった。
code point方式はV=2,048、未知token率0.00502で、dense embedding/output headを大きくする。
coverage、再現性、CPU計算量から、性能評価前にUTF-8 byteをprimaryへ固定した。
その代償としてT=32の平均有効contextは約12.6文字と短く、生成途中の不完全なUTF-8を
生じうる。この制約はmodelの結論と分離する。

## 事前登録とモデル

競合仮説は、自然なcontext多様性による安定化、subset不足、tokenizer影響、L19の過深さ、
capacity不足、teacher-forcedは安定してfree-runningだけ分岐、checkpoint選択、評価器、
optimizer/初期化の影響とした。primaryはheld-out token NLL、perplexity、seed間NLL範囲、
次token分布差とし、自然言語でsequence exactをprimaryにしなかった。

共通構成はV=256、T=32、D=16、FFN=32、H=2。浅いL6は20,864 parameter、深いL19は
48,320 parameterで、parameter数は一致していない。Adamはlearning rate 0.003、
beta1 0.9、beta2 0.999、epsilon 1e-8。各runは1,000 step、batch 8、256,000 target token、
100 stepごとにvalidation cacheの固定256 chunk（8,192 token）を評価した。checkpointは
validation NLLだけで選び、developmentは選択後に固定512 chunk（16,384 token）を一度
評価した。全runがstep 1,000を選択し、lossはまだplateauしていない。

smokeは各深さ1 run、5 step、batch 1で実測した。1 stepはL6 0.00355秒、L19 0.00863秒、
peak working setは約32.4 MiBと33.7 MiB、checkpoint書き込みは約1.1 msと1.3 msだった。
これに基づきformal 6 runへ固定し、追加対照は行わなかった。smoke 2、formal 6、追加0、
主要実験cycle 1、CPU評価再生成1である。

## 正式pilot

| depth | seed | validation NLL | development NLL | perplexity | top-1 | top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| L6 | 1 | 2.58496 | 2.63499 | 13.9431 | 0.3250 | 0.5994 |
| L6 | 2 | 2.56455 | 2.63966 | 14.0084 | 0.3196 | 0.6045 |
| L6 | 4 | 2.56901 | 2.61872 | 13.7181 | 0.3271 | 0.5953 |
| L19 | 1 | 2.51080 | 2.59914 | 13.4522 | 0.3291 | 0.6031 |
| L19 | 2 | 2.54835 | 2.61721 | 13.6974 | 0.3300 | 0.6064 |
| L19 | 4 | 2.51663 | 2.58124 | 13.2135 | 0.3265 | 0.6099 |

teacher-forcedの3-seed argmax一致率はL6 0.587、L19 0.607、pairwise JSは
0.0369と0.0333だった。free-runningでは一致率が0.602と0.379、pairwise JSが
0.133と0.189へ拡大した。生成長到達率は両構成1.0だった。byte生成のinvalid UTF-8率は
L6 0.604、L19 0.667だが、短い固定長snippet末尾でmultibyte文字が途中終了する影響を
含むため、自然言語品質指標として解釈しない。free-runningは固定16 prompt×16生成位置
だけの測定であり、信頼区間を伴う母集団推論ではない。生成文自体は公開していない。

paired-prefixでは直近8 byteを保持し、別記事由来の遠方prefixへ置換した。seed別の
argmax flip率はL6 0.141--0.250、L19 0.141--0.219で、identity controlのlogit差は0だった。
遠方context感度の測定器としては動作したが、意味的不変性を保証できないため、
人工データ型の未拘束挙動を再現したとは判定しない。

## Reviewer修正

独立レビューで、private training trajectoryのheaderに`validation_tokens`列名が1つ
不足していることを検出した。6ファイルの既存data rowは変更せずheaderだけを補正し、
writer側も修正した。さらにsource/corpus/cacheのSHA-256、runnerが記録したcacheとtraining
orderのFNV、seed順、formal config、trajectoryからのcheckpoint再選択、development/生成/
paired-prefixのfinite・範囲、final-test cache不在をexport時に相互検証するようにした。
これらは妥当な入力でのmodel数値演算を変更しないguardであり、formal trainingや評価を
追加実行していない。

## 中心的な問いへの回答

1. held-out NLLの大きなseed差は、この自然な日本語pilotでは再現しなかった。ただし
   greedy生成挙動はseedで分岐した。
2. homogeneous TRAINに由来するmixed-prefix未拘束という人工データの因果機構は
   再現しなかった。paired-prefix感度だけでは同じ機構を証明できない。
3. 通常Attentionの全6 runはfiniteで、validation/development NLLも狭い範囲に収まった。
   ただしAttention内部の一意な学習機構まで証明したものではない。
4. 現時点で最も支持されるのはfree-running増幅である。短いbyte context、少ない更新、
   depthとparameter数の非一致、未収束は機構候補として残る。データ不足、capacity、
   optimizer、tokenizerの単独因果は追加対照0のため未同定である。
5. 次の最小構成は、同じdeterministic splitとbyte tokenizerを凍結し、L6/H2/D16/FFN32、
   T=32、seed 1/2/4を維持したままTRAIN tokenまたはstepだけを増やす対照である。
   validation選択と未開封final testを維持し、NLLとfree-running JSを再測定する。

## 範囲と限界

- 目的は高品質な日本語生成ではなく、小規模実テキストでのseed安定性測定である。
- L6とL19はparameter matchedでなく、深さだけの純粋な因果比較ではない。
- 全checkpointが最終stepを選んだため、学習量を増やしたときの結論は未確定である。
- paired-prefixは探索的で、近重複漏洩は完全重複以外を定量化していない。
- evaluationはdeterministic cacheの先頭固定部分を使用し、development全chunkを評価して
  いない。free-runningは16 prompt、paired-prefixは各seed最大128 pairに限られる。
- 実テキストfinal testと既存人工データfinal holdoutは未開封である。
- device、HTP、QNN、QAIRT、ADB、Android/JNI、UI、COUNT_FROM_ONEは使用していない。
- production既定値は変更していない。

集計値とhashは[公開results bundle](results/qnn-nicopedia-real-text-pilot-2026-08/README.md)
に収録した。
