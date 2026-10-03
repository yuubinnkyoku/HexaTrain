### pre-registered fresh seeds
| seed | step | d Val bpb | d Dev bpb | d Val NLL | d Dev NLL | d Val top-1 | d Dev top-1 | sup. heads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 500 | -0.034767 | -0.019663 | -0.062735 | -0.032464 | +72 | +77 | 13 |
| 2 1000 | -0.040609 | -0.016583 | -0.073278 | -0.027379 | -3 | -64 | 4 |
| 2 1500 | -0.040620 | -0.033357 | -0.073296 | -0.055073 | +9 | -8 | 6 |
| 2 1750 | -0.021960 | -0.034319 | -0.039626 | -0.056662 | -11 | +25 | 8 |
| 2 2000 | -0.039665 | -0.040111 | -0.071574 | -0.066225 | -17 | -8 | 5 |
| 2 2500 | -0.056329 | -0.032325 | -0.101642 | -0.053370 | +91 | -27 | 6 |
| 2 3000 | -0.017353 | -0.021594 | -0.031313 | -0.035652 | -35 | +0 | 5 |
| 4 500 | -0.025406 | -0.043629 | -0.045844 | -0.072034 | +74 | +109 | 13 |
| 4 1000 | -0.038116 | -0.033878 | -0.068779 | -0.055933 | +10 | +71 | 13 |
| 4 1500 | -0.026154 | -0.027331 | -0.047194 | -0.045124 | -11 | +12 | 5 |
| 4 1750 | -0.031777 | -0.026903 | -0.057340 | -0.044418 | +74 | -71 | 10 |
| 4 2000 | -0.027774 | -0.039366 | -0.050116 | -0.064995 | +10 | -9 | 8 |
| 4 2500 | -0.007641 | +0.001603 | -0.013788 | +0.002647 | +13 | -29 | 5 |
| 4 3000 | -0.013354 | -0.046454 | -0.024097 | -0.076698 | -108 | +76 | 11 |

pre-registered coverage: seeds [2, 4] of [2, 4]

| seed | role | criterion | verdict | value |
| ---: | :--- | :--- | :--- | :--- |
| 2 | preregistered | R1 | pass | 2/2 |
| 2 | preregistered | R2 | not_reproduced | 0/5 |
| 2 | preregistered | R3 | none | none |
| 4 | preregistered | R1 | pass | 2/2 |
| 4 | preregistered | R2 | reproduced | 1/5 |
| 4 | preregistered | R3 | located | 2500 |
| 4 | preregistered | R4 | mixed | val+13 dev-29 top-1 tokens |
| 4 | preregistered | R5 | no_peak | 5 <= 11 |

decision: ambiguous_tie_breaker (pre-registered seeds only)
R2 reproduced in 1/2 pre-registered fresh seeds (seeds [4]): ambiguous.  The documented tie-breaker is one seed 3 run via -AllowExploratorySeed; the analyzer reports it as exploratory and keeps it outside this decision, so the tie is broken by an explicit human call, not by an automatic merge.

rows=14 gate=14 verdicts=8 problems=0
NOTE: seed_role_source=registry
NOTE: preregistered_seeds=[2, 4]
NOTE: preregistered_coverage=[2, 4]
