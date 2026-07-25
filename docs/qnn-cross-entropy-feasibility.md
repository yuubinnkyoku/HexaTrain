# QNN cross entropy feasibility (QAIRT 2.48)

## Evidence base

This inventory uses the installed QAIRT 2.48.40 headers (`QnnOpDef.h`, QNN API 2.37.0) and finalize/execute results on the target HTP V81 device. Paths, device endpoints, callbacks, and binaries are intentionally omitted.

| Operation | Header classification | Device evidence | Initial LM decision |
|---|---|---|---|
| LogSoftmax | direct op | not finalized in this milestone | not on required path |
| Softmax | direct op | finalize/execute success | used on HTP |
| ElementWiseLog | direct op | not finalized in this milestone | CPU stable logsumexp scalar |
| ReduceMax | direct op | header evidence; not tested here | not on required path |
| ReduceSum | direct op | existing HTP backward tests succeed | supported primitive |
| ElementWiseSubtract | direct op | CE microtest succeeds | used on HTP |
| ElementWiseMultiply | direct op | CE microtest succeeds | used on HTP |
| ElementWiseDivide | direct op | header evidence; not needed | composition possible, untested |
| Gather | direct op | not finalized here | CPU one-hot boundary preferred |
| OneHot | direct op | not finalized here | CPU one-hot boundary preferred |
| Argmax | direct op | not finalized here | CPU inference boundary allowed |
| TopK | direct op | not finalized here | outside this milestone |

## Selected implementation

The verified path is method C:

- HTP: Softmax and `(P - Y) / N`, embedding MatMul, Transformer forward/backward, output projection backward, and SGD.
- CPU: token-rule generation, input/target one-hot construction, graph control, stable logsumexp cross-entropy scalar, and independent reference calculations.

This is explicit backward composition. QNN automatic differentiation is not used.
