# QNN Transformer op feasibility (QAIRT 2.48)

This inventory is based on the locally installed QAIRT 2.48.40 headers (`QnnOpDef.h`) and the HTP OpDef supplement, not on assumed framework coverage. A header symbol proves that a QNN op exists; HTP execution is claimed only where PhoneLM finalized and executed a graph on the device.

| Transformer need | QAIRT 2.48 evidence | Classification | PhoneLM status |
|---|---|---|---|
| LayerNorm | `QNN_OP_LAYER_NORM`; epsilon and axes parameters | direct op | HTP finalized/executed, FP32 input/gamma/beta/output |
| RMSNorm | `QNN_OP_RMS_NORM` | direct op | available, not device-tested |
| Softmax | `QNN_OP_SOFTMAX`; axis and beta | direct op | HTP finalized/executed on last axis |
| Exp | `QNN_OP_ELEMENT_WISE_EXP` | direct op / primitive composition | available, not device-tested here |
| ReduceMax | `QNN_OP_REDUCE_MAX` | direct op / primitive composition | available, not device-tested here |
| ReduceMean | `QNN_OP_REDUCE_MEAN` | direct op / primitive composition | already used by existing training micrographs |
| ReduceSum | `QNN_OP_REDUCE_SUM` | direct op / primitive composition | already used by existing training micrographs |
| Rsqrt / Sqrt | `QNN_OP_ELEMENT_WISE_RSQRT`, `QNN_OP_ELEMENT_WISE_SQUARE_ROOT` | direct op / primitive composition | available, not needed because direct LayerNorm worked |
| Divide | `QNN_OP_ELEMENT_WISE_DIVIDE` | direct op / primitive composition | available, not device-tested here |
| Add / Subtract / Multiply | element-wise op definitions | direct op | HTP device-tested in training and Transformer graphs |
| MatMul | `QNN_OP_MAT_MUL`, transpose parameters | direct op | HTP device-tested, including QK transpose |
| Transpose | `QNN_OP_TRANSPOSE` with permutation parameter | direct op | available, not needed for the single-head rank-2 graph |
| Reshape | `QNN_OP_RESHAPE` | direct op | available, not needed in the rank-2 graph |
| Permute | no separate `QNN_OP_PERMUTE` definition | use Transpose | unsupported as a distinct op; no graph boundary required when Transpose suffices |
| Gather / embedding | `QNN_OP_GATHER` | direct op | available, not device-tested; CPU embedding remains an allowed future boundary |
| Cast | `QNN_OP_CAST` | direct op | available, not device-tested here |
| Select / Where | `QNN_OP_ELEMENT_WISE_SELECT` | direct op | HTP device-tested by existing backward graph |
| Causal mask | Add with a finite large-negative static mask | primitive composition | HTP device-tested; future probability was zero in the microtest |
| Cross entropy | LogSoftmax/Reduce primitives exist, but end-to-end signature was not tested | unknown | CPU required until a dedicated HTP correctness test exists |

The HTP supplement documents LayerNorm FP32 I/O and restricts normalization axes to the final dimension (or the last three dimensions of a 4-D input). PhoneLM uses the supported final dimension. Optional gamma and beta are supplied explicitly because the HTP finalizer rejected the one-input form (`1002`) while the documented three-input FP32 form finalized successfully.

## Boundaries and responsibility

The required forward milestones use no CPU compute inside the tested graphs. CPU code generates deterministic inputs and independently computes references. Attention and the tiny block each use one finalized HTP graph and one `graphExecute`. Backward, embedding, loss, and optimizer integration are outside this milestone and remain unverified.