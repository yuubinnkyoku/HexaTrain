# QNN HTP Softmax forward

PhoneLM uses the direct QAIRT `Softmax` op on the last axis with beta=1. The test covers normal values, values near +1000, and values near -1000 across 3x4 rows.

Device result: all three executions succeeded without fallback or NaN/Inf. Maximum CPU/HTP absolute error was 0.001908 and maximum row-sum error was 0.001908. Both acceptance limits are 0.002. The CPU reference subtracts each row maximum before exponentiation.

This is the same direct Softmax configuration used inside the causal attention and tiny Transformer graphs.