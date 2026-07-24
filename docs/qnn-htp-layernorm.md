# QNN HTP LayerNorm forward

PhoneLM uses the direct QAIRT `LayerNorm` op with FP32 input, gamma, beta, and output. Shape is B=2, T=3, D=8; normalization axis is D and epsilon is 1e-5. Gamma is one and beta is zero.

Device result: graph finalize and execute succeeded, CPU fallback was false, maximum absolute error was 0.000660, mean absolute error was 0.000191, maximum relative error (denominator floor 0.001) was 0.001940, and no NaN/Inf occurred. The acceptance limit is max absolute error < 0.002, chosen to cover the HTP FP32-interface/FP16-execution behavior observed in the direct op while remaining well below one percent of the normalized output scale.

The CPU side only creates deterministic input and calculates an independent double-accumulator reference. The HTP graph performs mean/variance normalization and affine application. QNN API results are recorded by the PhoneLM API trace; callback text is neither needed for the test nor published.