# QNN HTP Momentum SGD for the tiny language model

Momentum uses `velocity_next = momentum * velocity + gradient` and `weight_next = weight - learning_rate * velocity_next`, with zero initial velocity and separate current/next buffers.

The HTP implementation uses a second flattened optimizer graph after the existing training/backward graph. HTP executes all multiply/add/subtract optimizer arithmetic. CPU flattens and copies current weights, gradients, and velocity into fixed APP_WRITE buffers, controls the two graph executions, and unflattens the APP_READ results. There is no optimizer arithmetic fallback. The layout is the deterministic concatenation of the twelve public parameter groups and contains 3,136 FP32 elements (12,544 bytes of velocity state).

One-step device correctness passes with gradient maximum error 0.00240893, next-velocity error 0.00240893, and next-weight error 0.00011149. Two graphs are created/finalized and two executes are issued per update. Current and next buffers do not alias; callback capture is disabled, log level is WARN, API trace records successful HTP graph calls, and no QNN CPU backend fallback occurs.

The complete 24-configuration CPU search selected `lr=0.01`, momentum 0.95 at 1000 and 640 steps. Both HTP finalists became non-finite after repeated updates, despite their CPU references satisfying the additional condition. Momentum therefore did not establish a final optimizer and Adam evaluation was required.
