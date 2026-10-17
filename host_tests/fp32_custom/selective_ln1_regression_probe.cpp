// Reuse the existing regression graphs verbatim, with both packages loaded
// before context creation. No old graph/kernel/receipt/ABI is changed.
#define main selective_standalone_unused_main
#include "selective_ln1_probe.cpp"
#undef main
#define initializeProbe initializeSelective
#include "one_layer_probe.cpp"
#undef initializeProbe
