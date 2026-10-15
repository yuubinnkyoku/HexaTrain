#pragma once
// Diagnostic segment ABI v1. Existing Op signatures remain unchanged.
namespace closurecontract {
enum Kind { RawDQDK, FFN2, QKVHeads };
struct Contract { const char* name; unsigned inputs,outputs,inputWords,outputWords,marker; double gate; };
constexpr Contract contracts[]={
 {"RawDQDK",6,4,6144,4096,0x11,.01},
 {"FFN2",2,1,12288,2048,0x12,.002},
 {"QKVHeads",3,6,6144,6144,0x13,.002}
};
constexpr unsigned receiptWords(Kind k){return 8+contracts[k].inputWords+contracts[k].outputWords;}
}
