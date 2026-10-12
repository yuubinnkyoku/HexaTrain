// SPDX-License-Identifier: Apache-2.0
// Integer-only diagnostic fixture; identical generation on host and DSP.
#pragma once
#include <stdint.h>
#include <string.h>
#define HEXATRAIN_BOUNDARY_COUNT 256
static inline uint32_t hexatrain_boundary_bits(unsigned index) {
  static const uint32_t fixed[] = {
    0x00000000,0x80000000,0x3f800001,0xbf800001,0x3e7ef9db,0x3e808312,
    0x447ffff0,0x44800008,0x437fe000,0x43800000,0x44000001,0x45800001,
    0x00800000,0x00000001,0x007fffff,0x33800001,0x3f123456,0x7f7fffff
  };
  if(index < sizeof(fixed)/sizeof(fixed[0])) return fixed[index];
  uint32_t state=97;
  for(unsigned i=0;i<=index;++i) state=1664525u*state+1013904223u;
  // Replace all-ones exponents with a finite exponent. All other raw bits stay.
  if((state&0x7f800000u)==0x7f800000u) state^=0x00800000u;
  return state;
}
static inline void hexatrain_boundary_fill(float* out,unsigned count) {
  for(unsigned i=0;i<count;++i){uint32_t bits=hexatrain_boundary_bits(i);memcpy(out+i,&bits,4);}
}
