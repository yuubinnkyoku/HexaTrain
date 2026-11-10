# CB-FIX-1: Candidate B CPU access contract

Status: `CB-FIX-1_CPU_CONTRACT_CLOSED` (CPU model only).

The former check compared a planned last-MMA ordinal with a later Store
ordinal. It missed the sentinel fill, which is the first output write to
the reused Wlo and Ahi regions. The Candidate B CPU model now records pack,
MMA operand reads, bias writes, sentinel fills, HMX stores and HVX stores with
sequence, offset, size, direction and object identity. The independent trace
validator obtains each operand's last read from these events and checks every
subsequent overlapping write, including partial overlaps. It also rejects
out-of-range events, missing expected accesses, trace overflow and output
region overlap. Candidate B writes use the instrumented pack/fill wrappers or
the instrumented execution functions. State initialization writes are also
recorded; they occur before the candidate's operand lifetimes begin.

The fixed controls execute early writes to Wlo, Ahi and Alo and a one-byte
boundary overlap. They clear the legacy fault indicators before checking the
trace. Additional controls cover missing events, overflow, bounds and output
overlap; nominal Plain and B3 traces permit the post-read Wlo/Ahi reuse. The
existing 1,024-cell comparisons and registered digests remain unchanged.

GCC 15.2 at `-O0` and `-O2`, and Clang 22.1.8 at `-O2`, passed.
The working tree (including pre-existing C2 audit changes) reported 12,798
checks and zero failures, including the existing 27/27 injected faults; the
independently tested staged source reported 12,784 checks and zero failures.
The difference is the pre-existing, unstaged C2 priority tests. Fast completed
with 1/1 gates passing and Host with 2/2 gates passing. Logs are local build
artifacts and are not device evidence.

This describes CPU program order only. The CPU model does not establish when
an asynchronous HMX operand read completes on a device. D8 remains
`DEVICE_ONLY`, `B3_R2_SYNC_UNPROVEN` remains in force, and no device execution
or training integration is authorized by this audit.

Later audits extended the shared fault battery (27 at this commit, 30 after
2026-11-07) and added the all-six-array parity assertion for Candidate B; see
[the C2 CPU audit](hmx-c2-cpu-audit-2026-11-07.md). Neither changes the
`CB-FIX-1_CPU_CONTRACT_CLOSED` scope: it is a statement about the CPU model's
access trace, not about device read completion.
