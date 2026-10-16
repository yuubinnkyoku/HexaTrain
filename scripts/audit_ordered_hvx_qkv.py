"""Reuse core ELF checks and audit the fixed QKV invocation wrappers."""
import json
import re
from audit_ordered_hvx_ffn2 import main, require, sha


def wrappers(out, disasm):
    allowed = {'qhpi_tensor_shape', 'qhpi_tensor_raw_data', 'qhpi_tensor_type', 'memcpy',
               '__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes',
               '_ZN8lncustom5countEPK11QHPI_Tensor', 'hexatrain_ordered_hvx_qkv_nn64'}
    records = {}
    for name, body in re.findall(r'^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', disasm, re.M | re.S):
        if name.endswith('@plt'):
            continue
        for prefix, kind in (('_ZN6hvxqkv6kernelILb', 'hvx'), ('_ZN13segmentcustom6kernelILb1ELb', 'scalar')):
            m = re.match(re.escape(prefix) + r'([01])E', name)
            if not m:
                continue
            observed = m[1] == '1'; label = kind + ('_observed' if observed else '_no_receipt')
            calls = [c.removesuffix('@plt') for c in re.findall(r'\bcall\b[^\n]*<([^>]+)>', body)]
            require(set(calls) <= allowed and 'callr' not in body, 'UNEXPECTED_QKV_WRAPPER_CALL')
            frames = re.findall(r'allocframe\(#0x([0-9a-f]+)\)', body)
            require(len(frames) == 1, 'QKV_WRAPPER_FRAME_UNAVAILABLE')
            frame = int(frames[0], 16)
            require(frame == ((152 if kind == 'hvx' else 120) if observed else 96), 'QKV_WRAPPER_FRAME_CHANGED')
            vector_stack = len(re.findall(r'vmemu?\([^\n]*r29', body))
            require(vector_stack == 0, 'QKV_WRAPPER_VECTOR_SPILL')
            helper_calls = calls.count('hexatrain_ordered_hvx_qkv_nn64')
            if kind == 'hvx':
                require(helper_calls == 3, 'QKV_HELPER_CALL_STRUCTURE')
                require(not re.search(r'sfmpy|sfadd|sfmac', body), 'SCALAR_ARITHMETIC_IN_HVX_WRAPPER')
            else:
                require(helper_calls == 0, 'SCALAR_BASELINE_REPLACED')
            records[label] = dict(allocframe_operand_bytes=frame, vector_stack_accesses=vector_stack,
                                  arithmetic_helper_static_call_sites=helper_calls,
                                  unexpected_calls=0)
    require(len(records) == 4, 'QKV_WRAPPER_COVERAGE')
    audit = json.loads((out / 'codegen-audit.json').read_text())
    result = dict(schema='ORDERED_HVX_QKV_WRAPPERS_V1', package_sha256=audit['package_sha256'], wrappers=records,
                  linked_disassembly_sha256=sha(out / 'linked-disassembly-private.txt'),
                  note='allocframe operands exclude callee and QuRT total stack; both HVX wrappers contain three arithmetic helper call sites')
    (out / 'wrapper-codegen-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('qkv_wrapper_audit=PASS wrappers=4 vector_spills=0 unexpected_calls=0')

if __name__ == '__main__':
    wrappers(*main('hexatrain_ordered_hvx_qkv_nn64', 'QKV'))
