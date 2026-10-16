"""Inspect the linked DSP symbol; never infer device execution from codegen."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(symbol='hexatrain_ordered_hvx_ffn2', label='FFN2'):
    p = argparse.ArgumentParser()
    p.add_argument('--hexagon-tools-bin', required=True)
    p.add_argument('--package', required=True)
    p.add_argument('--private-root', required=True)
    a = p.parse_args()
    out = Path(a.private_root)
    out.mkdir(parents=True, exist_ok=True)
    tools = Path(a.hexagon_tools_bin)
    disasm = subprocess.check_output([str(tools / 'hexagon-llvm-objdump.exe'), '-d', a.package], text=True)
    elf = subprocess.check_output([str(tools / 'hexagon-llvm-readelf.exe'), '-h', a.package], text=True)
    (out / 'linked-disassembly-private.txt').write_text(disasm, encoding='utf-8')
    (out / 'linked-elf-private.txt').write_text(elf, encoding='utf-8')
    m = re.search(r'^([0-9a-f]+) <' + re.escape(symbol) + r'>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', disasm, re.M | re.S)
    require(m, 'HVX_' + label + '_SYMBOL_MISSING')
    body = m[2]
    require('Hexagon' in elf, 'NOT_HEXAGON_ELF')
    # Use the pinned reader's architecture tag; its encoding is 0x81, not 81 decimal.
    require(re.search(r'Flags:\s+0x81, V81\s*$', elf, re.M), 'NOT_V81_ELF')
    require(re.search(r'v\d+\.qf32 = vmpy\(v\d+\.qf32,v\d+\.qf32\)', body), 'NO_STRICT_VECTOR_MULTIPLY')
    require(re.search(r'v\d+\.qf32 = vadd\(v\d+\.sf,v\d+\.sf\)', body), 'NO_VECTOR_ADD')
    require(re.search(r'v\d+\.sf = v\d+\.qf32', body), 'NO_FP32_ROUNDING_CONVERSION')
    require('vsplat' in body and 'vmemu' in body, 'VECTOR_BROADCAST_OR_MEMORY_MISSING')
    require(not re.search(r'call|v\w*mpyacc|\+=.*vmpy|sfmpy|sfadd', body), 'FUSION_SCALAR_OR_CALL_IN_TARGET')
    require(not re.search(r'allocframe|dealloc|memw\(r29|vmem\(r29', body), 'UNEXPECTED_STACK_SPILL')
    k = 64 if label == 'QKV' else 128
    # Pinned lowering: three prologue/epilogue products plus K-3 loop trips.
    require(re.search(r'loop0\(0x[0-9a-f]+,#0x' + format(k - 3, 'x') + r'\)', body), 'ORDERED_Q_LOOP_TRIP_COUNT')
    require('memw(r7++#0x4)' in body and 'vmemu(r6++#0x2)' in body, 'ORDERED_INPUT_STRIDES')
    (out / 'target-disassembly-private.txt').write_text(m[0], encoding='utf-8')
    result = {
        'schema': 'ORDERED_HVX_' + label + '_CODEGEN_V1',
        'package_sha256': sha(a.package),
        'disassembler_sha256': sha(tools / 'hexagon-llvm-objdump.exe'),
        'target_disassembly_sha256': sha(out / 'target-disassembly-private.txt'),
        'architecture': 'V81', 'vector_bytes': 128, 'fp32_lanes': 32,
        'm': 32, 'k': k, 'n': 64, 'q_loop_trips': k - 3,
        'codegen': 'HVX_' + label + '_CODEGEN_VERIFIED',
        'multiply_sites': len(re.findall(r'= vmpy\(', body)),
        'add_sites': len(re.findall(r'= vadd\(', body)),
        'fp32_conversion_sites': len(re.findall(r'\.sf = v\d+\.qf32', body)),
        'normalization_sites': len(re.findall(r'\.qf32 = v\d+\.sf', body)),
        'broadcast_sites': body.count('vsplat'),
        'unaligned_memory_sites': body.count('vmemu'),
        'fma_sites': 0, 'calls_in_target': 0, 'stack_spill_sites': 0,
        'q_order_numeric_verification': 'UNRESOLVED', 'dsp_execution': 'UNRESOLVED',
    }
    (out / 'codegen-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    return out, disasm


if __name__ == '__main__':
    main()
