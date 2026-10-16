"""Audit the separately lowered K64 symbol using the existing ELF checks."""
from audit_ordered_hvx_ffn2 import main

if __name__ == '__main__':
    main('hexatrain_ordered_hvx_qkv_nn64', 'QKV')
