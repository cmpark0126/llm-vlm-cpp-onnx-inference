#!/usr/bin/env python3

def export_static_onnx():
    """
    Export static ONNX graphs for prefill and decode phases
    - Prefill: fixed 128 input tokens
    - Decode: single token generation
    - Max sequence length: 1024
    """
    print("TODO: Export static ONNX graphs")
    pass

if __name__ == "__main__":
    export_static_onnx()
