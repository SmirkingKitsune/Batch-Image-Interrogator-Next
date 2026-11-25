#!/usr/bin/env python3
"""
Image Interrogator - Batch Image Tagging Tool

An Electron application for batch image interrogation and tagging using
CLIP and Waifu Diffusion models with SQLite caching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main application entry point."""
    args = sys.argv[1:]
    if "--qt" in args:
        print("The Qt viewer has been removed. Launch without --qt to open Electron.", file=sys.stderr)
        sys.exit(2)
    print("Detecting compute devices...")
    from core.device_detector import detect_devices_early
    device_status = detect_devices_early()

    # Log detection results to console (use ASCII-safe characters for Windows)
    if device_status['pytorch_cuda_available']:
        print(f"[OK] PyTorch CUDA available - GPU acceleration enabled")
    else:
        print(f"[WARN] PyTorch CUDA not available - using CPU")
        if device_status['pytorch_error']:
            print(f"  Reason: {device_status['pytorch_error']}")

    if device_status['onnx_cuda_available']:
        print(f"[OK] ONNX Runtime CUDA available - GPU acceleration enabled")
    else:
        print(f"[WARN] ONNX Runtime CUDA not available - using CPU")
        if device_status['onnx_error']:
            print(f"  Reason: {device_status['onnx_error']}")

    print()  # Blank line for readability

    from bridge import start_bridge
    sys.exit(start_bridge(device_status, args))


if __name__ == '__main__':
    main()
