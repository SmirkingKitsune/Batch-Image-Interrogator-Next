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
    # CRITICAL: Detect devices BEFORE Qt initialization
    # This allows PyTorch to claim CUDA before Qt touches GPU resources
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

    args = sys.argv[1:]
    if "--qt" not in args:
        from bridge import start_bridge
        sys.exit(start_bridge(device_status, args))

    print("[WARN] The Qt viewer is deprecated. Electron is the default.", file=sys.stderr)
    # PyTorch has already claimed CUDA by the time the legacy UI imports Qt.
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("[X] Qt is not installed. Run setup.bat --qt or ./setup.sh --qt.", file=sys.stderr)
        sys.exit(1)
    from ui import MainWindow

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication([sys.argv[0], *(arg for arg in args if arg != "--qt")])
    app.setApplicationName("Image Interrogator")
    app.setOrganizationName("ImageInterrogator")

    # Create main window (pass device status for config defaults)
    window = MainWindow(device_status=device_status)
    window.show()

    # Defer model list population until event loop starts
    # This ensures window is fully rendered before heavy operations
    QTimer.singleShot(100, window.populate_model_lists)

    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
