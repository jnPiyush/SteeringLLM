"""
SteeringLLM Demo Launcher

This script preloads PyTorch **before** Streamlit (and its pyarrow dependency)
to avoid the Windows DLL conflict:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading "...\\torch\\lib\\c10.dll"

Root cause: pyarrow ships DLLs that conflict with torch's c10.dll on Windows.
If pyarrow loads first, torch cannot initialise.  Loading torch first avoids
the problem entirely.

Usage:
    python demo/launch.py                     # default: port 8501
    python demo/launch.py --port 8502         # custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SteeringLLM demo")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()

    # ---- Step 1: Preload torch BEFORE streamlit/pyarrow ----
    print("Pre-loading PyTorch (to avoid pyarrow DLL conflict) ...")
    try:
        import torch  # noqa: F401

        print(f"  torch {torch.__version__} loaded OK  (device: "
              f"{'CUDA' if torch.cuda.is_available() else 'CPU'})")
    except Exception as exc:
        print(f"  WARNING: torch failed to load: {exc}")
        print("  The demo UI will still launch but model loading will fail.")

    # ---- Step 2: Launch Streamlit ----
    # Resolve demo/app.py relative to this file
    app_path = str(Path(__file__).resolve().parent / "app.py")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless", "true",
        "--server.port", str(args.port),
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]

    from streamlit.web import cli as stcli  # noqa: E402

    stcli.main()


if __name__ == "__main__":
    main()
