"""
Convenience launcher. Runs training then starts the Streamlit app.

Usage:
    python run.py                          # train on sample CSV, then launch app
    python run.py --skip-train             # skip training, just launch app
    python run.py --csv path/to/data.csv   # train on custom CSV, then launch
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_training(csv_path=None, data_dir=None):
    cmd = [sys.executable, str(ROOT / "model" / "train_model.py")]
    if csv_path:
        cmd += ["--csv", csv_path]
    if data_dir:
        cmd += ["--data", data_dir]
    print(f"\n{'='*60}")
    print("Starting model training...")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[WARNING] Training exited with errors. Launching app anyway.")


def launch_app():
    print(f"\n{'='*60}")
    print("Launching Streamlit application...")
    print("Open your browser at: http://localhost:8501")
    print(f"{'='*60}\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "app.py"),
        "--server.headless=false",
        "--theme.base=light",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--csv", type=str, default=str(ROOT / "resume" / "resume.csv"))
    parser.add_argument("--data", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()

    if not args.skip_train:
        run_training(csv_path=args.csv, data_dir=args.data)

    launch_app()
