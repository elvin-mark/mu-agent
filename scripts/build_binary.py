"""Build standalone binary executable using PyInstaller."""

import os
import sys
import subprocess


def build():
    print("🚀 Building standalone binary for mu-agent...")
    entry_point = os.path.abspath("src/mu_agent/cli.py")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "mu",
        "--collect-all",
        "textual",
        "--collect-all",
        "rich",
        "--collect-all",
        "ddgs",
        "--collect-all",
        "httpx",
        entry_point,
    ]

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n✅ Standalone binary successfully compiled!")
        print("Executable path: ./dist/mu")
    else:
        print("\n❌ Build failed.")
        sys.exit(res.returncode)


if __name__ == "__main__":
    build()
