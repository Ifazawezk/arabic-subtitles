"""Render docs/banner.html to docs/images/banner.png with headless Edge or Chrome.

    python scripts/make_banner.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "banner.html"
OUT = ROOT / "docs" / "images" / "banner.png"

CANDIDATES = [
    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    "microsoft-edge",
    "google-chrome",
    "chromium",
]


def find_browser() -> str | None:
    for name in CANDIDATES:
        found = name if Path(name).is_file() else shutil.which(name)
        if found:
            return found
    return None


def main() -> int:
    browser = find_browser()
    if not browser:
        print("no Edge or Chrome found", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arsub-banner-") as profile:
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=1280,420",
                "--virtual-time-budget=3000",
                f"--user-data-dir={profile}",
                f"--screenshot={OUT}",
                PAGE.as_uri(),
            ],
            check=True,
            timeout=90,
        )
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
