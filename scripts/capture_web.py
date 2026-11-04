"""Real screenshots and a GIF of the web UI, driven by a headless browser.

Starts arsub-web, uploads a clip, runs the real model, edits one cue and burns
the video. Nothing is mocked, so what ends up in docs/images is what the app
really shows. Needs Playwright with Edge or Chrome (not a project dependency):

    pip install playwright
    python scripts/capture_web.py data/demo/excerpt.mp4 --model large-v3-turbo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
SHOTS = ROOT / "out" / "shots"


def wait_for_server(url: str, timeout: float = 120) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit("web UI did not start")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--port", type=int, default=7880)
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--server-python", default=sys.executable, help="Python that has arsub[web] installed")
    args = ap.parse_args()

    from PIL import Image
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"
    server = subprocess.Popen([args.server_python, "-m", "arsub.web", "--no-browser", "--port", str(args.port)], cwd=ROOT)
    frames: list[Path] = []

    def shot(page, name: str, keep: bool = True) -> None:
        path = SHOTS / f"{len(frames):02d}-{name}.png"
        page.screenshot(path=str(path))
        if keep:
            frames.append(path)

    try:
        wait_for_server(url)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 980})
            page.goto(url, wait_until="load")
            page.get_by_role("button", name="Make subtitles").wait_for(timeout=60000)
            shot(page, "empty")

            page.locator("input[type=file]").first.set_input_files(args.clip)
            page.wait_for_timeout(2500)
            page.get_by_text("Options").first.click()
            page.wait_for_timeout(600)
            model_box = page.get_by_label("Model")
            model_box.click()
            page.get_by_role("option", name=args.model, exact=True).click()
            page.wait_for_timeout(400)
            page.get_by_text("Options").first.click()
            page.wait_for_timeout(500)
            shot(page, "uploaded")

            page.get_by_role("button", name="Make subtitles").click()
            started = time.time()
            done = page.locator("#arsub-table td").filter(has_text="00:00")
            last = 0.0
            while time.time() - started < 900:
                if done.count() > 0:
                    break
                if page.get_by_text("Error", exact=True).count() > 0:
                    shot(page, "error")
                    raise SystemExit("the app showed an error, see out/shots")
                if time.time() - last > 4:
                    shot(page, "progress")
                    last = time.time()
                page.wait_for_timeout(500)
            else:
                raise SystemExit("no subtitles after 15 minutes")
            page.wait_for_timeout(2500)
            shot(page, "subtitles")

            # A real edit: change the first cue in the table, then export it.
            first = page.locator('#arsub-table [role="gridcell"][data-col="3"]').first
            first.dblclick()
            page.wait_for_timeout(500)
            page.keyboard.press("Control+A")
            page.keyboard.type("نص عدلته بيدي في الجدول")
            page.keyboard.press("Enter")
            page.wait_for_timeout(800)
            shot(page, "edited")
            page.get_by_role("button", name="Apply edits and export files").click()
            page.get_by_text(".srt").first.wait_for(timeout=60000)
            page.wait_for_timeout(1500)
            shot(page, "exported")

            page.get_by_role("button", name="Burn into video").click()
            page.wait_for_function("document.querySelectorAll('video').length >= 2", timeout=300000)
            page.wait_for_timeout(3000)
            shot(page, "burned")
            page.screenshot(path=str(IMAGES / "web-editor.png"), full_page=True)
            browser.close()
    finally:
        server.terminate()

    # A short GIF: keep a few progress frames, scale down, hold the last frame.
    progress = [f for f in frames if "progress" in f.name]
    stride = max(1, len(progress) // 5)
    picked = [f for f in frames if "progress" not in f.name]
    picked[2:2] = progress[::stride][:5]
    images = []
    for path in picked:
        image = Image.open(path).convert("RGB")
        image.thumbnail((980, 980))
        images.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    durations = [1300] * len(images)
    durations[-1] = 3500
    images[0].save(IMAGES / "web-demo.gif", save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(IMAGES / "web-demo.gif", (IMAGES / "web-demo.gif").stat().st_size // 1024, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
