"""Render docs/images/rtl-comparison.png with the real ffmpeg and libass.

Left column: a plain `ffmpeg -vf subtitles=file.srt` burn of an SRT with no
bidi marks. Right column: the same lines through arsub's ASS. Nothing is drawn
by hand, so the image can be regenerated and checked by anyone.

    python scripts/make_rtl_comparison.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from arsub.builder import Cue
from arsub.burn import FONTS_DIR, find_ffmpeg
from arsub.formats import AssStyle, to_ass, to_srt

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "rtl-comparison.png"
WIDTH, HEIGHT = 900, 720
STRIP = 96

LINES = [
    "في عام 2026 زاد العدد بنسبة 15%.",
    "استخدمنا Python و FastAPI في المشروع.",
    "هل تعرف الفرق بين SQL و NoSQL?",
]

LABELS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1800
PlayResY: 48
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: L,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,L,,0,900,0,,plain ffmpeg -vf subtitles=file.srt
Dialogue: 0,0:00:00.00,0:00:05.00,L,,900,0,0,,arsub --burn
"""


def run(ffmpeg: str, cwd: Path, *args: str) -> None:
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd, check=True)


def main() -> int:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    background = f"color=c=0x20252c:s={WIDTH}x{HEIGHT}:d=1"
    crop = f"crop={WIDTH}:{STRIP}:0:{HEIGHT - STRIP - 6}"
    with tempfile.TemporaryDirectory(prefix="arsub-cmp-") as tmp:
        work = Path(tmp)
        shutil.copytree(FONTS_DIR, work / "fonts")
        rows: list[str] = []
        for i, line in enumerate(LINES):
            cues = [Cue(0.0, 5.0, [line])]
            (work / f"n{i}.srt").write_text(to_srt(cues, rtl_marks="none"), encoding="utf-8")
            (work / f"a{i}.ass").write_text(to_ass(cues, AssStyle(font_size=54), WIDTH, HEIGHT), encoding="utf-8")
            force = "FontName=Noto Sans Arabic,FontSize=14,Bold=1,Outline=1.5,Shadow=0.5"
            run(ffmpeg, work, "-f", "lavfi", "-i", background,
                "-vf", f"subtitles=n{i}.srt:fontsdir=fonts:force_style='{force}',{crop}", "-frames:v", "1", f"n{i}.png")
            run(ffmpeg, work, "-f", "lavfi", "-i", background,
                "-vf", f"ass=a{i}.ass:fontsdir=fonts,{crop}", "-frames:v", "1", f"a{i}.png")
            run(ffmpeg, work, "-i", f"n{i}.png", "-i", f"a{i}.png", "-filter_complex",
                "[0:v]pad=iw+4:ih:0:0:color=0x4fb3bf[l];[l][1:v]hstack", f"row{i}.png")
            rows.append(f"row{i}.png")
        (work / "labels.ass").write_text(LABELS, encoding="utf-8")
        run(ffmpeg, work, "-f", "lavfi", "-i", f"color=c=0x101418:s={WIDTH * 2 + 4}x48:d=1",
            "-vf", "ass=labels.ass", "-frames:v", "1", "head.png")
        inputs: list[str] = []
        for name in ["head.png", *rows]:
            inputs += ["-i", name]
        run(ffmpeg, work, *inputs, "-filter_complex", f"vstack=inputs={len(rows) + 1}", "sheet.png")
        shutil.copy(work / "sheet.png", OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
