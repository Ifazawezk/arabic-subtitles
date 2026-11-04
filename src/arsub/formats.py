"""Subtitle file formats: SRT, WebVTT, ASS, plain text and JSON.

Writers take a list of :class:`~arsub.builder.Cue`. Readers for SRT and VTT
exist so an edited file can be loaded back (web editor, ``--subs``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import text as T
from .builder import Cue, Word

FORMATS = ("srt", "vtt", "ass", "txt", "json")


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """``3725.5`` -> ``01:02:05,500`` (SRT) or ``01:02:05.500`` with ``sep='.'``."""
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def to_srt(cues: list[Cue], rtl_marks: str = "embed") -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        lines = "\n".join(T.add_rtl_marks(line, rtl_marks) for line in cue.lines)
        blocks.append(f"{i}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{lines}\n")
    return "\n".join(blocks)


def _vtt_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(cues: list[Cue], rtl_marks: str = "embed") -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        lines = "\n".join(_vtt_escape(T.add_rtl_marks(line, rtl_marks)) for line in cue.lines)
        blocks.append(f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}\n{lines}\n")
    return "\n".join(blocks)


def to_txt(cues: list[Cue]) -> str:
    return "\n".join(" ".join(cue.lines) for cue in cues) + ("\n" if cues else "")


def to_json(cues: list[Cue], words: list[Word] | None = None, meta: dict | None = None) -> str:
    """Everything needed to rebuild cues later without running the model again."""
    data = {
        "meta": meta or {},
        "words": [
            {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text, "p": round(w.probability, 3)}
            for w in (words or [])
        ],
        "cues": [{"start": c.start, "end": c.end, "lines": c.lines} for c in cues],
    }
    return json.dumps(data, ensure_ascii=False, indent=1)


def load_json(path: str | Path) -> tuple[list[Word], list[Cue], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words = [Word(w["start"], w["end"], w["text"], w.get("p", 1.0)) for w in data.get("words", [])]
    cues = [Cue(c["start"], c["end"], list(c["lines"])) for c in data.get("cues", [])]
    return words, cues, data.get("meta", {})


@dataclass
class AssStyle:
    """Look of burned-in or .ass subtitles.

    ``font_size`` and ``margin_v`` are given for a 1080 pixel tall frame and
    scaled to the real video height.
    """

    font: str = "Noto Sans Arabic"
    font_size: int = 76  # Arabic reads smaller than Latin at the same point size
    bold: bool = True
    primary: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H80000000"
    outline: float = 3.0
    shadow: float = 1.0
    margin_v: int = 60
    margin_h: int = 80
    alignment: int = 2  # bottom centre
    # libass picks the paragraph direction from this field: any value other than
    # -1 forces left-to-right, which pushes Arabic punctuation to the wrong end.
    encoding: int = -1


def _ass_escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def to_ass(
    cues: list[Cue],
    style: AssStyle | None = None,
    width: int = 1920,
    height: int = 1080,
    rtl_marks: str = "embed",
    second_track: list[Cue] | None = None,
) -> str:
    """ASS that renders right-to-left text correctly in libass (ffmpeg, mpv).

    Two things make Arabic come out right: ``Encoding: -1`` in the style, which
    tells libass to detect the paragraph direction instead of assuming LTR, and
    RLE ... PDF around each line for renderers that ignore that field.
    ``second_track`` (for example an English translation) is drawn at the top.
    """
    style = style or AssStyle()
    scale = height / 1080
    size = max(18, round(style.font_size * scale))  # stays readable on tiny videos
    margin_v = round(style.margin_v * scale)
    margin_h = round(style.margin_h * scale)
    outline = round(style.outline * scale, 1)
    shadow = round(style.shadow * scale, 1)
    bold = -1 if style.bold else 0
    fmt = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding"
    )

    def style_line(name: str, alignment: int, font_size: int) -> str:
        return (
            f"Style: {name},{style.font},{font_size},{style.primary},&H000000FF,{style.outline_colour},"
            f"{style.back_colour},{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},"
            f"{margin_h},{margin_h},{margin_v},{style.encoding}"
        )

    header = [
        "[Script Info]",
        "; Written by arsub",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        fmt,
        style_line("Default", style.alignment, size),
        style_line("Top", 8, round(size * 0.85)),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for name, track in (("Default", cues), ("Top", second_track or [])):
        for cue in track:
            text = "\\N".join(_ass_escape(T.add_rtl_marks(line, rtl_marks)) for line in cue.lines)
            events.append(f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},{name},,0,0,0,,{text}")
    return "\n".join(header + events) + "\n"


_SRT_TIME = r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
_CUE_TIMING_RE = re.compile(_SRT_TIME + r"\s*-->\s*" + _SRT_TIME)


def _secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(content: str) -> list[Cue]:
    """Read SRT or WebVTT text back into cues. Bidi marks are removed."""
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    content = re.sub(r"(?<![:\d])(\d{2}:\d{2}\.\d{3})", r"00:\1", content)  # short VTT times
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            m = _CUE_TIMING_RE.search(line)
            if m:
                g = m.groups()
                text_lines = [T.strip_bidi_marks(x).strip() for x in lines[i + 1:]]
                text_lines = [re.sub(r"<[^>]+>", "", x) for x in text_lines if x.strip()]
                text_lines = [x.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&") for x in text_lines]
                if text_lines:
                    cues.append(Cue(_secs(*g[:4]), _secs(*g[4:]), text_lines))
                break
    return cues


def write(cues: list[Cue], fmt: str, path: str | Path, **kwargs) -> Path:
    """Write ``cues`` to ``path`` in format ``fmt``. Text files are UTF-8."""
    path = Path(path)
    if fmt == "srt":
        content = to_srt(cues, kwargs.get("rtl_marks", "embed"))
    elif fmt == "vtt":
        content = to_vtt(cues, kwargs.get("rtl_marks", "embed"))
    elif fmt == "ass":
        content = to_ass(
            cues,
            kwargs.get("style"),
            kwargs.get("width", 1920),
            kwargs.get("height", 1080),
            kwargs.get("ass_rtl_marks", "embed"),
            kwargs.get("second_track"),
        )
    elif fmt == "txt":
        content = to_txt(cues)
    elif fmt == "json":
        content = to_json(cues, kwargs.get("words"), kwargs.get("meta"))
    else:
        raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")
    # utf-8-sig for SRT keeps old Windows players from reading Arabic as mojibake.
    encoding = "utf-8-sig" if fmt == "srt" else "utf-8"
    path.write_text(content, encoding=encoding, newline="\n")
    return path
