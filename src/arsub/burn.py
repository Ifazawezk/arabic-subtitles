"""Burn subtitles into a video with ffmpeg and libass.

Arabic only renders correctly when three things line up:

* the ASS style says ``Encoding: -1`` so libass detects right-to-left
  paragraphs (with any other value it lays the line out left-to-right and the
  final "؟" or "." ends up on the wrong side),
* every line is wrapped in RLE ... PDF for renderers that ignore that field,
* the font actually has Arabic glyphs. A Noto Sans Arabic copy ships with the
  package and is handed to libass through ``fontsdir``.

ffmpeg is run from a temporary folder holding the .ass file and the fonts, so
filter arguments are plain relative names. That sidesteps ffmpeg's escaping
rules for Windows drive letters and non-ASCII folder names.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

log = logging.getLogger("arsub")

FONTS_DIR = Path(__file__).parent / "fonts"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".weba"}


class FFmpegError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    width: int = 0
    height: int = 0
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary.

    Order: ``ARSUB_FFMPEG`` environment variable, ``ffmpeg`` on PATH, then the
    static build that ships with the ``imageio-ffmpeg`` pip package.
    """
    env = os.environ.get("ARSUB_FFMPEG")
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=8)
def ffmpeg_filters(ffmpeg: str) -> frozenset[str]:
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, errors="replace")
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) <= 4 and parts[0].strip(".TSC|") == "":
            names.add(parts[1])
    return frozenset(names)


@lru_cache(maxsize=8)
def ffmpeg_encoders(ffmpeg: str) -> frozenset[str]:
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, errors="replace")
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])
    return frozenset(names)


def subtitle_filter(ffmpeg: str) -> str:
    """``ass`` if this ffmpeg build has libass, otherwise an error."""
    filters = ffmpeg_filters(ffmpeg)
    if "ass" in filters:
        return "ass"
    if "subtitles" in filters:
        return "subtitles"
    raise FFmpegError(
        f"{ffmpeg} was built without libass, so it cannot draw subtitles. "
        "Install a full build (for example from https://www.gyan.dev/ffmpeg/builds/ or "
        "https://github.com/BtbN/FFmpeg-Builds) and point ARSUB_FFMPEG at it."
    )


def probe(path: str | Path) -> MediaInfo:
    """Size, duration and stream types of a media file (via PyAV)."""
    import av

    info = MediaInfo()
    with av.open(str(path)) as container:
        if container.duration:
            info.duration = container.duration / 1_000_000
        videos = [s for s in container.streams.video if not _is_cover_art(s)]
        if videos:
            v = videos[0]
            info.has_video = True
            info.width, info.height = v.codec_context.width, v.codec_context.height
            if not info.duration and v.duration and v.time_base:
                info.duration = float(v.duration * v.time_base)
        info.has_audio = bool(container.streams.audio)
    return info


def _is_cover_art(stream) -> bool:
    disposition = getattr(stream, "disposition", None)
    try:
        return bool(disposition and "attached_pic" in str(disposition).lower())
    except Exception:  # noqa: BLE001
        return False


def is_audio_only(path: str | Path) -> bool:
    """Look inside the file first: an .ogg can hold video. The extension is the fallback."""
    try:
        return not probe(path).has_video
    except Exception:  # noqa: BLE001
        return Path(path).suffix.lower() in AUDIO_EXTS


def build_command(
    ffmpeg: str,
    input_path: str,
    output_path: str,
    sub_filter: str = "ass",
    audio_only: bool = False,
    size: tuple[int, int] = (1280, 720),
    video_codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
    audio_codec: str = "copy",
    background: str = "0x14181f",
) -> list[str]:
    """The ffmpeg command line. Expects to run with the subtitle folder as cwd."""
    subs = f"{sub_filter}=subs.ass:fontsdir=fonts"
    cmd = [ffmpeg, "-hide_banner", "-y", "-nostats", "-progress", "pipe:1"]
    if audio_only:
        w, h = size
        cmd += ["-f", "lavfi", "-i", f"color=c={background}:s={w}x{h}:r=25", "-i", input_path]
        wave_h = h // 4
        graph = (
            f"[1:a]aformat=channel_layouts=mono,showwaves=s={w}x{wave_h}:mode=cline:rate=25:scale=sqrt:colors=0x4fb3bf[w];"
            f"[0:v][w]overlay=0:{(h - wave_h) // 2 - h // 10}:shortest=1,{subs}[v]"
        )
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "1:a", "-shortest"]
        audio_codec = "aac" if audio_codec == "copy" else audio_codec
    else:
        cmd += ["-i", input_path, "-vf", subs, "-map", "0:v:0", "-map", "0:a?"]
    cmd += ["-c:v", video_codec]
    if video_codec == "libx264":
        cmd += ["-crf", str(crf), "-preset", preset]
    elif video_codec.endswith("_nvenc"):
        # NVENC's cq scale is not x264's crf: on a real film cq 26 matched the size of crf 20,
        # while cq 20 came out more than twice as large.
        cmd += ["-rc", "vbr", "-cq", str(crf + 6), "-b:v", "0", "-preset", "p5"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", audio_codec]
    if audio_codec == "aac":
        cmd += ["-b:a", "160k"]
    if str(output_path).lower().endswith((".mp4", ".m4v", ".mov")):
        cmd += ["-movflags", "+faststart"]
    cmd.append(output_path)
    return cmd


def _run_ffmpeg(cmd: list[str], cwd: str, duration: float, progress: Callable[[float, str], None] | None) -> None:
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    tail: list[str] = []

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip())
            del tail[:-30]

    import threading

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        if progress and duration > 0 and line.startswith("out_time_us="):
            try:
                done = int(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                continue
            progress(min(max(done / duration, 0.0), 1.0), "burning")
    proc.wait()
    t.join(timeout=5)
    if proc.returncode != 0:
        raise FFmpegError("ffmpeg failed:\n" + "\n".join(tail[-15:]))


def burn(
    input_path: str | Path,
    ass_text: str,
    output_path: str | Path,
    ffmpeg: str | None = None,
    video_codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
    size: tuple[int, int] = (1280, 720),
    progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Render ``ass_text`` onto ``input_path`` and write ``output_path``.

    Audio-only input becomes a video: a plain background with a waveform and
    the subtitles, handy for podcasts and voice notes.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise FFmpegError("ffmpeg not found. Run `pip install imageio-ffmpeg` or install ffmpeg and add it to PATH.")
    sub_filter = subtitle_filter(ffmpeg)
    if video_codec not in ffmpeg_encoders(ffmpeg):
        raise FFmpegError(f"this ffmpeg build has no {video_codec} encoder")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    audio_only = is_audio_only(input_path)
    try:
        duration = probe(input_path).duration
    except Exception:  # noqa: BLE001
        duration = 0.0

    with tempfile.TemporaryDirectory(prefix="arsub-") as tmp:
        Path(tmp, "subs.ass").write_text(ass_text, encoding="utf-8")
        shutil.copytree(FONTS_DIR, Path(tmp, "fonts"))
        common = dict(
            sub_filter=sub_filter,
            audio_only=audio_only,
            size=size,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )
        cmd = build_command(ffmpeg, str(input_path), str(output_path), **common)
        log.debug("running %s", " ".join(cmd))
        try:
            _run_ffmpeg(cmd, tmp, duration, progress)
        except FFmpegError:
            if audio_only:
                raise
            # Some audio codecs cannot be copied into the output container.
            cmd = build_command(ffmpeg, str(input_path), str(output_path), audio_codec="aac", **common)
            _run_ffmpeg(cmd, tmp, duration, progress)
    if progress:
        progress(1.0, "done")
    return output_path
