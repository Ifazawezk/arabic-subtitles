"""The whole job in one function, shared by the CLI and the web UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import formats as F
from .builder import BuildOptions, Cue, Word, build_cues, stats
from .burn import burn as burn_video
from .burn import probe
from .transcribe import TranscribeOptions, Transcript, transcribe

log = logging.getLogger("arsub")

ProgressFn = Callable[[float, str], None]


@dataclass
class JobResult:
    cues: list[Cue]
    words: list[Word] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)
    burned: Path | None = None
    stats: dict = field(default_factory=dict)


def _stage(progress: ProgressFn | None, lo: float, hi: float, label: str) -> ProgressFn | None:
    if progress is None:
        return None

    def inner(frac: float, msg: str) -> None:
        progress(lo + (hi - lo) * frac, f"{label}: {msg}")

    return inner


def output_paths(input_path: Path, out_dir: Path) -> dict[str, Path]:
    stem = input_path.stem
    for suffix in (".ar", ".en"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return {
        "ar": out_dir / f"{stem}.ar",
        "burned": out_dir / f"{stem}.subbed.mp4",
    }


def video_size(path: Path) -> tuple[int, int]:
    try:
        info = probe(path)
        if info.has_video and info.width and info.height:
            return info.width, info.height
    except Exception:  # noqa: BLE001
        pass
    return 1280, 720


def run(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    formats: tuple[str, ...] = ("srt",),
    transcribe_opts: TranscribeOptions | None = None,
    build_opts: BuildOptions | None = None,
    burn: bool = False,
    subs_path: str | Path | None = None,
    style: F.AssStyle | None = None,
    rtl_marks: str = "embed",
    video_codec: str = "libx264",
    crf: int = 20,
    progress: ProgressFn | None = None,
) -> JobResult:
    input_path = Path(input_path)
    out_dir = Path(out_dir) if out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    topts = transcribe_opts or TranscribeOptions()
    bopts = build_opts or BuildOptions()
    paths = output_paths(input_path, out_dir)
    media = input_path
    result = JobResult(cues=[])

    asr_share = 0.7 if burn else 1.0
    if subs_path is not None:
        subs_path = Path(subs_path)
        if subs_path.suffix.lower() not in (".srt", ".vtt", ".json"):
            raise ValueError(f"--subs takes .srt, .vtt or .json, not {subs_path.suffix or subs_path.name}")
        if subs_path.suffix.lower() == ".json":
            words, cues, meta = F.load_json(subs_path)
            result.cues = build_cues(words, bopts) if words else cues
            result.words, result.meta = words, meta
        else:
            result.cues = F.parse_srt(subs_path.read_text(encoding="utf-8-sig"))
    elif input_path.suffix.lower() == ".json":
        words, cues, meta = F.load_json(input_path)
        result.words, result.meta = words, meta
        result.cues = build_cues(words, bopts) if words else cues
        source = meta.get("source")
        media = Path(source) if source else input_path
    else:
        tr: Transcript = transcribe(input_path, topts, _stage(progress, 0.0, asr_share, "transcribing"))
        result.words = tr.words
        result.meta = {**tr.meta(), "source": str(input_path.resolve())}
        result.cues = build_cues(tr.words, bopts)
    if not result.cues:
        log.warning("no speech found, the subtitles are empty")
    result.stats = stats(result.cues, bopts)
    width, height = 1920, 1080
    if media.suffix.lower() != ".json" and media.exists():
        width, height = video_size(media)  # 1280x720 for audio-only input

    for fmt in formats:
        target = paths["ar"].with_name(paths["ar"].name + f".{fmt}")
        F.write(
            result.cues,
            fmt,
            target,
            rtl_marks=rtl_marks,
            style=style,
            width=width,
            height=height,
            words=result.words,
            meta=result.meta,
        )
        result.files[f"ar.{fmt}"] = target

    if burn:
        if media.suffix.lower() == ".json" or not media.exists():
            raise FileNotFoundError("--burn needs the original video; pass it as input together with --subs")
        ass_text = F.to_ass(result.cues, style, width, height)
        result.burned = burn_video(
            media,
            ass_text,
            paths["burned"],
            video_codec=video_codec,
            crf=crf,
            size=(width, height),
            progress=_stage(progress, asr_share, 1.0, "burning"),
        )
        result.files["burned"] = result.burned
    if progress:
        progress(1.0, "done")
    return result
