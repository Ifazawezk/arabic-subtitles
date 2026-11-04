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
from .text import is_rtl
from .transcribe import TRANSLATE_MODEL, TranscribeOptions, Transcript, transcribe

log = logging.getLogger("arsub")

ProgressFn = Callable[[float, str], None]


@dataclass
class JobResult:
    cues: list[Cue]
    words: list[Word] = field(default_factory=list)
    en_cues: list[Cue] = field(default_factory=list)
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


def english_options(build: BuildOptions) -> BuildOptions:
    """English lines can be a bit longer and read a bit faster."""
    return BuildOptions(
        max_chars=max(build.max_chars, 42),
        max_lines=build.max_lines,
        max_cps=max(build.max_cps, 20.0),
        min_duration=build.min_duration,
        max_duration=build.max_duration,
        min_gap=build.min_gap,
        pause=build.pause,
        digits="western",
    )


def translation_model(asr_model: str, requested: str | None = None) -> str:
    """The model for the English track: the one asked for, or one that can translate."""
    if requested:
        return requested
    return TRANSLATE_MODEL if "turbo" in asr_model else asr_model


def looks_untranslated(cues: list[Cue]) -> bool:
    """True when most of a translation is still Arabic."""
    lines = [line for cue in cues for line in cue.lines]
    return bool(lines) and sum(is_rtl(line) for line in lines) > len(lines) / 2


def output_paths(input_path: Path, out_dir: Path) -> dict[str, Path]:
    stem = input_path.stem
    for suffix in (".ar", ".en"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return {
        "ar": out_dir / f"{stem}.ar",
        "en": out_dir / f"{stem}.en",
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
    translate: bool = False,
    translate_model: str | None = None,
    burn: bool = False,
    burn_track: str = "ar",
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
        split = asr_share / 2 if translate else asr_share
        tr: Transcript = transcribe(input_path, topts, _stage(progress, 0.0, split, "transcribing"))
        result.words = tr.words
        result.meta = {**tr.meta(), "source": str(input_path.resolve())}
        result.cues = build_cues(tr.words, bopts)
        if translate:
            en_opts = TranscribeOptions(
                **{
                    **topts.__dict__,
                    "task": "translate",
                    "model": translation_model(tr.model, translate_model),
                    "dialect": None,
                    "initial_prompt": None,
                }
            )
            en = transcribe(input_path, en_opts, _stage(progress, split, asr_share, "translating"))
            result.en_cues = build_cues(en.words, english_options(bopts))
            result.meta["translation_model"] = en.model
            if looks_untranslated(result.en_cues):
                log.warning("the %s model returned Arabic instead of English, try --translate-model medium", en.model)

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
        if result.en_cues and fmt in ("srt", "vtt", "ass", "txt"):
            en_target = paths["en"].with_name(paths["en"].name + f".{fmt}")
            F.write(result.en_cues, fmt, en_target, rtl_marks="none", style=style, width=width, height=height)
            result.files[f"en.{fmt}"] = en_target

    if burn:
        if media.suffix.lower() == ".json" or not media.exists():
            raise FileNotFoundError("--burn needs the original video; pass it as input together with --subs")
        main_track, top = result.cues, None
        if burn_track == "en":
            main_track = result.en_cues
        elif burn_track == "both":
            top = result.en_cues
        ass_text = F.to_ass(main_track, style, width, height, second_track=top)
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
