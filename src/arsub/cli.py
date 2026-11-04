"""Command line interface: ``arsub video.mp4 --format srt --burn``."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

from . import __version__
from . import formats as F
from .builder import BuildOptions
from .dialects import DIALECTS
from .text import DIGIT_STYLES, RTL_MARK_MODES
from .transcribe import MODELS, TranscribeOptions

log = logging.getLogger("arsub")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arsub",
        description="Arabic subtitles for any video or audio file, offline on your own GPU or CPU.",
        epilog="examples:\n"
        "  arsub talk.mp4                      write talk.ar.srt\n"
        "  arsub talk.mp4 -f srt,vtt,ass --burn\n"
        "  arsub talk.ar.json --max-chars 36   rebuild subtitles without running the model\n"
        "  arsub talk.mp4 --subs edited.srt --burn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="video or audio file, or a .json file written by an earlier run")
    p.add_argument("-f", "--format", default="srt", help="comma separated list of srt,vtt,ass,txt,json or 'all' (default: srt)")
    p.add_argument("-o", "--output-dir", help="where to write files (default: next to the input)")
    p.add_argument("--burn", action="store_true", help="also write a copy of the video with the subtitles burned in")
    p.add_argument("--subs", help="use this .srt/.vtt/.json instead of transcribing (for burning an edited file)")
    p.add_argument("--doctor", action="store_true", help="check GPU, CUDA libraries and ffmpeg, then exit")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"arsub {__version__}")

    g = p.add_argument_group("speech recognition")
    g.add_argument("-m", "--model", default="auto", help=f"auto, one of {', '.join(MODELS)}, or a local CTranslate2 model folder")
    g.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    g.add_argument("--compute-type", default="auto", help="auto = float16 on GPU, int8 on CPU")
    g.add_argument("--language", default="ar", help="spoken language code, or 'auto' (default: ar)")
    g.add_argument("--dialect", choices=DIALECTS, default="none", help="nudge spelling toward a dialect with a primed prompt")
    g.add_argument("--prompt", help="custom initial prompt (names, terms, style)")
    g.add_argument("--hotwords", help="words to favour, e.g. names that are often misheard")
    g.add_argument("--beam-size", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=0, help="batched GPU inference, e.g. 8 (faster on long files)")
    g.add_argument("--no-vad", action="store_true", help="do not skip silence with voice activity detection")

    s = p.add_argument_group("subtitle layout")
    s.add_argument("--max-chars", type=int, default=42, help="characters per line, diacritics not counted (default: 42)")
    s.add_argument("--max-lines", type=int, default=2)
    s.add_argument("--max-cps", type=float, default=20.0, help="reading speed limit in characters per second (default: 20)")
    s.add_argument("--min-duration", type=float, default=1.0)
    s.add_argument("--max-duration", type=float, default=7.0)
    s.add_argument("--digits", choices=DIGIT_STYLES, default="western", help="western 0-9, arabic-indic, or keep")
    s.add_argument("--strip-diacritics", action="store_true", help="remove harakat")
    s.add_argument("--rtl-marks", choices=RTL_MARK_MODES, default="embed", help="invisible marks in .srt/.vtt for players that guess LTR")

    b = p.add_argument_group("burn-in")
    b.add_argument("--font", default="Noto Sans Arabic", help="font family (Noto Sans Arabic ships with arsub)")
    b.add_argument("--font-size", type=int, default=76, help="size for a 1080p frame, scaled to the video")
    b.add_argument("--video-codec", default="libx264", help="libx264 (default) or h264_nvenc for GPU encoding")
    b.add_argument("--crf", type=int, default=20, help="quality, lower is better (default: 20)")
    return p


def parse_formats(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return F.FORMATS
    out = []
    for part in value.split(","):
        part = part.strip().lower().lstrip(".")
        if not part:
            continue
        if part not in F.FORMATS:
            raise SystemExit(f"arsub: unknown format {part!r} (choose from {', '.join(F.FORMATS)})")
        if part not in out:
            out.append(part)
    return tuple(out) or ("srt",)


class ProgressBar:
    """A one-line progress bar on stderr."""

    def __init__(self, enabled: bool = True, width: int = 28) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.width = width
        self.start = time.perf_counter()
        self.last = ""

    def __call__(self, frac: float, msg: str) -> None:
        if not self.enabled:
            return
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{bar}] {frac * 100:5.1f}%  {msg[:48]:<48}"
        if line != self.last:
            sys.stderr.write(line)
            sys.stderr.flush()
            self.last = line

    def close(self) -> None:
        if self.enabled and self.last:
            sys.stderr.write("\n")


def doctor() -> int:
    """Print what arsub can see: Python, GPU, CUDA libraries, ffmpeg."""
    from .burn import find_ffmpeg, ffmpeg_encoders, ffmpeg_filters
    from .transcribe import cuda_device_count, register_cuda_libraries

    print(f"arsub {__version__} on Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    try:
        import ctranslate2
        import faster_whisper

        print(f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}")
    except ImportError as exc:
        print(f"faster-whisper missing: {exc}")
    dirs = register_cuda_libraries()
    if dirs:
        print("CUDA libraries from pip:")
        for d in dirs:
            print(f"  {d}")
    n = cuda_device_count()
    print(f"CUDA devices: {n}" + ("" if n else "  (will run on CPU)"))
    ff = find_ffmpeg()
    if not ff:
        print("ffmpeg: not found (pip install imageio-ffmpeg, or put ffmpeg on PATH)")
        return 1
    filters, encoders = ffmpeg_filters(ff), ffmpeg_encoders(ff)
    print(f"ffmpeg: {ff}")
    print(f"  libass (needed for --burn): {'yes' if 'ass' in filters else 'NO'}")
    print(f"  libx264: {'yes' if 'libx264' in encoders else 'no'}   h264_nvenc: {'yes' if 'h264_nvenc' in encoders else 'no'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="arsub: %(message)s")
    if args.doctor:
        return doctor()
    if not args.input:
        build_parser().print_usage(sys.stderr)
        print("arsub: error: an input file is required", file=sys.stderr)
        return 2
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"arsub: error: no such file: {input_path}", file=sys.stderr)
        return 2
    from .pipeline import run

    topts = TranscribeOptions(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        dialect=args.dialect,
        initial_prompt=args.prompt,
        hotwords=args.hotwords,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        vad=not args.no_vad,
    )
    bopts = BuildOptions(
        max_chars=args.max_chars,
        max_lines=args.max_lines,
        max_cps=args.max_cps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        digits=args.digits,
        remove_diacritics=args.strip_diacritics,
    )
    style = F.AssStyle(font=args.font, font_size=args.font_size)
    bar = ProgressBar(enabled=not args.quiet)
    t0 = time.perf_counter()
    try:
        result = run(
            input_path,
            out_dir=args.output_dir,
            formats=parse_formats(args.format),
            transcribe_opts=topts,
            build_opts=bopts,
            burn=args.burn,
            subs_path=args.subs,
            style=style,
            rtl_marks=args.rtl_marks,
            video_codec=args.video_codec,
            crf=args.crf,
            progress=bar,
        )
    except KeyboardInterrupt:
        bar.close()
        print("arsub: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        bar.close()
        if args.verbose:
            raise
        print(f"arsub: error: {exc}", file=sys.stderr)
        return 1
    bar.close()

    elapsed = time.perf_counter() - t0
    meta = result.meta
    if not args.quiet:
        if meta.get("device"):
            dur = meta.get("duration", 0.0)
            speed = f", {dur / meta['elapsed']:.1f}x realtime" if meta.get("elapsed") else ""
            print(f"{meta.get('model')} on {meta.get('device')} ({meta.get('compute_type')}): {dur:.0f}s of audio{speed}")
        st = result.stats
        print(f"{st.get('cues', 0)} cues, {st.get('too_fast', 0)} above {args.max_cps:g} chars/s, total {elapsed:.1f}s")
    for path in result.files.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
