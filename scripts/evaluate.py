"""Measure accuracy and speed on the test set from fetch_testset.py.

Accuracy: every clip is transcribed on its own, then WER and CER are computed
over the whole source (total errors / total reference words), not averaged per
clip. Text is normalized first, because a subtitle and a reference that differ
only in hamza, teh marbuta, diacritics or punctuation are not a real error.
Both the raw and the normalized score are stored.

Speed: all clips of the run are joined into one long file with short silences,
so the number reflects a real video (VAD, chunking, word timestamps) rather
than per-call overhead. "x realtime" is audio seconds per wall second, and it
excludes model load time, which is reported separately.

    python scripts/evaluate.py --model small --device cuda
    python scripts/evaluate.py --model small --device cpu --limit 20
    python scripts/evaluate.py --report
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from arsub import __version__
from arsub.dialects import DIALECTS, prompt_for
from arsub.text import convert_digits, strip_bidi_marks, strip_diacritics
from arsub.transcribe import TranscribeOptions, load_model, register_cuda_libraries, resolve_compute_type

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "testset" / "manifest.jsonl"
RESULTS = ROOT / "eval" / "results.json"
SAMPLE_RATE = 16000

SOURCE_LABELS = {
    "fleurs": "FLEURS (read MSA)",
    "cv18": "Common Voice 18 (crowd, read)",
    "masc": "MASC (YouTube, many dialects)",
}


def normalize(text: str) -> str:
    """Standard Arabic ASR scoring normalization."""
    text = convert_digits(strip_diacritics(strip_bidi_marks(text)), "western")
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        text = text.replace(a, b)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def light(text: str) -> str:
    """Raw scoring: only punctuation and spacing removed, letters kept as written."""
    text = re.sub(r"[^\w\sؐ-ًؚ-ٰٟ]|_", " ", strip_bidi_marks(text))
    return " ".join(text.split())


def score(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    keep = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    rr, hh = [r for r, _ in keep], [h if h.strip() else " " for _, h in keep]
    out = jiwer.process_words(rr, hh)
    chars = jiwer.process_characters(rr, hh)
    return {
        "wer": round(out.wer * 100, 2),
        "cer": round(chars.cer * 100, 2),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "reference_words": out.hits + out.substitutions + out.deletions,
    }


def score_items(items: list[dict], hyps: list[str]) -> tuple[dict, dict]:
    """Scores per source, per MASC recording type, and over everything."""
    groups: dict[str, list[int]] = {}
    for n, item in enumerate(items):
        groups.setdefault(item["source"], []).append(n)
        if item.get("type"):
            kind = "clean" if item["type"] == "c" else "noisy"
            groups.setdefault(f"{item['source']}-{kind}", []).append(n)
    by_source = {}
    for name, idx in groups.items():
        refs = [items[n]["text"] for n in idx]
        hh = [hyps[n] for n in idx]
        by_source[name] = {
            "clips": len(idx),
            "audio_seconds": round(sum(items[n]["duration"] for n in idx), 1),
            "normalized": score([normalize(r) for r in refs], [normalize(h) for h in hh]),
            "raw": score([light(r) for r in refs], [light(h) for h in hh]),
        }
    refs_all = [i["text"] for i in items]
    overall = {
        "normalized": score([normalize(r) for r in refs_all], [normalize(h) for h in hyps]),
        "raw": score([light(r) for r in refs_all], [light(h) for h in hyps]),
    }
    return by_source, overall


def rescore() -> None:
    """Score stored hypotheses again, for example after changing the normalization."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    manifest = {i["id"]: i for i in load_manifest(None, None)}
    for run_entry in data["runs"]:
        stored = run_entry.get("hypotheses") or []
        if not stored:
            continue
        items = [manifest[h["id"]] for h in stored]
        run_entry["by_source"], run_entry["overall"] = score_items(items, [h["text"] for h in stored])
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_manifest(limit: int | None, sources: list[str] | None) -> list[dict]:
    items = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sources:
        items = [i for i in items if i["source"] in sources]
    if limit:
        picked: list[dict] = []
        for source in dict.fromkeys(i["source"] for i in items):
            picked += [i for i in items if i["source"] == source][:limit]
        items = picked
    return items


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def transcribe_all(model, audio: np.ndarray, opts: TranscribeOptions) -> tuple[str, int]:
    kwargs = dict(
        language="ar",
        beam_size=opts.beam_size,
        word_timestamps=True,
        vad_filter=opts.vad,
        vad_parameters={"min_silence_duration_ms": opts.min_silence_ms} if opts.vad else None,
        initial_prompt=opts.initial_prompt,
    )
    if opts.batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        segments, _ = BatchedInferencePipeline(model=model).transcribe(audio, batch_size=opts.batch_size, **kwargs)
    else:
        segments, _ = model.transcribe(audio, condition_on_previous_text=False, **kwargs)
    parts: list[str] = []
    words = 0
    for seg in segments:
        parts.append(seg.text.strip())
        words += len(seg.words or [])
    return " ".join(parts), words


def run(args: argparse.Namespace) -> dict:
    from faster_whisper.audio import decode_audio

    register_cuda_libraries()
    items = load_manifest(args.limit, args.sources.split(",") if args.sources else None)
    compute = resolve_compute_type(args.compute_type, args.device)
    print(f"{args.model} on {args.device} ({compute}), {len(items)} clips", flush=True)

    t0 = time.perf_counter()
    model = load_model(args.model, args.device, compute)
    load_seconds = time.perf_counter() - t0

    clips = [decode_audio(str(ROOT / i["file"]), sampling_rate=SAMPLE_RATE) for i in items]
    total_audio = sum(len(c) for c in clips) / SAMPLE_RATE
    prompt = args.prompt or prompt_for(args.dialect)
    opts = TranscribeOptions(beam_size=args.beam_size, vad=not args.no_vad, initial_prompt=prompt, batch_size=args.batch_size)

    # The first call on a GPU pays for kernel setup, so warm up before timing.
    warm = time.perf_counter()
    transcribe_all(model, clips[0], opts)
    warmup_seconds = time.perf_counter() - warm

    gap = np.zeros(int(SAMPLE_RATE * 0.5), dtype=clips[0].dtype)
    joined = np.concatenate([x for c in clips for x in (c, gap)])
    t1 = time.perf_counter()
    _, long_words = transcribe_all(model, joined, opts)
    long_seconds = time.perf_counter() - t1
    long_audio = len(joined) / SAMPLE_RATE
    print(f"  long file: {long_audio:.0f}s of audio in {long_seconds:.1f}s = {long_audio / long_seconds:.1f}x", flush=True)

    hyps: list[str] = []
    t2 = time.perf_counter()
    for n, clip in enumerate(clips, 1):
        text, _ = transcribe_all(model, clip, opts)
        hyps.append(text)
        if n % 30 == 0:
            print(f"  {n}/{len(clips)} clips", flush=True)
    clip_seconds = time.perf_counter() - t2

    by_source, overall = score_items(items, hyps)
    for name, entry in by_source.items():
        print(f"  {name}: WER {entry['normalized']['wer']}%  CER {entry['normalized']['cer']}%", flush=True)
    print(f"  all: WER {overall['normalized']['wer']}%  CER {overall['normalized']['cer']}%", flush=True)
    return {
        "model": Path(args.model).name,  # a local model folder is named after the model
        "device": args.device,
        "compute_type": compute,
        "beam_size": args.beam_size,
        "vad": not args.no_vad,
        "prompt": bool(prompt),
        "dialect": args.dialect,
        "batch_size": args.batch_size,
        "min_silence_ms": opts.min_silence_ms,
        "label": args.label or "",
        "clips": len(items),
        "audio_seconds": round(total_audio, 1),
        "model_load_seconds": round(load_seconds, 2),
        "warmup_seconds": round(warmup_seconds, 2),
        "long_file_audio_seconds": round(long_audio, 1),
        "long_file_seconds": round(long_seconds, 2),
        "long_file_x_realtime": round(long_audio / long_seconds, 2),
        "long_file_words": long_words,
        "per_clip_seconds": round(clip_seconds, 2),
        "per_clip_x_realtime": round(total_audio / clip_seconds, 2),
        "by_source": by_source,
        "overall": overall,
        "hypotheses": [{"id": i["id"], "text": h} for i, h in zip(items, hyps)] if args.keep_text else None,
    }


def save(entry: dict) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"runs": []}
    data["machine"] = {
        "gpu": gpu_name(),
        "cpu": cpu_name(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "arsub": __version__,
    }
    key = lambda r: (r["model"], r["device"], r["compute_type"], r.get("label", ""), r["clips"])  # noqa: E731
    data["runs"] = [r for r in data["runs"] if key(r) != key(entry)] + [entry]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def report() -> str:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = [
        "| Model | Device | Clips | WER | CER | FLEURS WER | Common Voice WER | MASC WER | Speed | Load |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = {"small": 0, "medium": 1, "large-v3-turbo": 2, "large-v3": 3}
    for r in sorted(data["runs"], key=lambda r: (r["device"] != "cuda", order.get(r["model"], 9), r.get("label", ""))):
        src = r["by_source"]
        cell = lambda s: f"{src[s]['normalized']['wer']:.1f}%" if s in src else "-"  # noqa: E731
        name = r["model"] + (f" ({r['label']})" if r.get("label") else "")
        rows.append(
            f"| {name} | {r['device']} {r['compute_type']} | {r['clips']} | {r['overall']['normalized']['wer']:.1f}% "
            f"| {r['overall']['normalized']['cer']:.1f}% | {cell('fleurs')} | {cell('cv18')} | {cell('masc')} "
            f"| {r['long_file_x_realtime']:.1f}x | {r['model_load_seconds']:.1f}s |"
        )
    videos = data.get("videos") or []
    if videos:
        rows += ["", "| Video | Model | Mode | Length | Wall time | Speed | Words | Cues | Too fast |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in videos:
            mode = f"{v['device']}, batch {v['batch_size']}" if v["batch_size"] else f"{v['device']}, sequential"
            rows.append(
                f"| {v['name']} | {v['model']} | {mode} | {v['audio_seconds'] / 60:.1f} min | {v['wall_seconds']:.0f}s "
                f"| {v['x_realtime']:.1f}x | {v['words']} | {v['cues']} | {v['too_fast']} |"
            )
    return "\n".join(rows)


def run_video(args: argparse.Namespace) -> dict:
    """Real file, real settings. Wall time includes loading the model."""
    from arsub.builder import build_cues, stats
    from arsub.transcribe import transcribe

    opts = TranscribeOptions(model=args.model, device=args.device, batch_size=args.batch_size)
    result = transcribe(args.video, opts)
    cues = build_cues(result.words)
    st = stats(cues)
    entry = {
        "name": args.label or Path(args.video).stem,
        "model": Path(result.model).name,
        "device": result.device,
        "batch_size": args.batch_size,
        "audio_seconds": round(result.duration, 1),
        "wall_seconds": round(result.elapsed, 1),
        "x_realtime": round(result.duration / result.elapsed, 1),
        "words": len(result.words),
        "cues": st["cues"],
        "too_fast": st["too_fast"],
    }
    print(entry, flush=True)
    return entry


def save_video(entry: dict) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"runs": []}
    key = lambda v: (v["name"], v["model"], v["device"], v["batch_size"])  # noqa: E731
    data["videos"] = [v for v in data.get("videos", []) if key(v) != key(entry)] + [entry]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--limit", type=int, help="clips per source (for slow CPU runs)")
    ap.add_argument("--sources", help="comma separated subset of fleurs,cv18,masc")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--dialect", choices=DIALECTS, default="none")
    ap.add_argument("--prompt", help="initial prompt, for testing other priming text")
    ap.add_argument("--label", help="tag for this run in the results table")
    ap.add_argument("--video", help="time one real file instead of the test set")
    ap.add_argument("--keep-text", action="store_true", help="store hypotheses in results.json")
    ap.add_argument("--report", action="store_true", help="print the markdown table and exit")
    ap.add_argument("--rescore", action="store_true", help="score stored hypotheses again, without any model")
    args = ap.parse_args()
    if args.rescore:
        rescore()
    if args.report or args.rescore:
        print(report())
        return 0
    if args.video:
        save_video(run_video(args))
        return 0
    if not MANIFEST.exists():
        print("no test set yet, run: python scripts/fetch_testset.py", file=sys.stderr)
        return 1
    save(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
