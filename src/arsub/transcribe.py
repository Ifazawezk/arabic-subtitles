"""Speech to timed words with faster-whisper.

GPU first, CPU when there is no usable GPU. On Windows the CUDA libraries
can come from pip (``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12``); they are
added to the DLL search path before CTranslate2 needs them. If the GPU run
fails for any CUDA reason (missing DLL, out of memory, unsupported card) the
same file is transcribed again on the CPU with a smaller model.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .builder import Word
from .dialects import prompt_for

log = logging.getLogger("arsub")

MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo")
DEFAULT_GPU_MODEL = "large-v3-turbo"
DEFAULT_CPU_MODEL = "small"
# large-v3-turbo ignores the translate task and returns the source language,
# which is easy to miss, so translation always uses a model that really translates.
TRANSLATE_MODEL = "medium"

ProgressFn = Callable[[float, str], None]


@dataclass
class TranscribeOptions:
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "ar"
    task: str = "transcribe"
    beam_size: int = 5
    vad: bool = True
    # Lower values cut words from narration that sits over music: 500 lost words
    # on a real documentary that 2000 keeps, and it also ran slower.
    min_silence_ms: int = 2000
    dialect: str | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None
    condition_on_previous_text: bool = False
    batch_size: int = 0
    cpu_fallback_model: str = DEFAULT_CPU_MODEL
    download_root: str | None = None


@dataclass
class Transcript:
    words: list[Word]
    segments: list[dict] = field(default_factory=list)
    language: str = "ar"
    duration: float = 0.0
    device: str = "cpu"
    model: str = ""
    compute_type: str = ""
    elapsed: float = 0.0

    @property
    def speed(self) -> float:
        """How many seconds of audio were processed per second of wall time."""
        return self.duration / self.elapsed if self.elapsed > 0 else 0.0

    def meta(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "device": self.device,
            "model": self.model,
            "compute_type": self.compute_type,
            "elapsed": round(self.elapsed, 3),
        }


_cuda_paths_done = False


def register_cuda_libraries() -> list[str]:
    """Put pip-installed NVIDIA DLL folders on the search path (Windows only).

    ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` drops the DLLs into
    ``site-packages/nvidia/*/bin``, which Windows does not search on its own.
    Returns the folders that were added.
    """
    global _cuda_paths_done
    if _cuda_paths_done or sys.platform != "win32":
        return []
    _cuda_paths_done = True
    added: list[str] = []
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            path = str(bin_dir)
            if path in added:
                continue
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            added.append(path)
    return added


def cuda_device_count() -> int:
    register_cuda_libraries()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001 - any failure here means "no GPU"
        return 0


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if cuda_device_count() > 0 else "cpu"
    return device


def resolve_model(model: str, device: str) -> str:
    if model == "auto":
        return DEFAULT_GPU_MODEL if device == "cuda" else DEFAULT_CPU_MODEL
    return model


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    return compute_type


_MODEL_CACHE: dict[tuple, object] = {}


def _not_cached_errors() -> tuple[type[BaseException], ...]:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError

        return (LocalEntryNotFoundError,)
    except ImportError:  # very old huggingface_hub
        return (FileNotFoundError,)


def load_model(name: str, device: str, compute_type: str, download_root: str | None = None):
    """Load (and cache) a ``faster_whisper.WhisperModel``."""
    key = (name, device, compute_type, download_root)
    if key not in _MODEL_CACHE:
        register_cuda_libraries()
        from faster_whisper import WhisperModel

        log.info("loading model %s on %s (%s)", name, device, compute_type)
        common = dict(device=device, compute_type=compute_type, download_root=download_root)
        try:
            # Local copy first: once a model is cached, no request leaves the machine.
            _MODEL_CACHE[key] = WhisperModel(name, local_files_only=True, **common)
        except _not_cached_errors():
            log.info("model %s is not on this machine yet, downloading it once", name)
            _MODEL_CACHE[key] = WhisperModel(name, **common)
    return _MODEL_CACHE[key]


def _is_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    hints = ("cuda", "cublas", "cudnn", "gpu", "out of memory", "no kernel image", "dll")
    return isinstance(exc, (RuntimeError, OSError, ValueError)) and any(h in msg for h in hints)


def _run(audio, opts: TranscribeOptions, device: str, model_name: str, progress: ProgressFn | None) -> Transcript:
    compute_type = resolve_compute_type(opts.compute_type, device)
    t0 = time.perf_counter()
    model = load_model(model_name, device, compute_type, opts.download_root)
    prompt = opts.initial_prompt or prompt_for(opts.dialect)
    kwargs = dict(
        language=None if opts.language in (None, "auto") else opts.language,
        task=opts.task,
        beam_size=opts.beam_size,
        word_timestamps=True,
        vad_filter=opts.vad,
        vad_parameters={"min_silence_duration_ms": opts.min_silence_ms} if opts.vad else None,
        initial_prompt=prompt,
        hotwords=opts.hotwords,
        condition_on_previous_text=opts.condition_on_previous_text,
    )
    if opts.batch_size and opts.batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        kwargs.pop("condition_on_previous_text")
        segments, info = BatchedInferencePipeline(model=model).transcribe(audio, batch_size=opts.batch_size, **kwargs)
    else:
        segments, info = model.transcribe(audio, **kwargs)

    words: list[Word] = []
    segs: list[dict] = []
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    for seg in segments:
        segs.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            }
        )
        for w in seg.words or []:
            words.append(Word(start=w.start, end=w.end, text=w.word, probability=w.probability))
        if progress and duration > 0:
            progress(min(seg.end / duration, 1.0), f"{seg.end:.0f}s / {duration:.0f}s")
    if progress:
        progress(1.0, "done")
    return Transcript(
        words=words,
        segments=segs,
        language=getattr(info, "language", opts.language or "ar"),
        duration=duration,
        device=device,
        model=model_name,
        compute_type=compute_type,
        elapsed=time.perf_counter() - t0,
    )


def transcribe(audio, opts: TranscribeOptions | None = None, progress: ProgressFn | None = None) -> Transcript:
    """Transcribe (or translate) ``audio``: a path to any audio/video file or a numpy array.

    Tries the GPU when available. Any CUDA failure falls back to the CPU and,
    when the model was chosen automatically or is bigger than
    ``opts.cpu_fallback_model``, to that smaller model.
    """
    opts = opts or TranscribeOptions()
    if isinstance(audio, Path):
        audio = str(audio)
    device = resolve_device(opts.device)
    model_name = resolve_model(opts.model, device)
    if device != "cuda":
        return _run(audio, opts, "cpu", model_name, progress)
    try:
        return _run(audio, opts, "cuda", model_name, progress)
    except Exception as exc:  # noqa: BLE001
        if opts.device == "cuda" or not _is_cuda_error(exc):
            raise
        fallback = _smaller(model_name, opts.cpu_fallback_model)
        log.warning("GPU run failed (%s). Falling back to CPU with model %s.", str(exc).splitlines()[0], fallback)
        _MODEL_CACHE.pop((model_name, "cuda", resolve_compute_type(opts.compute_type, "cuda"), opts.download_root), None)
        cpu_opts = TranscribeOptions(**{**opts.__dict__, "compute_type": "auto"})
        return _run(audio, cpu_opts, "cpu", fallback, progress)


def _smaller(model: str, fallback: str) -> str:
    """The smaller of two model names (unknown names count as big)."""
    order = {name: i for i, name in enumerate(MODELS)}
    return model if order.get(model, len(MODELS)) <= order.get(fallback, len(MODELS)) else fallback
