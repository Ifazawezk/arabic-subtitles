"""Transcriber logic with a fake WhisperModel, so no model is downloaded."""

import sys
import types
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from arsub import transcribe as tr
from arsub.dialects import DIALECTS, prompt_for


class FakeModel:
    instances: list = []
    fail_on_cuda = False
    not_cached: set = set()

    def __init__(self, name, device, compute_type, download_root=None, local_files_only=False):
        if local_files_only and name in FakeModel.not_cached:
            raise LocalEntryNotFoundError("not in cache")
        self.name, self.device, self.compute_type = name, device, compute_type
        self.local_files_only = local_files_only
        self.calls = []
        FakeModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if self.device == "cuda" and FakeModel.fail_on_cuda:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        words = [
            SimpleNamespace(start=0.0, end=0.4, word=" مرحبا", probability=0.9),
            SimpleNamespace(start=0.5, end=0.9, word=" بكم", probability=0.8),
        ]
        segs = iter([SimpleNamespace(start=0.0, end=0.9, text=" مرحبا بكم", words=words, avg_logprob=-0.2, no_speech_prob=0.01)])
        return segs, SimpleNamespace(duration=2.0, language="ar")


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    FakeModel.instances = []
    FakeModel.fail_on_cuda = False
    FakeModel.not_cached = set()
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    tr._MODEL_CACHE.clear()
    yield
    tr._MODEL_CACHE.clear()


def test_cpu_run_returns_words_and_meta(monkeypatch):
    seen = []
    result = tr.transcribe("clip.wav", tr.TranscribeOptions(device="cpu"), progress=lambda f, m: seen.append(f))
    assert [w.text for w in result.words] == [" مرحبا", " بكم"]
    assert result.device == "cpu" and result.model == tr.DEFAULT_CPU_MODEL and result.compute_type == "int8"
    assert result.duration == 2.0
    assert seen[-1] == 1.0
    kwargs = FakeModel.instances[0].calls[0]
    assert kwargs["language"] == "ar" and kwargs["word_timestamps"] is True and kwargs["vad_filter"] is True


def test_auto_uses_gpu_defaults_when_cuda_is_there(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    result = tr.transcribe("clip.wav")
    assert result.device == "cuda"
    assert result.model == tr.DEFAULT_GPU_MODEL
    assert result.compute_type == "float16"


def test_cuda_failure_falls_back_to_cpu_and_smaller_model(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    FakeModel.fail_on_cuda = True
    result = tr.transcribe("clip.wav", tr.TranscribeOptions(model="large-v3"))
    assert result.device == "cpu"
    assert result.model == "small"
    assert [m.device for m in FakeModel.instances] == ["cuda", "cpu"]


def test_explicit_cuda_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)
    FakeModel.fail_on_cuda = True
    with pytest.raises(RuntimeError):
        tr.transcribe("clip.wav", tr.TranscribeOptions(device="cuda"))


def test_non_cuda_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 1)

    def boom(self, audio, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(FakeModel, "transcribe", boom)
    with pytest.raises(FileNotFoundError):
        tr.transcribe("missing.wav")


def test_models_are_cached():
    opts = tr.TranscribeOptions(device="cpu", model="tiny")
    tr.transcribe("a.wav", opts)
    tr.transcribe("b.wav", opts)
    assert len(FakeModel.instances) == 1


def test_vad_keeps_short_pauses():
    # 500 ms lost words on narration over music, see the comment in TranscribeOptions.
    assert tr.TranscribeOptions().min_silence_ms == 2000


def test_cached_model_never_touches_the_network():
    tr.transcribe("a.wav", tr.TranscribeOptions(device="cpu", model="tiny"))
    assert FakeModel.instances[0].local_files_only is True


def test_missing_model_is_downloaded_once():
    FakeModel.not_cached = {"tiny"}
    tr.transcribe("a.wav", tr.TranscribeOptions(device="cpu", model="tiny"))
    assert [m.local_files_only for m in FakeModel.instances] == [False]


def test_dialect_prompt_and_translate_task():
    tr.transcribe("a.wav", tr.TranscribeOptions(device="cpu", dialect="egyptian", task="translate", language="auto"))
    kwargs = FakeModel.instances[0].calls[0]
    assert kwargs["initial_prompt"] == prompt_for("egyptian")
    assert kwargs["task"] == "translate"
    assert kwargs["language"] is None


def test_dialects():
    assert "none" in DIALECTS and "gulf" in DIALECTS
    assert prompt_for("none") is None and prompt_for(None) is None
    with pytest.raises(ValueError):
        prompt_for("klingon")


def test_smaller():
    assert tr._smaller("large-v3", "small") == "small"
    assert tr._smaller("tiny", "small") == "tiny"
    assert tr._smaller("/models/custom", "small") == "small"


def test_resolve_helpers(monkeypatch):
    monkeypatch.setattr(tr, "cuda_device_count", lambda: 0)
    assert tr.resolve_device("auto") == "cpu"
    assert tr.resolve_model("auto", "cpu") == "small"
    assert tr.resolve_compute_type("auto", "cuda") == "float16"
    assert tr.resolve_compute_type("int8_float16", "cuda") == "int8_float16"
