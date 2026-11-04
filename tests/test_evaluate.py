"""The scoring rules behind every number in the README."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_normalize_ignores_spelling_variants_that_are_not_real_errors():
    assert evaluate.normalize("أَحْمَد ذهب إلى المدرسة.") == evaluate.normalize("احمد ذهب الي المدرسه")


def test_normalize_keeps_real_differences():
    assert evaluate.normalize("ذهب الولد") != evaluate.normalize("ذهبت البنت")


def test_normalize_unifies_digits():
    assert evaluate.normalize("عام ٢٠٢٦") == "عام 2026"


def test_light_keeps_letters_as_written():
    assert evaluate.light("إلى المدرسة!") == "إلى المدرسة"
    assert evaluate.light("إلى") != evaluate.light("الى")


def test_score_is_corpus_level():
    pytest.importorskip("jiwer")
    # 1 error in 1 word plus 0 errors in 3 words is 25%, not the 50% a per-clip average gives.
    out = evaluate.score(["واحد", "اثنان ثلاثة اربعة"], ["خطا", "اثنان ثلاثة اربعة"])
    assert out["wer"] == 25.0
    assert out["reference_words"] == 4


def test_score_counts_an_empty_hypothesis_as_deletions():
    pytest.importorskip("jiwer")
    out = evaluate.score(["واحد اثنان"], [""])
    assert out["wer"] == 100.0


def test_score_items_splits_masc_by_recording_type():
    pytest.importorskip("jiwer")
    items = [
        {"source": "masc", "type": "c", "duration": 1.0, "text": "واحد اثنان"},
        {"source": "masc", "type": "n", "duration": 2.0, "text": "ثلاثة اربعة"},
        {"source": "cv18", "duration": 3.0, "text": "خمسة ستة"},
    ]
    by_source, overall = evaluate.score_items(items, ["واحد اثنان", "ثلاثة", "خمسة ستة"])
    assert by_source["masc-clean"]["normalized"]["wer"] == 0.0
    assert by_source["masc-noisy"]["normalized"]["wer"] == 50.0
    assert by_source["masc"]["clips"] == 2
    assert round(overall["normalized"]["wer"], 2) == round(100 / 6, 2)
