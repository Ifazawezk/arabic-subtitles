from arsub import text as T
from arsub.builder import (
    BuildOptions,
    Word,
    build_cues,
    clean_words,
    cue_from_text,
    split_lines,
    stats,
)


def timed(sentence: str, start: float = 0.0, step: float = 0.35, gap: float = 0.05) -> list[Word]:
    """Words with even timing, like a steady speaker."""
    words, t = [], start
    for w in sentence.split():
        words.append(Word(t, t + step, w))
        t += step + gap
    return words


LONG = (
    "بسم الله الرحمن الرحيم أهلا وسهلا بكم في هذه الحلقة الجديدة من البرنامج "
    "حيث نتحدث اليوم عن تاريخ مدينة الدوحة وكيف تغيرت خلال الخمسين سنة الماضية "
    "من قرية صغيرة لصيد اللؤلؤ إلى عاصمة حديثة يعيش فيها أكثر من مليون شخص"
)


def test_short_text_stays_on_one_line():
    assert split_lines("مرحبا بكم جميعا".split(), 42) == ["مرحبا بكم جميعا"]


def test_split_never_breaks_words_and_respects_length():
    words = LONG.split()[:12]
    lines = split_lines(words, max_chars=42, max_lines=2)
    assert len(lines) == 2
    assert " ".join(lines).split() == words
    assert all(T.display_len(line) <= 42 for line in lines)


def test_line_does_not_end_with_preposition():
    words = "ذهب الولد الصغير مع والده إلى السوق في صباح يوم الجمعة الماضي".split()
    lines = split_lines(words, max_chars=36)
    assert len(lines) == 2
    first_last = T.bare_word(lines[0].split()[-1])
    assert first_last not in {"في", "إلى", "مع", "من", "على"}


def test_compound_name_kept_together():
    words = "وقال الشيخ عبد الله بن محمد إن الاجتماع سيعقد غدا في الدوحة".split()
    for max_chars in (26, 30, 34, 38):
        lines = split_lines(words, max_chars=max_chars)
        for line in lines[:-1]:
            assert T.bare_word(line.split()[-1]) not in {"عبد", "بن"}, (max_chars, lines)


def test_split_prefers_punctuation():
    words = "شكرا لكم جميعا على الحضور، والآن نبدأ الحديث عن الموضوع".split()
    lines = split_lines(words, max_chars=36)
    assert lines[0].endswith("الحضور،")


def test_diacritics_do_not_count_toward_length():
    plain = "كتب الطالب الدرس في الدفتر بعناية كبيرة"
    voweled = "كَتَبَ الطَّالِبُ الدَّرْسَ فِي الدَّفْتَرِ بِعِنَايَةٍ كَبِيرَةٍ"
    assert T.display_len(plain) <= 42
    assert len(split_lines(voweled.split(), 42)) == 1


def test_build_cues_limits():
    opts = BuildOptions(max_chars=42, max_lines=2, max_duration=7.0)
    cues = build_cues(timed(LONG), opts)
    assert len(cues) >= 3
    for c in cues:
        assert 1 <= len(c.lines) <= 2
        assert all(T.display_len(line) <= 42 for line in c.lines)
        assert c.end > c.start
    # every word ends up somewhere, in order
    joined = " ".join(" ".join(c.lines) for c in cues).split()
    assert joined == LONG.split()


def test_cues_never_overlap_and_keep_a_gap():
    opts = BuildOptions(min_gap=0.08)
    cues = build_cues(timed(LONG, step=0.25, gap=0.0), opts)
    for a, b in zip(cues, cues[1:]):
        assert b.start - a.end >= 0.08 - 1e-6


def test_pause_starts_new_cue():
    words = timed("مرحبا بكم") + timed("كيف حالكم", start=5.0)
    cues = build_cues(words)
    assert [c.text for c in cues] == ["مرحبا بكم", "كيف حالكم"]


def test_sentence_end_starts_new_cue():
    words = timed("هذه جملة أولى طويلة بما يكفي لتملأ نصف السطر. وهذه جملة ثانية")
    cues = build_cues(words)
    assert cues[0].lines[-1].endswith("السطر.")


def test_min_duration_extends_into_silence():
    words = [Word(0.0, 0.3, "نعم")] + timed("ثم بدأ الكلام من جديد", start=4.0)
    cues = build_cues(words, BuildOptions(min_duration=1.0))
    assert cues[0].text == "نعم"
    assert abs(cues[0].duration - 1.0) < 1e-6


def test_reading_speed_extension_stops_before_next_cue():
    fast = [Word(0.0, 0.2, "هذه"), Word(0.2, 0.4, "جملة"), Word(0.4, 0.6, "سريعة"), Word(0.6, 0.8, "جدا.")]
    nxt = timed("وهذه جملة تأتي بعدها مباشرة", start=1.5)
    opts = BuildOptions(max_cps=15, pause=0.5)
    cues = build_cues(fast + nxt, opts)
    assert cues[0].end <= cues[1].start - opts.min_gap + 1e-6
    assert cues[0].end > 0.8


def test_tiny_cue_merges_into_previous():
    words = timed("قال المدير في الاجتماع") + [Word(1.75, 1.9, "نعم")]
    cues = build_cues(words, BuildOptions(pause=0.7))
    assert len(cues) == 1
    assert cues[0].text.endswith("نعم")


def test_stray_punctuation_attaches_to_previous_word():
    words = clean_words([Word(0, 0.5, " مرحبا"), Word(0.5, 0.6, "،"), Word(0.6, 1.0, " صديقي")], BuildOptions())
    assert [w.text for w in words] == ["مرحبا،", "صديقي"]


def test_options_digits_and_diacritics_apply():
    opts = BuildOptions(digits="arabic-indic", remove_diacritics=True)
    cues = build_cues(timed("وُلِدَ في عام 1990"), opts)
    assert cues[0].text == "ولد في عام ١٩٩٠"


def test_latin_punctuation_fixed_in_cues():
    cues = build_cues(timed("كيف حالك ?"))
    assert cues[0].text == "كيف حالك؟"


def test_cue_from_text_resplits_and_cleans():
    cue = cue_from_text(1.0, 4.0, "هذا نص طويل كتبه المستخدم في المحرر , ويجب أن يقسم على سطرين")
    assert len(cue.lines) == 2
    assert "،" in cue.text


def test_stats():
    cues = build_cues(timed(LONG))
    s = stats(cues)
    assert s["cues"] == len(cues)
    assert s["too_long_lines"] == 0
    assert stats([]) == {"cues": 0, "too_fast": 0, "too_long_lines": 0, "max_cps": 0.0}


def test_empty_input():
    assert build_cues([]) == []
    assert split_lines([]) == []
