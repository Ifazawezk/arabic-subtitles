import pytest

from arsub import text as T


def test_latin_punctuation_becomes_arabic_after_arabic_words():
    assert T.fix_punctuation("مرحبا , كيف حالك ?") == "مرحبا، كيف حالك؟"
    assert T.fix_punctuation("أولا; ثانيا") == "أولا؛ ثانيا"


def test_punctuation_left_alone_in_english_and_numbers():
    assert T.fix_punctuation("Hello, how are you?") == "Hello, how are you?"
    assert T.fix_punctuation("السعر 3,5 ريال") == "السعر 3,5 ريال"


def test_comma_after_number_in_arabic_sentence():
    assert T.fix_punctuation("في عام 2020, بدأنا") == "في عام 2020، بدأنا"


def test_spacing_around_punctuation():
    assert T.fix_punctuation("نعم ،قال ذلك .") == "نعم، قال ذلك."
    assert T.fix_punctuation("كيف؟؟ حالك") == "كيف؟ حالك"
    assert T.fix_punctuation("( مرحبا )") == "(مرحبا)"


def test_strip_diacritics_and_tatweel():
    assert T.strip_diacritics("مَرْحَبًا بِكُمْ") == "مرحبا بكم"
    assert T.strip_diacritics("جمـــيل") == "جميل"


def test_digits_to_arabic_indic():
    assert T.convert_digits("عام 2026", "arabic-indic") == "عام ٢٠٢٦"
    assert T.convert_digits("12.5%", "arabic-indic") == "١٢٫٥٪"


def test_digits_to_western_including_persian():
    assert T.convert_digits("٢٠٢٦ و ۳", "western") == "2026 و 3"
    assert T.convert_digits("١٢٫٥٪", "western") == "12.5%"


def test_digits_keep_and_bad_style():
    assert T.convert_digits("٢٠٢٦", "keep") == "٢٠٢٦"
    with pytest.raises(ValueError):
        T.convert_digits("1", "roman")


def test_persian_letters_are_mapped():
    assert T.normalize_letters("کتاب فی") == "كتاب في"


def test_display_len_ignores_marks():
    plain = "مرحبا"
    assert T.display_len("مَرْحَبًا") == len(plain)
    assert T.display_len(T.RLE + plain + T.PDF) == len(plain)


def test_is_rtl():
    assert T.is_rtl("مرحبا بالعالم")
    assert T.is_rtl("استخدمنا Python هنا")
    assert not T.is_rtl("Hello world")
    assert not T.is_rtl("2026")


def test_rtl_marks():
    line = "كيف حالك؟"
    assert T.add_rtl_marks(line, "none") == line
    assert T.add_rtl_marks(line, "rlm") == T.RLM + line + T.RLM
    assert T.add_rtl_marks(line, "embed") == T.RLE + line + T.PDF
    assert T.add_rtl_marks("Hello", "embed") == "Hello"
    assert T.strip_bidi_marks(T.add_rtl_marks(line, "embed")) == line


def test_clean_text_full():
    raw = "مَرحباً , عام ٢٠٢٦ ?"
    assert T.clean_text(raw) == "مَرحباً، عام 2026؟"
    assert T.clean_text(raw, digits="arabic-indic", remove_diacritics=True) == "مرحبا، عام ٢٠٢٦؟"


def test_sentence_and_clause_ends():
    assert T.ends_sentence("انتهى.")
    assert T.ends_sentence("حقا؟")
    assert T.ends_clause("أولا،")
    assert not T.ends_sentence("كلمة")
    assert T.bare_word("«مرحباً،»") == "مرحبا"
