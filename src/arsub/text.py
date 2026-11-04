"""Arabic text helpers for subtitles.

Everything here is pure string work so it can be tested without a model:
punctuation fixes, digit styles, diacritics, Persian look-alike letters,
display length and the invisible bidi marks that keep punctuation on the
correct side of right-to-left lines.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, Quranic marks, superscript alef and friends. They sit on top of
# letters, so they never add width to a subtitle line.
DIACRITICS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed]")
TATWEEL = "\u0640"

ARABIC_LETTER_RE = re.compile("[\u0620-\u064a\u0660-\u0669\u066e-\u06d3\u06fa-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_LETTER_RE = re.compile("[A-Za-z]")

# Bidi control characters.
LRM = "\u200e"
RLM = "\u200f"
ALM = "\u061c"
RLE = "\u202b"
PDF = "\u202c"
RLI = "\u2067"
PDI = "\u2069"
BIDI_MARKS_RE = re.compile("[\u200e\u200f\u061c\u202a-\u202e\u2066-\u2069]")
ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\ufeff]")

WESTERN_DIGITS = "0123456789"
ARABIC_INDIC_DIGITS = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
PERSIAN_DIGITS = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"

DIGIT_STYLES = ("western", "arabic-indic", "keep")
RTL_MARK_MODES = ("none", "rlm", "embed")

# Whisper sometimes writes Arabic words with Persian/Urdu look-alike letters.
PERSIAN_TO_ARABIC = str.maketrans({
    "\u06a9": "\u0643",  # keheh -> kaf
    "\u06cc": "\u064a",  # farsi yeh -> yeh
    "\u06c0": "\u0629",  # heh with yeh above -> teh marbuta
})

_TO_WESTERN = str.maketrans(ARABIC_INDIC_DIGITS + PERSIAN_DIGITS, WESTERN_DIGITS * 2)
_TO_ARABIC_INDIC = str.maketrans(WESTERN_DIGITS + PERSIAN_DIGITS, ARABIC_INDIC_DIGITS * 2)

SENTENCE_END = (".", "!", "\u061f", "?", "\u2026")
CLAUSE_END = ("\u060c", ",", "\u061b", ";", ":")


def is_arabic_char(ch: str) -> bool:
    return bool(ARABIC_LETTER_RE.match(ch))


def is_rtl(text: str) -> bool:
    """True when Arabic letters outnumber Latin letters in ``text``."""
    arabic = len(ARABIC_LETTER_RE.findall(text))
    latin = len(LATIN_LETTER_RE.findall(text))
    return arabic > 0 and arabic >= latin


def strip_diacritics(text: str) -> str:
    """Remove harakat and tatweel, keep the letters."""
    return DIACRITICS_RE.sub("", text).replace(TATWEEL, "")


def strip_bidi_marks(text: str) -> str:
    return BIDI_MARKS_RE.sub("", text)


def convert_digits(text: str, style: str = "western") -> str:
    """Rewrite digits as ``western`` (0-9), ``arabic-indic`` or leave them (``keep``).

    Arabic-Indic output also uses the Arabic percent sign and decimal separator
    so "12.5%" becomes "١٢٫٥٪" instead of a mix of scripts.
    """
    if style == "keep":
        return text
    if style == "western":
        text = text.translate(_TO_WESTERN)
        text = text.replace("\u066a", "%")
        text = re.sub("(?<=[0-9])\u066b(?=[0-9])", ".", text)
        text = re.sub("(?<=[0-9])\u066c(?=[0-9])", ",", text)
        return text
    if style == "arabic-indic":
        text = text.translate(_TO_ARABIC_INDIC)
        text = re.sub("(?<=[\u0660-\u0669])\\.(?=[\u0660-\u0669])", "\u066b", text)
        text = re.sub("(?<=[\u0660-\u0669]),(?=[\u0660-\u0669])", "\u066c", text)
        text = re.sub("(?<=[\u0660-\u0669])\\s?%", "\u066a", text)
        return text
    raise ValueError(f"unknown digit style {style!r}, expected one of {DIGIT_STYLES}")


_LATIN_TO_ARABIC_PUNCT = {",": "\u060c", "?": "\u061f", ";": "\u061b"}


def _arabic_punctuation(text: str) -> str:
    """Swap Latin , ? ; for the Arabic forms when they follow Arabic text.

    A comma between two digits ("3,5") or after a Latin word is left alone.
    """
    rtl_line = is_rtl(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in _LATIN_TO_ARABIC_PUNCT:
            prev = strip_diacritics(text[:i]).rstrip()
            prev_ch = prev[-1] if prev else ""
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            between_digits = prev_ch.isdigit() and next_ch.isdigit()
            after_arabic = bool(prev_ch) and is_arabic_char(prev_ch)
            after_number_in_rtl = prev_ch.isdigit() and rtl_line
            if (after_arabic or after_number_in_rtl) and not between_digits:
                out.append(_LATIN_TO_ARABIC_PUNCT[ch])
                continue
        out.append(ch)
    return "".join(out)


_SPACE_BEFORE_PUNCT_RE = re.compile("\\s+([\u060c\u061b\u061f,;:!?.\u2026)\\]\u00bb])")
_SPACE_AFTER_OPEN_RE = re.compile("([(\\[\u00ab])\\s+")
_NO_SPACE_AFTER_PUNCT_RE = re.compile("([\u060c\u061b\u061f!\u2026])(?=[^\\s\u060c\u061b\u061f!.\u2026)\\]\u00bb\"'])")
_REPEATED_PUNCT_RE = re.compile("([\u060c\u061b\u061f!?,;])\\1+")


def fix_punctuation(text: str) -> str:
    """Tidy spacing and use Arabic punctuation marks in Arabic text."""
    text = _arabic_punctuation(text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _NO_SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_letters(text: str) -> str:
    """NFC, Persian look-alikes to Arabic letters, drop zero-width junk."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(PERSIAN_TO_ARABIC)
    return ZERO_WIDTH_RE.sub("", text)


def clean_text(text: str, digits: str = "western", remove_diacritics: bool = False) -> str:
    """Full cleanup used on every subtitle line."""
    text = strip_bidi_marks(normalize_letters(text))
    if remove_diacritics:
        text = strip_diacritics(text)
    text = convert_digits(text, digits)
    return fix_punctuation(text)


def display_len(text: str) -> int:
    """Characters a viewer actually reads.

    Diacritics, tatweel, bidi marks and zero-width characters take no space on
    screen, so they do not count toward line length or reading speed.
    """
    text = strip_bidi_marks(ZERO_WIDTH_RE.sub("", text))
    return len(strip_diacritics(text))


def add_rtl_marks(line: str, mode: str = "embed") -> str:
    """Make a right-to-left line render correctly in players that guess LTR.

    ``rlm`` puts a RIGHT-TO-LEFT MARK at both ends, which fixes trailing
    punctuation and leading numbers in most players. ``embed`` wraps the line in
    RLE ... PDF, the stronger form some TVs need. Lines that are mostly Latin
    are returned unchanged.
    """
    if mode == "none" or not line or not is_rtl(line):
        return line
    if mode == "rlm":
        return f"{RLM}{line}{RLM}"
    if mode == "embed":
        return f"{RLE}{line}{PDF}"
    raise ValueError(f"unknown rtl mark mode {mode!r}, expected one of {RTL_MARK_MODES}")


def bare_word(word: str) -> str:
    """A word without diacritics or surrounding punctuation, for lookups."""
    word = strip_diacritics(strip_bidi_marks(word))
    return word.strip(" \t\u060c\u061b\u061f,;:!?.\u2026\"'()[]\u00ab\u00bb-")


def ends_sentence(word: str) -> bool:
    return strip_bidi_marks(word).rstrip("\"'\u00bb)").endswith(SENTENCE_END)


def ends_clause(word: str) -> bool:
    return strip_bidi_marks(word).rstrip("\"'\u00bb)").endswith(CLAUSE_END)
