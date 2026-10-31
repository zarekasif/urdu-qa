"""Text normalisation for Urdu QA.

Used in two places:
- `clean_text` repairs the raw translated data (HTML entities, stray whitespace) before
  answer spans are located, and never changes the characters of the text otherwise.
- `normalize_answer` is the comparison form used by the metrics, so that a prediction and
  a gold answer that differ only in spelling variants, diacritics or punctuation still match.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata

# Arabic-script code points that machine translation often emits in place of the Urdu ones.
_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh -> Farsi/Urdu yeh
        "ى": "ی",  # alef maksura -> Urdu yeh
        "ك": "ک",  # Arabic kaf -> keheh
        "ه": "ہ",  # Arabic heh -> heh goal
        "ۀ": "ہ",  # heh with yeh above -> heh goal
        "ۓ": "ے",  # yeh barree with hamza -> yeh barree
    }
)

# Eastern Arabic-Indic (Urdu) and Arabic-Indic digits -> ASCII.
_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)

# Harakat, superscript alef, tatweel.
_DIACRITICS = re.compile("[ً-ٰٟـ]")

_URDU_PUNCT = "۔،؟؛٪«»“”‘’…"
_PUNCT = set(string.punctuation) | set(_URDU_PUNCT)

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Unescape HTML entities (e.g. ``&quot;``, ``&#39;``) and collapse whitespace."""
    text = html.unescape(html.unescape(str(text)))
    return _WS.sub(" ", text).strip()


def normalize_answer(text: str) -> str:
    """Normalise an answer string for EM/F1 comparison."""
    text = clean_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP).translate(_DIGIT_MAP)
    text = _DIACRITICS.sub("", text)
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = text.lower()
    return _WS.sub(" ", text).strip()
