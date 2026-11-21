"""Which language a chunk is written in, so it can go to the right keyword index (Italian or English)."""
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

MIN_CONFIDENCE = 0.85  # below this the detector is not sure enough to pick a side
MIN_LETTERS = 15  # a page number, a case reference or a short heading has too little text to judge


@lru_cache
def _detector():
    return LanguageDetectorBuilder.from_languages(Language.ITALIAN, Language.ENGLISH).build()


def detect(text: str) -> str | None:
    """'it' or 'en' when sure, None when not (the chunk then goes into both keyword indexes)."""
    if sum(c.isalpha() for c in text) < MIN_LETTERS:
        return None
    for c in _detector().compute_language_confidence_values(text):
        if c.value >= MIN_CONFIDENCE:
            return "it" if c.language == Language.ITALIAN else "en"
    return None
