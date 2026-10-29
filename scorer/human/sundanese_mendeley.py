ALPHABET = "sundanese"
VARIANT = "print"
SOURCE = ("Indonesian Local Script Characters (Ihsan), Mendeley Data doi:10.17632/vfj32bpjsf.3, Sunda.zip; "
          "bare consonants from Vowel A/, independent vowels from each vowel folder")
LICENSE = "CC BY 4.0"
INVERT = True  # items are background-subtracted ink maps

import subprocess  # noqa: E402
import unicodedata  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402
from ._scan import clean  # noqa: E402

DIR = DATA / "sundanese_mendeley"
URL = "https://data.mendeley.com/public-files/datasets/vfj32bpjsf/files/fe673703-fe0d-4fd7-8ab8-9edee981aff5/file_downloaded"
# é = ᮆ (AE) and e = ᮈ (pepet), checked by eye; kha and sya (ᮮ ᮯ) aren't in our inventory
CHARS = {"a": "ᮃ", "i": "ᮄ", "u": "ᮅ", "é": "ᮆ", "o": "ᮇ", "e": "ᮈ", "eu": "ᮉ",
         "ka": "ᮊ", "qa": "ᮋ", "ga": "ᮌ", "nga": "ᮍ", "ca": "ᮎ", "ja": "ᮏ", "za": "ᮐ", "nya": "ᮑ", "ta": "ᮒ", "da": "ᮓ",
         "na": "ᮔ", "pa": "ᮕ", "fa": "ᮖ", "va": "ᮗ", "ba": "ᮘ", "ma": "ᮙ", "ya": "ᮚ", "ra": "ᮛ", "la": "ᮜ", "wa": "ᮝ",
         "sa": "ᮞ", "xa": "ᮟ", "ha": "ᮠ"}


def fetch():
    if (DIR / "Sunda").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    if not (DIR / "Sunda.zip").exists():
        subprocess.run(["curl", "-sSLfo", DIR / "Sunda.zip", URL], check=True)
    zipfile.ZipFile(DIR / "Sunda.zip").extractall(DIR)


def items():
    for v in sorted((DIR / "Sunda").iterdir()):
        for d in sorted(v.iterdir()) if v.is_dir() else []:
            ch = CHARS.get(unicodedata.normalize("NFC", d.name).lower())
            # consonant folders outside Vowel A hold syllables (ka + vowel sign), not bare letters
            if ch and (v.name == "Vowel A" or ch in "ᮃᮄᮅᮆᮇᮈᮉ"):
                for f in sorted(d.glob("*.*")):
                    yield clean(f), ch, None, None
