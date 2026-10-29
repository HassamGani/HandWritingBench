ALPHABET = "bengali"
VARIANT = "print"
SOURCE = ("BanglaLekha-Isolated, Mendeley Data hf6sf8zrkc "
          "(Biswas et al. 2017, Data in Brief 12)")
LICENSE = "CC BY 4.0 (Mendeley Data)"
INVERT = True

import hashlib  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://data.mendeley.com/public-files/datasets/hf6sf8zrkc/files/ffe73a3a-5999-4ad2-970f-3e14c141cdb1/file_downloaded"
HOME = DATA / "bengali_banglalekha"
DIR = HOME / "raw" / "BanglaLekha-Isolated" / "Images"
# folders 1-11 vowels, 12-43 consonants; 44-84 (more consonant signs, digits, conjuncts) are dropped
LETTERS = "অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ".split()
KEEP = 0.76  # share of forms kept, ~1500 images per letter out of ~1980


def fetch():
    if DIR.exists():
        return
    zp = HOME / "banglalekha.zip"
    if not zp.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(HOME / "raw")


def items():
    for n, ch in enumerate(LETTERS, 1):
        for f in sorted((DIR / str(n)).glob("*.png")):
            # district_institution_gender_age_date_formserial_class; one form = one writer
            form = f.stem.split("_")[5]
            # md5, not the sha256 split_of uses, so the cap doesn't skew the test share
            if int(hashlib.md5(form.encode()).hexdigest(), 16) % 1000 < KEEP * 1000:
                yield f, ch, form, None
