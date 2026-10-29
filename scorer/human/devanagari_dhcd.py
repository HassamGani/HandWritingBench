ALPHABET = "devanagari"
VARIANT = "print"
SOURCE = ("DHCD, Devanagari Handwritten Character Dataset, UCI ML repo #389 "
          "(Acharya, Pant, Gyawali 2015, SKIMA)")
LICENSE = "CC BY 4.0 (UCI)"
INVERT = True

import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"
HOME = DATA / "devanagari_dhcd"
DIR = HOME / "raw" / "DevanagariHandwrittenCharacterDataset"
# character_1_ka .. character_33_ha; 34-36 are the conjuncts क्ष त्र ज्ञ, dropped with the digits
CONSONANTS = "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह".split()


def fetch():
    if DIR.exists():
        return
    zp = HOME / "dhcd.zip"
    if not zp.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(HOME / "raw")


def items():
    # no writer ids in the files; the official Train/Test split is used as is
    for split in ("Train", "Test"):
        for d in sorted((DIR / split).glob("character_*")):
            n = int(d.name.split("_")[1])
            if n > 33:
                continue
            for f in sorted(d.glob("*.png")):
                yield f, CONSONANTS[n - 1], None, split.lower()
