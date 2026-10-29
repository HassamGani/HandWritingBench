ALPHABET = "devanagari"
VARIANT = "print"
SOURCE = ("NHCD, Nepali Handwritten Character Dataset, github: ashokpant/nhcr data/datasets/nhcd.rar "
          "(Pant, Gyawali, Acharya 2012, SKIMA)")
LICENSE = "none stated in the GitHub repo (a Kaggle mirror labels it ODbL)"

import subprocess  # noqa: E402
import urllib.request  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://raw.githubusercontent.com/ashokpant/nhcr/master/data/datasets/nhcd.rar"
HOME = DATA / "devanagari_nhcd"
DIR = HOME / "raw"
# vowels 11-12 (अं अः), consonants 34-36 (conjuncts) and numerals are not in the inventory
VOWELS = "अ आ इ ई उ ऊ ए ऐ ओ औ".split()
CONSONANTS = "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह".split()


def fetch():
    if DIR.exists():
        return
    rar = HOME / "nhcd.rar"
    if not rar.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, rar)
    DIR.mkdir()
    subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(DIR)], check=True)


def items():
    for group, letters in (("vowels", VOWELS), ("consonants", CONSONANTS)):
        for n, ch in enumerate(letters, 1):
            for f in sorted((DIR / group / str(n)).glob("*.jpg")):
                # NNN_kk.jpg: NNN is the writer
                yield f, ch, f.name.split("_")[0], None
