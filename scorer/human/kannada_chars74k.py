ALPHABET = "kannada"
VARIANT = "print"
SOURCE = ("Chars74K, Kannada handwritten (tablet) characters, KannadaHnd.tgz from "
          "info-ee.surrey.ac.uk/CVSSP/demos/chars74k (de Campos, Babu, Varma 2009, VISAPP)")
LICENSE = "free for research use (per the Chars74K page)"

import tarfile  # noqa: E402
import urllib.request  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/KannadaHnd.tgz"
HOME = DATA / "kannada_chars74k"
DIR = HOME / "raw" / "Kannada" / "Hnd" / "Img"
# Sample001-016 are the vowels (008 ೠ, 015 ಅಂ, 016 ಅಃ aren't ours). Then 17 classes per consonant
# with the bare one first; the consonants run ಕ..ರ ಱ ಲ ಳ ೞ ವ ಶ ಷ ಸ ಹ ಕ್ಷ (all checked by eye).
VOWELS = dict(zip([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], "ಅ ಆ ಇ ಈ ಉ ಊ ಋ ಎ ಏ ಐ ಒ ಓ ಔ".split()))
CONSONANTS = "ಕ ಖ ಗ ಘ ಙ ಚ ಛ ಜ ಝ ಞ ಟ ಠ ಡ ಢ ಣ ತ ಥ ದ ಧ ನ ಪ ಫ ಬ ಭ ಮ ಯ ರ ಱ ಲ ಳ ೞ ವ ಶ ಷ ಸ ಹ".split()
CLASSES = VOWELS | {17 + 17 * i: c for i, c in enumerate(CONSONANTS) if c not in "ಱೞ"}


def fetch():
    if DIR.exists():
        return
    tgz = HOME / "KannadaHnd.tgz"
    if not tgz.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(HOME / "raw", filter="data")


def items():
    for n, ch in sorted(CLASSES.items()):
        for f in sorted((DIR / f"Sample{n:03d}").glob("*.png")):
            # imgNNN-KKK.png: the same K has the same pen and hand in every class, so K is the writer
            yield f, ch, f.stem.split("-")[1], None
