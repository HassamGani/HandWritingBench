ALPHABET = "tamil"
VARIANT = "print"
SOURCE = ("uTHCD, unconstrained Tamil Handwritten Character Database, kaggle: "
          "faizalhajamohideen/uthcdtamil-handwritten-database (Shaffi & Hajamohideen 2021, IEEE Access 9)")
LICENSE = "none stated (Kaggle lists it as Unknown)"

import csv  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/faizalhajamohideen/uthcdtamil-handwritten-database"
HOME = DATA / "tamil_uthcd"
DIR = HOME / "raw" / "uTHCD_a(70-30-split)" / "70-30-split"
# uTHCD's class ids don't follow the lipitk TamilChar table. Read off by eye: 0 is the aa-sign,
# 1-13 the vowels and aytham, then six classes per consonant (C+pulli, C, Ci, Cii, Cu, Cuu) from 14,
# with the consonants in the order below (note ச before ங, and ள ழ வ). Grantha and ligatures follow.
CLASSES = dict(zip(range(1, 14), "அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ".split()))
CLASSES.update({15 + 6 * i: c for i, c in enumerate("க ச ங ஞ ட ண த ந ப ம ய ர ல ள ழ வ ற ன".split())})


def fetch():
    if DIR.exists():
        return
    zp = HOME / "uthcd.zip"
    if not zp.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zp)
    z = zipfile.ZipFile(zp)
    z.extractall(HOME / "raw", [n for n in z.namelist() if n.startswith("uTHCD_a(70-30-split)/70-30-split/") and "classwise" not in n])


def items():
    # The official 70/30 split isn't by writer (the same form id shows up on both sides), so the
    # split is left to split_of. File names are <form>_<class>.bmp. "NNs" forms hold all 156 classes;
    # numbered forms come in pairs, odd ones with classes 1-79 and even ones with 0 and 81-155,
    # so 2k and 2k+1 are taken as one writer.
    for part in ("Train", "Test"):
        for _, name, label in list(csv.reader(open(DIR / f"GroundTruth-{part}.csv")))[1:]:
            ch = CLASSES.get(int(label))
            if ch is None:
                continue
            form = name.rsplit("_", 1)[0]
            writer = form if form.endswith("s") else f"p{int(form) // 2}"
            yield DIR / part / name, ch, writer, None
