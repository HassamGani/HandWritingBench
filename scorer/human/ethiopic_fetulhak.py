ALPHABET = "ethiopic"
VARIANT = "base"
SOURCE = "Handwritten Amharic character dataset (Fetulhak Abdurahman), github: Fetulhak/Handwritten-Amharic-character-Dataset"
LICENSE = "GPL-3.0 (repo license)"
INVERT = True

import subprocess  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://raw.githubusercontent.com/Fetulhak/Handwritten-Amharic-character-Dataset/master/Amharic%20Character%20Dataset%20{}.rar"
DIR = DATA / "ethiopic_fetulhak"
# classes run 7 vowel orders per consonant (001he..007ho, 008le, ...), consonants in the same
# order as ethiopic_yared, so the base form of consonant k is class 1 + 7k
BASE = "ሀለሐመሠረሰሸቀበቨተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ"


def fetch():
    if (DIR / "raw").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        rar = DIR / f"raw{i}.rar"
        if not rar.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(rar), URL.format(i)], check=True)
        (DIR / "raw").mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(DIR / "raw")], check=True)


def items():
    # file names are <class><romanized>.<n>.jpg; n is a per-class sample number, not a known writer,
    # so this set only trains (the Ethiopic human test stays the writer-split Yared set). A near-duplicate
    # check against Yared found no shared images.
    for k, ch in enumerate(BASE):
        for f in sorted((DIR / "raw").glob(f"*/{1 + 7 * k:03d}*.jpg")):
            yield f, ch, None, "train"
