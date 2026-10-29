ALPHABET = "thai"
VARIANT = "print"
SOURCE = ("Burapha-TH, character subset (1,072 writers, official train/test), "
          "https://services.informatics.buu.ac.th/datasets/Burapha-TH/ ; Onuean et al., "
          "\"Burapha-TH: A Multi-Purpose Character, Digit, and Syllable Handwriting Dataset\", Applied Sciences 2022")
LICENSE = "not stated on the download page (the paper calls it free to use for research)"
INVERT = True  # items are despeckled ink maps

import collections  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import zipfile  # noqa: E402

from . import DATA, to_ink  # noqa: E402
from ._scan import despeckle  # noqa: E402

DIR = DATA / "thai_burapha"
URL = "https://services.informatics.buu.ac.th/datasets/Burapha-TH/character/20210306-{}.zip"
LETTERS = set("กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")


def fetch():
    for split in ("train", "test"):
        z = DIR / f"20210306-{split}.zip"
        if (DIR / split).exists():
            continue
        DIR.mkdir(parents=True, exist_ok=True)
        if not z.exists():
            subprocess.run(["curl", "-sSLfo", z, URL.format(split)], check=True)
        zipfile.ZipFile(z).extractall(DIR)


def items():
    # folders are NN-DEC-HEX-NAME with HEX the TIS-620 code; files are <writer>_<n>.jpg.
    # The official split shares ~100 writers (a handful of images each), so every writer
    # goes to the split that holds most of their images.
    files = []
    for split in ("train", "test"):
        for d in sorted((DIR / split).iterdir()):
            ch = chr(0x0E00 + int(d.name.split("-")[2], 16) - 0xA0) if d.is_dir() else ""
            if ch in LETTERS:
                files += [(f, ch, re.sub(r"_\d+$", "", f.stem), split) for f in sorted(d.glob("*.jpg")) if " - Copy" not in f.stem]
    votes = collections.defaultdict(collections.Counter)
    for f, ch, w, split in files:
        votes[w][split] += 1
    for f, ch, w, split in files:
        # dust specks on the scans would otherwise stretch the letter's bounding box
        yield despeckle(to_ink(f)), ch, w, votes[w].most_common(1)[0][0]
