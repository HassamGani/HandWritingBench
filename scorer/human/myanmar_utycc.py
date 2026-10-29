ALPHABET = "myanmar"
VARIANT = "print"
SOURCE = ("UTYCC Burmese handwritten character dataset (BurmeseCharacterDataSets, one glyph per 310x219 photo), "
          "Google Drive id 1JbD-xMQmcq6lT5gDjE5WiwJJ13ywe7J6")
LICENSE = "CC BY-SA 4.0"
INVERT = True

import hashlib  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402

from . import DATA  # noqa: E402
from ._scan import ink_map  # noqa: E402

DIR = DATA / "myanmar_utycc"
URL = "https://drive.usercontent.google.com/download?id=1JbD-xMQmcq6lT5gDjE5WiwJJ13ywe7J6&export=download&confirm=t"
CHARS = {"ka_kji": "က", "kha_khway": "ခ", "ga_nge": "ဂ", "ga_khi": "ဃ", "nga": "င", "sah_lone": "စ", "hsa_lain": "ဆ",
         "za_kwear": "ဇ", "za_myin_hsware": "ဈ", "nya_kyi": "ည", "ta_thun_lyin_chate": "ဋ", "hta_wun_beare": "ဌ",
         "da_yin_kout": "ဍ", "da_yay_hmote": "ဎ", "na_kji": "ဏ", "ta_wun_pu": "တ", "hta_hsin_htu": "ထ", "da_htway": "ဒ",
         "da_out_chite": "ဓ", "na_ngear": "န", "pa_sout": "ပ", "pfa_u_htoat": "ဖ", "ba_htoat_chite": "ဗ", "ba_kone": "ဘ",
         "ma": "မ", "yah_pet_let": "ယ", "yah_kout": "ရ", "la": "လ", "wa": "ဝ", "tha": "သ", "ha": "ဟ", "la_kji": "ဠ", "Ah": "အ"}


def fetch():
    if (DIR / "img").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    if not (DIR / "utycc.rar").exists():
        subprocess.run(["curl", "-sSLfo", DIR / "utycc.rar", URL], check=True)
    # a rar of per-class rars (plus one big rar repeating all of them)
    (DIR / "outer").mkdir(exist_ok=True)
    (DIR / "img").mkdir()
    subprocess.run(["bsdtar", "-xf", DIR / "utycc.rar", "-C", DIR / "outer"], check=True)
    for k in CHARS:
        subprocess.run(["bsdtar", "-xf", DIR / "outer" / "BurmeseCharacterDataSets" / f"{k}.rar", "-C", DIR / "img"], check=True)


def crop(f):
    # small pen glyph on a photo of paper: subtract a smoothed background, keep the biggest
    # dark blob plus whatever lies near it, and cut that out. Paper edges and folds (long thin
    # lines, or anything touching the photo's edge) are skipped
    ink = ink_map(f, median=False)
    lab = label(ink > 50)
    h, w = ink.shape

    def junk(p):
        y0, x0, y1, x1 = p.bbox
        return min(y0, x0, h - y1, w - x1) < 3 or (max(y1 - y0, x1 - x0) > 60 and min(y1 - y0, x1 - x0) < 12)

    parts = [p for p in regionprops(lab) if p.area >= 20 and not junk(p)]
    if not parts:
        return ink.astype(np.uint8)
    y0, x0, y1, x1 = max(parts, key=lambda p: p.area).bbox
    keep = [p for p in parts if p.bbox[0] < y1 + 25 and p.bbox[2] > y0 - 25 and p.bbox[1] < x1 + 25 and p.bbox[3] > x0 - 25]
    y0, x0 = min(p.bbox[0] for p in keep), min(p.bbox[1] for p in keep)
    y1, x1 = max(p.bbox[2] for p in keep), max(p.bbox[3] for p in keep)
    mask = ndimage.binary_dilation(np.isin(lab, [p.label for p in keep]), iterations=2)
    return (ink * mask)[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4].astype(np.uint8)


def items():
    # files are <n><class>.jpg and <n><class>_up.jpg; n numbers the contributor (pen and paper
    # stay the same across classes), so it's the writer id. Some classes were padded with
    # byte-identical copies under other numbers; only the first of each is kept.
    for k, ch in CHARS.items():
        seen = set()
        for f in sorted((DIR / "img" / k).glob("*.jpg"), key=lambda f: (len(f.name), f.name)):
            h = hashlib.md5(f.read_bytes()).digest()
            if h not in seen:
                seen.add(h)
                yield crop(f), ch, int(re.match(r"\d+", f.name).group()), None
