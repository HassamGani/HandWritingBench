ALPHABET = "greek"
VARIANT = "print"
SOURCE = ("GCDB, Greek Character Database (Vamvakas et al. 2008, isolated handwritten Greek letters), "
          "kaggle: vrushalipatel/handwritten-greek-characters-from-gcdb")
LICENSE = "ODC-By (as stated on Kaggle)"

import urllib.request  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/vrushalipatel/handwritten-greek-characters-from-gcdb"
DIR = DATA / "greek_gcdb"
NAMES = dict(zip("ALPHA BETA GAMMA DELTA EPSILON ZETA HETA THETA IOTA KAPA LAMDA MI NI KSI OMIKRON PII RO SIGMA TAU YPSILON FI XI PSI OMEGA".split(),
                 range(24)))
UPPER, LOWER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ", "αβγδεζηθικλμνξοπρστυφχψω"  # no final sigma in GCDB


def fetch():
    if (DIR / "raw" / "Query").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    zp = DIR / "greek_gcdb.zip"
    if not zp.exists():
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(DIR / "raw")


def clean(f):
    # each crop carries bits of the form's dotted cell borders along its edges, which would wreck
    # the bounding box: drop thin pieces lying in the outer band, and specks
    ink = np.asarray(Image.open(f).convert("L")) < 128
    h, w = ink.shape
    lab = label(ink, connectivity=2)
    regions = regionprops(lab)
    big = max((r.area for r in regions), default=0)
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        edge = y0 >= h - 8 or y1 <= 8 or x0 >= w - 8 or x1 <= 8
        thin = min(y1 - y0, x1 - x0) <= 3
        if r.area <= 3 or r.area < 0.08 * big or (edge and thin):
            ink[lab == r.label] = False
    return np.where(ink, 0, 255).astype(np.uint8)


def items():
    # folders are LETT_{CAP,SML}_{NORM,SUFF}.<name>; NORM and SUFF are just two collection rounds,
    # both kept. form_N is the scanned sheet, one writer per sheet.
    for d in sorted((DIR / "raw" / "Query").iterdir()):
        case, name = d.name.split("_")[1], d.name.split(".")[1]
        ch = (UPPER if case == "CAP" else LOWER)[NAMES[name]]
        for f in sorted(d.glob("*.bmp")):
            yield clean(f), ch, int(f.stem.split("_")[1]), None
