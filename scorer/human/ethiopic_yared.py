ALPHABET = "ethiopic"
VARIANT = "base"
SOURCE = "Ge'ez handwritten characters (OCR_dataset.zip), hf: Yaredoffice/geez-characters"
LICENSE = "Apache-2.0"

import urllib.request  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://huggingface.co/datasets/Yaredoffice/geez-characters/resolve/main/OCR_dataset.zip"
DIR = DATA / "ethiopic_yared"
# folders 0-33 are the first-order (base) forms; the rest are other vowel orders and punctuation
BASE = "ሀለሐመሠረሰሸቀበቨተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ"


def fetch():
    if (DIR / "raw" / "OCR_dataset").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    zp = DIR / "OCR_dataset.zip"
    if not zp.exists():
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(DIR / "raw")


def clean(f):
    # 25x25 scans, many on grey speckled paper: binarize with Otsu and drop specks much smaller
    # than the main stroke, else the speckle gets skeletonized into dots and shrinks the letter
    im = Image.open(f).convert("RGBA")
    ink = 255 - np.asarray(Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("L"), np.float32)
    mask = ink > threshold_otsu(ink)
    lab = label(mask, connectivity=2)
    big = max((r.area for r in regionprops(lab)), default=0)
    for r in regionprops(lab):
        if r.area < 0.15 * big:
            mask[lab == r.label] = False
    return np.where(mask, 0, 255).astype(np.uint8)


def items():
    # official split is writer-disjoint (48 train writers, 5 test); file names are <class>_<writer>.
    # train_form/test_form hold the same images again, so only the numbered folders are read
    for split in ("train", "test"):
        for i, ch in enumerate(BASE):
            for f in sorted((DIR / "raw" / "OCR_dataset" / split / str(i)).glob("*.png")):
                yield clean(f), ch, f.stem.split("_")[1], split
