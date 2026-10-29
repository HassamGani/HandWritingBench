ALPHABET = "arabic"
VARIANT = "isolated"
SOURCE = ("AHCD, Arabic Handwritten Characters Dataset, kaggle: mloey1/ahcd1 "
          "(El-Sawy, Loey, El-Bakry 2017, WSEAS Trans. Computer Research 5)")
LICENSE = "ODbL 1.0 (Open Database License, per the Kaggle page)"
INVERT = True

import re  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/mloey1/ahcd1"
HOME = DATA / "arabic_ahcd"
DIR = HOME / "raw"
LETTERS = "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي".split()


def fetch():
    if DIR.exists():
        return
    zp = HOME / "ahcd1.zip"
    if not zp.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(DIR)


def items():
    # 60 writers, the official train/test split is by writer (48/12), but the files carry no writer id
    for split, sub in (("train", "Train Images 13440x32x32/train"), ("test", "Test Images 3360x32x32/test")):
        for f in sorted((DIR / sub).glob("*.png")):
            k = int(re.search(r"label_(\d+)", f.name).group(1))
            yield f, LETTERS[k - 1], None, split
