ALPHABET = "tifinagh"
VARIANT = "print"
SOURCE = ("AMHCD, Amazigh Handwritten Character Database (Es-Saady et al. 2011, IRCAM Tifinagh, 60 writers), "
          "kaggle: benaddym/amazigh-handwritten-character-database-amhcd")
LICENSE = "not stated on Kaggle; academic database, cite Es-Saady et al. 2011"

import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/benaddym/amazigh-handwritten-character-database-amhcd"
DIR = DATA / "tifinagh_amhcd"
# romanized folder names -> letters, from the dataset's own label list (checked by eye);
# yagw and yakw are labialized ⴳⵯ ⴽⵯ, not in our inventory
NAMES = {"ya": "ⴰ", "yab": "ⴱ", "yach": "ⵛ", "yad": "ⴷ", "yadd": "ⴹ", "yae": "ⵄ", "yaf": "ⴼ", "yag": "ⴳ",
         "yagh": "ⵖ", "yah": "ⵀ", "yahh": "ⵃ", "yaj": "ⵊ", "yak": "ⴽ", "yal": "ⵍ", "yam": "ⵎ", "yan": "ⵏ",
         "yaq": "ⵇ", "yar": "ⵔ", "yarr": "ⵕ", "yas": "ⵙ", "yass": "ⵚ", "yat": "ⵜ", "yatt": "ⵟ", "yaw": "ⵡ",
         "yax": "ⵅ", "yay": "ⵢ", "yaz": "ⵣ", "yazz": "ⵥ", "yey": "ⴻ", "yi": "ⵉ", "yu": "ⵓ"}


def fetch():
    if (DIR / "raw" / "AMHCD_64").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    zp = DIR / "tifinagh_amhcd.zip"
    if not zp.exists():
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(DIR / "raw")


def items():
    for name, ch in NAMES.items():
        for f in sorted((DIR / "raw" / "AMHCD_64" / "AMHCD_64" / name).glob("*.jpeg")):
            # <name>_1..780: 60 writers x 13 samples in order (style changes every 13 files,
            # aligned at 1, in every class), so (n-1)//13 is the writer
            yield f, ch, (int(f.stem.split("_")[1]) - 1) // 13, None
