ALPHABET = "japanese"
VARIANT = "katakana"
SOURCE = ("ETL Character Database, ETL1 katakana part (1411 writers), AIST/ETL; kaggle mirror: "
          "nhtminhhongnguyn/etlcdb, files ETLCDB/ETLCDB/ETL1/ETL1_7 .. ETL1_13")
LICENSE = "AIST ETL database terms, research use only (the Kaggle mirror's MIT tag is the uploader's, not AIST's)"
INVERT = True

import subprocess  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402

from . import DATA, to_ink  # noqa: E402
from ._scan import despeckle  # noqa: E402

DIR = DATA / "japanese_etl1"
URL = "https://www.kaggle.com/api/v1/datasets/download/nhtminhhongnguyn/etlcdb?file_name=ETLCDB%2FETLCDB%2FETL1%2F"
FILES = [f"ETL1_{i}" for i in range(7, 14)]
KATAKANA = set("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (DIR / f).exists():
            subprocess.run(["curl", "-sSLfo", DIR / f, URL + f], check=True)


def items():
    # same M-type records as ETL7. The sheets have the 51-slot kana grid, so イ ウ エ come twice
    # per writer (the yi/wu/ye slots) and are kept; ヰ ヱ (coded as small ィ ェ) are dropped
    for f in FILES:
        b = (DIR / f).read_bytes()
        for i in range(0, len(b), 2052):
            r = b[i:i + 2052]
            kana = unicodedata.normalize("NFKC", bytes([r[6]]).decode("cp932"))
            if kana not in KATAKANA:
                continue
            px = np.frombuffer(r[32:2048], np.uint8)
            img = np.stack([px >> 4, px & 15], 1).reshape(63, 64)
            # the scans carry specks and bits of the neighbouring cells
            yield despeckle(to_ink(img, True)), kana, int.from_bytes(r[4:6], "big"), None
