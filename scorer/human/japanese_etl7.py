ALPHABET = "japanese"
VARIANT = "hiragana"
SOURCE = ("ETL Character Database, ETL7L (hiragana, 175 writers), AIST/ETL; kaggle mirror: "
          "nhtminhhongnguyn/etlcdb, files ETLCDB/ETLCDB/ETL7/ETL7_1, ETL7_2")
LICENSE = "AIST ETL database terms, research use only (the Kaggle mirror's MIT tag is the uploader's, not AIST's)"
INVERT = True

import subprocess  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402

from . import DATA  # noqa: E402

DIR = DATA / "japanese_etl7"
URL = "https://www.kaggle.com/api/v1/datasets/download/nhtminhhongnguyn/etlcdb?file_name=ETLCDB%2FETLCDB%2FETL7%2F"
# ETL7_3/4 (ETL7S) hold the same samples as ETL7_1/2 (ETL7L), rescanned, so only L is used
FILES = ["ETL7_1", "ETL7_2"]
HIRAGANA = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (DIR / f).exists():
            subprocess.run(["curl", "-sSLfo", DIR / f, URL + f], check=True)


def items():
    # 2052-byte M-type records: sheet number at bytes 4-5, JIS X 0201 (half-width katakana)
    # code at byte 6 (mapped to hiragana), 64x63 4-bit image at bytes 32-2047, light ink on dark
    for f in FILES:
        b = (DIR / f).read_bytes()
        for i in range(0, len(b), 2052):
            r = b[i:i + 2052]
            kana = unicodedata.normalize("NFKC", bytes([r[6]]).decode("cp932"))
            hira = chr(ord(kana) - 0x60)
            if hira not in HIRAGANA:  # dakuten and handakuten marks
                continue
            px = np.frombuffer(r[32:2048], np.uint8)
            img = np.stack([px >> 4, px & 15], 1).reshape(63, 64)
            # every writer filled two consecutive sheets (1-2, 3-4, ...)
            writer = (int.from_bytes(r[4:6], "big") + 1) // 2
            yield img, hira, writer, None
