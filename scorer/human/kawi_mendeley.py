ALPHABET = "kawi"
VARIANT = "print"
SOURCE = "Indonesian Local Script Characters (Ihsan), Mendeley Data doi:10.17632/vfj32bpjsf.3, Kawi.zip"
LICENSE = "CC BY 4.0"
INVERT = True  # items are background-subtracted ink maps

import subprocess  # noqa: E402
import unicodedata  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

from . import DATA, to_ink  # noqa: E402
from ._scan import clean  # noqa: E402

DIR = DATA / "kawi_mendeley"
URL = "https://data.mendeley.com/public-files/datasets/vfj32bpjsf/files/33331ef3-499e-4e1d-8ed1-e18aa0a74699/file_downloaded"
# folders use IAST: ṭa ḍa ṇa are the retroflex row, tha dha the dental aspirates
CHARS = {"ka": "𑼒", "kha": "𑼓", "ga": "𑼔", "gha": "𑼕", "ṅa": "𑼖", "ca": "𑼗", "cha": "𑼘", "ja": "𑼙", "jha": "𑼚", "ña": "𑼛",
         "ṭa": "𑼜", "ḍa": "𑼞", "ṇa": "𑼠", "ta": "𑼡", "tha": "𑼢", "da": "𑼣", "dha": "𑼤", "na": "𑼥", "pa": "𑼦", "pha": "𑼧",
         "ba": "𑼨", "bha": "𑼩", "ma": "𑼪", "ya": "𑼫", "ra": "𑼬", "la": "𑼭", "wa": "𑼮", "śa": "𑼯", "ṣa": "𑼰", "sa": "𑼱", "ha": "𑼲"}


def fetch():
    if (DIR / "Kawi").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    if not (DIR / "Kawi.zip").exists():
        subprocess.run(["curl", "-sSLfo", DIR / "Kawi.zip", URL], check=True)
    zipfile.ZipFile(DIR / "Kawi.zip").extractall(DIR)


def items():
    from hwbench.preprocess import normalize

    # most images were rotated with black fill, leaving dark wedges at the edges (dropped by clean);
    # strokes are thin and faint, so no median filter
    rows = [(clean(f, 5, median=False), CHARS[unicodedata.normalize("NFC", d.name)])
            for d in sorted((DIR / "Kawi").iterdir()) for f in sorted(d.glob("*.*"))]
    # about 40% are rotated/zoomed/blurred copies of another image. Copies of one drawing (near
    # identical once normalized) share a group id, used as the writer so they stay in one split
    sig = [normalize(to_ink(x, True), frame=160, pen=5, out=40) for x, _ in rows]
    root = list(range(len(rows)))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i

    for ch in set(CHARS.values()):
        idx = [i for i, (_, c) in enumerate(rows) if c == ch and sig[i] is not None]
        a = np.stack([sig[i].ravel() for i in idx])
        sq = (a * a).sum(1)
        for i, j in zip(*np.nonzero(sq[:, None] + sq[None] - 2 * a @ a.T < 5)):
            root[find(idx[i])] = find(idx[j])
    for i, (x, ch) in enumerate(rows):
        yield x, ch, f"g{find(i)}", None
