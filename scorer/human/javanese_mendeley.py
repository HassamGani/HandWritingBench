ALPHABET = "javanese"
VARIANT = "carakan"
SOURCE = ("Indonesian Local Script Characters (Ihsan), Mendeley Data doi:10.17632/vfj32bpjsf.3, Jawa.zip; "
          "bare consonants from all_class/Vokal A plus the variations folder")
LICENSE = "CC BY 4.0"
INVERT = True  # items are background-subtracted ink maps

import re  # noqa: E402
import subprocess  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402
from ._scan import clean  # noqa: E402

DIR = DATA / "javanese_mendeley"
URL = "https://data.mendeley.com/public-files/datasets/vfj32bpjsf/files/8aa88926-486e-465b-a5fe-62e907488910/file_downloaded"
# dha = ꦝ (DDA) and tha = ꦛ (TTA), the retroflex pair in Javanese spelling
CHARS = {"ha": "ꦲ", "na": "ꦤ", "ca": "ꦕ", "ra": "ꦫ", "ka": "ꦏ", "da": "ꦢ", "ta": "ꦠ", "sa": "ꦱ", "wa": "ꦮ", "la": "ꦭ",
         "pa": "ꦥ", "dha": "ꦝ", "ja": "ꦗ", "ya": "ꦪ", "nya": "ꦚ", "ma": "ꦩ", "ga": "ꦒ", "ba": "ꦧ", "tha": "ꦛ", "nga": "ꦔ"}


def fetch():
    if (DIR / "Jawa").exists():
        return
    DIR.mkdir(parents=True, exist_ok=True)
    if not (DIR / "Jawa.zip").exists():
        subprocess.run(["curl", "-sSLfo", DIR / "Jawa.zip", URL], check=True)
    zipfile.ZipFile(DIR / "Jawa.zip").extractall(DIR)


def items():
    for d in sorted((DIR / "Jawa" / "all_class" / "Vokal A").iterdir()):
        for f in sorted(d.glob("*.*")):
            yield clean(f), CHARS[d.name], None, None
    # most variations/ files are "<sheet> (<slot>).jpg", slot = the letter's place in ha-na-ca-ra-ka
    # order, so the sheet number is one writer's form; ra/ and the "000" files have no sheet number
    for d in sorted((DIR / "Jawa" / "variations").iterdir()):
        for f in sorted(d.glob("*.*")):
            m = re.match(r"(\d+) \(\d+\)", f.name)
            yield clean(f), CHARS[d.name], f"v{m.group(1)}" if m and m.group(1) != "000" else None, None
