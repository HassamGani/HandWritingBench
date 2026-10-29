ALPHABET = "oriya"
VARIANT = "print"
SOURCE = "Odia Handwritten Character Dataset, github: Swatimahapatra07/Odia-Handwritten-Character-Dataset"
LICENSE = "none stated in the GitHub repo"

import re  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

from . import DATA  # noqa: E402

URL = "https://codeload.github.com/Swatimahapatra07/Odia-Handwritten-Character-Dataset/zip/refs/heads/main"
HOME = DATA / "odia_swati"
DIR = HOME / "raw" / "Odia-Handwritten-Character-Dataset-main"
# folders are named by the letter. The form has no ଠ; its "୦" folder holds plain circles that could be
# ଠ or a zero, so it's skipped along with the conjuncts, ୟ and ୠ.
LETTERS = set("ଅଆଇଈଉଊଋଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଡଢଣତଥଦଧନପଫବଭମଯରଲଳଶଷସହ")


def fetch():
    if DIR.exists():
        return
    zp = HOME / "repo.zip"
    if not zp.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zp)
    zipfile.ZipFile(zp).extractall(HOME / "raw")


def items():
    for d in sorted(DIR.iterdir()):
        if d.name not in LETTERS:
            continue
        for f in sorted(d.glob("*.png")):
            # "doc (N)", "doc(N)" and "page_N" share one form number N across folders (same hand
            # checked by eye). Folders saved under scanner-app names (କ-ଞ, ଟ) have no usable form number.
            m = re.fullmatch(r"(?:doc ?\(|page_)(\d+)\)?", f.stem)
            yield f, d.name, int(m.group(1)) if m else None, None
