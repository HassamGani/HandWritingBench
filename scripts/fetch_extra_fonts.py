"""Download free fonts that aren't on Google Fonts, for alphabets that are still weak.

uv run python scripts/fetch_extra_fonts.py

Sources are below with the license each one states. Files land in data/fonts/<alphabet>/
(gitignored); fonts.extra.json records url, archive member and sha256 for each.
"""

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "data" / "fonts"
CACHE = ROOT / "data" / "cache" / "extra_fonts"
OUT = ROOT / "fonts.extra.json"

GNU = "https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip"
QUIVIRA = "http://www.quivira-font.com/files/Quivira.otf"
DAMASE = "https://deb.debian.org/debian/pool/main/f/fonts-mph-2b-damase/fonts-mph-2b-damase_001.000.dfsg.2+ds1-7_all.deb"
CULMUS = "https://sourceforge.net/projects/culmus/files/ancient_fonts/AncientSemiticFonts-0.06-1/AncientSemiticFonts-0.06-1.TTF.tgz/download"
NENAD = "http://nenad.bplaced.net/lib/exe/fetch.php/"
BALI = "https://raw.githubusercontent.com/longnow/bali-fonts/master/release/"
LG = "https://www.languagegeek.com/font/"
DOUROS = "https://deb.debian.org/debian/pool/main/t/ttf-ancient-fonts/fonts-ancient-scripts_2.60-5_all.deb"
NAU = "https://apagreekkeys.org/fonts/NAU5_008.zip"
DEJAVU = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-sans-ttf-2.37.zip"
BABEL = "https://www.babelstone.co.uk/Fonts/Download/"
SHAV = "https://2gd4.me/tidbit/shavian-fonts/"

# (family, url, member inside the archive or None, license)
SOURCES = {
    "glagolitic": [
        ("VeleBit", NENAD + ":velebit-regular_13.03.2025.zip", "VeleBit-Regular 13.03.2025.ttf", "CC BY-SA 3.0 DE"),
        ("Glagolica Missal DPG", NENAD + ":glagolica_missal_dpg.zip", "Glagolica Missal DPG.ttf", "CC BY-SA 3.0 DE"),
        ("Vinodolski Zakon", NENAD + ":vinodolski_zakon_23112015_hp.zip", "Vinodolski Zakon_23112015_signiert.ttf", "CC BY-SA 3.0 DE"),
        ("Dilyana", "https://www.fontspace.com/get/family/zm8po", "Dilyana-n3pJ.ttf", "freeware, non-commercial"),
        ("Quivira", QUIVIRA, None, "public domain"),
        ("MPH 2B Damase", DAMASE, "damase.ttf", "public domain"),
        ("FreeSerif", GNU, "FreeSerif.ttf", "GPL-3.0 with font exception"),
    ],
    "cherokee": [
        ("Tsulehisanvhi", "https://language.cherokee.gov/media/x0lp0ruq/tsulehisanvhi.ttf", None, "not stated (Cherokee Nation)"),
        ("Donisiladv", "https://language.cherokee.gov/media/zzcb3sf3/donisiladv.ttf", None, "not stated (Cherokee Nation)"),
        ("Digohweli", LG + "Digohweli.zip", "Digohweli_1_7.ttf", "freeware"),
        ("Aboriginal Sans", LG + "AboriginalSans.zip", "AboriginalSansREGULAR.ttf", "free"),
        ("Aboriginal Serif", LG + "AboriginalSerif.zip", "AboriginalSerifREGULAR.ttf", "free"),
        ("FreeSerif", GNU, "FreeSerif.ttf", "GPL-3.0 with font exception"),
        ("Quivira", QUIVIRA, None, "public domain"),
        ("MPH 2B Damase", DAMASE, "damase.ttf", "public domain"),
    ],
    "phoenician": [
        ("Phoenician Ahiram", CULMUS, "Phoenician-Ahiram.ttf", "GPL-2.0"),
        ("Hebrew Paleo Mesha", CULMUS, "Hebrew-Paleo-Mesha.ttf", "GPL-2.0"),
        ("Hebrew Paleo Siloam", CULMUS, "Hebrew-Paleo-Siloam.ttf", "GPL-2.0"),
        ("Hebrew Paleo Gezer", CULMUS, "Hebrew-Paleo-Gezer.ttf", "GPL-2.0"),
        ("Hebrew Paleo Lachish", CULMUS, "Hebrew-Paleo-Lachish.ttf", "GPL-2.0"),
        ("Hebrew Paleo Qumran", CULMUS, "Hebrew-Paleo-Qumran.ttf", "GPL-2.0"),
        ("Aramaic Early Br Rkb", CULMUS, "Aramaic-Early-Br-Rkb.ttf", "GPL-2.0"),
        ("FreeSans", GNU, "FreeSans.ttf", "GPL-3.0 with font exception"),
        ("Quivira", QUIVIRA, None, "public domain"),
        ("MPH 2B Damase", DAMASE, "damase.ttf", "public domain"),
    ],
    "balinese": [
        ("Vimala", BALI + "Vimala.ttf", None, "OFL-1.1"),
        ("Pustaka Bali", BALI + "PustakaBali.ttf", None, "OFL-1.1"),
        ("Kadiri", BALI + "Kadiri.ttf", None, "OFL-1.1"),
    ],
    "kayahli": [
        ("FreeMono", GNU, "FreeMono.ttf", "GPL-3.0 with font exception"),
    ],
    # BabelStone and DejaVu draw the vowels as notches across the stem, like people write them;
    # Noto, Segoe and Apple Symbols draw diamonds, which skeletonize to a plain stem line
    "ogham": [
        ("BabelStone Ogham", BABEL + "BabelStoneOghamR.ttf", None, "OFL-1.1"),
        ("BabelStone Ogham Bound", BABEL + "BabelStoneOghamBoundR.ttf", None, "OFL-1.1"),
        ("BabelStone Ogham Fixed", BABEL + "BabelStoneOghamFixed.ttf", None, "OFL-1.1"),
        ("BabelStone Pictish Ogham", BABEL + "BabelStoneOghamPictishR.ttf", None, "OFL-1.1"),
        ("DejaVu Sans", DEJAVU, "DejaVuSans.ttf", "Bitstream Vera / DejaVu"),
    ],
    "old-italic": [
        ("Italica Vetus", "https://scholarsfonts.net/ItalicaVetus_1-201.zip", "Italica Vetus 1-201.otf", "OFL-1.1"),
        ("Aegean", DOUROS, "Aegean_hint.ttf", "free for any use (Debian fonts-ancient-scripts 2.60)"),
        ("New Athena Unicode", NAU, "newathu5_8.ttf", "OFL-1.1"),
        ("DejaVu Sans", DEJAVU, "DejaVuSans.ttf", "Bitstream Vera / DejaVu"),
        ("FreeMono", GNU, "FreeMono.ttf", "GPL-3.0 with font exception"),
    ],
    "coptic": [
        ("Antinoou", "https://evertype.com/fonts/coptic/AntinoouFont.zip", "Antinoou.ttf", "freeware, not to be altered or sold"),
        ("IFAO-Grec Unicode", "https://www.ifao.egnet.net/uploads/polices/IFAOGrecUnicode-v3.zip", "IFAOGrec.ttf", "free, royalty-free (IFAO)"),
        ("Analecta", DOUROS, "Analecta_hint.ttf", "free for any use (Debian fonts-ancient-scripts 2.60)"),
        ("Nilus", DOUROS, "Nilus_hint.ttf", "free for any use (Debian fonts-ancient-scripts 2.60)"),
        ("New Athena Unicode", NAU, "newathu5_8.ttf", "OFL-1.1"),
        ("Fairfax HD", "https://www.kreativekorp.com/swdownload/fonts/core/fairfaxhd.zip", "FairfaxHD.ttf", "OFL-1.1"),
        ("Quivira", QUIVIRA, None, "public domain"),
        ("FreeSerif", GNU, "FreeSerif.ttf", "GPL-3.0 with font exception"),
    ],
    # Marin (Ogham), Aramaic VIIBCE (7th-century BCE forms, not Imperial) and Damase's Old Italic
    # (mirrored, right-to-left Etruscan forms) were tried and left out
    "imperial-aramaic": [
        ("Aramaic Imperial Yeb", CULMUS, "Aramaic-Imperial-Yeb.ttf", "GPL-2.0"),
        ("Quivira", QUIVIRA, None, "public domain"),
    ],
    "meroitic-cursive": [
        ("Nilus", DOUROS, "Nilus_hint.ttf", "free for any use (Debian fonts-ancient-scripts 2.60)"),
    ],
    # outline designs only (the pixel-style Shavian fonts are skipped)
    "shavian": [
        ("Inter Alia", "https://raw.githubusercontent.com/Shavian-info/interalia/master/instance_otf/InterAlia-Regular.otf", None, "OFL-1.1"),
        ("Doolittle Garamond", "https://raw.githubusercontent.com/SpaceLatvija/Shavian-kettu-font-pack/main/DoolittleGaramond-Regular.otf", None, "OFL-1.1"),
        ("Andagii", "http://www.i18nguy.com/unicode/andagii.zip", "ANDAGII_.TTF", "freeware"),
        ("Salsa Tequila", SHAV + "SalsaTequila.otf", None, "OFL-1.1"),
        ("Orelega One Shavian", SHAV + "orelega-one.ttf", None, "OFL-1.1"),
        ("Playwrite Shavian", SHAV + "playwrite.otf", None, "OFL-1.1"),
        ("Space Mono Shavian", SHAV + "space-mono-shavian-r.otf", None, "OFL-1.1"),
        ("Audio Nugget Shavian", SHAV + "audio-nugget.ttf", None, "OFL-1.1"),
        ("Americium Shavian", SHAV + "americium-shavian.otf", None, "not stated"),
        ("Banter Shavian", SHAV + "bantershavianunicase.otf", None, "not stated"),
        ("Consistency Shavian", SHAV + "consistency-shavian.ttf", None, "not stated"),
        ("Impetus Shavian", SHAV + "impetusshavian.ttf", None, "not stated"),
        ("Couth Shavian", SHAV + "couth43shavian.ttf", None, "not stated"),
        ("Iotacism Shavian", SHAV + "iotacism43shavian.otf", None, "not stated"),
    ],
}


def fetch(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / hashlib.sha1(url.encode()).hexdigest()
    if not p.exists():
        # curl, since a couple of hosts refuse urllib (languagegeek 406, sourceforge 403)
        subprocess.run(["curl", "-sSfL", "-o", str(p), url], check=True, timeout=300)
    return p.read_bytes()


def member(blob, name):
    if blob[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return z.read(next(n for n in z.namelist() if Path(n).name == name))
    if blob[:8] == b"!<arch>\n":  # .deb: an ar archive holding data.tar.*
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.deb").write_bytes(blob)
            subprocess.run(["ar", "x", "x.deb"], cwd=d, check=True)
            data = next(Path(d).glob("data.tar.*")).read_bytes()
        return member(data, name)
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        return t.extractfile(next(m for m in t.getmembers() if Path(m.name).name == name)).read()


def main():
    out = {"alphabets": {}}
    for key, fonts in SOURCES.items():
        (FONTS / key).mkdir(parents=True, exist_ok=True)
        for fam, url, name, lic in fonts:
            try:
                blob = fetch(url)
                data = member(blob, name) if name else blob
            except Exception as e:
                print(f"{key}: {fam} failed ({e})")
                continue
            file = name or Path(url).name
            (FONTS / key / file).write_bytes(data)
            out["alphabets"].setdefault(key, []).append({"family": fam, "url": url, "member": name, "file": file,
                                                         "license": lic, "sha256": hashlib.sha256(data).hexdigest()})
            print(f"{key:<11} {fam}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
