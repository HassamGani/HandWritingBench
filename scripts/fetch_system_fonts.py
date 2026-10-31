"""Copy extra fonts from Microsoft Word and macOS for alphabets with few open fonts.

uv run python scripts/fetch_system_fonts.py            # scan, copy, write fonts/system.json
uv run python scripts/fetch_system_fonts.py --locked   # copy exactly what fonts/system.json lists

These fonts aren't ours to redistribute, so they're only copied into data/fonts/ (gitignored)
and never uploaded. fonts/system.json records family, source and sha256 so a rebuild on a
machine with the same Word/macOS fonts is checkable. Without them the trainer just uses the
open fonts in fonts/lock.json.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from fontTools.ttLib import TTFont

from hwbench.alphabets import ALPHABETS

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "data" / "fonts"
OUT = ROOT / "fonts" / "system.json"
SOURCES = {
    "Microsoft Word": [Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")],
    "macOS": [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path("/System/Library/AssetsV2")],
}
# only alphabets this thin on open fonts get system fonts
MAX_OPEN = 3
# Chinese system fonts that also carry Mongolian: all three draw it as the same tiny design
SKIP = {"Heiti TC", "STHeiti", "STFangsong"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(name):
    return name.lower().replace(" ", "")


def scan():
    found = []
    for src, dirs in SOURCES.items():
        for d in dirs:
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                    continue
                try:
                    # the trainer and PIL read face 0 of a collection, so that's the only face used
                    f = TTFont(p, fontNumber=0, lazy=True)
                    fam = f["name"].getDebugName(16) or f["name"].getDebugName(1)
                    cmap = set(f.getBestCmap() or {})
                except Exception:
                    continue
                if fam and not fam.startswith(".") and "LastResort" not in fam and fam not in SKIP:
                    found.append((src, fam, p, cmap))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    a = ap.parse_args()
    if a.locked:
        data = json.loads(OUT.read_text())
    else:
        lock = json.loads((ROOT / "fonts" / "lock.json").read_text())["alphabets"]
        fonts = scan()
        data = {"alphabets": {}}
        for alpha in ALPHABETS:
            have = lock.get(alpha.key, [])
            if alpha.key == "latin" or len(have) > MAX_OPEN:
                continue
            names = {norm(f["family"]) for f in have}
            cps = [ord(c) for c in alpha.chars]
            picked = {}
            for src, fam, p, cmap in fonts:
                n = norm(fam)
                if n in picked or any(n.startswith(h) or h.startswith(n) for h in names):
                    continue
                cov = sum(c in cmap for c in cps) / len(cps)
                if cov >= 0.9:
                    picked[n] = {"family": fam, "source": src, "original": str(p), "file": p.name,
                                 "sha256": sha(p), "coverage": round(cov, 3)}
            if picked:
                data["alphabets"][alpha.key] = sorted(picked.values(), key=lambda f: f["family"])
    for key, fs in data["alphabets"].items():
        (FONTS / key).mkdir(parents=True, exist_ok=True)
        for f in fs:
            src = Path(f["original"])
            if not src.exists() or sha(src) != f["sha256"]:
                print(f"{key}: {f['family']} missing or changed at {src}, skipped")
                continue
            shutil.copy(src, FONTS / key / f["file"])
        print(f"{key:<22} + {', '.join(f['family'] for f in fs)}")
    if not a.locked:
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
