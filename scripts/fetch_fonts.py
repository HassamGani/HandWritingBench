"""Download open-licensed training fonts for every alphabet from google/fonts.

uv run python scripts/fetch_fonts.py            # resolve + download, writes fonts/lock.json
uv run python scripts/fetch_fonts.py --locked   # re-download exactly what fonts/lock.json lists

Fonts land in data/fonts/ (gitignored). The lock file pins the google/fonts commit
and every file's sha256, so the scorer training set is reproducible.
"""

import argparse
import hashlib
import json
import re
import ssl
import subprocess
import urllib.request
from pathlib import Path

import certifi
from fontTools.ttLib import TTFont

from hwbench.alphabets import ALPHABETS
from hwbench.trace import render

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "data" / "fonts"
LOCK = ROOT / "fonts" / "lock.json"
META_URL = "https://fonts.google.com/metadata/fonts"
MAX_FAMILIES = 28
MAX_FAMILIES_CJK = 10
SKIP_CATEGORIES = {"Symbols"}
CTX = ssl.create_default_context(cafile=certifi.where())
# decorative families whose glyphs aren't plain letterforms
SKIP_FAMILIES = re.compile(r"(Flag|Barcode|Blocks|Icons|Emoji|Bitmap|Pixel|Outline|Shadow|Stencil|Mono Color|Dots|Wavy)")


def gh(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def dirname(family):
    return re.sub(r"[^a-z0-9]", "", family.lower())


def pick_file(files):
    ttf = [f for f in files if f["name"].endswith(".ttf") and "italic" not in f["name"].lower()]
    for pat in (r"-Regular\.ttf$", r"\[.*\]\.ttf$", r"\.ttf$"):
        hit = [f for f in ttf if re.search(pat, f["name"])]
        if hit:
            return sorted(hit, key=lambda f: len(f["name"]))[0]
    return None


def coverage(path, chars):
    try:
        cmap = TTFont(path, fontNumber=0, lazy=True).getBestCmap()
    except Exception:
        return 0.0
    return sum(ord(c) in cmap for c in chars) / len(chars)


def distinct_glyphs(path, chars):
    """Share of letters that render to a distinct image; catches fonts whose cmap lists a
    script but draw the same placeholder for every letter."""
    seen = set()
    for ch in chars:
        try:
            m = render(str(path), ch, 64)
        except OSError:
            continue
        if m is not None:
            seen.add(hashlib.md5(m.tobytes() + bytes(m.shape)).digest())
    return len(seen) / len(chars)


def download(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120, context=CTX) as r:
        dst.write_bytes(r.read())


def resolve(only_new=False):
    meta = json.loads(urllib.request.urlopen(META_URL, timeout=60, context=CTX).read())["familyMetadataList"]
    if only_new and LOCK.exists():
        lock = json.loads(LOCK.read_text())  # keep the pinned commit and existing picks
        sha = lock["google_fonts_commit"]
    else:
        sha = gh("repos/google/fonts/commits/main")["sha"]
        lock = {"google_fonts_commit": sha, "alphabets": {}}
    for a in ALPHABETS:
        if a.key == "latin" or (only_new and a.key in lock["alphabets"]):
            continue
        fams = [m for m in meta if set(m.get("subsets", [])) & set(a.gf) and m.get("category") not in SKIP_CATEGORIES
                and not SKIP_FAMILIES.search(m["family"])]
        fams.sort(key=lambda m: m.get("defaultSort", 10**6))
        cap = MAX_FAMILIES_CJK if a.key in ("japanese", "hangul") else MAX_FAMILIES
        got = []
        for m in fams:
            if len(got) >= cap:
                break
            files = None
            for lic in ("ofl", "apache", "ufl"):
                files = gh(f"repos/google/fonts/contents/{lic}/{dirname(m['family'])}?ref={sha}")
                if isinstance(files, list):
                    break
            f = pick_file(files) if isinstance(files, list) else None
            if not f:
                continue
            dst = FONTS / a.key / f["name"]
            if not dst.exists():
                try:
                    download(f"https://raw.githubusercontent.com/google/fonts/{sha}/{f['path']}", dst)
                except Exception as e:
                    print("  fail", m["family"], e)
                    continue
            cov = coverage(dst, a.chars)
            if cov < 0.9 or distinct_glyphs(dst, a.chars) < 0.9:
                dst.unlink()
                continue
            got.append({"family": m["family"], "category": m.get("category"), "path": f["path"], "file": f["name"],
                        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "coverage": round(cov, 3)})
        lock["alphabets"][a.key] = got
        print(f"{a.key:<11} {len(got):>2} families  ({len(fams)} candidates)", flush=True)
        LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False))


def fetch_locked():
    lock = json.loads(LOCK.read_text())
    sha = lock["google_fonts_commit"]
    for key, fonts in lock["alphabets"].items():
        for f in fonts:
            dst = FONTS / key / f["file"]
            if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest() != f["sha256"]:
                download(f"https://raw.githubusercontent.com/google/fonts/{sha}/{f['path']}", dst)
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == f["sha256"], dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--only-new", action="store_true", help="add alphabets missing from the lock, same commit")
    a = ap.parse_args()
    fetch_locked() if a.locked else resolve(a.only_new)
