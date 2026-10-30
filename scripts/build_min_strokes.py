"""Build src/hwbench/data/min_strokes.json: the fewest strokes each letter can be drawn in
without retracing (hwbench.continuity.min_strokes on font skeletons).

uv run --extra train python scripts/build_min_strokes.py

Skeleton spurs at serifs and sharp corners only ever add strokes, so fonts are rendered at their
lightest weight, sans first, and each letter takes a low percentile over fonts rather than the
median. A single odd font that joins two parts can't pull it under either.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scorer"))

from gap_latin import latin_fonts  # noqa: E402
from train_alphabet import covered, fonts_for  # noqa: E402

from hwbench.alphabets import ALPHABETS  # noqa: E402
from hwbench.continuity import min_strokes  # noqa: E402

OUT = ROOT / "src" / "hwbench" / "data" / "min_strokes.json"
SIZE = 320
MAX_FONTS = 16
PCT = 20


def _light(path):
    f = ImageFont.truetype(path, SIZE)
    try:
        axes = f.get_variation_axes()
        f.set_variation_by_axes([a["minimum"] if a.get("name", b"") in (b"Weight", "Weight") else a["default"] for a in axes])
    except (OSError, AttributeError):
        pass  # not a variable font
    return f


def _counts(args):
    path, chars = args
    f = _light(path)
    out = {}
    for ch in chars:
        img = Image.new("L", (SIZE * 3, SIZE * 3), 0)
        ImageDraw.Draw(img).text((SIZE, SIZE), ch, font=f, fill=255)
        m = np.asarray(img) > 127
        if m.any():
            ys, xs = np.nonzero(m)
            out[ch] = min_strokes(m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    return out


def fonts_of(a):
    if a.key == "latin":
        return latin_fonts()[:MAX_FONTS * 2]
    fs = fonts_for(a.key)
    return sorted(fs, key=lambda f: f.get("category") != "Sans Serif")


def main():
    table = json.loads(OUT.read_text()) if OUT.exists() and len(sys.argv) > 1 else {}
    keys = sys.argv[1:] or [a.key for a in ALPHABETS]
    with Pool(10) as pool:
        for a in ALPHABETS:
            if a.key not in keys:
                continue
            chars = "".join(dict.fromkeys(c for v in a.variants if not v.style for c in v.chars))
            fonts = [(f["local"], covered(f, chars)) for f in fonts_of(a)]
            fonts = [(p, c) for p, c in fonts if c][:MAX_FONTS]
            if not fonts:
                continue
            per = {}
            for counts in pool.map(_counts, fonts):
                for ch, k in counts.items():
                    per.setdefault(ch, []).append(k)
            table[a.key] = {ch: int(np.percentile(v, PCT, method="lower")) for ch, v in per.items()}
            print(f"{a.key:<22} {len(fonts):>2} fonts  mean {np.mean(list(table[a.key].values())):.2f}", flush=True)
    OUT.write_text(json.dumps(table, ensure_ascii=False, indent=0, sort_keys=True))


if __name__ == "__main__":
    main()
