"""Build src/hwbench/data/positions.json: where each letter sits relative to the
baseline and top line in open fonts, and which same-shape case pairs the guide
lines can actually tell apart.

uv run --extra train python scripts/build_positions.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scorer"))

from gap_latin import latin_fonts  # noqa: E402
from train_alphabet import MANUAL_CASE, candidate_pairs, covered, fonts_for, split_fonts  # noqa: E402

from hwbench.alphabets import ALPHABETS  # noqa: E402
from hwbench.positions import FLOOR_SD  # noqa: E402
from hwbench.preprocess import normalize  # noqa: E402
from hwbench.tasks import SAME_SHAPE  # noqa: E402
from hwbench.trace import render, render_on_baseline  # noqa: E402

OUT = ROOT / "src" / "hwbench" / "data" / "positions.json"
PROFILE = {"frame": 160, "pen": 5, "out": 40}
# a case pair is judged strictly only if position alone splits it in fonts this often,
# and the two sit far enough apart (in top-line heights) that a careful writer can show it
STRICT_ACC = 0.9
STRICT_GAP = 0.2


def _font_boxes(args):
    # from rendered ink: PIL's getbbox clips the bottom at the baseline, so a floating י would sit on it
    path, chars = args
    out = {}
    for ch in chars:
        r = render_on_baseline(path, ch, 200)
        if r is not None:
            m, top = r
            out[ch] = (-top, -(top + m.shape[0]))  # top and bottom above the baseline, font px
    return out


def _shape(args):
    path, ch = args
    m = render(path, ch, 200)
    return None if m is None else normalize(m.astype(np.float32), **PROFILE).ravel()


def fonts_of(a):
    if a.key == "latin":
        fs = latin_fonts()[:80]
        return [{**f, "family": f["family"]} for f in fs]
    tr, va, _ = split_fonts(fonts_for(a.key), a.key)
    return tr + va  # test fonts stay unseen, the trace baseline draws them


def build(a, pool):
    chars = a.chars
    fonts = fonts_of(a)
    if not fonts:
        return None
    cov = [covered(f, chars) for f in fonts]
    boxes = pool.map(_font_boxes, [(f["local"], c) for f, c in zip(fonts, cov)])
    upper = [c for c in chars if c.isupper()]
    lower = [c for c in chars if c.islower()]
    vals, xhs = {c: [] for c in chars}, []
    for bx in boxes:
        if not bx:
            continue
        if a.cased:
            caps = [bx[c][0] for c in upper if c in bx]
            if not caps:
                continue
            ref = np.median(caps)
            low = [bx[c][0] for c in lower if c in bx]
            if low:
                xhs.append(np.percentile(low, 30) / ref)
        else:
            ref = np.median([v[0] for v in bx.values()])
        for c, (t, b) in bx.items():
            vals[c].append((t / ref, b / ref))
    classes = {}
    for c, v in vals.items():
        if v:
            v = np.array(v)
            classes[c] = [round(float(v[:, 0].mean()), 3), round(float(v[:, 0].std()), 3),
                          round(float(v[:, 1].mean()), 3), round(float(v[:, 1].std()), 3)]

    # same-shape case pairs (and kana), as in scorer/train_alphabet.py
    jobs = [(f["local"], ch) for f, cs in zip(fonts[:12], cov[:12]) for ch in cs]
    imgs = dict(zip(jobs, pool.map(_shape, jobs, chunksize=32)))
    per_font = {}
    for (path, ch), v in imgs.items():
        if v is not None:
            per_font.setdefault(path, {})[ch] = v

    def corr(x, y):
        cs = [np.corrcoef(d[x], d[y])[0, 1] for d in per_font.values() if x in d and y in d]
        return float(np.median(cs)) if cs else 0.0

    if a.key == "latin":
        case = [(c.upper(), c) for c in SAME_SHAPE] + [("I", "l")]
    elif a.key in MANUAL_CASE:
        case = [(c, c.lower()) for c in MANUAL_CASE[a.key]]
    else:
        case = [(x, y) for x, y in candidate_pairs(a) if corr(x, y) > 0.75]

    # which case pairs can the guide lines split? classify each font's glyphs by position alone
    loose = []
    for x, y in case:
        if x not in classes or y not in classes:
            loose.append([x, y])
            continue
        ok, n = _separable(vals, classes, x, y)
        gap = np.hypot(classes[x][0] - classes[y][0], classes[x][2] - classes[y][2])
        if n == 0 or ok / n < STRICT_ACC or gap < STRICT_GAP:
            loose.append(sorted([x, y]))
    return {"xh": round(float(np.median(xhs)), 3) if a.cased and xhs else None, "n_fonts": len(fonts),
            "classes": classes, "case_groups": _components(case), "loose": loose}


def _components(pairs):
    groups = []
    for x, y in pairs:
        hit = [g for g in groups if x in g or y in g]
        groups = [g for g in groups if g not in hit] + [set().union({x, y}, *hit)]
    return [sorted(g) for g in groups]


def _separable(vals, classes, x, y):
    def ll(v, c):
        mt, st, mb, sb = classes[c]
        st, sb = np.hypot(st, FLOOR_SD), np.hypot(sb, FLOOR_SD)
        return -0.5 * ((v[0] - mt) / st) ** 2 - np.log(st) - 0.5 * ((v[1] - mb) / sb) ** 2 - np.log(sb)

    ok = n = 0
    for c, o in ((x, y), (y, x)):
        for v in vals[c]:
            ok += ll(v, c) > ll(v, o)
            n += 1
    return ok, n


if __name__ == "__main__":
    keys = sys.argv[1:]
    data = json.loads(OUT.read_text()) if OUT.exists() and keys else {}
    with Pool() as pool:
        for a in ALPHABETS:
            if keys and a.key not in keys:
                continue
            r = build(a, pool)
            if r is None:
                print(f"{a.key}: no fonts")
                continue
            data[a.key] = r
            print(f"{a.key:<14} fonts {r['n_fonts']:<3} xh {r['xh']}  loose {[''.join(p) for p in r['loose']]}", flush=True)
    OUT.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1))
