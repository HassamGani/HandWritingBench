"""Where letters sit on the guide lines, and how that splits same-shape letters.

The shape classifier only sees the letter cropped and scaled, so c/C or Hebrew ו/ן
look the same to it. The canvas has guide lines (baseline, top line, and a dashed
lowercase line for cased scripts), and fonts tell us where each letter's top and
bottom should be relative to them, so the shape probability gets weighted by how
well the ink's top and bottom fit each letter.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

GUIDE_TOP = 160  # canvas px; the top line is cap height (cased) or typical letter height
GUIDE_BASE = 400
CANVAS = 560
MARGIN = 20
# how sloppy we allow placement to be, as a fraction of the top-to-baseline height
FLOOR_SD = 0.1

_PATH = Path(__file__).parent / "data" / "positions.json"


@cache
def table() -> dict:
    return json.loads(_PATH.read_text()) if _PATH.exists() else {}


def guides(alphabet: str) -> dict | None:
    a = table().get(alphabet)
    if a is None:
        return None
    top, base = GUIDE_TOP, GUIDE_BASE
    hi, lo, w = a.get("span", [0, 0, 0])
    h, room = base - top, CANVAS - 2 * MARGIN
    if base - hi * h < MARGIN or base - lo * h > CANVAS - MARGIN or w * h > room:
        # some letters are much taller, deeper or wider than the usual height (Mandaic, Tai Tham, ...),
        # so tighten the lines until every letter fits and center the whole range
        h = min(h, room / (hi - lo), room / w)
        base = round(MARGIN + hi * h + (room - (hi - lo) * h) / 2, 1)
        top = round(base - h, 1)
    g = {"top": top, "base": base}
    if a.get("xh"):
        g["mid"] = round(base - a["xh"] * (base - top), 1)
    return g


def ink_features(strokes, ink: float, g: dict) -> tuple[float, float]:
    """Top and bottom of the ink, in units of top-line height above the baseline."""
    ys = np.concatenate([np.asarray(s, float)[:, 1] for s in strokes])
    h = g["base"] - g["top"]
    return float((g["base"] - (ys.min() - ink / 2)) / h), float((g["base"] - (ys.max() + ink / 2)) / h)


def reweight(p: np.ndarray, classes: list[str], alphabet: str, feats: tuple[float, float]) -> np.ndarray:
    """p(letter | shape, position) ~ p(letter | shape) * p(position | letter). Letters with
    clearly different shapes barely move. Same-shape case pairs are split by position
    alone: the classifier's preference between them comes from font quirks, not shape."""
    a = table().get(alphabet)
    if a is None:
        return p
    idx = {c: i for i, c in enumerate(classes)}
    p = p.copy()
    for g in a.get("case_groups", []):
        ix = [idx[c] for c in g if c in idx]
        p[ix] = p[ix].mean()
    t, b = feats
    ll = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if c in a["classes"]:
            mt, st, mb, sb = a["classes"][c]
            st, sb = np.hypot(st, FLOOR_SD), np.hypot(sb, FLOOR_SD)
            ll[i] = -0.5 * ((t - mt) / st) ** 2 - np.log(st) - 0.5 * ((b - mb) / sb) ** 2 - np.log(sb)
        else:
            ll[i] = np.nan  # e.g. the "not a letter" class: position says nothing about it
    if np.isnan(ll).all():
        return p
    ll[np.isnan(ll)] = np.nanmax(ll)
    w = p * np.exp(ll - ll.max())
    return w / w.sum() if w.sum() > 0 else p
