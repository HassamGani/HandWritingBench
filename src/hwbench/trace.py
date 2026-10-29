"""Turn a font glyph into pen strokes: render, skeletonize, walk the skeleton graph."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

SIZE = 320
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@lru_cache(maxsize=64)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


def render(path: str, ch: str, size: int = SIZE) -> np.ndarray | None:
    """Glyph as a boolean mask, cropped to ink."""
    f = _font(str(path), size)
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, size), ch, font=f, fill=255)
    m = np.asarray(img) > 127
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _rdp(p: np.ndarray, eps: float) -> np.ndarray:
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.hypot(*ab)
    d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n if n > 0 else np.hypot(*(p - a).T)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([_rdp(p[: i + 1], eps)[:-1], _rdp(p[i:], eps)])
    return np.array([a, b])


def skeleton_paths(mask: np.ndarray, eps: float = 1.5, min_spur: float = 0.06) -> list[np.ndarray]:
    """Skeleton -> list of (x, y) polylines. Short spurs from skeletonization are dropped."""
    sk = skeletonize(np.pad(mask, 2))
    pix = set(zip(*np.nonzero(sk)))
    if not pix:
        return []

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in NB if (p[0] + dy, p[1] + dx) in pix]

    deg = {p: len(nbrs(p)) for p in pix}
    nodes = {p for p, k in deg.items() if k != 2}
    seen = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        seen.add(frozenset((a, b)))
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in nbrs(cur) if q != prev and frozenset((cur, q)) not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(frozenset((prev, cur)))
            path.append(cur)
            if cur == a:
                break
        return path

    for n in nodes:
        if deg[n] == 0:
            paths.append([n])
        for m in nbrs(n):
            if frozenset((n, m)) not in seen:
                paths.append(walk(n, m))
    for p in pix:  # loops with no junctions (O, 0)
        for m in nbrs(p):
            if frozenset((p, m)) not in seen:
                nodes.add(p)
                paths.append(walk(p, m))

    size = max(mask.shape)
    out = []
    for path in paths:
        a = np.array([(x - 2, y - 2) for y, x in path], dtype=float)
        length = np.hypot(*np.diff(a, axis=0).T).sum() if len(a) > 1 else 0
        ends = [tuple(path[0]), tuple(path[-1])]
        is_spur = any(deg[e] == 1 for e in ends) and any(deg[e] > 2 for e in ends)
        if is_spur and length < min_spur * size:
            continue
        out.append(_rdp(a, eps))
    return _join(out)


def _join(paths, tol=2.5):
    """Chain polylines that meet end to end, so a pen would draw them in one go."""
    paths = [p for p in paths if len(p)]
    merged = True
    while merged:
        merged = False
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j or len(paths[i]) < 2 or len(paths[j]) < 2:
                    continue
                if np.hypot(*(paths[i][-1] - paths[j][0])) <= tol:
                    paths[i] = np.vstack([paths[i], paths[j][1:]])
                    paths.pop(j)
                    merged = True
                    break
                if np.hypot(*(paths[i][-1] - paths[j][-1])) <= tol:
                    paths[i] = np.vstack([paths[i], paths[j][::-1][1:]])
                    paths.pop(j)
                    merged = True
                    break
            if merged:
                break
    return paths


def glyph_strokes(path: str, ch: str) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    """Strokes in glyph pixel coords plus the glyph (w, h)."""
    m = render(path, ch)
    if m is None:
        return None
    return skeleton_paths(m), (m.shape[1], m.shape[0])
