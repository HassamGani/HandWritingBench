"""Clean-up shared by the scanned letter sets (Mendeley Indonesian scripts, UTYCC Burmese)."""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops


def ink_map(img, median: bool = True) -> np.ndarray:
    """Scan or photo -> ink darkness over a smoothed paper background (removes shading)."""
    im = Image.open(img) if not isinstance(img, Image.Image) else img
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
    g = np.asarray(im.convert("L"), np.float32)
    bg = ndimage.gaussian_filter(ndimage.maximum_filter(g, 15), 5)
    ink = np.clip(bg - g, 0, 255)
    return ndimage.median_filter(ink, 3) if median else ink


def clean(img, band: int = 3, median: bool = True) -> np.ndarray:
    """Letter scan -> ink map (bright ink on black) without shading, specks, frame or ruled lines."""
    ink = ink_map(img, median)
    h, w = ink.shape
    side = max(h, w)
    lab = label(ink > 0.35 * max(np.percentile(ink, 99.5), 1))
    keep = []
    for p in regionprops(lab):
        y0, x0, y1, x1 = p.bbox
        # frame edges, ruled lines and rotation wedges are long thin slivers near the edge
        sliver = max(y1 - y0, x1 - x0) > 0.25 * side and min(y1 - y0, x1 - x0) < 0.12 * side
        near = min(y0, x0, h - y1, w - x1) < 0.15 * side
        if p.area >= 0.0006 * h * w and not (sliver and near):
            keep.append(p.label)
    out = ink * ndimage.binary_dilation(np.isin(lab, keep))
    out[:band], out[-band:], out[:, :band], out[:, -band:] = 0, 0, 0, 0
    out = Image.fromarray(out.astype(np.uint8))
    out.thumbnail((160, 160))  # a whole set is held in memory by load()
    return np.asarray(out)


def despeckle(ink: np.ndarray, small: float = 0.05, tiny: float = 0.01) -> np.ndarray:
    """[0, 1] ink map -> uint8 crop of the letter, minus dots under `tiny` of all ink anywhere and
    specks under `small` of it away from the letter. Thresholds as normalize() will after to_ink's
    contrast stretch."""
    lo, hi = np.percentile(ink, 1), np.percentile(ink, 99.5)
    ink = np.clip((ink - lo) / max(hi - lo, 1e-3), 0, 1)
    lab = label(ink > 0.35)
    parts = regionprops(lab)
    if not parts:
        return (ink * 255).astype(np.uint8)
    total = sum(p.area for p in parts)
    big = [p for p in parts if p.area >= small * total]
    m = 0.1 * max(ink.shape)
    y0, x0 = min(p.bbox[0] for p in big) - m, min(p.bbox[1] for p in big) - m
    y1, x1 = max(p.bbox[2] for p in big) + m, max(p.bbox[3] for p in big) + m
    keep = [p.label for p in parts if p in big or (p.area >= tiny * total and p.bbox[0] >= y0 and p.bbox[1] >= x0
                                                   and p.bbox[2] <= y1 and p.bbox[3] <= x1)]
    # only the kept pixels: faint background left in would get stretched up again by to_ink
    out = ink * np.isin(lab, keep)
    ys, xs = np.nonzero(out)
    return (out[max(ys.min() - 4, 0):ys.max() + 5, max(xs.min() - 4, 0):xs.max() + 5] * 255).astype(np.uint8)
