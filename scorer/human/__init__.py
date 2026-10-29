"""Human handwriting datasets, used to train and test the per-alphabet scorers.

Each module here is one dataset and defines:
  ALPHABET  benchmark alphabet key ("greek")
  VARIANT   task variant the images belong to ("print")
  SOURCE    where it comes from (URL / hf id / kaggle ref, plus citation)
  LICENSE   as stated by the source
  INVERT    optional, True if the images are light ink on a dark background
  fetch()   download into data/human/<module name>/ if it isn't there yet
  items()   yields (image, char, writer, split): a file path, PIL image or 2-D array of one letter,
            the benchmark character it shows, a writer id (or None), and "train"/"test"
            if the source has an official split (else None)

Splits are by writer wherever writers are known, so the human test numbers measure
reading new people's handwriting, not new samples from people seen in training.
"""

import hashlib
import importlib
import pkgutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "human"
CACHE = ROOT / "data" / "cache" / "human"
TEST_SHARE = 0.2


def modules():
    return [importlib.import_module(f"{__name__}.{m.name}") for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")]


def to_ink(img, invert: bool = False) -> np.ndarray:
    """Any letter image -> ink map in [0, 1] (1 = ink), big enough to skeletonize."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if isinstance(img, np.ndarray):
        a = img.astype(np.float32)
        a = a / (a.max() or 1)
        img = Image.fromarray((a * 255).astype(np.uint8))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img)
    g = np.asarray(img.convert("L"), np.float32) / 255
    ink = g if invert else 1 - g
    # stretch contrast: scans and photos rarely have pure black ink on pure white
    lo, hi = np.percentile(ink, 1), np.percentile(ink, 99.5)
    ink = np.clip((ink - lo) / max(hi - lo, 1e-3), 0, 1)
    h, w = ink.shape
    if max(h, w) < 112:
        k = 112 / max(h, w)
        ink = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize((round(w * k), round(h * k)), Image.BILINEAR), np.float32) / 255
    return ink


def split_of(writer, official, key) -> str:
    if official in ("train", "test"):
        return official
    h = int(hashlib.sha256(str(writer if writer is not None else key).encode()).hexdigest(), 16)
    return "test" if (h % 1000) < TEST_SHARE * 1000 else "train"


def _norm(args):
    from hwbench.preprocess import normalize

    img, invert, profile = args
    out = normalize(to_ink(img, invert), **profile)
    return None if out is None else (out * 255).round().astype(np.uint8)


def load(mod, profile: dict) -> dict:
    """Normalized images for one dataset, cached per profile."""
    from multiprocessing import Pool

    name = mod.__name__.rsplit(".", 1)[-1]
    cache = CACHE / f"{name}-{profile['frame']}-{profile['pen']}-{profile['out']}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    mod.fetch()
    imgs, chars, writers, splits = [], [], [], []
    for i, (img, ch, writer, split) in enumerate(mod.items()):
        imgs.append(img)
        chars.append(ch)
        writers.append("" if writer is None else str(writer))
        splits.append(split_of(writer, split, f"{name}:{i}"))
    invert = getattr(mod, "INVERT", False)
    with Pool() as p:
        xs = p.map(_norm, [(im, invert, profile) for im in imgs], chunksize=64)
    keep = [i for i, x in enumerate(xs) if x is not None]
    out = {"x": np.stack([xs[i] for i in keep]), "char": np.array([chars[i] for i in keep]),
           "writer": np.array([writers[i] for i in keep]), "split": np.array([splits[i] for i in keep])}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def for_alphabet(alphabet: str, variant: str, profile: dict) -> list[tuple[str, dict, dict]]:
    """(dataset name, info, data) for every dataset of this alphabet/variant."""
    out = []
    for m in modules():
        if m.ALPHABET == alphabet and getattr(m, "VARIANT", "print") == variant:
            info = {"name": m.__name__.rsplit(".", 1)[-1], "source": m.SOURCE, "license": m.LICENSE}
            out.append((info["name"], info, load(m, profile)))
    return out
