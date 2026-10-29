"""Train the letter classifier used as the legibility scorer.

uv run --extra train python scorer/train.py
Needs EMNIST byclass idx files in data/emnist (hf download Heliosoph/EMNIST).
"""

import gzip
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import synth
from hwbench.preprocess import emnist_to_ink, normalize
from hwbench.tasks import CLASSES, accept_set

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/emnist"
CACHE = ROOT / "data/cache"
OUT = ROOT / "models"
NAME = "scorer-v0.2"
SEED = 0
EPOCHS = 8
BS = 448  # EMNIST per batch
BS_SYN = 64  # synthetic print letters per batch
SYN_PER_GLYPH = {"train": 40, "val": 5, "test": 1}


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _norm(raw):
    out = normalize(emnist_to_ink(raw))
    return np.zeros((28, 28), np.float32) if out is None else out


def load_split(split):
    cache = CACHE / f"byclass_letters_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    imgs = read_idx(RAW / f"emnist-byclass-{split}-images-idx3-ubyte.gz")
    labels = read_idx(RAW / f"emnist-byclass-{split}-labels-idx1-ubyte.gz").astype(np.int64)
    keep = labels >= 10  # drop digits; 10-35 = A-Z, 36-61 = a-z, same order as CLASSES
    imgs, labels = imgs[keep], labels[keep] - 10
    t = time.time()
    with Pool() as p:
        x = np.stack(p.map(_norm, imgs, chunksize=2048))
    print(f"normalized {split}: {len(x)} in {time.time() - t:.0f}s")
    x = (x * 255).round().astype(np.uint8)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=labels)
    return x, labels


def load_synth(split):
    cache = CACHE / f"synth_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"], list(z["fonts"])
    x, y, fonts = synth.build(split, SYN_PER_GLYPH[split], seed={"train": 1, "val": 2, "test": 3}[split])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y, fonts=np.array(fonts))
    return x, y, fonts


class Net(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.f = nn.Sequential(block(1, 48), block(48, 96), block(96, 192))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(192 * 3 * 3, 384), nn.ReLU(), nn.Dropout(0.3), nn.Linear(384, n))

    def forward(self, x):
        return self.head(self.f(x))


def augment(x):
    # small random affine per sample: rotation, scale, shear, shift
    n = x.shape[0]
    d = x.device
    ang = (torch.rand(n, device=d) - 0.5) * 2 * np.deg2rad(12)
    sc = 1 + (torch.rand(n, device=d) - 0.5) * 0.25
    sh = (torch.rand(n, device=d) - 0.5) * 0.4
    tx, ty = [(torch.rand(n, device=d) - 0.5) * 0.15 for _ in range(2)]
    c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([c, -s + sh, tx], 1), torch.stack([s, c, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.no_grad()
def predict(model, x, dev):
    model.eval()
    out = []
    for i in range(0, len(x), 4096):
        xb = torch.from_numpy(x[i : i + 4096]).float().div(255).unsqueeze(1).to(dev)
        out.append(F.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out)


def report(probs, y):
    pred = probs.argmax(1)
    acc_strict = float((pred == y).mean())
    acc_sets = np.array([CLASSES[p] in accept_set(CLASSES[t]) for p, t in zip(pred, y)])
    per = {CLASSES[k]: float(acc_sets[y == k].mean()) for k in range(len(CLASSES))}
    # balanced = mean over classes, so frequent letters like e don't dominate
    return {"top1_strict": acc_strict, "top1_accept": float(acc_sets.mean()), "balanced_accept": float(np.mean(list(per.values()))), "per_class_accept": per}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    x, y = load_split("train")
    xt, yt = load_split("test")
    idx = rng.permutation(len(x))
    nval = len(x) // 20
    xv, yv = x[idx[:nval]], y[idx[:nval]]
    x, y = x[idx[nval:]], y[idx[nval:]]
    xs, ys, fonts_train = load_synth("train")
    xsv, ysv, fonts_val = load_synth("val")
    xst, yst, fonts_test = load_synth("test")
    print(f"synth train {len(xs)} ({len(fonts_train)} fonts), val {len(xsv)} ({len(fonts_val)}), test {len(xst)} ({len(fonts_test)})")

    # balanced sampling, otherwise the scorer learns a prior that favours e/t/a
    counts = np.bincount(y, minlength=len(CLASSES))
    w = torch.from_numpy(1.0 / counts[y]).double()
    steps = len(y) // BS

    xg, yg = torch.from_numpy(x), torch.from_numpy(y)
    xsg, ysg = torch.from_numpy(xs), torch.from_numpy(ys)
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * steps)

    best, best_state, best_ep = -1, None, -1
    for ep in range(EPOCHS):
        model.train()
        t = time.time()
        sel = torch.multinomial(w, steps * BS, replacement=True)
        tot = 0.0
        for i in range(steps):
            b = sel[i * BS : (i + 1) * BS]
            bs = torch.randint(len(ys), (BS_SYN,))
            xb = torch.cat([xg[b], xsg[bs]]).float().div(255).unsqueeze(1).to(dev)
            yb = torch.cat([yg[b], ysg[bs]]).to(dev)
            loss = F.cross_entropy(model(augment(xb)), yb, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        r = report(predict(model, xv, dev), yv)
        rs = report(predict(model, xsv, dev), ysv)
        sel_metric = (r["balanced_accept"] + rs["balanced_accept"]) / 2
        print(f"ep {ep} loss {tot / steps:.3f} emnist-val balanced {r['balanced_accept']:.4f}  font-val balanced {rs['balanced_accept']:.4f} ({time.time() - t:.0f}s)", flush=True)
        if sel_metric > best:
            best, best_ep = sel_metric, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = report(predict(model, xt, dev), yt)
    font_val = report(predict(model, xsv, dev), ysv)
    hershey = report(predict(model, xst, dev), yst)
    print(f"EMNIST test accept {test['top1_accept']:.4f} balanced {test['balanced_accept']:.4f}")
    print(f"font val balanced {font_val['balanced_accept']:.4f}   hershey sans (held out) accept {hershey['top1_accept']:.4f}")
    worst = sorted(test["per_class_accept"].items(), key=lambda a: a[1])[:6]
    print("weakest EMNIST classes", [(k, round(v, 3)) for k, v in worst])
    print("hershey misses", [CLASSES[k] for k in range(len(CLASSES)) if hershey["per_class_accept"][CLASSES[k]] < 1])

    OUT.mkdir(exist_ok=True)
    model = model.cpu().eval()
    onnx_path = OUT / f"{NAME}.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), onnx_path, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    meta = {
        "name": NAME,
        "classes": CLASSES,
        "input": "1x28x28 float, ink=1, from hwbench.preprocess.normalize",
        "train_data": "EMNIST byclass letters (train minus 5% val) + synthetic print letters (scorer/synth.py), split by font family",
        "train_files_sha256": {f.name: sha(f) for f in sorted(RAW.glob("emnist-byclass-*.gz"))},
        "synth_fonts": {"train": synth.font_hashes(fonts_train) | {f: "hershey" for f in synth.HERSHEY_TRAIN},
                        "val": synth.font_hashes(fonts_val), "test": synth.HERSHEY_TEST},
        "epochs": EPOCHS,
        "best_epoch": best_ep,
        "seed": SEED,
        "selection": "mean of EMNIST-val and font-val balanced accuracy",
        "test": test,
        "font_val": {k: font_val[k] for k in ("top1_accept", "balanced_accept")},
        "hershey_sans_test": hershey,
        "onnx_sha256": sha(onnx_path),
    }
    (OUT / f"{NAME}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
