"""Pretrain one network on every non-Latin alphabet at once, to initialize the per-alphabet scorers.

uv run --extra train python scorer/pretrain_backbone.py

Scripts with one or two fonts can't learn from their own data what varies between designs;
a network that has seen 137 scripts' worth of strokes, curves and loops can. Only each alphabet's
training fonts and training writers go in (the same splits train_alphabet.py uses), so validation
and test fonts/writers stay unseen and per-alphabet model selection stays honest.
"""

import time

import numpy as np

import train_alphabet as T
from hwbench.alphabets import ALPHABETS

CACHE = T.ROOT / "data" / "cache" / "backbone.npz"
OUT = T.ROOT / "data" / "cache" / "backbone.pt"
EPOCHS = 8
BS = 512
HUMAN_PER = 300  # per letter


def dataset():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["x"], z["y"], int(z["n"])
    xs, ys, base = [], [], 0
    rng = np.random.default_rng(0)
    for a in ALPHABETS:
        if a.key == "latin":
            continue
        t0 = time.time()
        fonts = T.fonts_for(a.key)
        if not fonts:
            continue
        tr, _, _ = T.split_fonts(fonts, a.key)
        x, y = T.build(tr, a.chars, max(4, min(60, 300 // len(tr))), 1.0, seed=11)
        hum, _ = T.human_data(a.key, a.chars)
        if "train" in hum:
            hx, hy = hum["train"]
            sel = np.concatenate([rng.permutation(np.where(hy == k)[0])[:HUMAN_PER] for k in np.unique(hy)])
            x, y = np.concatenate([x, hx[sel]]), np.concatenate([y, hy[sel]])
        xs.append(x)
        ys.append(y + base)
        base += len(a.chars)
        print(f"{a.key:<22} {len(x):>6} samples  ({time.time() - t0:.0f}s)", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, x=x, y=y, n=base)
    return x, y, base


def main():
    import torch
    import torch.nn.functional as F

    from train import Net, augment

    x, y, n = dataset()
    print(f"{len(x)} samples, {n} classes", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = Net(n, T.PROFILE["out"]).to(dev)
    xg, yg = torch.from_numpy(x), torch.from_numpy(y)
    steps = len(x) // BS
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * steps)
    for ep in range(EPOCHS):
        t0, tot = time.time(), 0.0
        model.train()
        perm = torch.randperm(len(y))
        for i in range(steps):
            b = perm[i * BS : (i + 1) * BS]
            xb = xg[b].float().div(255).unsqueeze(1).to(dev)
            loss = F.cross_entropy(model(augment(xb)), yg[b].to(dev), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep}  loss {tot / steps:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
