# Shared backbone experiment (2026-10-30)

Idea: scripts with one or two fonts can't learn from their own data what changes between font designs, so pretrain one network on every non-Latin alphabet at once and fine-tune each scorer from it. `scorer/pretrain_backbone.py` trains it on each alphabet's train fonts and train writers only (827,729 samples, 5,444 classes, 8 epochs), and `HWB_INIT=data/cache/backbone.pt` makes `train_alphabet.py` start from it. 70 alphabets were fine-tuned this way: every tier A/B script plus the ones with human data.

Per alphabet, the backbone model replaced v0.4.1 only when it beat it by at least 0.1 point on validation (mean of val fonts and val writers where there are writers). 37 of 70 passed.

Validation looked good: mean 95.3% → 95.9% over the 70. The baselines didn't hold up:

| | v0.4.1 | backbone picks |
|---|---|---|
| trace, mean of 138 | 97.82% | 97.84% |
| scribble acceptance, mean | 3.09% | 3.10% |
| wrong_letter | 0.12% | 0.10% |

Trace went up on Balinese (87→89), Brahmi (93→96), Coptic (96→100), Georgian (97→100), Imperial Aramaic (64→68), Lisu, Lycian and Tibetan (83→87). It went down on Hebrew (93→85), Sinhala (98→92), Gothic (96→92), Glagolitic (98→96), Carian and Old Turkic. Scribble acceptance went up on Cyrillic, Devanagari, Gujarati, Oriya and Imperial Aramaic. On human test writers it was mostly small gains (Oriya +1.7, Arabic +0.8, Lao +0.7, Javanese +0.7), with Ethiopic −0.6.

Why: for font-only scripts, validation is one or two fonts, so a 0.1 to 1 point validation gain is a letter or two, which is noise. Selecting on that just swaps which letters fail. Averaging the v0.4.1 and backbone models (log-prob mean) matched the backbone alone on validation (95.9%) and only helped the human-data scripts (Greek, Oriya and Japanese +0.25 to 0.4 on val writers).

Not released. v0.4.1 stays. Full diff in `runs/backbone-bldiff.txt` (local).
