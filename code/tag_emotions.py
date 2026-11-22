# CIS 5930 - Emotion tagging with DReAMy (run in Google Colab, GPU runtime recommended)
# Tags the five Hall & Van de Castle emotions in each dream using the DReAMy annotator
# (Bertolini et al., 2024). Input: dreams_clean.csv from clean_dreams.py.
# Data: DreamBank.net, CC BY-NC-SA 4.0.

!pip install -q dreamy
import os, time
import numpy as np
import pandas as pd
import dreamy

THRESH = 0.5      # score at which a dream "has" an emotion (a choice we will test)
CHUNK = 250       # dreams per batch of work, so progress is visible
LABELS = {"AN": "anger", "AP": "apprehension", "CO": "confusion", "SD": "sadness", "HA": "happiness"}

# ---------- 1. Load the cleaned dreams ----------
if not os.path.exists("dreams_clean.csv"):
    from google.colab import files
    files.upload()                      # choose dreams_clean.csv
d = pd.read_csv("dreams_clean.csv", parse_dates=["date", "date_est"])

# Tag every dream that passes the 25-word check (the 50-word main set is a subset of these)
todo = d[d.keep_25].copy()
texts = todo["text"].fillna("").tolist()
print("Dreams to tag:", len(texts))

# ---------- 2. Run the annotator ----------
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print("Device:", device, "(switch Runtime > Change runtime type > T4 GPU if this says cpu)")

preds, t0 = [], time.time()
for i in range(0, len(texts), CHUNK):
    preds += dreamy.annotate_reports(texts[i:i + CHUNK], task="SA", device=device, batch_size=16)
    print(f"  tagged {min(i + CHUNK, len(texts))}/{len(texts)}  ({time.time() - t0:.0f}s)")

if not (isinstance(preds[0], list) and isinstance(preds[0][0], dict)):
    raise SystemExit(f"Unexpected output format, stopping. First prediction: {preds[0]}")
print("Example output:", preds[0])

# ---------- 3. Turn scores into columns (labels come sorted by score, so map by name) ----------
for code, name in LABELS.items():
    todo["emo_" + name] = [next(p["score"] for p in pred if p["label"] == code) for pred in preds]
emo_cols = ["emo_" + n for n in LABELS.values()]
todo["emo_max"] = todo[emo_cols].max(axis=1)
todo["emo_dominant"] = np.where(todo.emo_max >= THRESH,
                                todo[emo_cols].idxmax(axis=1).str.replace("emo_", ""), "none")
for c in emo_cols:
    todo[c.replace("emo_", "has_")] = todo[c] >= THRESH
has_cols = [c.replace("emo_", "has_") for c in emo_cols]
todo["has_any"] = todo[has_cols].any(axis=1)

out = d.merge(todo[["journal", "dream_no"] + emo_cols + ["emo_max", "emo_dominant"] + has_cols + ["has_any"]],
              on=["journal", "dream_no"], how="left")
out.to_csv("dreams_tagged.csv", index=False)
print("Saved dreams_tagged.csv")

# ---------- 4. Sanity checks ----------
main = out[out.keep]                                    # the 50-word main set
print("\n% of dreams with each emotion at threshold", THRESH, "(50-word set):")
print((main.groupby("journal")[has_cols + ["has_any"]].mean() * 100).round(1).to_string())

b = main[main.journal == "Barb"].copy()
b["set"] = np.where(b.dream_no <= 3114, "set1 (series b)", "set2 (series b2)")
print("\nBarb, first part vs later part. Domhoff's published figures (different method, rough comparison only):")
print(" set1: anger 17.5, apprehension 27.6, sadness 10.1, confusion 10.8, happiness 15.6, any 54.6")
print(" set2: anger 14.2, apprehension 28.1, sadness  8.6, confusion 13.6, happiness 20.2, any 56.6")
print((b.groupby("set")[has_cols + ["has_any"]].mean() * 100).round(1).to_string())

print("\nDoes the share of dreams with an emotion depend on dream length? (both journals, 25+ word set)")
t = out[out.keep_25].copy()
t["length"] = pd.cut(t.words, [24, 49, 100, 200, 400, 5000], labels=["25-49", "50-100", "101-200", "201-400", "400+"])
print((t.groupby(["journal", "length"], observed=True)["has_any"].agg(["mean", "size"]).round(3)).to_string())

print("\nThreshold sensitivity, share of 50-word dreams with any emotion:")
for th in [0.3, 0.5, 0.7]:
    print(" ", th, (main[emo_cols].max(axis=1) >= th).groupby(main.journal).mean().round(3).to_dict())

print("\nDreams over ~350 words (may exceed the model's input limit):", int((out.words > 350).sum()))
try:
    from google.colab import files
    files.download("dreams_tagged.csv")
except ImportError:
    pass
