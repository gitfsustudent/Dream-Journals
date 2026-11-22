# CIS 5930 - Truncation check: re-tag dreams over 180 words in pieces of <=180 words (Colab, GPU runtime)
# Input: dreams_tagged_v2.csv (from retag_chunks.py). Output: dreams_tagged_v3.csv
import os
import numpy as np
import pandas as pd
import dreamy

MAXW = 180
LABELS = {"AN": "anger", "AP": "apprehension", "CO": "confusion", "SD": "sadness", "HA": "happiness"}
emo = ["emo_" + n for n in LABELS.values()]
import shutil
shutil.copy("/kaggle/input/dreams-v2/dreams_tagged_v2.csv", "dreams_tagged_v2.csv")
d = pd.read_csv("dreams_tagged_v2.csv", parse_dates=["date", "date_est"])
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print("Device:", device)

def split_text(text, maxw=MAXW):
    w = str(text).split(); n = max(1, int(np.ceil(len(w) / maxw))); size = int(np.ceil(len(w) / n))
    return [" ".join(w[i * size:(i + 1) * size]) for i in range(n)]

idx = d.index[d.keep_25 & (d.words > MAXW)]
pieces, owner, wts = [], [], []
for i in idx:
    for ch in split_text(d.at[i, "text"]):
        pieces.append(ch); owner.append(i); wts.append(len(ch.split()))
print(f"{len(idx)} dreams over {MAXW} words -> {len(pieces)} pieces to tag")

out = []
for i in range(0, len(pieces), 250):
    out += dreamy.annotate_reports(pieces[i:i + 250], task="SA", device=device, batch_size=16)
    if (i // 250) % 4 == 3: print(f"  tagged {min(i + 250, len(pieces))}/{len(pieces)}")
sc = pd.DataFrame([{"emo_" + LABELS[p["label"]]: p["score"] for p in pred} for pred in out])[emo]
sc["owner"], sc["w"] = owner, wts
for e in emo: sc[e + "_w"] = sc[e] * sc["w"]
agg = sc.groupby("owner")[[e + "_w" for e in emo]].sum().div(sc.groupby("owner")["w"].sum(), axis=0)
agg.columns = emo
nch = sc.groupby("owner").size()

for e in emo:
    d[e + "_v2"] = d[e]
    d.loc[agg.index, e] = agg[e]
d["n_chunks_v2"] = d["n_chunks"]
d.loc[nch.index, "n_chunks"] = nch
d["emo_max"] = d[emo].max(axis=1)

# ---------- comparison ----------
d["band"] = pd.cut(d.words, [180, 280, 400, 5000], labels=["181-280 (was 1 piece)", "281-400 (was 2 pieces)", "400+ (was 2-4 pieces)"])
print("\nPer band: mean score v2 -> v3, and correlation (dreams over 180 words)")
rows = []
for b, g in d.loc[idx].groupby("band", observed=True):
    for e in emo:
        rows.append({"band": b, "emotion": e[4:], "n": len(g), "mean_v2": g[e + "_v2"].mean(), "mean_v3": g[e].mean(),
                     "mean_abs_change": (g[e] - g[e + "_v2"]).abs().mean(), "corr": g[e].corr(g[e + "_v2"])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nHow to read: if scores for 181-280-word dreams barely change (corr high, means equal), the old tagging was NOT truncating.")
d.drop(columns=["band"]).to_csv("dreams_tagged_v3.csv", index=False)
print("Saved dreams_tagged_v3.csv")
try:
    from google.colab import files
    files.download("dreams_tagged_v3.csv")
except ImportError:
    pass
