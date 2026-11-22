# CIS 5930 - Re-tag long dreams in chunks so the whole dream is read (run in Colab, GPU runtime)
# Input: dreams_tagged.csv from tag_emotions.py. Output: dreams_tagged_v2.csv
# Dreams over MAXW words are split into equal chunks; chunk scores are averaged, weighted by chunk length.

!pip install -q dreamy
import os
import numpy as np
import pandas as pd
import dreamy

MAXW = 280      # words per chunk (about 400 tokens, safely under a 512-token limit)
THRESH = 0.5
LABELS = {"AN": "anger", "AP": "apprehension", "CO": "confusion", "SD": "sadness", "HA": "happiness"}
emo = ["emo_" + n for n in LABELS.values()]

if not os.path.exists("dreams_tagged.csv"):
    from google.colab import files
    files.upload()                      # choose dreams_tagged.csv
d = pd.read_csv("dreams_tagged.csv", parse_dates=["date", "date_est"])

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print("Device:", device)

def split_text(text, maxw=MAXW):
    w = str(text).split()
    n = max(1, int(np.ceil(len(w) / maxw)))
    size = int(np.ceil(len(w) / n))
    return [" ".join(w[i * size:(i + 1) * size]) for i in range(n)]

def tag(texts):
    out = []
    for i in range(0, len(texts), 250):
        out += dreamy.annotate_reports(texts[i:i + 250], task="SA", device=device, batch_size=16)
        print(f"  tagged {min(i + 250, len(texts))}/{len(texts)}")
    return pd.DataFrame([{"emo_" + LABELS[p["label"]]: p["score"] for p in pred} for pred in out])[emo]

# ---------- split the long dreams ----------
long_idx = d.index[d.keep_25 & (d.words > MAXW)]
pieces, owner, wts = [], [], []
for i in long_idx:
    for ch in split_text(d.at[i, "text"]):
        pieces.append(ch); owner.append(i); wts.append(len(ch.split()))
print(f"{len(long_idx)} dreams over {MAXW} words -> {len(pieces)} chunks to tag")

sc = tag(pieces)
sc["owner"], sc["w"] = owner, wts
for e in emo:
    sc[e + "_w"] = sc[e] * sc["w"]
tot = sc.groupby("owner")["w"].sum()
agg = sc.groupby("owner")[[e + "_w" for e in emo]].sum().div(tot, axis=0)
agg.columns = emo
nch = sc.groupby("owner").size()

# ---------- replace scores for long dreams, keep the originals ----------
for e in emo:
    d[e + "_orig"] = d[e]
    d.loc[agg.index, e] = agg[e]
d["n_chunks"] = 1
d.loc[nch.index, "n_chunks"] = nch
d["emo_max"] = d[emo].max(axis=1)
d["emo_dominant"] = pd.Series([None] * len(d), index=d.index, dtype="object")
tagged = d.keep_25
top = d.loc[tagged, emo].idxmax(axis=1).str.replace("emo_", "")
d.loc[tagged, "emo_dominant"] = np.where(d.loc[tagged, "emo_max"] >= THRESH, top, "none")
for e in emo:
    d[e.replace("emo_", "has_")] = d[e] >= THRESH
d["has_any"] = d[[e.replace("emo_", "has_") for e in emo]].any(axis=1)

# ---------- how much did chunking change things? ----------
print("\nFor the long dreams: original (whole-dream) score vs chunked score")
rows = [{"emotion": e.replace("emo_", ""),
         "mean_original": d.loc[long_idx, e + "_orig"].mean(),
         "mean_chunked": d.loc[long_idx, e].mean(),
         "mean_abs_change": (d.loc[long_idx, e] - d.loc[long_idx, e + "_orig"]).abs().mean(),
         "correlation": d.loc[long_idx, e].corr(d.loc[long_idx, e + "_orig"])} for e in emo]
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nChunks per long dream:", nch.value_counts().sort_index().to_dict())

d.to_csv("dreams_tagged_v2.csv", index=False)
print("Saved dreams_tagged_v2.csv")
try:
    from google.colab import files
    files.download("dreams_tagged_v2.csv")
except ImportError:
    pass
