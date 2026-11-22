# CIS 5930 - Follow-up checks on the DReAMy emotion tags (run in Colab, GPU runtime)
# Input: dreams_tagged.csv from tag_emotions.py.

!pip install -q dreamy
import os
import numpy as np
import pandas as pd
import dreamy

LABELS = {"AN": "anger", "AP": "apprehension", "CO": "confusion", "SD": "sadness", "HA": "happiness"}
emo = ["emo_" + n for n in LABELS.values()]
if not os.path.exists("dreams_tagged.csv"):
    from google.colab import files
    files.upload()                      # choose dreams_tagged.csv
d = pd.read_csv("dreams_tagged.csv", parse_dates=["date", "date_est"])
main = d[d.keep].copy()

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"

def tag(texts):
    out = []
    for i in range(0, len(texts), 250):
        out += dreamy.annotate_reports(texts[i:i + 250], task="SA", device=device, batch_size=16)
    return pd.DataFrame([{"emo_" + LABELS[p["label"]]: p["score"] for p in pred} for pred in out])[emo]

# ---------- Part 1: does the model ignore the end of long dreams? ----------
long_ = main[main.words > 350]
long_ = long_.sample(min(150, len(long_)), random_state=0)
full = long_[emo].reset_index(drop=True)
first = tag([" ".join(t.split()[:250]) for t in long_.text])
last = tag([" ".join(t.split()[-250:]) for t in long_.text])
print(f"PART 1: {len(long_)} long dreams. Correlation of full-text scores with the score from")
print("only the first 250 words vs only the last 250 words (if 'first' is much higher, the end is probably ignored):")
rows = [{"emotion": e.replace("emo_", ""), "corr_first": full[e].corr(first[e]), "corr_last": full[e].corr(last[e])} for e in emo]
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------- Part 2: average scores by dream length ----------
t = d[d.keep_25 & d.emo_max.notna()].copy()
t["length"] = pd.cut(t.words, [24, 49, 100, 200, 400, 5000], labels=["25-49", "50-100", "101-200", "201-400", "400+"])
print("\nPART 2: average emotion scores by journal and dream length")
print(t.groupby(["journal", "length"], observed=True)[emo].mean().round(2).to_string())

# ---------- Part 3: first look at Barb over time (dates partly estimated) ----------
b = main[(main.journal == "Barb") & main.date_est.notna()].copy()
b["year"] = b.date_est.dt.year
print("\nPART 3a: Barb, average score per estimated year (n = dreams that year)")
g = b.groupby("year")[emo].mean().round(2)
g["n"] = b.groupby("year").size()
print(g.to_string())

def window(label, lo, hi):
    w = b[(b.date_est >= lo) & (b.date_est <= hi)]
    return pd.Series({**w[emo].mean().round(3).to_dict(), "n": len(w)}, name=label)
print("\nPART 3b: Barb around the Derek period and Howard's death (estimated dates)")
print(pd.DataFrame([
    window("before Derek (to 1994-10-06)", "1979-01-01", "1994-10-06"),
    window("Derek period (1994-10-07 to 1996-09-26)", "1994-10-07", "1996-09-26"),
    window("after Derek, before Howard died (to 1997-03-31)", "1996-09-27", "1997-03-31"),
    window("after Howard's death (from 1997-05-01)", "1997-05-01", "2001-12-31"),
]).to_string())
