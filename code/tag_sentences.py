# CIS 5930 - Tag every sentence in the Dream Machine pool with DReAMy's five emotions (Colab, GPU runtime)
# Input: sentences_to_tag.csv. Output: sentences_tagged.csv
# Saves progress as it goes, so if the session drops you can re-run and it will resume.

!pip install -q dreamy
import os, time
import numpy as np
import pandas as pd
import dreamy

LABELS = {"AN": "anger", "AP": "apprehension", "CO": "confusion", "SD": "sadness", "HA": "happiness"}
emo = ["emo_" + n for n in LABELS.values()]
CHUNK = 2000

if not os.path.exists("sentences_to_tag.csv"):
    from google.colab import files
    files.upload()                      # choose sentences_to_tag.csv
S = pd.read_csv("sentences_to_tag.csv")
S["people"] = S["people"].fillna("")

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print("Device:", device, "| sentences:", len(S))

# resume from a partial file if there is one
parts = []
if os.path.exists("sentences_partial.csv"):
    parts.append(pd.read_csv("sentences_partial.csv"))
    print("Resuming: already tagged", len(parts[0]))
done = set(parts[0].sid) if parts else set()
todo = S[~S.sid.isin(done)].sort_values("words")      # similar lengths together runs faster
t0 = time.time()
for n, i in enumerate(range(0, len(todo), CHUNK)):
    chunk = todo.iloc[i:i + CHUNK]
    preds = dreamy.annotate_reports(chunk.sentence.tolist(), task="SA", device=device, batch_size=64)
    if not (isinstance(preds[0], list) and isinstance(preds[0][0], dict)):
        raise SystemExit(f"Unexpected output format. First prediction: {preds[0]}")
    df = pd.DataFrame([{"emo_" + LABELS[p["label"]]: p["score"] for p in pred} for pred in preds])[emo]
    df.insert(0, "sid", chunk.sid.values)
    parts.append(df)
    if n % 5 == 4:
        pd.concat(parts).to_csv("sentences_partial.csv", index=False)
    print(f"  tagged {min(i + CHUNK, len(todo))}/{len(todo)}  ({time.time() - t0:.0f}s)")
sc = pd.concat(parts).drop_duplicates("sid")
S = S.merge(sc, on="sid", how="left")
assert S[emo].notna().all().all(), "some sentences were not tagged"
S["emo_max"] = S[emo].max(axis=1)
S["emo_top"] = S[emo].idxmax(axis=1).str.replace("emo_", "")
S.to_csv("sentences_tagged.csv", index=False)
print("Saved sentences_tagged.csv")

# ---------- checks ----------
print("\nAverage sentence score and % of sentences scoring 0.5+, by journal:")
print(pd.concat([S.groupby("journal")[emo].mean().round(3).add_prefix("mean_"),
                 (S.groupby("journal")[emo].agg(lambda x: (x >= 0.5).mean()) * 100).round(1).add_prefix("pct50_")], axis=1).T.to_string())
print("\nShare of sentences where NO emotion reaches 0.5:", round((S.emo_max < 0.5).mean(), 3))
print("\nAverage score by sentence length:")
S["len"] = pd.cut(S.words, [5, 9, 14, 24, 40], labels=["6-9", "10-14", "15-24", "25-40"])
print(S.groupby("len", observed=True)[emo].mean().round(3).to_string())
print("\nDoes the average of a dream's sentence scores track the dream's own score? (correlation, dreams with 4+ sentences)")
g = S.groupby(["journal", "dream_no"])
avg = g[emo].mean(); par = g[["dream_" + n for n in LABELS.values()]].first(); cnt = g.size()
ok = cnt >= 4
print({n: round(avg.loc[ok, "emo_" + n].corr(par.loc[ok, "dream_" + n]), 2) for n in LABELS.values()})
print("\nTop 3 sentences per emotion (for a quick read):")
for n in LABELS.values():
    top = S.sort_values("emo_" + n, ascending=False).head(3)
    for _, r in top.iterrows(): print(f"  [{n} {r['emo_' + n]:.2f}] {r.journal}: {r.sentence[:120]}")
try:
    from google.colab import files
    files.download("sentences_tagged.csv")
except ImportError:
    pass
