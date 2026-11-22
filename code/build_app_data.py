"""Build the four JSON files the web app loads: dreams_viz.json, cast.json, trends.json, pool.json.

Inputs (not in the repo; see README for how to produce them):
  dreams_tagged_v3.csv   - one row per dream, emotion scores (retag_180.py output)
  sentences_tagged.csv   - one row per sentence, emotion scores (tag_sentences.py output)
Usage: python build_app_data.py dreams_tagged_v3.csv sentences_tagged.csv OUT_DIR
"""
import sys, re, json
import numpy as np
import pandas as pd
from masking import *

dreams_path, sent_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
E5 = ["anger", "apprehension", "confusion", "sadness", "happiness"]
d = pd.read_csv(dreams_path, parse_dates=["date", "date_est"])
s = pd.read_csv(sent_path)

# ---- recompute dominant emotion from the final (v3) scores: strongest score, or "none" if nothing is above 0.5
M = d[["emo_" + e for e in E5]].values
dom = np.array(E5)[M.argmax(1)]
dom[M.max(1) <= 0.5] = "none"
d["emo_dominant_v3"] = dom

names, maskset = build_maskset(s.sentence, [s.dream_people, s.people])
mask = make_masker(names, maskset)

# ================= dreams_viz.json : one row per dream =================
# [journal(0 Barb,1 Pegasus), dream_no, year, dominant code, masked excerpt, date_recorded]
CODE = {"anger": 0, "happiness": 1, "confusion": 2, "sadness": 3, "apprehension": 4, "none": 5}
ex = {}
for (j, n), g in s.groupby(["journal", "dream_no"]):
    g = g.sort_values("sent_idx")
    c = g[g.words >= 9]
    ex[(j, int(n))] = (c.iloc[0] if len(c) else g.iloc[0]).sentence
dk = d[d.keep & d.journal.isin(["Barb", "Pegasus"])]
viz = []
for j in ["Barb", "Pegasus"]:
    for r in dk[dk.journal == j].sort_values("dream_no").itertuples():
        y = None if pd.isna(r.date_est) or j == "Pegasus" else int(r.date_est.year)
        e = ex.get((j, int(r.dream_no)))
        e = mask(e) if e else None
        if e and BAD_EXCERPT.search(e):
            e = None
        viz.append([0 if j == "Barb" else 1, int(r.dream_no), y, CODE[r.emo_dominant_v3], e, 1 if r.date_source == "recorded" else 0])
json.dump(viz, open(f"{out_dir}/dreams_viz.json", "w"), separators=(",", ":"))

# ================= cast.json : recurring people (names never leave this script) =================
# Roles come from Domhoff's public write-up of Barb's journal. Unidentified people get generic labels.
def find(j, n):
    g = d[(d.journal == j) & d.keep].sort_values("dream_no").reset_index(drop=True)
    rx = re.compile(r"\b" + n + r"(?:'s)?\b")
    return [i for i, t in enumerate(g.text) if rx.search(str(t))], len(g)
B = [("Howard", "Her ex-husband"), ("Dwight", "Her favorite brother"), ("Ginny", "Her closest friend"), ("Lucy", "A friend from theater"),
     ("Darryl", "Her first love"), ("Ellie", "A recurring figure"), ("Derek", "A man she was infatuated with"), ("Charla", "A young child")]
P = [("Ann", "A recurring figure"), ("Bob", "A recurring figure"), ("Steve", "A recurring figure"), ("John", "A recurring figure")]
cast = {}
for key, j, L in (("barb", "Barb", B), ("pegasus", "Pegasus", P)):
    cast[key] = []
    for n, role in L:
        idx, N = find(j, n)
        cast[key].append({"role": role, "n": len(idx), "idx": idx})
    cast[key + "_N"] = N
gb = d[(d.journal == "Barb") & d.keep].sort_values("dream_no").reset_index(drop=True)
cast["death_idx"] = int(gb.index[gb.date_est >= pd.Timestamp("1997-04-01")][0])
json.dump(cast, open(f"{out_dir}/cast.json", "w"), separators=(",", ":"))

# ================= trends.json : length-adjusted yearly means, month-block bootstrap =================
TE = ["anger", "apprehension", "confusion", "sadness", "happiness"]
rng = np.random.default_rng(5)
trends = {}
for j in ["Barb", "Pegasus"]:
    g = d[(d.journal == j) & d.keep_25 & (d.words >= 50)].copy()
    if j == "Barb":
        g = g.dropna(subset=["date_est"]).sort_values("date_est")
        g["bin"] = g.date_est.dt.year
        blk = g.date_est.dt.to_period("M").astype(str).values
        t = (g.date_est - pd.Timestamp("1990-01-01")).dt.days.values / 3652.5
    else:
        g = g.sort_values("dream_no"); n = len(g)
        g["bin"] = (np.arange(n) * 12 // n)
        blk = (np.arange(n) // 20).astype(str)          # 20-dream blocks (no dates)
        t = np.arange(n) / n
    pc = (g.words / g.n_chunks).values.astype(float)    # words per scored piece
    L = np.log(pc); L = (L - L.mean()) / L.std()
    XL = np.column_stack([np.ones(len(g)), L, L ** 2]); Xt = np.column_stack([XL, t])
    bins = sorted(g.bin.unique()); cnt = g.groupby("bin").size()
    if j == "Barb":
        bins = [b for b in bins if cnt[b] >= 40]
    ub = np.unique(blk); idx = {b: np.where(blk == b)[0] for b in ub}
    res = {"bins": [int(b) for b in bins], "n": [int(cnt[b]) for b in bins], "N": int(len(g)), "emo": {}}
    if j == "Pegasus":
        res["label"] = [f"{int(g[g.bin == b].dream_no.min())}–{int(g[g.bin == b].dream_no.max())}" for b in bins]
    for e in TE:
        y = g["emo_" + e].values * 100
        def adj(ii):
            beta = np.linalg.lstsq(XL[ii], y[ii], rcond=None)[0]
            return y[ii] - XL[ii] @ beta + y[ii].mean()
        def binmeans(ii):
            a = adj(ii); b = g.bin.values[ii]
            return np.array([a[b == k].mean() if (b == k).any() else np.nan for k in bins])
        m = binmeans(np.arange(len(g)))
        bs, sl = [], []
        for _ in range(400):
            ii = np.concatenate([idx[k] for k in rng.choice(ub, len(ub))])
            bs.append(binmeans(ii)); sl.append(np.linalg.lstsq(Xt[ii], y[ii], rcond=None)[0][3])
        bs = np.array(bs)
        lo = np.nanpercentile(bs, 2.5, axis=0); hi = np.nanpercentile(bs, 97.5, axis=0)
        s_ = np.linalg.lstsq(Xt, y, rcond=None)[0]
        res["emo"][e] = {"m": np.round(m, 2).tolist(), "lo": np.round(lo, 2).tolist(), "hi": np.round(hi, 2).tolist(),
                         "slope": [round(float(s_[3]), 2), *np.round(np.percentile(sl, [2.5, 97.5]), 2).tolist()],
                         "b0": round(float(s_[0]), 2),
                         "line": [round(float(s_[0] + s_[3] * t.min()), 2), round(float(s_[0] + s_[3] * t.max()), 2)]}
    trends[j.lower()] = res
json.dump(trends, open(f"{out_dir}/trends.json", "w"), separators=(",", ":"))

# ================= pool.json : sentences for the Dream Machine =================
# [journal, dream_no, year, masked sentence, anger, happiness, confusion, sadness, apprehension, historical_flag]
PE = ["anger", "happiness", "confusion", "sadness", "apprehension"]
sp = s[s.words.between(8, 30) & ~s.sentence.str.contains(BAD)]
rows = []
for j, name in ((0, "Barb"), (1, "Pegasus")):
    g = sp[sp.journal == name]
    for e in PE:
        cap = {"apprehension": 900, "happiness": 900}.get(e, 2000) if j == 0 else 200
        sel = g[(g["emo_" + e] >= .7) & (g[["emo_" + x for x in PE]].idxmax(axis=1) == "emo_" + e)]
        rows.append(sel.sample(min(cap, len(sel)), random_state=5))
    neu = g[g[["emo_" + e for e in PE]].max(axis=1) <= .3]
    rows.append(neu.sample(min(500 if j == 0 else 150, len(neu)), random_state=5))
P_ = pd.concat(rows).drop_duplicates("sid")
pool = []
for r in P_.itertuples():
    t = mask(r.sentence).strip()
    if t.count("[name]") > 2:
        continue
    pool.append([0 if r.journal == "Barb" else 1, int(r.dream_no), None if pd.isna(r.year_est) else int(r.year_est), t,
                 *[int(round(100 * getattr(r, "emo_" + e))) for e in PE], 1 if HIST.search(r.sentence) else 0])
json.dump(pool, open(f"{out_dir}/pool.json", "w"), separators=(",", ":"))
print("wrote", len(viz), "dreams,", len(pool), "pool sentences")
