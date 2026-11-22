# CIS 5930 - Dream journal cleaning script (run in Google Colab)
# Data: DreamBank.net (UC Santa Cruz), CC BY-NC-SA 4.0, via the DReAMy Hugging Face copy.
# Journals: Barb Sanders (series "b" then "b2") and Pegasus ("pegasus").
# Dreams were cleaned and filtered as described below.

!pip install -q dreamy
import re
import numpy as np
import pandas as pd
import dreamy

MIN_WORDS = 50        # main cutoff (published standard)
CHECK_WORDS = 25      # sensitivity check

# ---------- 1. Load and assemble the two journals ----------
df = dreamy.get_HF_DreamBank(database="base", as_dataframe=True)
df["raw_row"] = df.index

barb = pd.concat([df[df.series == "b"], df[df.series == "b2"]]).copy()   # b first, then b2
barb["journal"] = "Barb"
peg = df[df.series == "pegasus"].copy()
peg["journal"] = "Pegasus"
d = pd.concat([barb, peg]).reset_index(drop=True)
d["dream_no"] = d.groupby("journal").cumcount() + 1
d["raw_text"] = d["dreams"]
d["raw_words"] = d["raw_text"].str.split().str.len()

# ---------- 2. Pull the leading date out of the text ----------
LEAD = re.compile(r"^\s*\((\d{1,2})/(\d{1,2})/(\d{2})\)")
def parse_date(s):
    m = LEAD.match(s)
    if not m:
        return pd.NaT
    mo, da, yy = map(int, m.groups())
    yr = 1900 + yy if yy >= 40 else 2000 + yy
    return pd.to_datetime(f"{yr}-{mo:02d}-{da:02d}", errors="coerce")
d["date"] = d["raw_text"].apply(parse_date)
d["text"] = d["raw_text"].str.replace(LEAD, "", regex=True)

# ---------- 3. Move [bracketed] items into their own columns ----------
BRACKET = re.compile(r"\[([^\]]*)\]")
def split_brackets(s):
    items = [x.strip() for x in BRACKET.findall(s)]
    bl = any(x == "BL" for x in items)
    note = "; ".join(x for x in items if x.upper().startswith("NOTE"))
    title = "; ".join(x.strip('"“” ') for x in items if x.startswith(('"', "“")))
    other = "; ".join(x for x in items if x != "BL" and not x.upper().startswith("NOTE")
                      and not x.startswith(('"', "“")))
    return pd.Series({"is_baseline250": bl, "note": note, "title": title, "bracket_other": other})
d = pd.concat([d, d["text"].apply(split_brackets)], axis=1)
d["text"] = d["text"].str.replace(BRACKET, " ", regex=True)
d["text"] = d["text"].str.replace(r"[\[\]]", " ", regex=True)   # stray unmatched brackets

# ---------- 4. Normalize spacing, recount words ----------
d["text"] = d["text"].str.replace(r"\s+", " ", regex=True).str.strip()
d["words"] = d["text"].str.split().str.len().fillna(0).astype(int)

# ---------- 5. Dates: recorded vs estimated ----------
d["date_flag_backwards"] = False
d["date_est"] = pd.NaT
d["date_source"] = "none"
for j, g in d.groupby("journal"):
    dated = g[g.date.notna()].sort_values("dream_no")
    if dated.empty:
        continue
    backwards = dated.date < dated.date.cummax().shift(fill_value=dated.date.iloc[0])
    bad = dated.index[backwards.values]
    d.loc[bad, "date_flag_backwards"] = True
    d.loc[dated.index, "date_est"] = dated.date
    d.loc[dated.index, "date_source"] = "recorded"
    d.loc[bad, "date_est"] = pd.NaT                      # out-of-order dates are not trusted
    d.loc[bad, "date_source"] = "out_of_order"
    # Interpolate between recorded dates for Barb only (Pegasus has just a handful)
    if j == "Barb":
        anchors = dated[~backwards.values]
        x = anchors.dream_no.values
        EPOCH = pd.Timestamp("1970-01-01")
        y = (anchors.date - EPOCH).dt.days.values.astype(float)   # days since 1970
        gg = d.loc[g.index]   # fresh copy so the out-of-order flags set above are visible
        inside = gg[(gg.dream_no >= x.min()) & (gg.dream_no <= x.max()) & (gg.date.isna() | gg.date_flag_backwards)]
        est = np.interp(inside.dream_no.values, x, y)
        d.loc[inside.index, "date_est"] = EPOCH + pd.to_timedelta(np.round(est), unit="D")
        d.loc[inside.index, "date_source"] = np.where(d.loc[inside.index, "date_flag_backwards"], "out_of_order_estimated", "estimated")

# ---------- 6. Length filter ----------
d["keep"] = d["words"] >= MIN_WORDS
d["keep_25"] = d["words"] >= CHECK_WORDS

# ---------- 7. Report ----------
print("DREAMS PER JOURNAL (all / kept at", MIN_WORDS, "words / kept at", CHECK_WORDS, "words)")
print(d.groupby("journal").agg(all=("keep", "size"), kept=("keep", "sum"), kept_25=("keep_25", "sum")))
print("\nShare dropped at", MIN_WORDS, "words:")
print((1 - d.groupby("journal")["keep"].mean()).round(3))
print("\nDate source by journal:")
print(d.groupby(["journal", "date_source"]).size())
print("\nDates that go backwards (dream_no, date):")
print(d[d.date_flag_backwards][["journal", "dream_no", "date"]].to_string())
print("\nBaseline-250 flag (all / kept):", int(d.is_baseline250.sum()), "/", int(d[d.keep].is_baseline250.sum()))
print("Dreams with a title:", int((d.title != "").sum()), "| with a note:", int((d.note != "").sum()),
      "| other bracket text:", int((d.bracket_other != "").sum()))
print("Brackets left in cleaned text:", int(d.text.str.contains(r"[\[\]]").sum()))
print("Empty cleaned text:", int((d.text == "").sum()))
print("\nBarb estimated date range:", d[d.journal == "Barb"].date_est.min(), "to", d[d.journal == "Barb"].date_est.max())

# ---------- 8. Save ----------
raw_cols = ["journal", "series", "raw_row", "dream_no", "raw_text", "gender", "year"]
d[raw_cols].to_csv("dreams_raw_subset.csv", index=False)
clean_cols = ["journal", "series", "raw_row", "dream_no", "date", "date_est", "date_source",
              "date_flag_backwards", "text", "words", "is_baseline250", "title", "note",
              "bracket_other", "keep", "keep_25"]
d[clean_cols].to_csv("dreams_clean.csv", index=False)
print("\nSaved dreams_raw_subset.csv and dreams_clean.csv")
try:
    from google.colab import files
    files.download("dreams_raw_subset.csv")
    files.download("dreams_clean.csv")
except ImportError:
    pass
