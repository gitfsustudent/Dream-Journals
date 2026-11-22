"""Split kept dreams into sentences and filter them for the Dream Machine tagging step.

Input : a dreams table with columns journal, dream_no, date_est, text, people, keep, emo_* (people = recurring
        named characters found in the dream by name matching, separated by ';').
Output: sentences_to_tag.csv  (then run tag_sentences.py on a GPU)
Filters: 6-40 words; sentences with sexual or self-harm/abuse vocabulary are removed.
NOTE: the `people` column was produced in an earlier name-matching step that is not in this repo as a script.
      Everything else in this file is the code that was run.
"""
import re, sys
import pandas as pd

m = pd.read_csv(sys.argv[1], parse_dates=["date", "date_est"]); m["people"] = m.people.fillna("")
k = m[m.keep].copy()
ABBR = set("mr mrs ms dr st jr sr vs etc inc no mt ft".split())
SPLIT = re.compile(r"([.!?]+[\"”')]*)\s+(?=[A-Z\"“'(])")
def sents(t):
    t = str(t).strip(); out = []; last = 0
    for mt in SPLIT.finditer(t):
        prev = re.findall(r"[A-Za-z]+", t[max(0, mt.start() - 6):mt.start()])
        pw = prev[-1].lower() if prev else ""
        if (len(pw) == 1 and pw.isalpha()) or pw in ABBR: continue
        out.append(t[last:mt.end()].strip()); last = mt.end()
    out.append(t[last:].strip()); return [s for s in out if s]
SEX = re.compile(r"\b(sex|sexual|sexually|rape|raped|raping|naked|nude|penis|breast|breasts|orgasm|intercourse|genital\w*|erect\w*|vagina\w*|masturbat\w*|fuck\w*|molest\w*|incest\w*|porn\w*|horny|lust\w*|whore|slut|prostitut\w*)\b", re.I)
HARM = re.compile(r"\b(suicide|suicidal|kill(ed|ing)? (myself|himself|herself|themselves)|hang(ed|ing)? (myself|himself|herself)|abus(e|ed|ing|ive)|overdos\w*)\b", re.I)
rows = []
for _, r in k.iterrows():
    ss = sents(r.text)
    for i, s in enumerate(ss):
        rows.append((r.journal, int(r.dream_no), i, len(ss), r.date_est.year if pd.notna(r.date_est) else "", s, len(s.split()), r.people)
                    + tuple(r["emo_" + e] for e in ["anger", "apprehension", "confusion", "sadness", "happiness"]))
S = pd.DataFrame(rows, columns=["journal", "dream_no", "sent_idx", "n_sents", "year_est", "sentence", "words", "dream_people",
                                "dream_anger", "dream_apprehension", "dream_confusion", "dream_sadness", "dream_happiness"])
f = S[(S.words >= 6) & (S.words <= 40)].copy()
f = f[~f.sentence.str.contains(SEX) & ~f.sentence.str.contains(HARM)].copy()
pe = {j: set(";".join(k[k.journal == j].people).split(";")) - {""} for j in ["Barb", "Pegasus"]}
f["people"] = [";".join(sorted(n for n in pe[j] if re.search(rf"\b{n}\b", s))) for j, s in zip(f.journal, f.sentence)]
f = f.reset_index(drop=True); f.insert(0, "sid", f.index)
f.to_csv("sentences_to_tag.csv", index=False)
print("sentences kept:", len(f), f.groupby("journal").size().to_dict())
