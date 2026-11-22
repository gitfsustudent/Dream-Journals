"""Name masking and sentence filters shared by the app-data builders.

DreamBank already replaces real names with pseudonyms. We still hide names in everything shown on the public
page: any capitalized word that is not on a short allow-list of ordinary capitalized words becomes "[name]".
This over-masks on purpose (some place names and ordinary capitalized words are hidden too).
"""
import re
import pandas as pd

ALLOW = set("""Oh You City No OK It's Aunt The We TV He What It Well How This A Hey Look Don't But Yes Indian She Why Now Let's If Then They That's Maybe Are Do My Is French Come King Here And There That So Native University God VW See Just Christmas Go XXX Wait He's Mr Mrs Dr Doctor You're Did Mexico New Get Let Queen PA Spanish Xmas Ah Dad Can Help Hi Street California House Star German What's Will Not Good White Chinese Indians American Grandma Grandpa Uncle Mexican English Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December Thanksgiving Easter Mom Mother Father Daddy Mommy I I'm I'd I'll I've St Mother's Where When Who Which Please Thanks Sorry Okay Right Sure Anyway Also As At In On To For With From All Some Any One Two Three Our Your His Her Their Its Mama Papa Sir Madam Miss Ms Rev Sister Brother Bishop Reverend Judge Captain Officer Nurse Teacher Professor Santa Jesus Christ Bible Lord Heaven Hell Nazi Nazis Germans Jewish Jews Holocaust""".split())

# sentences containing any of these are never shown (sexual, crude, slurs, self-harm)
BAD = re.compile(r"make love|making love|\bsex|naked|nude|\brape|suicid|kill (?:my|him|her)self|breast|penis|vagina|orgasm|erotic|masturbat|molest|genital|porn|slut|whore|fuck|shit|nigger|retard|\bdamn|\bbitch|\bcunt|\bcock|\bpee\b|\bpoop|\bfart|\bhomo|\bfag|\bqueer|\bgay|lesbian|prostitut|bedroom|undress|\bbra\b|panties|underwear|\bbare\b|\bstrip", re.I)
BAD_EXCERPT = re.compile(r"make love|making love|\bsex|naked|nude|rape|raped|suicid|kill (?:my|him|her)self|breast|penis|vagina|orgasm|erotic|masturbat|molest|genital|porn|slut|whore|fuck|shit|nigger|retard", re.I)
HIST = re.compile(r"concentration camp|Nazi|Holocaust", re.I)

def base(w):
    return re.sub(r"['’]s$", "", w)

def build_maskset(sentences: pd.Series, people_cols):
    names = set()
    for col in people_cols:
        for v in col.dropna().unique():
            names.update(x.strip() for x in str(v).split(";"))
    names -= {"Mexican", "English", "Ave"}
    mid = set()
    for t in sentences:
        toks = re.findall(r"[A-Za-z'’]+", t)
        for w in toks[1:]:
            if w[0].isupper():
                mid.add(w)
    maskset = {w for w in mid if w not in ALLOW and base(w) not in ALLOW} | names | {base(w) for w in mid if base(w) not in ALLOW}
    return names, maskset

def make_masker(names, maskset):
    def f(m):
        w = m.group(0)
        if w in ALLOW:
            return w
        b = base(w)
        if len(b) == 1 and b not in "IA":
            return "[name]"
        if w in maskset or b in maskset or b in names:
            return "[name]" + ("'s" if w != b else "")
        return w
    def mask(t):
        return re.sub(r"\b[A-Za-z][A-Za-z'’]*\b", lambda m: f(m) if m.group(0)[0].isupper() else m.group(0), t)
    return mask
