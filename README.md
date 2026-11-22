What Do We Dream About?

Following two dream journals across decades. CIS 5930 data visualization final project, Joshua Okafor.

Live page: https://gitfsustudent.github.io/Dream-Journals/
Code: https://github.com/gitfsustudent/Dream-Journals

A scrolling web page that opens with the "This Man" internet legend, then reads two real dream journals from DreamBank: Barb Sanders (a journal kept for about two decades, mostly 1979-2001) and Pegasus (a factory worker, 1949-64, no dates). It shows:

Dream barcode. One window per dream, colored by its strongest emotion. Hover for a (name-masked) line from the dream.
Who keeps coming back. A timeline of recurring people in each journal, one tick per dream.
Does the mood change? Length-adjusted yearly averages with 95% bands, for five emotions.
Dream Machine. Real sentences stitched into a new "dream" by the feelings you choose. Nobody dreamed the result.
Run it locally

The page is static. Serve the folder with any web server (opening index.html directly will not load the data):

python3 -m http.server 8000      # then open http://localhost:8000
Files
Path	What it is
index.html	The whole web app (HTML, CSS, JavaScript).
dreams_viz.json, cast.json, trends.json, pool.json	Data the page loads. Built by code/build_app_data.py. Names are masked.
thisman.jpg	The widely circulated sketch from the 2006 legend (author unknown).
data/dreams_scores.csv	Per-dream emotion scores and cleaning flags, without the dream text.
code/	The analysis pipeline, in the order below.
Pipeline (code/)
clean_dreams.py - load the DReAMy copy of DreamBank, parse dates and bracketed notes, drop dreams under 50 words and baseline duplicates, estimate missing dates (Barb only).
tag_emotions.py - tag each dream with the DReAMy annotator (five Hall and Van de Castle emotions, scores 0-1). GPU recommended.
check_tagger.py, retag_chunks.py - found that long dreams were over-scored; re-tag in word-limited pieces.
retag_180.py (retag_180_kaggle.py for Kaggle) - final check: re-tag every dream over 180 words in pieces of at most 180 words, weighted by length. This is the final tagging (v3).
make_sentences.py, tag_sentences.py - split dreams into sentences and tag each one, for the Dream Machine.
build_app_data.py (+ masking.py) - recompute each dream's dominant emotion from the final scores, compute length-adjusted trends with month-block bootstrap, build the recurring-people timeline, mask names, and write the four JSON files.

Inputs the repo does not contain (large, and they hold full dream text): dreams_tagged_v3.csv and sentences_tagged.csv. Regenerate them with steps 1-5, then run python code/build_app_data.py dreams_tagged_v3.csv sentences_tagged.csv .

Data and license
Source: DreamBank.net (G. William Domhoff, UC Santa Cruz), accessed through the DReAMy Hugging Face copy (dreamy.get_HF_DreamBank).
License: DreamBank is CC BY-NC-SA 4.0. The derived data in this repository (the JSON files and data/dreams_scores.csv) is shared under CC BY-NC-SA 4.0: attribute DreamBank, non-commercial use only, share alike. The code is MIT (see LICENSE).
Names: DreamBank already uses pseudonyms. The page additionally hides every capitalized word that is not on a short allow-list, so some ordinary capitalized words also appear as "[name]".
Filtering: dreams under 50 words are left out. Sentences with sexual, self-harm or abuse vocabulary, or crude language, are removed before tagging or display. Two sentences that mention a historical atrocity are kept and marked.
Roles: the roles shown for recurring people in Barb's journal (ex-husband, brother, friend and so on) come from Domhoff's public write-up of her series, not from this analysis.
Key results
Barb's tone is stable across two decades except for less anger: about 4 fewer angry dreams per 100 each decade (95% range -5.8 to -2.5).
Dreams with a recurring named person are angrier in Barb's journal (about 4 more angry dreams per 100). Pegasus shows no clear difference.
No detectable change in anger or sadness in the year after her ex-husband's death (April 1997).
Pegasus's trends (less anger, more confusion) assume his dream order is time order; his journal has no dates.
Limitations
Two dreamers, not random samples; written dreams are memories, not the dreams themselves.
Dates for Barb are partly estimated; Pegasus has none.
The emotion tagger is a machine model. It makes errors and works best on short passages, so long dreams are scored in pieces of up to 180 words. A check showed the first whole-dream scores were inflated for mid-length dreams.
Only named recurring characters are counted, so people referred to as "my mother" or "he" are missed.
Five emotions were tested; only anger survives a strict multiple-comparison correction in Barb's journal.
Known gap: the step that found recurring named people for make_sentences.py (a name-matching pass) is not included as a standalone script.
Credits
DreamBank.net, G. William Domhoff, University of California, Santa Cruz.
DReAMy, the Python library for dream report analysis used for emotion tagging (Hugging Face: DReAMy-lib).
Visual mood inspired by the 2010 film Inception; no film artwork or lettering is used.
