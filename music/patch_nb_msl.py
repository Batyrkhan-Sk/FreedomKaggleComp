"""Patch the deliverable notebook to min_samples_leaf=1000, to match submission18.

Run ONLY after deciding to submit submission18.csv. The notebook currently
reproduces submission_final.csv (LB 0.38415) byte-identically, and patching it
breaks that pairing -- so a .bak is kept and verify_nb.py must be re-run against
the NEW csv before the notebook is sent to the organizers.
"""
import json, shutil, sys
NB = "music-recommender-solution.ipynb"
shutil.copy(NB, NB + ".pre_msl1000")
nb = json.load(open(NB))
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    for i, ln in enumerate(c["source"]):
        if "min_samples_leaf=200" in ln:
            c["source"][i] = ln.replace("min_samples_leaf=200", "min_samples_leaf=1000"); n += 1
assert n == 1, f"expected exactly 1 occurrence, patched {n}"
json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
print(f"patched {n} occurrence; backup at {NB}.pre_msl1000")
