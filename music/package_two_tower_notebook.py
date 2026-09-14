"""Create the self-contained Kaggle notebook for the two-tower experiment."""
import json
from pathlib import Path

src = Path("two_tower.py").read_text()
setup = '''# Kaggle GPU configuration.  Change --cut to 2025-08-31 and remove
# --eval only after the Aug 16-30 holdout experiment shows a real improvement.
import sys
sys.argv = [
    "two_tower.py",
    # The code auto-detects the attached competition directory. This placeholder
    # also permits a local run by replacing it with the data folder.
    "--input-dir", "/kaggle/input/competition-data",
    "--cut", "2025-08-16", "--eval",
    "--epochs", "8", "--dim", "96", "--batch-size", "8192",
    "--retrieval-k", "500",
    "--output", "/kaggle/working/two_tower_holdout_candidates.csv",
    "--device", "cuda",
]
'''

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}},
    "cells": [
        md("""# Two-tower retrieval experiment — Personalized Music Recommender

This is an **experiment**, not the current submission pipeline. It learns
user/item embeddings from pre-cutoff listening events and retrieves from the
entire pre-cutoff catalogue. The notebook first validates exactly as a temporal
holdout; only a material lift should justify blending it with the established
GBM submission notebook.

Run on a Kaggle T4/P100 GPU. It uses only the competition's supplied CSVs and
does not download a pretrained model or depend on local cache files."""),
        code(setup),
        md("""## Train, retrieve, and score

The code fixes random seeds. GPU arithmetic can still vary slightly by runtime,
so use the reported holdout score—not one run alone—to decide whether to submit."""),
        code(src),
    ],
}
Path("music-two-tower-experiment.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote music-two-tower-experiment.ipynb")
