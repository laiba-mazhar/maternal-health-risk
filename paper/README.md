# The write-up

`paper.md` is the manuscript. `generated_tables.md` holds every table, produced
from `artifacts/` by a script rather than typed by hand:

```bash
python scripts/train.py            # produces artifacts/
python scripts/make_figures.py     # produces artifacts/figures/
python scripts/make_paper_tables.py
```

## Two things to do before this goes anywhere

**1. Re-run on the real dataset.** Every number currently in the manuscript comes
from the synthetic stand-in and is marked as such. Nothing in it is a clinical or
scientific finding until:

```bash
python scripts/download_data.py
python scripts/train.py --source real
python scripts/make_figures.py && python scripts/make_paper_tables.py
```

The prose then needs re-reading, not just the tables — several claims in the
Results and Discussion are conditional on the values, and the honest ones may
reverse. In particular, whether any learned model beats the guideline baseline at
matched referral load is an empirical question whose answer could flip.

**2. Verify the citations.** `references.bib` contains real published works, but
volume, issue, page, and DOI fields were written from memory and have not been
checked against publisher records. Confirm every field, and every author list,
against the DOI before submission.

## Converting to PDF

With pandoc:

```bash
pandoc paper.md -o paper.pdf --citeproc --bibliography=references.bib
```

Figures are referenced relative to the repository root, so run pandoc from there
(`pandoc paper/paper.md ...`) or adjust the paths.
