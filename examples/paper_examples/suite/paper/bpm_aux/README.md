# Local-only material for the BPM paper

The paper itself lives in `../bpm`, a clone of the Overleaf project registered as a git
submodule of this repository; nothing in this folder is synced to Overleaf.

- `make_figures.py`: regenerates `../bpm/figures/*.pdf` from the cells in `../../results/`
  (run it, then commit and push inside `../bpm` to update Overleaf).
- `sections_v1/`: the draft before the restructuring into the A-E PN paper layout.
- `2306.02910.md`: the original A-E PN paper, the structure and style reference.
