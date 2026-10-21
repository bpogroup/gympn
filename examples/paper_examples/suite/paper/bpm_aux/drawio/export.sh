#!/bin/bash
# Regenerate the .drawio sources and export each to PDF (for the paper) and PNG (for review).
cd "$(dirname "$0")"
../../../../../../.venv/Scripts/python make_drawio.py
DIO="/c/Program Files/draw.io/draw.io.exe"
for f in ${1:-fig_*.drawio}; do
  b=${f%.drawio}
  "$DIO" -x -f pdf --crop -o "$b.pdf" "$f" >/dev/null 2>&1
  "$DIO" -x -f png -s 2 -o "$b.png" "$f" >/dev/null 2>&1
  echo "$b"
done
