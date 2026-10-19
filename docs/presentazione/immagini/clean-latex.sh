#!/bin/sh
set -eu

# Remove LaTeX build files below this script's directory.
cd -P -- "$(dirname -- "$0")"
find . -type f \( \
  -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o \
  -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.lof' -o \
  -name '*.log' -o -name '*.lot' -o -name '*.out' -o \
  -name '*.synctex.gz' -o -name '*.toc' \
\) -print -delete
