# Tetris Perfect Clear Analysis

## Purpose
Analyzes which four-piece Tetris opening sequences (from the second seven-piece bag)
allow a Perfect Clear, using descriptive statistics and a decision-tree classifier.
Course project for CS 2104, Week 13 milestone.

## Data
`data/tetris_perfect_clear_sequences.csv`: 840 rows, one for every ordered sequence of
four different pieces. Columns: `piece_1`-`piece_4` (I, J, L, O, S, T, Z) and
`success` (1 = can produce a Perfect Clear, 0 = cannot).
Source: TETRIS Perfect Clear Challenge (https://swng.github.io/ttpc/). I built the CSV myself.

## Setup
python -m venv .venv
source .venv/bin/activate
pip install pandas==2.2.3 matplotlib==3.9.2 scikit-learn==1.5.2 pytest==9.2.0

## Reproduce the results
Run from the repo root:
python code/analysis.py            # quick overview
python code/descriptive_stats.py   # tables -> docs/tables, figures -> docs/figures
python code/tune_depth.py          # cross-validation comparison of tree depths
python code/model.py               # decision tree, metrics, confusion matrix
python -m pytest tests -v          # tests

## Technologies
Python, pandas, matplotlib, scikit-learn, pytest, git

## Repository layout
code/ scripts | data/ dataset | tests/ pytest tests and saved results |
docs/ generated tables and figures | report/ final report (Week 14)