# SWEET.TV programme recommendation — FKN midterm

Author: Timur Khayrislamov. New, original exam implementation with AI assistance.

Compares popularity, metadata TF-IDF content profiles, and personalized PageRank
on the same temporal holdout. Code and synthetic tests are included; competition
data and private session logs are not redistributed.

## Measured validation

Train: 9 March–17 May 2020; validation: 18 May–26 July 2020, on the
source's unspecified local wall-clock time. Explicit midnight boundaries clip
schedule-file spillovers. Candidates: 6,384 valid scheduled programme IDs.
Repeated recommendations are allowed for every method. The fixed cohort has
1,246 template users with observable holdout targets, including 8 cold users;
14 template users have no eligible observed targets and are excluded from all
validation means. All 1,260 template rows receive final predictions.

| Method | MAP@5 | Fit (s) | Predict cohort (s) |
|---|---:|---:|---:|
| Popularity | 0.777052 | 1.072 | 5.472 |
| Content cosine | 0.274944 | 1.221 | 7.171 |
| PPR, continuation 0.50 | **0.791118** | 1.690 | 8.521 |
| PPR, continuation 0.85 | 0.780254 | 1.733 | 10.050 |

Choose PPR 0.50 by this validation only. The selected model was refitted on
9 March–26 July histories and predicts scheduled programmes for
27 July–18 October. The exact final file is `submission.csv`, SHA-256
`84f933f62856b1e764fce5cd4fccca1abcee9a5818fdf9297d39d49762812aae`.
Its headers, raw user strings/order, 1,260 rows, five distinct positive IDs per
row, spacing, and membership in the 6,632-item target pool have been checked.
This repository does not claim a returned Kaggle score until an actual submission
record establishes one. Local MAP follows the slides; hidden-evaluator equivalence
is not established. No enjoyment, revenue or online causal effect is measured.

## Reproduce

Python 3.9.6 on macOS ARM64 was used. The actual environment is pinned.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify.py
```

The tests use invented data. Real inputs must be downloaded through your own
authorized Kaggle account after you personally accept the competition rules:
https://www.kaggle.com/competitions/sweettv-tv-program-recommender
Place the five official CSV files under `data/raw/`, preserving originals.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.cli run --raw-dir data/raw --run-dir experiments/reproduce-01 --calendar-config config/calendar-boundaries.json
PYTHONPATH=src .venv/bin/python -m sweettv.cli validate --run-dir experiments/reproduce-01 --template data/raw/submission.csv
```

Runs are immutable. Exports include hashes, data and interval audits, fixed cohorts,
per-user AP, recommendations, exact-rational independent AP checks, model states,
score variance, PPR convergence, timings/memory, and final submission validation.
The predeclared watch threshold is 120 seconds after unioning fragments of an
airing; it was not tuned by MAP. Vocabulary, IDF, profiles and co-view graph use
training-period information only. Programme ID 0 is unlabelled context, not a
preference or recommendation. Cold/short lists share training-popularity fallback.
The graph uses sparse row-block products with immediate top-100 neighbour pruning.
See `docs/methods.md` and `results/validation-summary.json` for definitions/settings.

## Limitations and provenance

Only 10.76%, 11.31%, and 11.18% of raw schedule rows are labelled in the three
files. Full-history usable views cover a biased subset of logged viewing; missing
labels and absence from the log do not establish dislike or never watching.
The view timestamps lack timezone annotations, and 117 declared durations differ
from their wall-clock endpoints by one hour. This limitation is preserved.

Implementation was generated and reviewed with Codex; automated checks are not
student personal verification. Report, genuine private-score screenshot and
unmodified native session exports are course materials kept outside this repo.
No raw competition data, supplied course PDFs/slides, tokens or private logs
are included. Dependencies retain their upstream licenses.
