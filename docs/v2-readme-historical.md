# SWEET.TV programme recommendation — FKN midterm

Author: Timur Khayrislamov (Хайрисламов Тимур). Original exam implementation with AI assistance.

The current model uses personalized PageRank with square-root viewing-time restart weights and link-following probability 0.25. It was selected on an earlier development period, checked on the later period, and refitted on all allowed historical viewing. Popularity and metadata TF-IDF content filtering are evaluated separately.

## Measured improvement

| Method | Top5 seconds MAP@5 | Top5 watches MAP@5 | All watched MAP@5 | Fit s | Predict s |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.177472 | 0.310072 | 0.777052 | 1.245 | 5.883 |
| Content TF-IDF | 0.024564 | 0.027860 | 0.274944 | 1.410 | 8.046 |
| Original PPR (alpha 0.50) | 0.190441 | 0.323826 | 0.791118 | 1.970 | 9.338 |
| Selected PPR (alpha 0.25, sqrt seconds) | 0.304660 | 0.391758 | 0.774628 | 2.598 | 8.567 |

These later-period results use the **same 1,246 users and candidates** for all methods (1,238 warm users, 8 cold users; 14 template users without observable targets excluded once). The new model raises primary MAP@5 from 0.190441 to 0.304660 (60.0% relative). It recommends 326 distinct validation programmes versus 88 for the original PPR. Its broad all-watched score is lower: the target definition matters.

Primary relevance is each user's at-most-five eligible held-out programmes with the most accepted viewing seconds. Secondary relevance uses accepted-airing counts. The original all-watched metric is retained separately. These are explicit local proxies: the competition's exact hidden target construction is not established, and local numbers do not predict its returned score.

User-provided Kaggle screenshots show the following completed submissions:

| Visible filename | Public score | Private score |
|---|---:|---:|
| `submission.csv` | 0.29103 | 0.28587 |
| `submission_v2.csv` | 0.32589 | 0.28919 |

The displayed private gain is **0.00332 (1.16% relative)**, while the public gain is 0.03486 (11.98%). The large local gain transferred weakly to the private score. This does not establish equivalence between local target proxies and the hidden evaluator. The second filename and user context match the released v2 file, but neither crop identifies an account, submission ID, description, absolute upload time or submitted file hash; that exact linkage remains unverified. Original screenshot bytes are preserved privately. No new model or validation search was run after receiving the v2 score.

## Current submission and preserved first version

- `submission.csv` and `submissions/v2-ppr-0.25-seconds-sqrt.csv`: current v2, SHA-256 `d5060c1949c8a555bc6c607b2ba7f5e93806add1ab7991c9e2a5b9d242537f85`.
- `submissions/v1-ppr-0.5.csv`: preserved original, SHA-256 `84f933f62856b1e764fce5cd4fccca1abcee9a5818fdf9297d39d49762812aae`.

The new file preserves all 1,260 exact template user strings and their order. Every row contains five distinct positive IDs in the 6,632-programme future pool, with exact headers and spacing. It has 269 distinct recommended programmes; 1,246 user rows differ from v1. Cold users share the same training-popularity fallback.

## Frozen evaluation and actual checks

Development fits 9 March–12 April 2020 and labels 13 April–17 May (1,229 eligible template users). A fixed 14-method grid compares the three original methods, two diagnostic personal-history rankings, eight PPR seed/continuation variants and one fixed 28-day recency variant. Only development Top5-seconds MAP selects the method, with fixed order for ties. The chosen configuration is saved before confirmation scores exist.

Confirmation fits 9 March–17 May and labels 18 May–26 July. It evaluates only the selected model and three original baselines. This block had previously been inspected under broad targets; it is not an untouched independent test. No reselection uses confirmation. The final model fits 9 March–26 July and predicts programmes scheduled 27 July–18 October. Naive source wall-clock values are preserved without inferring a timezone.

The complete v2 model run took 148.74 seconds with 1.072 GB peak resident memory on the recorded machine. Final refit/prediction took 4.164/9.324 seconds. All 70 synthetic correctness tests pass. Independent actual-output checks reconstruct labels and rational AP, reproduce the original three baselines exactly, check every new restart against historical quantities, verify unchanged graph edges and stationary solutions, and validate the final CSV. A separate preparation rerun reproduces all model inputs exactly; only measured runtime differs.

## Reproduce

Python 3.9.6 on macOS ARM64 was used. Dependencies are pinned. Tests use invented data.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify.py
```

Obtain the five official CSV files through your own authorized Kaggle account after personally accepting the competition rules, and extract them unchanged under `data/raw/extracted/`.
Competition: https://www.kaggle.com/competitions/sweettv-tv-program-recommender

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.cli run --raw-dir data/raw/extracted --run-dir experiments/competition-20261030-03 --calendar-config config/calendar-boundaries.json
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/prepare_improvement.py --base-run experiments/competition-20261030-03 --output-root experiments/improvement-v2-data
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.improve --development-prepared experiments/improvement-v2-data/development --confirmation-prepared experiments/competition-20261030-03/prepared --recency-dir experiments/improvement-v2-data/recency --template data/raw/extracted/submission.csv --run-dir experiments/improvement-v2-01 --config config/improvement-v2.json
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_improvement.py
```

Run directories are immutable; use a fresh checkout or new names for a rerun. The preparation helper opens the existing SQLite database read-only. Early tuning uses only its train/valid interactions, metadata and candidates; emitted full/future sections are unused. Recency aggregates historical accepted airings only, with exact raw seconds/watches parity against each training window. No future user viewing enters any model.

## Methods and limitations

See `docs/methods.md` for the original methods, `docs/improved-methods.md` for the new seed definition and fixed grid, `config/improvement-v2.json` for the protocol, and `results/` for actual aggregate exports. The PPR graph is unchanged; only restart weights and continuation change. The two personal-history diagnostics do not count as a qualifying recommendation family.

The half-open interval join unions overlapping user/channel views, resolves duplicate/ambiguous airings, and requires 120 accepted seconds per canonical airing after aggregating fragments. ID 0 is unlabelled context, never a preference or recommendation. Repeated TV programmes may be recommended for all methods; ties use ascending programme IDs. All fitted representations use training history only.

Only 10.76%, 11.31% and 11.18% of schedule rows are labelled. Missing labels and absence from a log do not establish dislike or never watching. Timestamp timezone is unspecified; 117 declared viewing durations differ from their wall-clock endpoints by one hour. No enjoyment, revenue or online causal effect is measured.

Implementation and automated audits used Codex; they are not student personal verification. Course reports, genuine score screenshots and complete unmodified native logs remain private. Raw competition data, supplied course sources, credentials and private session logs are excluded. Dependencies retain their upstream licenses.
