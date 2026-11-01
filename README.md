# SWEET.TV programme recommender

Author: Timur Khayrislamov (Хайрисламов Тимур). Implementation and automated experiments/checks were performed with Codex. No personal student verification is invented.

## Verified current submission

Completed own-account Kaggle submission **57875732**, returned **PRIVATE 0.39000**, public **0.40643**. The exact scored file is [submission.csv](submission.csv), method `mfshare_rate_16_50`. The own-submission server download matched its bytes exactly, and a fresh refit without fitted-factor reuse reproduced the same file.

SHA-256: `8b722b51734d6b7d6b46be0b563afce424dbbf5214dd7e18170ca82152b9b7e1`.

All1260 exact template user strings, order and headers are retained; each row has five distinct positive eligible programme IDs. Original v1/v2 files remain under `submissions/`. Raw competition data, trained user factors, viewing histories, credentials and authentic private session records are excluded from this repository.

The requested PRIVATE>=0.40000 target is NOT achieved; a public score above0.40 does not meet it.

## Methods, local comparison and feedback

The finite rounds compare popularity separately from content TF-IDF, personalized PageRank and implicit matrix factorization. PPR uses a training-only co-view graph and explicit personalized restarts. MF compares normalized count-intensity KL reconstruction, paired count/duration heads sharing user factors, squared-error reconstruction and declared input transforms. Rank fusion uses weight/(60+rank), missing scores contribute0, and the same training-popularity fallback follows every method. Histories and permitted channel/time/broadcast availability are auxiliary signals, not additional qualifying families.

See [execution status](docs/target040-status.md), the individual round reports, [all genuine returned results](docs/target040-scoreboard.md), and `results/target040-best/` for exact candidate configuration, genuine score linkage and fresh-refit proof. Top5 accepted seconds and Top5 accepted-airing counts are separate local proxy targets; the hidden target definition is unknown. The three long-horizon folds use8/12/14-week histories and12/8/6-week horizons, with identical cohorts/pools/repeat policy and cold users retained. Folds overlap and were previously inspected. Local selection maximizes the exact mean of0.5S+0.5W and also requires wins in at least2/3 folds. Neither local nor public MAP is claimed as a private score.

The user's explicit final batch instruction authorized many distinct submissions within the real100/day quota. Portfolios are finite and frozen before their new metrics; exact duplicate hashes are skipped. All external results, including losses, are retained. Choosing the best returned private score follows the exam acceptance rule and is exposed to repeated feedback; it is not an independent untouched evaluation. There are no individual-user probes, leaked targets or weeks31–42 viewing features.

## Reproduce

Recorded environment: Python3.9.6 on macOS ARM64, dependencies pinned in `requirements-v3.lock`. Start with a fresh checkout and your own authorized unchanged five competition CSVs under `data/raw/extracted/`. Personally accept Kaggle rules where required; no credentials are needed for local fitting. Preparation manifests contain fresh timestamps/runtime hashes; the one-candidate refitter freezes actual rebuilt input hashes before fitting while preserving the original model settings/windows and requiring the exact expected CSV hash.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-v3.lock
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/verify_target040.py --destination evidence/new-verification
.venv/bin/python -m sweettv.cli run --raw-dir data/raw/extracted --run-dir experiments/competition-20261030-03 --calendar-config config/calendar-boundaries.json
.venv/bin/python scripts/prepare_improvement_v3.py --base-run experiments/competition-20261030-03 --config config/target040-round05-preparation.json --output-root experiments/target040-transfer-data-01
.venv/bin/python scripts/prepare_target040_exposure.py --config config/target040-round05-preparation.json --prepared experiments/target040-transfer-data-01 --output experiments/target040-transfer-exposure-01
.venv/bin/python scripts/prepare_target040_engagement.py --output-dir experiments/target040-engagement-data-01
.venv/bin/python scripts/prepare_target040_schedule_factors.py --output experiments/target040-schedule-factors-01
.venv/bin/python scripts/refit_target040_candidate.py --source-run results/target040-best --prepared experiments/target040-transfer-data-01 --output experiments/new-best-refit --expected-sha256 8b722b51734d6b7d6b46be0b563afce424dbbf5214dd7e18170ca82152b9b7e1
```

All destinations are immutable; use a fresh checkout or new names for reruns. The refitter accepts explicit alternative feature paths. It loads no saved factors and performs no new quality search; it fails if the final CSV differs from the scored hash. The original120s canonical-airing acceptance, interval unions/conflict handling and source-calendar bounds remain intact. Item0 is unlabelled context, never an item preference. Future schedule data supplies permitted identity/channel/time/duration availability only; every fitted preference, graph and factor ends before its cutoff.

105 synthetic correctness tests passed in the development workspace and the separately staged source package. Actual independent round10/11 checks recomputed761940 AP rows and verified218 original protected artifacts unchanged. All portfolio files have independent full-user ranking/fallback/CSV audits and persisted-model round-trip checks. These checks were automated by Codex. The exact final source-package verification is under `results/`.

## Scope of evidence

Real API evidence proves the score/CSV linkage. It is not claimed as a substitute for the original exam's required Kaggle Submissions-page screenshot. Earlier report drafts/native session material remain private and are not silently relabelled as a complete updated course hand-in. [Historical v2 documentation](docs/v2-readme-historical.md) is retained for provenance; its current-model wording refers to its original publication date.
