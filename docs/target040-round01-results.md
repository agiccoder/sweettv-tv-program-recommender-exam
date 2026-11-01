# target040-round01-opportunity: actual results

Run `experiments/target040-round01-20261101-01`; protocol frozen at 2026-11-01T12:50:27.544066+00:00 with SHA-256 `6634025c4db35a54303b8580891a279ee3fb06dc259de1930e3fb40475f4dda4`. 39 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **history_watches**. Replaces the previous local best: **False**. Raw aggregate winner: `history_watches`. Strict wins over the reference: 0/3.

## Question and protocol

Existing exponent0 channel_time retained. Only candidate availability changes; raw training-only profile/smoothing/ID vocabulary unchanged.

The three cutoffs and targets are reused, overlapping and previously inspected; they are not independent untouched tests. Same frozen eligible template users, candidate pools, allowed repeats and training-viewer-popularity fallback across methods. User IDs are exact strings. Candidate availability uses deduplicated nonconflicting schedule seconds, never future viewing or content fitting. All trained profiles, graph edges, vocabularies, factor states and statistics end before the respective cutoff.

S = Top5 accepted seconds; W = Top5 accepted-airing count; A = all observed programmes, diagnostic only. The hidden target formula is unknown. Selection maximizes the exact rational mean across folds of .5 S + .5 W; exact ties prefer the previous best, then frozen order. A replacement must improve the mean and win at least2/3 folds. No returned private score chooses local parameters.

| Fold | Train end | Target end | Users | Candidates |
| --- | --- | --- | --- | --- |
| weeks5 | 2020-04-13T00:00:00 | 2020-05-18T00:00:00 | 1229 | 4412 |
| weeks10 | 2020-04-20T00:00:00 | 2020-06-29T00:00:00 | 1240 | 6449 |
| weeks12 | 2020-05-04T00:00:00 | 2020-07-27T00:00:00 | 1247 | 7049 |

## Components and parent families

Content and PPR are evaluated as separate permitted families, with a separate popularity baseline. Personal histories and channel availability are auxiliary signals; neither counts as another family. The earlier v3 ALS32/64 evaluations remain preserved, including their negative results.

Opportunity context retains the v3 historical unioned channel × weekday ×4h counts, including unlabelled airtime, and training-global21600s smoothing. Candidate score = v3 mean context affinity × (unique target broadcast hours)^power, power .5 or1. Exponent0 is the existing mean-affinity ablation. True empty-history users have no native affinity. Multiplicity after schedule deduplication is intentional broadcast opportunity; accidental row duplication cannot boost it. Raw scores are not calibrated probabilities.

All PPR methods use the same training-only co-view graph: X=row-L1(log1p(seconds)); XᵀX without diagonal, row top100, row normalization. Dangling mass returns to the personal restart; L1 tolerance1e-8 and cap300. Native lists have at most50 positive finite scores. RRF uses fixed weight/(60+rank), missing=0, no per-item renormalization, and fallback only after fusion. Hybrid candidate unions include each history native list, PPR and the selected channel list, <=250 IDs.

Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.

## All actual comparisons

Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).

| Method | weeks5 | weeks10 | weeks12 | Aggregate | Δ previous best |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.199015 / 0.327608 / 0.764383 | 0.206625 / 0.336254 / 0.798988 | 0.199612 / 0.335432 / 0.827062 | 0.267424 | -0.130471 |
| content | 0.048400 / 0.055433 / 0.363651 | 0.044133 / 0.047318 / 0.388974 | 0.030845 / 0.033855 / 0.355314 | 0.043330 | -0.354565 |
| ppr_v2 | 0.339388 / 0.397791 / 0.741257 | 0.349435 / 0.418597 / 0.755901 | 0.332665 / 0.408097 / 0.794755 | 0.374329 | -0.023566 |
| history_watches | 0.331904 / 0.465018 / 0.791503 | 0.336613 / 0.476296 / 0.803108 | 0.320099 / 0.457442 / 0.830476 | 0.397895 | +0.000000 |
| channel_time | 0.021684 / 0.013816 / 0.176455 | 0.016473 / 0.010815 / 0.141331 | 0.012171 / 0.009912 / 0.158351 | 0.014145 | -0.383750 |
| personal_history_fusion | 0.347418 / 0.394773 / 0.747841 | 0.359491 / 0.413612 / 0.764940 | 0.339850 / 0.396354 / 0.789789 | 0.375249 | -0.022646 |
| hybrid_70_20_10 | 0.290005 / 0.323398 / 0.735560 | 0.308485 / 0.341444 / 0.751124 | 0.293066 / 0.341184 / 0.781649 | 0.316264 | -0.081631 |
| channel_opportunity_sqrt | 0.178251 / 0.162078 / 0.582165 | 0.204004 / 0.192434 / 0.637652 | 0.199019 / 0.193261 / 0.670249 | 0.188175 | -0.209721 |
| hybrid_opportunity_sqrt_70_20_10 | 0.353011 / 0.391598 / 0.782436 | 0.362012 / 0.412958 / 0.803267 | 0.340620 / 0.397915 / 0.825929 | 0.376352 | -0.021543 |
| hybrid_opportunity_sqrt_80_15_05 | 0.353354 / 0.400492 / 0.769915 | 0.361585 / 0.420233 / 0.786413 | 0.342660 / 0.408952 / 0.818899 | 0.381213 | -0.016683 |
| channel_opportunity_total | 0.329789 / 0.363150 / 0.789486 | 0.335712 / 0.369370 / 0.798951 | 0.318140 / 0.349425 / 0.820585 | 0.344264 | -0.053631 |
| hybrid_opportunity_total_70_20_10 | 0.359570 / 0.422776 / 0.804916 | 0.370394 / 0.443770 / 0.825609 | 0.349877 / 0.427068 / 0.849415 | 0.395576 | -0.002319 |
| hybrid_opportunity_total_80_15_05 | 0.351734 / 0.414387 / 0.784824 | 0.361831 / 0.435540 / 0.802505 | 0.343087 / 0.419821 / 0.830265 | 0.387733 | -0.010162 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks5 | history_watches | 0.643545 / 0.822389 | 0.331904 / 0.465018 | 0/1170 |
| weeks10 | history_watches | 0.676250 / 0.834637 | 0.336613 / 0.476296 | 0/1037 |
| weeks12 | history_watches | 0.678268 / 0.830152 | 0.320099 / 0.457442 | 0/739 |

Measured evaluation/refit wall time 170.782s, peak process RSS 1.044GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 144,924 AP rows, 3,519,144 fusion terms and 370,700 opportunity scores, with exact selection reproduction and 130,095 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round01-20261101-01/final/model.pkl`; `experiments/target040-round01-20261101-01/final/submission.csv`. SHA-256 `dc2da045bdea6d589826f73816bde48cb499c984f6084423c4d76364aa1bc1b9`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The original audited v3 prepared data and cached fitted states must be present and hash-identical. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round01.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
