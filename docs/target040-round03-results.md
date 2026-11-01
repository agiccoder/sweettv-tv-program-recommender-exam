# target040-round03-exposure-rate: actual results

Run `experiments/target040-round03-20261101-01`; protocol frozen at 2026-11-01T13:08:01.195254+00:00 with SHA-256 `03922b57082e3a23ce7df023939b6a7ec31cfeed145a9c08f894060a8ef9ca58`. 48 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **rate_mix_seconds_50**. Replaces the previous local best: **True**. Raw aggregate winner: `rate_mix_seconds_50`. Strict wins over the reference: 3/3.

## Question and protocol

Raw train favourite ranks ignore changing broadcast opportunity. Reused v3 watch-target misses have lower median future/historical airing-rate change than hits; apply bounded availability-rate reranking to independently evaluated PPR seeds.

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

Mixed PPR changes only restart: v=(1-c)L1(sqrt(seconds)) + cL1(watches or recent_watches), c in {.5,.75,1}, link-following alpha=.1. Recent watches sum one28-day decay per accepted canonical airing using the last clipped positive overlap end, bounded by cutoff. It does not decay one last programme timestamp. This is personalized PageRank, not a new unrelated architecture. Two additional PPR/history hybrids use .5/.5 and .25/.75 reciprocal-rank weights.

v3 private .36630 was first observed and exact submitted bytes downloaded after round01 freeze. It confirms the target remains unmet; no returned score is used to fit or select this round.

This round's new availability correction is score_base(u,i) * clip(((target_exposure_i+s)/target_days)/((historical_exposure_i+s)/train_days),.25,4)**power. Exposure is either total nonconflicting broadcast seconds (3600s smoothing) or canonical airings with >=120 nonconflicting seconds (one-airing smoothing). Powers .5/1 and the [.25,4] gain cap were frozen. The full base score over eligible known items is transformed before native top50 retrieval; no padding enters this transformation. Target airtime is a fixed allowed availability feature, not future user behaviour or a learned future normalization.

Round02 locally selected candidate returned private .35452, below already verified v3 .36630. Rate features and error diagnosis were prepared separately; parameter choices and local objective do not optimize returned scores.

Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.

## All actual comparisons

Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).

| Method | weeks5 | weeks10 | weeks12 | Aggregate | Δ previous best |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.199015 / 0.327608 / 0.764383 | 0.206625 / 0.336254 / 0.798988 | 0.199612 / 0.335432 / 0.827062 | 0.267424 | -0.136085 |
| content | 0.048400 / 0.055433 / 0.363651 | 0.044133 / 0.047318 / 0.388974 | 0.030845 / 0.033855 / 0.355314 | 0.043330 | -0.360179 |
| ppr_v2 | 0.339388 / 0.397791 / 0.741257 | 0.349435 / 0.418597 / 0.755901 | 0.332665 / 0.408097 / 0.794755 | 0.374329 | -0.029181 |
| history_watches | 0.331904 / 0.465018 / 0.791503 | 0.336613 / 0.476296 / 0.803108 | 0.320099 / 0.457442 / 0.830476 | 0.397895 | -0.005615 |
| channel_opportunity_total | 0.329789 / 0.363150 / 0.789486 | 0.335712 / 0.369370 / 0.798951 | 0.318140 / 0.349425 / 0.820585 | 0.344264 | -0.059245 |
| hybrid_opportunity_total_70_20_10 | 0.359570 / 0.422776 / 0.804916 | 0.370394 / 0.443770 / 0.825609 | 0.349877 / 0.427068 / 0.849415 | 0.395576 | -0.007934 |
| ppr_mix_rw_75 | 0.340619 / 0.465561 / 0.790145 | 0.346866 / 0.478368 / 0.805414 | 0.330254 / 0.459391 / 0.835605 | 0.403510 | +0.000000 |
| rate_v2_seconds_50 | 0.344483 / 0.405980 / 0.727337 | 0.361272 / 0.438581 / 0.776595 | 0.341318 / 0.419765 / 0.807469 | 0.385233 | -0.018276 |
| rate_v2_seconds_100 | 0.269466 / 0.305514 / 0.537215 | 0.302910 / 0.354792 / 0.650343 | 0.272938 / 0.312641 / 0.675849 | 0.303043 | -0.100466 |
| rate_v2_airings_50 | 0.348807 / 0.410868 / 0.733805 | 0.358482 / 0.440702 / 0.781060 | 0.335581 / 0.426814 / 0.805164 | 0.386876 | -0.016634 |
| rate_v2_airings_100 | 0.277878 / 0.311333 / 0.560625 | 0.306069 / 0.373958 / 0.664754 | 0.273226 / 0.337052 / 0.678621 | 0.313253 | -0.090257 |
| rate_mix_seconds_50 | 0.349985 / 0.481525 / 0.805781 | 0.356648 / 0.493306 / 0.828118 | 0.338749 / 0.473486 / 0.848781 | 0.415616 | +0.012107 |
| rate_mix_seconds_100 | 0.336580 / 0.456184 / 0.747481 | 0.348223 / 0.475884 / 0.794849 | 0.325728 / 0.438794 / 0.804552 | 0.396899 | -0.006611 |
| rate_mix_airings_50 | 0.349654 / 0.482163 / 0.810264 | 0.354207 / 0.495548 / 0.835720 | 0.333911 / 0.472812 / 0.851318 | 0.414716 | +0.011206 |
| rate_mix_airings_100 | 0.337624 / 0.457906 / 0.758020 | 0.343478 / 0.480242 / 0.800457 | 0.315536 / 0.443280 / 0.804828 | 0.396344 | -0.007165 |
| history_rate_airings | 0.334977 / 0.468289 / 0.777654 | 0.337907 / 0.483265 / 0.814501 | 0.318241 / 0.463470 / 0.833766 | 0.401025 | -0.002485 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks5 | ppr_mix_rw_75 | 0.656604 / 0.847979 | 0.340619 / 0.465561 | 0/1170 |
| weeks5 | rate_mix_seconds_50 | 0.656116 / 0.848468 | 0.349985 / 0.481525 | 0/1170 |
| weeks10 | ppr_mix_rw_75 | 0.683992 / 0.846734 | 0.346866 / 0.478368 | 0/1037 |
| weeks10 | rate_mix_seconds_50 | 0.688347 / 0.852218 | 0.356648 / 0.493306 | 0/1037 |
| weeks12 | ppr_mix_rw_75 | 0.677466 / 0.829511 | 0.330254 / 0.459391 | 0/739 |
| weeks12 | rate_mix_seconds_50 | 0.679711 / 0.832398 | 0.338749 / 0.473486 | 0/739 |

Measured evaluation/refit wall time 114.210s, peak process RSS 1.259GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 178,368 AP rows, 600,201 fusion terms and 185,350 opportunity scores, with exact selection reproduction and 74,340 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 14,411 full restart rows, 600 sampled PPR scores, 161,190 availability gains and 5,390 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round03-20261101-01/final/model.pkl`; `experiments/target040-round03-20261101-01/final/submission.csv`. SHA-256 `7d738470463a9b1804398e9e858d5531e5481b79cd969f757b24bbca4c233d3c`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The original audited v3 prepared data and cached fitted states must be present and hash-identical. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round03.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
