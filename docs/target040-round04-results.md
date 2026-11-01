# target040-round04-count-intensity-mf: actual results

Run `experiments/target040-round04-20261101-01`; protocol frozen at 2026-11-01T13:17:36.471250+00:00 with SHA-256 `7e2bb424a75a59b5e0f200dcb5b3a1e3ffc00a053f7aadde7688f2949d2d51d0`. 48 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **rrf_ppr_nmf16_10**. Replaces the previous local best: **True**. Raw aggregate winner: `rrf_ppr_nmf16_10`. Strict wins over the reference: 3/3.

## Question and protocol

Binary preference ALS poorly recovers repeats. Direct relative count-intensity factorization may preserve preference strength and complement PPR/history.

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

The new implicit MF fits q_ui=watches_ui/sum_j watches_uj with nonnegative factors under generalized KL(q||WH), plus scaled L2 penalties alpha_W=alpha_H=1e-6. This models relative recorded count intensity, not fabricated ratings. Zeros influence the objective and approximate equal observation opportunity; they are not verified dislikes. Fixed ranks16/32, random seed20261030, multiplicative updates, cap100, tol1e-4; no validation early stopping. Observed items remain eligible; unknown IDs get no invented factors. Official [scikit-learn1.6.1 NMF documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.NMF.html) and installed orientation/persistence were checked. Per-state diagnostics retain convergence warnings, actual iterations, reconstruction error and factor variance. The independent auditor reconstructs normalized input, all factor shapes/nonnegativity and the full KL loss including the installed solver's numerical epsilon convention, then checks sampled scores/retrieval without production scoring. Four fixed rank fusions per factor use either the previous PPR winner or personal watches, with MF weights .10/.25.

Round02 private .35452 and round03 .35327, both below existing v3 .36630; all retained. The new objective is motivated by local ALS repeat/ranking failures and uses the same frozen dual-proxy selection, not returned scores.

Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.

## All actual comparisons

Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).

| Method | weeks5 | weeks10 | weeks12 | Aggregate | Δ previous best |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.199015 / 0.327608 / 0.764383 | 0.206625 / 0.336254 / 0.798988 | 0.199612 / 0.335432 / 0.827062 | 0.267424 | -0.148192 |
| content | 0.048400 / 0.055433 / 0.363651 | 0.044133 / 0.047318 / 0.388974 | 0.030845 / 0.033855 / 0.355314 | 0.043330 | -0.372286 |
| ppr_v2 | 0.339388 / 0.397791 / 0.741257 | 0.349435 / 0.418597 / 0.755901 | 0.332665 / 0.408097 / 0.794755 | 0.374329 | -0.041288 |
| history_watches | 0.331904 / 0.465018 / 0.791503 | 0.336613 / 0.476296 / 0.803108 | 0.320099 / 0.457442 / 0.830476 | 0.397895 | -0.017721 |
| ppr_mix_rw_75 | 0.340619 / 0.465561 / 0.790145 | 0.346866 / 0.478368 / 0.805414 | 0.330254 / 0.459391 / 0.835605 | 0.403510 | -0.012107 |
| rate_mix_seconds_50 | 0.349985 / 0.481525 / 0.805781 | 0.356648 / 0.493306 / 0.828118 | 0.338749 / 0.473486 / 0.848781 | 0.415616 | +0.000000 |
| count_nmf16 | 0.283447 / 0.435414 / 0.769835 | 0.276728 / 0.433102 / 0.786239 | 0.275766 / 0.433456 / 0.825250 | 0.356319 | -0.059298 |
| rrf_ppr_nmf16_10 | 0.349016 / 0.489086 / 0.822175 | 0.355411 / 0.501723 / 0.841175 | 0.339169 / 0.483456 / 0.865814 | 0.419644 | +0.004027 |
| rrf_ppr_nmf16_25 | 0.341053 / 0.491091 / 0.827671 | 0.346728 / 0.504540 / 0.846202 | 0.331115 / 0.487207 / 0.866597 | 0.416956 | +0.001339 |
| rrf_history_nmf16_10 | 0.332148 / 0.470968 / 0.796230 | 0.334454 / 0.479398 / 0.809610 | 0.320430 / 0.462997 / 0.836576 | 0.400066 | -0.015551 |
| rrf_history_nmf16_25 | 0.327101 / 0.474682 / 0.801104 | 0.329556 / 0.486790 / 0.819919 | 0.318190 / 0.470791 / 0.843675 | 0.401185 | -0.014431 |
| count_nmf32 | 0.290093 / 0.436540 / 0.769266 | 0.300371 / 0.453032 / 0.791309 | 0.289781 / 0.438589 / 0.817094 | 0.368068 | -0.047549 |
| rrf_ppr_nmf32_10 | 0.349119 / 0.487820 / 0.818468 | 0.355387 / 0.499747 / 0.839379 | 0.339153 / 0.480414 / 0.863397 | 0.418607 | +0.002990 |
| rrf_ppr_nmf32_25 | 0.341689 / 0.487989 / 0.823408 | 0.349626 / 0.501355 / 0.842062 | 0.336140 / 0.486028 / 0.865557 | 0.417138 | +0.001521 |
| rrf_history_nmf32_10 | 0.331378 / 0.468549 / 0.794321 | 0.334145 / 0.478860 / 0.807688 | 0.319530 / 0.460056 / 0.834167 | 0.398753 | -0.016863 |
| rrf_history_nmf32_25 | 0.327699 / 0.470745 / 0.798841 | 0.331422 / 0.482538 / 0.811000 | 0.319107 / 0.465384 / 0.837263 | 0.399482 | -0.016134 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks5 | rate_mix_seconds_50 | 0.656116 / 0.848468 | 0.349985 / 0.481525 | 0/1170 |
| weeks5 | rrf_ppr_nmf16_10 | 0.691592 / 0.891592 | 0.349016 / 0.489086 | 0/1170 |
| weeks10 | rate_mix_seconds_50 | 0.688347 / 0.852218 | 0.356648 / 0.493306 | 0/1037 |
| weeks10 | rrf_ppr_nmf16_10 | 0.725645 / 0.893548 | 0.355411 / 0.501723 | 0/1037 |
| weeks12 | rate_mix_seconds_50 | 0.679711 / 0.832398 | 0.338749 / 0.473486 | 0/739 |
| weeks12 | rrf_ppr_nmf16_10 | 0.732478 / 0.883079 | 0.339169 / 0.483456 | 0/739 |

Measured evaluation/refit wall time 122.754s, peak process RSS 1.232GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 178,368 AP rows, 2,910,852 fusion terms and 0 opportunity scores, with exact selection reproduction and 74,340 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 14,411 full restart rows, 600 sampled PPR scores, 17,910 availability gains and 600 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round04-20261101-01/final/model.pkl`; `experiments/target040-round04-20261101-01/final/submission.csv`. SHA-256 `072d7c50e5d9b3d6da83c4f0fa0ae956382ed23c2898daeafbc24333e66cc01d`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The original audited v3 prepared data and cached fitted states must be present and hash-identical. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round04.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
