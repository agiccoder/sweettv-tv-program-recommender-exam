# target040-round05-history-length-transfer: actual results

Run `experiments/target040-round05-20261101-01`; protocol frozen at 2026-11-01T13:33:02.988880+00:00 with SHA-256 `f4f319e4be7274faa4d2fb67ac73a3e980bf7f3bd6f76e3957ab0c7deb75bbaf`. 48 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **rrf_ppr_nmf16_10**. Replaces the previous local best: **False**. Raw aggregate winner: `rrf_ppr_nmf16_10`. Strict wins over the reference: 0/3.

## Question and protocol

Finite transfer diagnosis: same16 previously defined models,12/8/6-week horizons and8/12/14-week training histories, vs20-week final history.

The three cutoffs and targets are reused, overlapping and previously inspected; they are not independent untouched tests. Same frozen eligible template users, candidate pools, allowed repeats and training-viewer-popularity fallback across methods. User IDs are exact strings. Candidate availability uses deduplicated nonconflicting schedule seconds, never future viewing or content fitting. All trained profiles, graph edges, vocabularies, factor states and statistics end before the respective cutoff.

S = Top5 accepted seconds; W = Top5 accepted-airing count; A = all observed programmes, diagnostic only. The hidden target formula is unknown. Selection maximizes the exact rational mean across folds of .5 S + .5 W; exact ties prefer the previous best, then frozen order. A replacement must improve the mean and win at least2/3 folds. No returned private score chooses local parameters.

| Fold | Train end | Target end | Users | Candidates |
| --- | --- | --- | --- | --- |
| weeks12 | 2020-05-04T00:00:00 | 2020-07-27T00:00:00 | 1247 | 7049 |
| late8 | 2020-06-01T00:00:00 | 2020-07-27T00:00:00 | 1244 | 5742 |
| late6 | 2020-06-15T00:00:00 | 2020-07-27T00:00:00 | 1244 | 5023 |

## Components and parent families

Content and PPR are evaluated as separate permitted families, with a separate popularity baseline. Personal histories and channel availability are auxiliary signals; neither counts as another family. The earlier v3 ALS32/64 evaluations remain preserved, including their negative results.

Opportunity context retains the v3 historical unioned channel × weekday ×4h counts, including unlabelled airtime, and training-global21600s smoothing. Candidate score = v3 mean context affinity × (unique target broadcast hours)^power, power .5 or1. Exponent0 is the existing mean-affinity ablation. True empty-history users have no native affinity. Multiplicity after schedule deduplication is intentional broadcast opportunity; accidental row duplication cannot boost it. Raw scores are not calibrated probabilities.

All PPR methods use the same training-only co-view graph: X=row-L1(log1p(seconds)); XᵀX without diagonal, row top100, row normalization. Dangling mass returns to the personal restart; L1 tolerance1e-8 and cap300. Native lists have at most50 positive finite scores. RRF uses fixed weight/(60+rank), missing=0, no per-item renormalization, and fallback only after fusion. Hybrid candidate unions include each history native list, PPR and the selected channel list, <=250 IDs.

Observed v3 .36630; rounds02 .35452,03 .35327,04 .37018. Failure of local gains to transfer motivates training-history-length diagnosis. No new weights/models or private-score selection.

All cutoff-specific states in this round are fitted fresh. The old cached-run manifest is provenance only; no old fold fit or training inputs are reused. Each new prepared interaction/context/recency/availability file is hashed in this run. Trial exports inherited a generic cached-fit timing label; for fresh folds this label does not apply: the actual nonzero component-fit/shared-graph timing fields and manifest reuse_cached_fits=false are authoritative. The12-week fold retains its old source periods for a reproducibility comparison; later cutoffs diagnose training-history length and remain overlapping previously observed periods.

Mixed PPR changes only restart: v=(1-c)L1(sqrt(seconds)) + cL1(watches or recent_watches), c in {.5,.75,1}, link-following alpha=.1. Recent watches sum one28-day decay per accepted canonical airing using the last clipped positive overlap end, bounded by cutoff. It does not decay one last programme timestamp. This is personalized PageRank, not a new unrelated architecture. Two additional PPR/history hybrids use .5/.5 and .25/.75 reciprocal-rank weights.

v3 private .36630 was first observed and exact submitted bytes downloaded after round01 freeze. It confirms the target remains unmet; no returned score is used to fit or select this round.

This round's new availability correction is score_base(u,i) * clip(((target_exposure_i+s)/target_days)/((historical_exposure_i+s)/train_days),.25,4)**power. Exposure is either total nonconflicting broadcast seconds (3600s smoothing) or canonical airings with >=120 nonconflicting seconds (one-airing smoothing). Powers .5/1 and the [.25,4] gain cap were frozen. The full base score over eligible known items is transformed before native top50 retrieval; no padding enters this transformation. Target airtime is a fixed allowed availability feature, not future user behaviour or a learned future normalization.

Round02 locally selected candidate returned private .35452, below already verified v3 .36630. Rate features and error diagnosis were prepared separately; parameter choices and local objective do not optimize returned scores.

The new implicit MF fits q_ui=watches_ui/sum_j watches_uj with nonnegative factors under generalized KL(q||WH), plus scaled L2 penalties alpha_W=alpha_H=1e-6. This models relative recorded count intensity, not fabricated ratings. Zeros influence the objective and approximate equal observation opportunity; they are not verified dislikes. Fixed ranks16/32, random seed20261030, multiplicative updates, cap100, tol1e-4; no validation early stopping. Observed items remain eligible; unknown IDs get no invented factors. Official [scikit-learn1.6.1 NMF documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.NMF.html) and installed orientation/persistence were checked. Per-state diagnostics retain convergence warnings, actual iterations, reconstruction error and factor variance. The independent auditor reconstructs normalized input, all factor shapes/nonnegativity and the full KL loss including the installed solver's numerical epsilon convention, then checks sampled scores/retrieval without production scoring. Four fixed rank fusions per factor use either the previous PPR winner or personal watches, with MF weights .10/.25.

Round02 private .35452 and round03 .35327, both below existing v3 .36630; all retained. The new objective is motivated by local ALS repeat/ranking failures and uses the same frozen dual-proxy selection, not returned scores.

Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.

## All actual comparisons

Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).

| Method | weeks12 | late8 | late6 | Aggregate | Δ previous best |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.199612 / 0.335432 / 0.827062 | 0.166653 / 0.291760 / 0.746498 | 0.152923 / 0.275247 / 0.711174 | 0.236938 | -0.135016 |
| content | 0.030845 / 0.033855 / 0.355314 | 0.023617 / 0.024526 / 0.243025 | 0.022245 / 0.025115 / 0.217497 | 0.026701 | -0.345253 |
| ppr_v2 | 0.332665 / 0.408097 / 0.794755 | 0.282717 / 0.373408 / 0.739609 | 0.259464 / 0.365389 / 0.733320 | 0.336957 | -0.034997 |
| history_watches | 0.320099 / 0.457442 / 0.830476 | 0.271455 / 0.423432 / 0.767157 | 0.250043 / 0.410464 / 0.743483 | 0.355489 | -0.016465 |
| ppr_mix_rw_75 | 0.330254 / 0.459391 / 0.835605 | 0.282757 / 0.426675 / 0.766618 | 0.261702 / 0.416680 / 0.752460 | 0.362910 | -0.009044 |
| rate_mix_seconds_50 | 0.338749 / 0.473486 / 0.848781 | 0.288524 / 0.431503 / 0.771399 | 0.266005 / 0.417162 / 0.753832 | 0.369238 | -0.002716 |
| count_nmf16 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057322 |
| rrf_ppr_nmf16_10 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788641 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | +0.000000 |
| rrf_ppr_nmf16_25 | 0.331115 / 0.487207 / 0.866597 | 0.275362 / 0.437401 / 0.799419 | 0.255734 / 0.427540 / 0.775726 | 0.369060 | -0.002894 |
| rrf_history_nmf16_10 | 0.320430 / 0.462997 / 0.836576 | 0.268331 / 0.424569 / 0.772744 | 0.249274 / 0.412923 / 0.747527 | 0.356421 | -0.015533 |
| rrf_history_nmf16_25 | 0.318190 / 0.470791 / 0.843675 | 0.260946 / 0.424788 / 0.778199 | 0.242677 / 0.412162 / 0.752688 | 0.354926 | -0.017028 |
| count_nmf32 | 0.289781 / 0.438589 / 0.817094 | 0.247947 / 0.406562 / 0.760171 | 0.225016 / 0.391747 / 0.731115 | 0.333274 | -0.038680 |
| rrf_ppr_nmf32_10 | 0.339153 / 0.480414 / 0.863397 | 0.288328 / 0.436171 / 0.788124 | 0.264871 / 0.422781 / 0.766763 | 0.371953 | -0.000001 |
| rrf_ppr_nmf32_25 | 0.336140 / 0.486028 / 0.865557 | 0.283438 / 0.439177 / 0.799266 | 0.260595 / 0.424930 / 0.773012 | 0.371718 | -0.000236 |
| rrf_history_nmf32_10 | 0.319530 / 0.460056 / 0.834167 | 0.270828 / 0.423660 / 0.770908 | 0.250257 / 0.412291 / 0.746739 | 0.356104 | -0.015850 |
| rrf_history_nmf32_25 | 0.319107 / 0.465384 / 0.837263 | 0.270166 / 0.428631 / 0.776881 | 0.249668 / 0.416626 / 0.753553 | 0.358264 | -0.013690 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks12 | rrf_ppr_nmf16_10 | 0.732478 / 0.883079 | 0.339169 / 0.483456 | 0/739 |
| late8 | rrf_ppr_nmf16_10 | 0.679662 / 0.866961 | 0.286262 / 0.434520 | 0/753 |
| late6 | rrf_ppr_nmf16_10 | 0.646383 / 0.858923 | 0.264041 / 0.424274 | 0/706 |

Measured evaluation/refit wall time 160.564s, peak process RSS 1.230GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 179,280 AP rows, 2,951,980 fusion terms and 0 opportunity scores, with exact selection reproduction and 24,940 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 14,519 full restart rows, 600 sampled PPR scores, 17,814 availability gains and 600 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round05-20261101-01/final/model.pkl`; `experiments/target040-round05-20261101-01/final/submission.csv`. SHA-256 `072d7c50e5d9b3d6da83c4f0fa0ae956382ed23c2898daeafbc24333e66cc01d`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The prepared data must match the frozen preparation hash. Cached states are used only when reuse_cached_fits is true; otherwise every cutoff is fitted fresh. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round05.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
