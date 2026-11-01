# target040-round11-input-intensity: actual results

Run `experiments/target040-round11-20261101-01`; protocol frozen at 2026-11-01T15:11:04.997143+00:00 with SHA-256 `5f2bfd179d8a008567000e29111a44698aa2783b898375fe3e77241247f7aecf`. 93 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **eng_rrf_recent_capped600**. Replaces the previous local best: **False**. Raw aggregate winner: `eng_rrf_recent_capped600`. Strict wins over the reference: 0/3.

## Question and protocol

Finite implicit MF input ablation: reduce heavy-tail intensity or retain activity weighting; compare square-root count, unnormalized count, and square-root seconds at inherited ranks16/32.

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

Batch01 best private .37949 is visible; these periods/results are not independent. Joint MF rank/score-view plan was frozen before batch01 scores; squared loss is a separate head-error hypothesis. Numerical losses/ranks/solvers/fixed fusions frozen before all new metrics; no algorithm fits to private scores.

Accepted-airing engagement features retain the original120s acceptance and canonical join. Thresholding happens before any transform. Each feature is training-only, with exact seconds/watches parity and bounded event endpoints. The following raw auxiliary histories are separately evaluated; they do not count as another qualifying family:

| Signal | Definition |
| --- | --- |
| capped600 | sum min(accepted canonical airing seconds/600,1) |
| capped1800 | sum min(accepted canonical airing seconds/1800,1) |
| fractional_watches | sum min(accepted canonical airing seconds/canonical airing duration,1) |
| active_days | number of distinct arithmetic source days with an accepted canonical airing; day defined by last clipped positive overlap end minus1 microsecond; union across channels |
| recent_capped600 | sum min(accepted canonical airing seconds/600,1)*2**(-(cutoff-last_overlap_end)/(28*86400)) |

Each signal also forms a normalized PPR restart on the unchanged training graph, alpha=.1. Fixed availability power.5 and PPR/MF RRF weights.9/.1 are inherited without tuning. All frozen trials remain in the actual table.

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
| popularity | 0.199612 / 0.335432 / 0.827062 | 0.166653 / 0.291760 / 0.746498 | 0.152923 / 0.275247 / 0.711174 | 0.236938 | -0.135270 |
| content | 0.030845 / 0.033855 / 0.355314 | 0.023617 / 0.024526 / 0.243025 | 0.022245 / 0.025115 / 0.217497 | 0.026701 | -0.345507 |
| ppr_v2 | 0.332665 / 0.408097 / 0.794755 | 0.282717 / 0.373408 / 0.739609 | 0.259464 / 0.365389 / 0.733320 | 0.336957 | -0.035251 |
| history_watches | 0.320099 / 0.457442 / 0.830476 | 0.271455 / 0.423432 / 0.767157 | 0.250043 / 0.410464 / 0.743483 | 0.355489 | -0.016719 |
| ppr_mix_rw_75 | 0.330254 / 0.459391 / 0.835605 | 0.282757 / 0.426675 / 0.766618 | 0.261702 / 0.416680 / 0.752460 | 0.362910 | -0.009298 |
| rate_mix_seconds_50 | 0.338749 / 0.473486 / 0.848781 | 0.288524 / 0.431503 / 0.771399 | 0.266005 / 0.417162 / 0.753832 | 0.369238 | -0.002970 |
| count_nmf16 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057576 |
| rrf_ppr_nmf16_10 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788641 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | -0.000254 |
| eng_ppr_recent_capped600 | 0.336057 / 0.454138 / 0.833427 | 0.287023 / 0.421683 / 0.763668 | 0.264140 / 0.412243 / 0.749349 | 0.362547 | -0.009661 |
| eng_rate_recent_capped600 | 0.342580 / 0.468407 / 0.844852 | 0.292851 / 0.426919 / 0.769954 | 0.268550 / 0.411683 / 0.747929 | 0.368498 | -0.003710 |
| eng_rrf_recent_capped600 | 0.342526 / 0.479882 / 0.863336 | 0.290418 / 0.431573 / 0.786768 | 0.267511 / 0.421337 / 0.764960 | 0.372208 | +0.000000 |
| count_nmf32 | 0.289781 / 0.438589 / 0.817094 | 0.247947 / 0.406562 / 0.760171 | 0.225016 / 0.391747 / 0.731115 | 0.333274 | -0.038934 |
| count_nmf64 | 0.307116 / 0.451147 / 0.821537 | 0.259019 / 0.416426 / 0.769177 | 0.234477 / 0.402071 / 0.741455 | 0.345043 | -0.027165 |
| mfinput_sqrtcount_16 | 0.250051 / 0.403777 / 0.824822 | 0.204861 / 0.358888 / 0.754708 | 0.187789 / 0.344252 / 0.712618 | 0.291603 | -0.080605 |
| rrf_mfinput_sqrtcount_16_10 | 0.336333 / 0.481590 / 0.868038 | 0.284279 / 0.432878 / 0.787361 | 0.259887 / 0.420769 / 0.765772 | 0.369289 | -0.002918 |
| rrf_mfinput_sqrtcount_16_50 | 0.299642 / 0.466632 / 0.873630 | 0.251085 / 0.417020 / 0.801484 | 0.227044 / 0.397781 / 0.769598 | 0.343201 | -0.029007 |
| mfinput_sqrtcount_32 | 0.260024 / 0.403681 / 0.817319 | 0.223424 / 0.370011 / 0.755155 | 0.201200 / 0.359440 / 0.724724 | 0.302963 | -0.069244 |
| rrf_mfinput_sqrtcount_32_10 | 0.338540 / 0.481460 / 0.864659 | 0.286002 / 0.434566 / 0.786667 | 0.261444 / 0.421308 / 0.764767 | 0.370553 | -0.001655 |
| rrf_mfinput_sqrtcount_32_50 | 0.308500 / 0.471152 / 0.865079 | 0.261602 / 0.423569 / 0.804711 | 0.235383 / 0.408706 / 0.771554 | 0.351485 | -0.020722 |
| mfinput_rawcount_16 | 0.275736 / 0.435592 / 0.822499 | 0.232023 / 0.389887 / 0.757189 | 0.209416 / 0.375410 / 0.724770 | 0.319677 | -0.052530 |
| rrf_mfinput_rawcount_16_10 | 0.339142 / 0.482566 / 0.865873 | 0.287208 / 0.435667 / 0.786758 | 0.262093 / 0.422567 / 0.766332 | 0.371540 | -0.000667 |
| rrf_mfinput_rawcount_16_50 | 0.312168 / 0.480396 / 0.864403 | 0.264169 / 0.431050 / 0.802224 | 0.240614 / 0.417401 / 0.774416 | 0.357633 | -0.014575 |
| mfinput_rawcount_32 | 0.294501 / 0.441398 / 0.816068 | 0.244539 / 0.406908 / 0.762808 | 0.216364 / 0.380863 / 0.721455 | 0.330762 | -0.041446 |
| rrf_mfinput_rawcount_32_10 | 0.339337 / 0.481353 / 0.865295 | 0.288111 / 0.435142 / 0.786943 | 0.264879 / 0.424287 / 0.765748 | 0.372185 | -0.000023 |
| rrf_mfinput_rawcount_32_50 | 0.323836 / 0.483823 / 0.863053 | 0.271399 / 0.438044 / 0.801174 | 0.247395 / 0.420857 / 0.774386 | 0.364226 | -0.007982 |
| mfinput_sqrtseconds_16 | 0.248062 / 0.365162 / 0.804210 | 0.213119 / 0.326289 / 0.724043 | 0.192178 / 0.314392 / 0.697586 | 0.276534 | -0.095674 |
| rrf_mfinput_sqrtseconds_16_10 | 0.339997 / 0.476707 / 0.865127 | 0.286452 / 0.429252 / 0.785748 | 0.261265 / 0.417538 / 0.766594 | 0.368535 | -0.003673 |
| rrf_mfinput_sqrtseconds_16_50 | 0.306028 / 0.444012 / 0.864924 | 0.258357 / 0.400764 / 0.796718 | 0.234338 / 0.387366 / 0.769879 | 0.338478 | -0.033730 |
| mfinput_sqrtseconds_32 | 0.284528 / 0.384015 / 0.806407 | 0.224239 / 0.336870 / 0.721656 | 0.197781 / 0.320991 / 0.697283 | 0.291404 | -0.080804 |
| rrf_mfinput_sqrtseconds_32_10 | 0.340102 / 0.479099 / 0.862473 | 0.287875 / 0.431645 / 0.785107 | 0.262296 / 0.418596 / 0.765110 | 0.369936 | -0.002272 |
| rrf_mfinput_sqrtseconds_32_50 | 0.328097 / 0.459396 / 0.864726 | 0.269271 / 0.408615 / 0.792363 | 0.239826 / 0.391777 / 0.761093 | 0.349497 | -0.022711 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks12 | eng_rrf_recent_capped600 | 0.731676 / 0.882438 | 0.342526 / 0.479882 | 0/739 |
| late8 | eng_rrf_recent_capped600 | 0.678537 / 0.865997 | 0.290418 / 0.431573 | 0/753 |
| late6 | eng_rrf_recent_capped600 | 0.645257 / 0.857476 | 0.267511 / 0.421337 | 0/706 |

Measured evaluation/refit wall time 585.478s, peak process RSS 1.349GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 347,355 AP rows, 5,199,600 fusion terms and 0 opportunity scores, with exact selection reproduction and 74,700 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 29,038 full restart rows, 1,200 sampled PPR scores, 35,628 availability gains and 1,200 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round11-20261101-01/final/model.pkl`; `experiments/target040-round11-20261101-01/final/submission.csv`. SHA-256 `8cefdc8efeeeebd2d6a97959d37f32afe4e222ba2701ef70b2f8aea917d63789`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The prepared data must match the frozen preparation hash. Cached states are used only when reuse_cached_fits is true; otherwise every cutoff is fitted fresh. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round11.json --prepared experiments/target040-transfer-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
