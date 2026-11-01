# target040-round10-mf-objective-ablation: actual results

Run `experiments/target040-round10-20261101-01`; protocol frozen at 2026-11-01T15:07:30.483152+00:00 with SHA-256 `c82604f4b77c5e04fb15dab9afa53613eaca78e5706e89cf3d27323cafc86402`. 111 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **eng_rrf_recent_capped600**. Replaces the previous local best: **False**. Raw aggregate winner: `rrf_joint_nmf32_seconds_original`. Strict wins over the reference: 1/3.

## Question and protocol

Finite implicit-MF objective ablation: shared count/duration KL factors and head-sensitive squared count-intensity reconstruction, ranks32/64 inherited, same fixed PPR/MF fusion.

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

Shared-user implicit MF: J=[.5 L1_user(watches),.5 L1_user(seconds)], shared U, two item blocks. Generalized KL plus inherited L2.

Readouts are the count head, duration head, or an equal mean after separate full-candidate-pool L1 normalization. All views keep the same fitted shared user factors. Unknown programme IDs have no invented factors; common fallback follows ranking. Ranks32/64 and fixed fusion weights are frozen before metrics. The independent checker reconstructs both training blocks, the complete KL loss including unobserved cells, factor orientation and sampled full-pool readouts.

Normalized count KL reconstruction emphasizes relative errors throughout long positive tails. Squared intensity reconstruction directly penalizes head magnitude errors; compare using the same target proxies, no hidden preferences.

Squared count-intensity NMF uses coordinate descent in pinned scikit-learn1.6.1, with the same fixed regularization, iteration limit and seed. Independent sparse reconstruction checks ||Q||²-2<Q,UVᵀ>+trace((UᵀU)(VᵀV)) against sklearn reconstruction error.

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
| joint_nmf32_watches | 0.299022 / 0.436779 / 0.810543 | 0.246763 / 0.400600 / 0.755571 | 0.231077 / 0.384507 / 0.728100 | 0.333125 | -0.039083 |
| rrf_joint_nmf32_watches_original | 0.339893 / 0.480310 / 0.863186 | 0.287763 / 0.436377 / 0.787320 | 0.265096 / 0.422320 / 0.764628 | 0.371960 | -0.000248 |
| rrf_joint_nmf32_watches_engagement | 0.344103 / 0.477875 / 0.861080 | 0.291779 / 0.432267 / 0.784657 | 0.268719 / 0.419156 / 0.764022 | 0.372317 | +0.000109 |
| joint_nmf32_seconds | 0.306271 / 0.389209 / 0.785242 | 0.256310 / 0.361117 / 0.714059 | 0.248617 / 0.353237 / 0.710281 | 0.319127 | -0.053081 |
| rrf_joint_nmf32_seconds_original | 0.341534 / 0.479260 / 0.862109 | 0.289745 / 0.435769 / 0.786267 | 0.266680 / 0.421517 / 0.764443 | 0.372418 | +0.000210 |
| rrf_joint_nmf32_seconds_engagement | 0.345830 / 0.476319 / 0.859489 | 0.292956 / 0.431096 / 0.784558 | 0.269309 / 0.417663 / 0.762125 | 0.372195 | -0.000012 |
| joint_nmf32_mean | 0.308484 / 0.419765 / 0.801540 | 0.256664 / 0.387251 / 0.743832 | 0.240699 / 0.374068 / 0.721192 | 0.331155 | -0.041053 |
| rrf_joint_nmf32_mean_original | 0.340842 / 0.479963 / 0.862900 | 0.288789 / 0.436061 / 0.786785 | 0.265941 / 0.422039 / 0.765038 | 0.372272 | +0.000065 |
| rrf_joint_nmf32_mean_engagement | 0.345119 / 0.477982 / 0.861008 | 0.292176 / 0.431492 / 0.785032 | 0.268861 / 0.418625 / 0.762961 | 0.372376 | +0.000168 |
| count_nmf_frob32 | 0.301545 / 0.457709 / 0.844248 | 0.248237 / 0.405595 / 0.769263 | 0.234826 / 0.399928 / 0.740201 | 0.341307 | -0.030901 |
| rrf_count_nmf_frob32_original | 0.337995 / 0.482649 / 0.864820 | 0.285997 / 0.434271 / 0.784845 | 0.263148 / 0.423489 / 0.766415 | 0.371258 | -0.000950 |
| rrf_count_nmf_frob32_engagement | 0.342716 / 0.480433 / 0.864520 | 0.289542 / 0.430268 / 0.783384 | 0.266565 / 0.419477 / 0.761519 | 0.371500 | -0.000708 |
| count_nmf64 | 0.307116 / 0.451147 / 0.821537 | 0.259019 / 0.416426 / 0.769177 | 0.234477 / 0.402071 / 0.741455 | 0.345043 | -0.027165 |
| joint_nmf64_watches | 0.317193 / 0.438081 / 0.820369 | 0.263655 / 0.399721 / 0.757286 | 0.241798 / 0.389486 / 0.731744 | 0.341656 | -0.030552 |
| rrf_joint_nmf64_watches_original | 0.339602 / 0.478222 / 0.861933 | 0.288802 / 0.435410 / 0.784068 | 0.266101 / 0.423084 / 0.764904 | 0.371870 | -0.000338 |
| rrf_joint_nmf64_watches_engagement | 0.344504 / 0.475541 / 0.860436 | 0.293009 / 0.430506 / 0.783156 | 0.269143 / 0.418197 / 0.762079 | 0.371817 | -0.000391 |
| joint_nmf64_seconds | 0.325531 / 0.386766 / 0.791382 | 0.271310 / 0.358671 / 0.717744 | 0.256876 / 0.353333 / 0.707176 | 0.325414 | -0.046793 |
| rrf_joint_nmf64_seconds_original | 0.340209 / 0.477829 / 0.861727 | 0.289392 / 0.433607 / 0.782519 | 0.267438 / 0.422942 / 0.765238 | 0.371903 | -0.000305 |
| rrf_joint_nmf64_seconds_engagement | 0.345245 / 0.474689 / 0.859217 | 0.293714 / 0.429113 / 0.782559 | 0.270397 / 0.418328 / 0.761742 | 0.371914 | -0.000294 |
| joint_nmf64_mean | 0.324231 / 0.417658 / 0.808164 | 0.271383 / 0.383633 / 0.743794 | 0.251407 / 0.374314 / 0.721977 | 0.337104 | -0.035103 |
| rrf_joint_nmf64_mean_original | 0.339305 / 0.478281 / 0.861460 | 0.288797 / 0.434555 / 0.783432 | 0.266565 / 0.423395 / 0.765316 | 0.371816 | -0.000392 |
| rrf_joint_nmf64_mean_engagement | 0.344713 / 0.475207 / 0.860433 | 0.293708 / 0.430611 / 0.783422 | 0.269595 / 0.418419 / 0.762647 | 0.372042 | -0.000166 |
| count_nmf_frob64 | 0.312037 / 0.455638 / 0.830650 | 0.266988 / 0.419376 / 0.770584 | 0.247318 / 0.413063 / 0.748317 | 0.352403 | -0.019805 |
| rrf_count_nmf_frob64_original | 0.339238 / 0.480064 / 0.864231 | 0.288071 / 0.434745 / 0.785456 | 0.265343 / 0.423741 / 0.764920 | 0.371867 | -0.000341 |
| rrf_count_nmf_frob64_engagement | 0.343849 / 0.477068 / 0.862660 | 0.292318 / 0.430346 / 0.785059 | 0.268722 / 0.419681 / 0.761616 | 0.371997 | -0.000211 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks12 | eng_rrf_recent_capped600 | 0.731676 / 0.882438 | 0.342526 / 0.479882 | 0/739 |
| weeks12 | rrf_joint_nmf32_seconds_original | 0.723176 / 0.864314 | 0.341534 / 0.479260 | 0/739 |
| late8 | eng_rrf_recent_capped600 | 0.678537 / 0.865997 | 0.290418 / 0.431573 | 0/753 |
| late8 | rrf_joint_nmf32_seconds_original | 0.674357 / 0.849598 | 0.289745 / 0.435769 | 0/753 |
| late6 | eng_rrf_recent_capped600 | 0.645257 / 0.857476 | 0.267511 / 0.421337 | 0/706 |
| late6 | rrf_joint_nmf32_seconds_original | 0.640434 / 0.842042 | 0.266680 / 0.421517 | 0/706 |

Measured evaluation/refit wall time 656.864s, peak process RSS 1.321GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 414,585 AP rows, 6,685,200 fusion terms and 0 opportunity scores, with exact selection reproduction and 74,700 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 29,038 full restart rows, 1,200 sampled PPR scores, 35,628 availability gains and 1,200 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round10-20261101-01/final/model.pkl`; `experiments/target040-round10-20261101-01/final/submission.csv`. SHA-256 `8cefdc8efeeeebd2d6a97959d37f32afe4e222ba2701ef70b2f8aea917d63789`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The prepared data must match the frozen preparation hash. Cached states are used only when reuse_cached_fits is true; otherwise every cutoff is fitted fresh. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round10.json --prepared experiments/target040-transfer-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
