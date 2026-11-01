# target040-round08-schedule-conditioned-cold-mf: actual results

Run `experiments/target040-round08-20261101-01`; protocol frozen at 2026-11-01T13:59:23.203646+00:00 with SHA-256 `d2db332a1cd61432b5230421bc223863fa482324f81988af1bee597c70395e8d`. 69 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **eng_rrf_recent_capped600**. Replaces the previous local best: **False**. Raw aggregate winner: `eng_rrf_recent_capped600`. Strict wins over the reference: 0/3.

## Question and protocol

Finite inductive-MF support ablation: historical schedule regression to fixed MF16 item factors, cold factor shrinkage, unchanged known factors and two already evaluated fixed PPR parents.

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

Known real private results: v3 .36630, round02 .35452,03 .35327,04 .37018,07 .36363. Round05/06 identical bytes were not submitted. Round07 remains the local reference despite worse private feedback. Globally unseen held-out positions and missing-factor policy motivate this candidate-support hypothesis; private scores do not select alphas/shrinkage or parent.

All cutoff-specific states in this round are fitted fresh. The old cached-run manifest is provenance only; no old fold fit or training inputs are reused. Each new prepared interaction/context/recency/availability file is hashed in this run. Actual measured component/shared fitting and prediction fields and manifest reuse_cached_fits=false describe the fresh work; shared costs are not additive across trials. The12-week fold retains its old source periods for a reproducibility comparison; later cutoffs diagnose training-history length and remain overlapping previously observed periods.

Accepted-airing engagement features retain the original120s acceptance and canonical join. Thresholding happens before any transform. Each feature is training-only, with exact seconds/watches parity and bounded event endpoints. The following raw auxiliary histories are separately evaluated; they do not count as another qualifying family:

| Signal | Definition |
| --- | --- |
| capped600 | sum min(accepted canonical airing seconds/600,1) |
| capped1800 | sum min(accepted canonical airing seconds/1800,1) |
| fractional_watches | sum min(accepted canonical airing seconds/canonical airing duration,1) |
| active_days | number of distinct arithmetic source days with an accepted canonical airing; day defined by last clipped positive overlap end minus1 microsecond; union across channels |
| recent_capped600 | sum min(accepted canonical airing seconds/600,1)*2**(-(cutoff-last_overlap_end)/(28*86400)) |

Each signal also forms a normalized PPR restart on the unchanged training graph, alpha=.1. Fixed availability power.5 and PPR/MF RRF weights.9/.1 are inherited without tuning. All28 trials, including losing fraction/day/cap variants, remain in the actual table.

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
| ind_mf_a1_c25 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057576 |
| ind_rrf_original_a1_c25 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788641 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | -0.000254 |
| ind_rrf_engagement_a1_c25 | 0.342526 / 0.479882 / 0.863336 | 0.290418 / 0.431573 / 0.786768 | 0.267511 / 0.421337 / 0.764960 | 0.372208 | +0.000000 |
| ind_mf_a1_c100 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057576 |
| ind_rrf_original_a1_c100 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788481 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | -0.000254 |
| ind_rrf_engagement_a1_c100 | 0.342526 / 0.479882 / 0.863336 | 0.290418 / 0.431573 / 0.786608 | 0.267511 / 0.421337 / 0.764960 | 0.372208 | +0.000000 |
| ind_mf_a10_c25 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057576 |
| ind_rrf_original_a10_c25 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788641 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | -0.000254 |
| ind_rrf_engagement_a10_c25 | 0.342526 / 0.479882 / 0.863336 | 0.290418 / 0.431573 / 0.786768 | 0.267511 / 0.421337 / 0.764960 | 0.372208 | +0.000000 |
| ind_mf_a10_c100 | 0.275766 / 0.433456 / 0.825250 | 0.222125 / 0.383875 / 0.748041 | 0.204086 / 0.368481 / 0.719879 | 0.314631 | -0.057576 |
| ind_rrf_original_a10_c100 | 0.339169 / 0.483456 / 0.865814 | 0.286262 / 0.434520 / 0.788641 | 0.264041 / 0.424274 / 0.767245 | 0.371954 | -0.000254 |
| ind_rrf_engagement_a10_c100 | 0.342526 / 0.479882 / 0.863336 | 0.290418 / 0.431573 / 0.786768 | 0.267511 / 0.421337 / 0.764960 | 0.372208 | +0.000000 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks12 | eng_rrf_recent_capped600 | 0.731676 / 0.882438 | 0.342526 / 0.479882 | 0/739 |
| late8 | eng_rrf_recent_capped600 | 0.678537 / 0.865997 | 0.290418 / 0.431573 | 0/753 |
| late6 | eng_rrf_recent_capped600 | 0.645257 / 0.857476 | 0.267511 / 0.421337 | 0/706 |

Measured evaluation/refit wall time 199.771s, peak process RSS 1.196GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 257,715 AP rows, 3,714,000 fusion terms and 0 opportunity scores, with exact selection reproduction and 24,940 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Additional independent checks: 29,038 full restart rows, 1,200 sampled PPR scores, 35,628 availability gains and 1,200 sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round08-20261101-01/final/model.pkl`; `experiments/target040-round08-20261101-01/final/submission.csv`. SHA-256 `8cefdc8efeeeebd2d6a97959d37f32afe4e222ba2701ef70b2f8aea917d63789`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The prepared data must match the frozen preparation hash. Cached states are used only when reuse_cached_fits is true; otherwise every cutoff is fitted fresh. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round08.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
