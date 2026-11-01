# target040-round02-mixed-restart: actual results

Run `experiments/target040-round02-20261101-01`; protocol frozen at 2026-11-01T12:57:56.133664+00:00 with SHA-256 `aa1c1aa5bfa1dafb0a57ed749847144412529a84eb37979f5e54bb1e4a508bae`. 48 completed trials, 0 failures. All automated work/checks are attributed to Codex, not the student's personal verification.

Selected: **ppr_mix_rw_75**. Replaces the previous local best: **True**. Raw aggregate winner: `ppr_mix_rw_75`. Strict wins over the reference: 3/3.

## Question and protocol

v3 duration-aware PPR/fusion beats watches on seconds; watches wins count proxy. Convex restart blends and two fixed RRF weights may improve their frozen compromise. Recency may downweight stale frequent programmes in the fixed heavy-user example.

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

Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.

## All actual comparisons

Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).

| Method | weeks5 | weeks10 | weeks12 | Aggregate | Δ previous best |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.199015 / 0.327608 / 0.764383 | 0.206625 / 0.336254 / 0.798988 | 0.199612 / 0.335432 / 0.827062 | 0.267424 | -0.130471 |
| content | 0.048400 / 0.055433 / 0.363651 | 0.044133 / 0.047318 / 0.388974 | 0.030845 / 0.033855 / 0.355314 | 0.043330 | -0.354565 |
| ppr_v2 | 0.339388 / 0.397791 / 0.741257 | 0.349435 / 0.418597 / 0.755901 | 0.332665 / 0.408097 / 0.794755 | 0.374329 | -0.023566 |
| history_watches | 0.331904 / 0.465018 / 0.791503 | 0.336613 / 0.476296 / 0.803108 | 0.320099 / 0.457442 / 0.830476 | 0.397895 | +0.000000 |
| personal_history_fusion | 0.347418 / 0.394773 / 0.747841 | 0.359491 / 0.413612 / 0.764940 | 0.339850 / 0.396354 / 0.789789 | 0.375249 | -0.022646 |
| channel_opportunity_total | 0.329789 / 0.363150 / 0.789486 | 0.335712 / 0.369370 / 0.798951 | 0.318140 / 0.349425 / 0.820585 | 0.344264 | -0.053631 |
| hybrid_opportunity_total_70_20_10 | 0.359570 / 0.422776 / 0.804916 | 0.370394 / 0.443770 / 0.825609 | 0.349877 / 0.427068 / 0.849415 | 0.395576 | -0.002319 |
| history_recent_watches | 0.334278 / 0.462206 / 0.787603 | 0.340353 / 0.475017 / 0.802343 | 0.325365 / 0.454916 / 0.831053 | 0.398689 | +0.000794 |
| ppr_mix_w_50 | 0.343323 / 0.457047 / 0.783261 | 0.348886 / 0.467738 / 0.793867 | 0.330358 / 0.453609 / 0.824499 | 0.400160 | +0.002265 |
| ppr_mix_w_75 | 0.338771 / 0.466810 / 0.787817 | 0.342969 / 0.476394 / 0.801555 | 0.324443 / 0.459436 / 0.830767 | 0.401471 | +0.003575 |
| ppr_mix_w_100 | 0.333478 / 0.472363 / 0.796681 | 0.338266 / 0.482495 / 0.807296 | 0.321492 / 0.463496 / 0.837522 | 0.401932 | +0.004036 |
| ppr_mix_rw_50 | 0.345275 / 0.456656 / 0.784348 | 0.353800 / 0.471630 / 0.798563 | 0.334237 / 0.451713 / 0.829714 | 0.402219 | +0.004323 |
| ppr_mix_rw_75 | 0.340619 / 0.465561 / 0.790145 | 0.346866 / 0.478368 / 0.805414 | 0.330254 / 0.459391 / 0.835605 | 0.403510 | +0.005615 |
| ppr_mix_rw_100 | 0.335322 / 0.466955 / 0.791834 | 0.340681 / 0.479910 / 0.809816 | 0.326514 / 0.460997 / 0.837902 | 0.401730 | +0.003835 |
| rrf_ppr_watches_50 | 0.347909 / 0.447846 / 0.787955 | 0.354397 / 0.460152 / 0.801149 | 0.335464 / 0.441018 / 0.830759 | 0.397798 | -0.000098 |
| rrf_ppr_watches_75 | 0.342815 / 0.465964 / 0.796805 | 0.346366 / 0.473113 / 0.807234 | 0.328573 / 0.455229 / 0.832817 | 0.402010 | +0.004115 |

## Retrieval, segments and error evidence

All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.

| Fold | Method | Native recall S/W | Final MAP S/W | Globally new S hits/positions |
| --- | --- | --- | --- | --- |
| weeks5 | history_watches | 0.643545 / 0.822389 | 0.331904 / 0.465018 | 0/1170 |
| weeks5 | ppr_mix_rw_75 | 0.656604 / 0.847979 | 0.340619 / 0.465561 | 0/1170 |
| weeks10 | history_watches | 0.676250 / 0.834637 | 0.336613 / 0.476296 | 0/1037 |
| weeks10 | ppr_mix_rw_75 | 0.683992 / 0.846734 | 0.346866 / 0.478368 | 0/1037 |
| weeks12 | history_watches | 0.678268 / 0.830152 | 0.320099 / 0.457442 | 0/739 |
| weeks12 | ppr_mix_rw_75 | 0.677466 / 0.829511 | 0.330254 / 0.459391 | 0/739 |

Measured evaluation/refit wall time 123.805s, peak process RSS 1.176GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.

Independent actual-export audit checked 178,368 AP rows, 1,833,766 fusion terms and 185,350 opportunity scores, with exact selection reproduction and 92,925 unchanged reference recommendation slots. It confirmed 218 protected pre-task files unchanged. Original states/runs/CSV files are not replaced.

Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.

## Designated local candidate

`experiments/target040-round02-20261101-01/final/model.pkl`; `experiments/target040-round02-20261101-01/final/submission.csv`. SHA-256 `8764aa5f7126ca2bde843216d7be70a60585176232ab3df3e56c5a4fb2a8beee`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).

A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.

## Reproduction

Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The original audited v3 prepared data and cached fitted states must be present and hash-identical. Exact source snapshots preserve earlier round implementations before later compatible extensions.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config config/target040-round02.json --prepared experiments/improvement-v3-data-01 --run-dir experiments/target040-reproduction
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction
```

Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.
