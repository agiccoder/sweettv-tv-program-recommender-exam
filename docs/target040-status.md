# Target040 execution status

Updated at 2026-11-01T15:30:50.202735+00:00. The genuine PRIVATE>=0.40000 objective is not yet achieved. Code, experiments and automated checks are performed by Codex; no personal student verification is invented.

## Confirmed best

PRIVATE **0.39000**, public **0.40643**, completed submission **57875732**. Method `mfshare_rate_16_50`: PPR with historical mixed duration/count restart and fixed broadcast-availability correction, fused with implicit KL count-intensity NMF16 through reciprocal ranks (weights0.5/0.5, constant60). This is a PPR/implicit-MF hybrid; personal histories and availability remain auxiliary signals.

Exact local CSV: `submission_target040_best_039000.csv`.
SHA-256: `8b722b51734d6b7d6b46be0b563afce424dbbf5214dd7e18170ca82152b9b7e1`.
The server-downloaded own-submission bytes exactly match the validated model output. A fresh refit with no fitted-factor reuse also reproduced the exact CSV SHA (`experiments/target040-reproduce-039000-01/reproduction-validation.json`). Proof: `evidence/target040-01/best-039000-57875732/linkage-and-real-score.json`. Original v1/v2/v3 and all subsequent candidates remain preserved.

## Actual account limits and batch authorization

The user's latest explicit instruction authorizes many distinct submissions within the remaining hour. The official live `submissions/limits` endpoint reports100/day for this late-submission account. The older competition-listing field10/day is stale. Evidence: `evidence/target040-01/live-submission-limits-01.json`. No rules acceptance or quota bypass was performed.

Portfolio01 predeclared27 candidates, generated all27, reproduced two previous complete1260-user CSV controls, and passed independent checks of170100 recommendation positions. Exactly26 distinct new hashes were submitted. Repeated history-list requests caused a genuine HTTP429; all logs were retained, a120s cooldown was respected, and remaining uploads resumed serially with5s pacing and per-batch history reads. All own returned scores, including losing methods, are saved in `portfolio01-resume01/results.json`.

Portfolio02 predeclares25 candidates: PPR/history parents fused with MF16/32/64 at0.5/0.75 MF weight, plus MF16. New local comparisons on all three long-horizon folds are complete in `experiments/target040-portfolio02-validation-02`, with predictions, exact AP and segment denominators. Best new dual proxy0.36878654 remains below the local reference0.37220786. These are diversification candidates under the user's explicit batch instruction, not a claimed local improvement. All25 final files passed independent checks of157500 positions; duplicate prior hashes are skipped. All24 new unique hashes were submitted and returned completed scores. Best PRIVATE0.39000/public0.40643, exact downloaded linkage verified. One prior MF16 hash was skipped.

The final choice by best returned private score follows the assignment's acceptance rule. It is explicitly exposed to external feedback and must not be described as a clean untouched holdout. Local fitting/parameters do not optimize a private-score loss; there are no per-user probes or hidden-label retrieval.

## Finite local rounds

| Round | Hypothesis | Comparisons | Outcome |
| --- | --- | --- | --- |
|01|Broadcast opportunity|13×3|History reference retained; same v3 CSV|
|02|Duration/count/recency restart|16×3|Local0.403510, PRIVATE0.35452|
|03|Schedule opportunity correction|16×3|Local0.415616, PRIVATE0.35327|
|04|Implicit count-intensity MF|16×3|Local0.419644, PRIVATE0.37018|
|05|Longer-history transfer|16×3|Local0.371954; same round04 bytes|
|06|Complete-pool intensity mixing|16×3|Negative; reference retained|
|07|Accepted-airing engagement|28×3|Local0.372208, PRIVATE0.36363|
|08|Schedule-conditioned unseen-item factors|23×3|Negative; independent audit passed|
|09|Cold-item MF completion without changing known PPR scores|23×3|Negative; independent audit passed|
|10|Joint count/duration factors and squared-error objective|37×3|Complete; raw best0.372418 wins only1/3 folds, reference retained;414585-AP audit passed|
|11|Square-root counts/duration and unnormalized activity intensity|31×3|Complete; reference retained, independent347355-AP audit passed|

Rounds01–04 reuse the earlier5/10/12-week targets. Rounds05 onward use8/12/14-week training histories and12/8/6-week prediction horizons. All folds overlap and are previously observed, not independent untouched tests. The hidden target formula is unknown. Report S=Top5 accepted seconds and W=Top5 accepted canonical-airing counts separately; A=all observed is diagnostic. Local selection uses the exact mean across folds of0.5S+0.5W, requires improvement and strict wins on at least2/3 folds, and prefers the prior reference on exact ties. The reference remains `eng_rrf_recent_capped600` until actual new results justify replacement.

Portfolio03 freezes24 joint/squared-MF candidates; portfolio04 freezes18 transformed-input candidates. Their source/config freezes precede new quality measurements. Both temporal rounds and all final candidate audits are complete; future refits use only allowed historical preferences. Portfolio03 contains23 unique unsubmitted hashes plus one already-submitted MF64 hash; portfolio04 contains18 unique hashes. Portfolio06 covers the omitted standalone joint-MF32 count head with the same fitted component, no new parameters. Portfolio05 predeclares three previously evaluated complete-pool intensity mixtures for the residual quota. Real timings, factors, all-user native lists, predictions, AP, recalls and segments remain in their experiment directories.

## Correctness and provenance

Latest complete verification: **105 tests passed**, zero failures/errors, stable source hashes (`evidence/target040-01/verification-20`). New checks cover paired-block MF orientation, distinct count/duration heads, full-pool normalization, unknown users/IDs, repeats, pickle reproduction, independent generalized-KL and Frobenius reconstruction, transformed inputs and complete synthetic rounds. No dependency was upgraded; `requirements-v3.lock` remains pinned.

Every serious candidate preserves all1260 exact template user strings/order/headers and five distinct positive eligible programme IDs per row. Every upload has an immutable CSV copy, SHA-256, model/config/source/input linkage, genuine API ID/status/scores and author/account description. A live API compatibility issue was corrected without rewriting prior evidence: the old SDK integer page argument was ignored and repeated the first50 rows. Direct official pageSize100 returned all57 unique own IDs. `portfolio02-submissions/results-complete-history.json` retains complete normalized observations, alongside the exact raw response and original duplicated list. New serial batches assert unique IDs and full expected coverage from one pageSize200 query. No duplicate CSV was uploaded because deduplication also uses immutable upload receipts.

The original218 protected artifacts remain unchanged through the independent round09 audit. Failed commands and negative model results are not erased. Native session records are never edited or merged. Credentials remain outside the project with mode0600 and are never printed or uploaded.

## Reproduction

Use new destination names and matching immutable source snapshots from the relevant run. Restore snapshots in an isolated project copy for byte-level reproduction of historical implementations.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-new-verification
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/target040_portfolio.py --config config/target040-portfolio01.json --output experiments/target040-new-portfolio01
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_portfolio.py --root experiments/target040-new-portfolio01
```

Best candidate output is `experiments/target040-portfolio02-20261101-01/candidates/mfshare_rate_16_50/final/submission.csv` and its persisted `model.pkl`. Genuine API evidence is not claimed to replace the original exam's required Kaggle Submissions-page screenshot. The older public GitHub/PDF artifacts do not yet describe the current best model; do not claim a complete new course hand-in from this status document.

## Final user-authorized batch outcome

All95 new distinct uploads are COMPLETE, with102 genuine own-account observations in the final history. The official live quota now reports100 submissions today and0 allowed. No additional upload is attempted beyond this real limit. The best remains PRIVATE0.39000/public0.40643, submission57875732; PRIVATE>=0.40000 is not achieved. No score is estimated or invented.

| Portfolio | New distinct uploads | Completed | Best returned PRIVATE within batch |
| --- | --- | --- | --- |
|1|26|26|0.37949|
|2|24|24|0.39000|
|3|23|23|0.37679|
|4|18|18|0.38973|
|5|3|3|0.38402|
|6|1|1|0.37135|

The final linkage audit independently rehashed all95 upload CSVs, source CSVs and persisted models and checked each candidate audit. Across98 generated candidate files, independent audits cover617400 recommendation positions. Original218 protected files remain unchanged. The strongest external candidate has local S/W MAP of0.314515/0.480839 (weeks12),0.255308/0.427420 (late8),0.238339/0.414633 (late6), dual mean0.3551755900, below the local reference0.3722078566. Thus the external gain is not presented as a local-selection gain or independent validation.

The round10 audit emitted matmul runtime warnings while returning finite small factors/products and passing its norm check. Six additional actual checks compared every Gram entry and the complete Frobenius norm through non-BLAS einsum, all passed (`frobenius-gram-additional-check.json`); the cause is unestablished. Prediction models/CSVs were unchanged. Future checker reductions now use that independently checked einsum path, with original checker bytes and warning logs preserved.

The final official quota response, exact server-downloaded best bytes, actual final score list and immutable batch/model linkage are under `evidence/target040-01/`. A fresh no-cache refit reproduces the best CSV exactly. The original exam's genuine Submissions-page screenshot and full updated PDF/native-session hand-in are not claimed complete by this model-improvement batch.

Final public source-package verification:105 tests passed with stable core source hashes (`results/target040-final-source-verification.json`). The portable account helper separately passed an offline parser check against the preserved authentic API response; optional past-account evidence is no longer required in a fresh checkout. This check performed no new upload.
