#!/usr/bin/env python3
"""Render actual audited v3 results into the requested local English document."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def number(value):
    return "—" if value is None else f"{value:.6f}"


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in row) + " |" for row in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/improvement-v3-results.md")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run_dir.resolve()
    manifest, config, selection = [read(run / name) for name in ["manifest.json", "frozen-config.json", "selection.json"]]
    audit, numerics = read(run / "independent-audit.json"), read(run / "independent-ALS-numerics.json")
    assert audit["passed"] and numerics["passed"] and manifest["status"] == "complete_local_v3_evaluation"
    slices = {block["name"]: read(run / block["name"] / "summary.json") for block in config["slices"]}
    selected = selection["selected_method"]
    freeze = read(ROOT / "evidence/improvement-v3-01/protocol-freeze.json")
    preparation = read(Path(manifest["prepared"]) / "preparation-manifest.json")
    verification = read(ROOT / "evidence/improvement-v3-01/verification-01/verification.json")
    validation = read(run / "final/submission-validation.json")
    bootstrap = read(run / "post-selection-bootstrap.json")
    text = ["# SWEET.TV v3: actual local experiments", "",
        f"Run: `{run.relative_to(ROOT)}`. Completed {manifest['finished_utc']}. All runs, checks and document generation were performed by Codex, not attributed to the student's personal verification.", "",
        f"Selected working specification: **{selected}**. New strategy passes the replacement gate: **{selection['recommended_replacement_of_v2']}**. Raw aggregate winner: `{selection['raw_best_method']}`; strictly better than v2 on {selection['raw_best_strict_slice_wins_over_v2']}/3 slices.", "",
        "This is a local candidate. Nothing was submitted to Kaggle or published; the previous PDF and public repository staging were not changed. These proxy metrics cannot establish hidden Kaggle quality.", "",
        "## Frozen protocol and preservation", "",
        f"Protocol was frozen before new quality metrics at `{freeze.get('created_utc', freeze.get('frozen_utc', freeze.get('timestamp_utc')))}`. Config SHA-256: `{manifest['config_sha256']}`. The frozen JSON is the authoritative specification. No parameter grid was extended after seeing results.", "",
        "For every method within a slice: identical eligible template cohort, candidate pool, allowed repeats and training-viewer-popularity fallback. Cohort/target exports were fixed during preparation, before model scoring. Empty-target template users are excluded once for all methods. Scores below are macro MAP@5 over those exact cohorts.", "",
        "S = Top5 by summed accepted seconds; W = Top5 by accepted canonical-airing count; A = all observed eligible programmes, a separate diagnostic. These are three local definitions, not a known Kaggle formula. Selection uses the unweighted mean across slices of 0.5 S + 0.5 W, exact rational AP values, v2-first deterministic tie order. A new aggregate winner replaces v2 only with a strictly higher aggregate and at least two strictly winning slices.", "",
        "The target windows overlap and periods were partially inspected in v1/v2. The old confirmation is also previously inspected. None is an independent untouched test, and the slices must not be treated as independent samples.", ""]
    text.append(table(["Slice", "Training [start, end)", "Target [start, end)", "Users", "Candidates"], [
        [b["name"], b["train_start"][:10] + " → " + b["train_end"][:10], b["train_end"][:10] + " → " + b["target_end"][:10], slices[b["name"]]["cohort_size"], slices[b["name"]]["candidate_count"]] for b in config["slices"]]))
    text += ["", f"Independent preservation audit confirmed **{audit['protected_original_files_unchanged']} of {audit['protected_original_files_checked']} protected original files** unchanged, including all raw data/SQLite, v1/v2 source/configs/models, both former CSVs, score evidence and old verification. The sole differing file is macOS directory metadata `experiments/competition-20261030-03/.DS_Store`: same 6148-byte length, changed hash, modification time 2026-11-01T12:14:52Z (before this actual model run). No intentional write to it was performed by v3 tooling; the cause is not established. Its current bytes and both hashes are preserved in `evidence/improvement-v3-01/changed-metadata/` and `protected-files-change-audit.json`. The first audit stopped on this discrepancy; its failed console is retained. The second audit accepts only that exact recorded metadata hash exception; changes to data/models/source remain fatal. The raw database was opened read-only. Preparation took {preparation['wall_seconds']:.2f} seconds. Original six reference methods reproduce {audit['legacy_prediction_slots_identical']:,} actual early-development recommendation slots including scores.", "",
        "The preflight audit independently recomputed old exported AP; confirmed 4251/6225 late seconds-target positions were seen by the same user, 692/6225 were globally absent from train, 3538/6632 final candidate IDs absent from the full graph, and 0.4115110011978921 of unioned historical channel-view time unlabelled. See `evidence/improvement-v3-01/preflight-audit.json` and the new preparation manifest for provenance.", "",
        "## Implemented signals", "",
        "Accepted programme interactions retain the original half-open intervals, same-channel join, interval union, conflict resolution and >=120-second threshold per canonical airing. ID 0 is never a programme or recommendation. All user IDs remain exact strings.", "",
        "Channel/time counts union raw user/channel intervals, including unlabelled and ambiguous viewing, clip to the training cutoff and split into 7 weekday × 6 four-hour bins per historically viewed channel. G is the training-only global slot distribution. The smoothed profile is (C_u + 21600 G)/(total_u + 21600). A raw-empty user has no native context score. Candidate exposure A is nonconflicting deduplicated labelled broadcast seconds in each slot divided by all such seconds for that programme. Unknown training channels contribute to the denominator but have zero affinity. Score = A · profile: a duration-weighted mean, so total repeats/length do not independently boost a programme. Schedule affinity represents availability, not an observed future view.", "",
        "Personal-history fusion uses equal reciprocal ranks from seconds, watches and recent seconds. Recent seconds sum accepted-airing seconds × 2^(-age/28 days) using each airing's last clipped positive overlap. RRF uses weight/(60 + rank), zero for missing components, with no per-candidate renormalization. Native lists have at most 50 positive finite scores and no popularity padding. History fusion ranks its union of at most 150 IDs. Ordinary hybrids union PPR-v2, each history and channel lists (<=250); the ALS hybrid adds ALS-32 (<=300). Fallback is appended only after ranking. Ties use score descending, ID ascending.", "",
        "Hybrid (PPR, history fusion, channel) weights are fixed at (.70,.20,.10), (.50,.25,.25), (.40,.20,.40). The ALS hybrid assigns .25 each to reciprocal ranks of PPR-v2, ALS-32, history fusion and channel. Future schedule access uses only ID, channel, start and duration. Content vocabulary, IDF, graph and preferences use training data only.", "",
        "The v2 graph remains X = row-L1(log1p(seconds)), C = XᵀX without the diagonal, row top-100 and transition normalization; restart L1(sqrt(seconds)), alpha=.25. Additional graphs fix all those settings except edge weights. With distinct-user support >=2, cosine = Cij/sqrt(Cii Cjj); lift = n_users Cij/((sum_u Xui+1)(sum_u Xuj+1)) × support/(support+10). Products are blocked with a 64 MiB payload budget. These are two graph normalizations, not association rules.", "",
        "ALS uses implicit 0.7.3 CPU, CSR users × items, factors 32/64, stored confidence 1 + 40 log1p(seconds/600), library alpha=1, regularization .1, 20 iterations, CG3, seed 20261030, float32 and one thread. Positive interactions represent preference with confidence; absence has baseline confidence 1, not a verified dislike. Scores are factor dot products on known IDs; positive finite scores are retrieved and no random factors are invented for unknown IDs. Repeats remain allowed.", "",
        "The confidence interpretation follows [Hu, Koren and Volinsky (2008)](https://yifanhu.net/PUB/cf.pdf). User/item orientation, confidence input, `filter_already_liked_items=False`, and official NPZ save/load were checked against the [official CPU ALS API](https://benfred.github.io/implicit/api/models/cpu/als.html), [primary implementation](https://github.com/benfred/implicit/blob/main/implicit/cpu/als.py) and installed 0.7.3 API. The documentation site labels its API 0.7.2; tests exercised installed 0.7.3. See [official release metadata](https://pypi.org/project/implicit/).", "",
        "## All actual trials", "", "Each slice cell is **S / W / A**. Aggregate is mean(0.5 S + 0.5 W). No row is a Kaggle score.", ""]
    text.append(table(["Method", *slices, "Aggregate", "Δ v2"], [
        [spec["name"], *[" / ".join(number(s["methods"][spec["name"]]["metrics"][kind]["map_at_5"]) for kind in ["primary", "secondary", "all_observed"]) if spec["name"] in s["methods"] else "FAILED" for s in slices.values()],
         number(selection["aggregate_criterion"].get(spec["name"])), number(selection["aggregate_criterion"].get(spec["name"], float("nan")) - selection["aggregate_criterion"]["ppr_v2"])] for spec in config["methods"]]))
    means = {method: {kind: sum(s["methods"][method]["metrics"][kind]["map_at_5"] for s in slices.values()) / len(slices)
                      for kind in ["primary", "secondary"]} for method in ["ppr_v2", selected]}
    text += ["", "All 48 actual trials completed; no optional dependency or resource failure occurred. Failed/unavailable trials would remain in slice `trials/` and `component-failures.json`; incomplete methods cannot win selection. There are no unreported replacements or additional searches.", "",
        "## Interpretation and bounded ablations", "",
        f"The selected method is **{selected}**, an existing simple history baseline evaluated under the new frozen compromise, not a newly successful channel/ALS ensemble. Its aggregate advantage is {number(selection['aggregate_criterion'][selected] - selection['aggregate_criterion']['ppr_v2'])}. Mean seconds MAP changes {number(means['ppr_v2']['primary'])} → {number(means[selected]['primary'])}; mean watches MAP changes {number(means['ppr_v2']['secondary'])} → {number(means[selected]['secondary'])}. Thus the gain in the chosen objective is a tradeoff between proxies, not a uniform relevance improvement. Preserve v2 for comparison; changing the proxy weights after this result would invalidate the frozen selection.", "",
        "The three-history fusion improves seconds MAP over v2 on each slice, but its aggregate advantage is only " + number(selection["aggregate_criterion"]["personal_history_fusion"] - selection["aggregate_criterion"]["ppr_v2"]) + ". It wins the combined criterion on two slices and loses on weeks12. It is not selected because the aggregate history-watches result is higher. This bounded ablation shows that history-rank blending changes the target tradeoff; it is not evidence that adding every signal helps.", "",
        "Channel/time alone has very low MAP. Across the presets with channel weights .10, .25 and .40, aggregate scores decrease; all three weights vary between presets, so this does not isolate a causal channel-weight effect. Combined history/channel candidate unions improve retrieval recall over PPR top50, but final hybrid MAP is worse: a larger candidate union does not solve ordering. The two normalized graphs also lose to original-graph v2. Both ALS dimensions and the fixed ALS hybrid lose substantially. These are retained negative results; no extra smoothing/half-life/factor/weight search followed them.", "",
        "The channel model hits 47/1170, 18/1037 and 10/739 globally new seconds-target positions on the three slices; v2 and the selected history baseline hit zero in those subsets. That limited new-item ability does not compensate for its much lower overall relevance. Context-only cohorts have only 1, 2 and 4 users; channel MAP is zero in each, so these data do not establish a useful context-only benefit. The no-history cohorts are also tiny (2, 3, 4); identical fallback is a correctness result, not proof of cold-start quality.", "",
        "Historical Kaggle screenshots remain observations of private scores 0.28587 and 0.28919; exact account/submission-byte linkage was not established. They were neither changed nor used for this selection. No v3 leaderboard score exists in this work.", "",
        "## Segments", "", "Short = 1–20 distinct accepted train programmes; medium = 21–99; long >=100; channel_only = no accepted items but raw channel context; no_history = neither. **n<30 is a small denominator**. Each method cell is S / W. All segment user IDs and all 16 methods remain in the run exports.", ""]
    compared = list(dict.fromkeys(["ppr_v2", selected, "channel_time", "personal_history_fusion"]))
    records = []
    for name, s in slices.items():
        for group, n in s["segments"].items():
            records.append([name, group, str(n) + (" (small)" if n < 30 else ""), *[" / ".join(number(s["methods"][method]["metrics"][kind]["segments"][group]["map_at_5"]) for kind in ["primary", "secondary"]) for method in compared]])
    text.append(table(["Slice", "Segment", "Users", *compared], records))
    text += ["", "## Retrieval and target subsets", "", "Native union recall is measured before fallback; augmented recall adds only the IDs actually used for fallback. This augmented recall bounds MAP for the at-most-five targets. Cells below show native recall S / W, then final MAP S / W. Retrieval and final ordering are different losses.", ""]
    retrieval_methods = list(dict.fromkeys(["ppr_v2", selected, "personal_history_fusion", "channel_time", "hybrid_70_20_10", "hybrid_als_ppr"]))
    text.append(table(["Slice", "Method", "Native recall S / W", "With fallback S / W", "Final MAP S / W"], [
        [name, method, " / ".join(number(s["methods"][method]["candidate_recall"][k]["macro_native_union_recall"]) for k in ["primary", "secondary"]),
         " / ".join(number(s["methods"][method]["candidate_recall"][k]["macro_union_plus_actual_fallback_recall"]) for k in ["primary", "secondary"]),
         " / ".join(number(s["methods"][method]["metrics"][k]["map_at_5"]) for k in ["primary", "secondary"])] for name, s in slices.items() for method in retrieval_methods]))
    records = []
    for name, s in slices.items():
        for kind in ["primary", "secondary"]:
            for subset in ["seen_by_same_user", "absent_from_all_train", "known_but_not_seen_by_user"]:
                baseline = s["methods"]["ppr_v2"]["metrics"][kind]["target_subsets"][subset]
                entries = []
                for method in list(dict.fromkeys(["ppr_v2", selected, "channel_time"])):
                    v = s["methods"][method]["metrics"][kind]["target_subsets"][subset]
                    entries.append(f"{method}: {v['hits_at_5']}/{v['target_positions']} ({number(v['micro_recall_at_5'])})")
                records.append([name, kind, subset, baseline["users_with_subset"], "; ".join(entries)])
    text += ["", table(["Slice", "Target", "Subset", "Users", "Top5 hits / target positions (micro recall)"], records), "",
        "Coverage is only a diagnostic; each trial records unique recommended IDs, catalogue coverage and concentration. Recommending more distinct IDs does not by itself improve relevance.", "",
        "## Costs and verification", "",
        f"Evaluation plus final refit: {manifest['wall_seconds']:.2f} seconds; process peak RSS {manifest['peak_rss_bytes'] / 1024**3:.3f} GiB. Apple M5, Python 3.9.6, one configured thread, full package versions in the manifest. Preparation is separate. Shared data, original graph and native lists are cached only within a cutoff/config. Times below are measured dependency-component fit / native prediction / ranking-and-export seconds, summed across slices; shared data/graph costs are separately recorded in each summary, so these are not isolated benchmarks or additive trial costs. RSS is cumulative process high-water, not per-method allocation.", ""]
    text.append(table(["Method", "Fit components s", "Predict components s", "Rank/export s", "Max recorded RSS GiB"], [
        [spec["name"], *[f"{sum(s['methods'][spec['name']][key] for s in slices.values() if spec['name'] in s['methods']):.3f}" for key in ["fit_seconds_cached_components", "predict_seconds_cached_components", "ranking_export_seconds"]],
         f"{max(s['methods'][spec['name']]['process_highwater_rss_bytes_so_far'] for s in slices.values() if spec['name'] in s['methods'])/1024**3:.3f}"] for spec in config["methods"]]))
    text += ["", f"**{verification['tests_run']} tests passed**, {verification['failures']} failures, {verification['errors']} errors, no skips. These include the original tests plus real SQLite cutoff perturbation, unlabelled context, calendar splitting, duplicate/permutation invariance, hand RRF/missing weights/ties, two independent graph formulas, non-square ALS orientation/confidence/repeat API/finite scores/official save-load, complete synthetic 16×3 pipeline and optional-dependency failure handling. Synthetic tests are correctness evidence, not real quality evidence.", "",
        f"The independent actual-export audit recalculated {sum(x['AP_rows_checked'] for x in audit['slices'].values()):,} per-user AP rows across three targets and {sum(x['fusion_contributions_checked'] for x in audit['slices'].values()):,} component contributions. It reconstructed targets, exact selection, native ranks, all final native/fallback recommendations, candidate recall, segment and target-subset counts/metrics. All native/final scores were checked finite. Model round-trip, template order/headers, distinct positive candidate IDs, exact string IDs and final SHA were verified.", "",
        "NumPy emitted divide/overflow/invalid warnings in ALS BLAS matmul during native inference; these remain in the console evidence. No nonfinite scores passed the strict runtime guard. `independent-ALS-numerics.json` additionally checks all saved factor entries finite, a safe bound on dot magnitudes, and sampled real user scores against independent float64 contraction without BLAS matmul. Across the six frozen models, 46,360 candidate scores were compared; maximum absolute discrepancy was below 2.90e-7, and every factor-based upper bound on absolute dot magnitude was below 90. The warnings recurred with finite, matching outputs; their underlying environment cause is not asserted. This diagnostic does not change scores, parameters or selection.", "",
        "Paired bootstrap uses the same sampled user multiplicity across overlapping slices, 2000 resamples, seed 20261030. These are descriptive **post-selection** intervals; they do not correct winner-selection bias or establish future Kaggle improvement.", "", table(["Method vs v2", "Observed aggregate Δ", "95% percentile interval"], [
            [method, number(v["observed_delta"]), "[" + ", ".join(number(x) for x in v["percentile_interval"]) + "]"] for method, v in bootstrap["comparisons_vs_v2"].items()]), "",
        "## The same two fixed users", "",
        "These IDs were fixed before metrics and are the two examples from v2; examples absent from a slice's eligible cohort are shown but never included in that slice MAP. Full exact histories, held-out programmes and all 16 Top5 lists remain in each slice's `two-user-examples.json`. The comparison is illustrative, not representative evidence. Titles are descriptive schedule labels and can vary for a reused programme ID; scoring uses exact IDs.", ""]
    for name in slices:
        examples = read(run / name / "two-user-examples.json")
        for user in config["examples"]:
            ex = examples[user]
            text += [f"### {name}: user {user}", "", f"Eligible cohort: {ex['in_cohort']}; train programmes: {ex['training_item_count']}; raw channel seconds: {ex['raw_context_seconds']:.0f}. Top seconds history: " + "; ".join(f"{r['item_id']} ({r['title']}, {r['seconds']:.0f}s)" for r in ex["training_history"][:5]) + ".", ""]
            text.append(table(["Method", "Ordered Top5 ID (title; S hit / W hit)"], [[method, "; ".join(f"{r['item_id']} ({r['title']}; {int(r['primary_hit'])}/{int(r['secondary_hit'])})" for r in ex["methods"][method])] for method in compared]))
            if selected in {"history_seconds", "history_watches"}:
                signal = "seconds" if selected == "history_seconds" else "watches"
                lookup = {int(r["item_id"]): r for r in ex["training_history"]}
                text += ["", "Selected-rank evidence: " + "; ".join(f"{r['item_id']}: {lookup[r['item_id']][signal]} train {signal}" if r["item_id"] in lookup else f"{r['item_id']}: {r['reason']}" for r in ex["methods"][selected]) + "."]
            text.append("")
    text += ["The heavy-user example is an actual counterexample to uniform improvement. History-watches promotes programme 12001682 (118 accepted training airings at every cutoff) above the news programme 2400480 (84/87/90 airings). The promoted programme ID is absent from both Top5 target sets in all three slices, while PPR keeps programme 6500479, a seconds-target hit. This user does not illustrate a personal improvement despite the cohort-level criterion win. The fixed cold example has no accepted or raw channel history at all three cutoffs, so all methods return the same fallback; it is excluded from weeks5 metrics because it has no eligible target there.", "",
        "## Final refit and local handoff", "",
        f"Exactly one selected specification was saved before final refit: `{json.dumps(selection['selected_specification'], sort_keys=True)}`. Final training is [2020-03-09, 2020-07-27); target schedule [2020-07-27, 2020-10-19). {validation['training_pairs']:,} accepted user/programme pairs; {validation['fit_seconds']:.3f}s fit, {validation['predict_seconds']:.3f}s prediction. The final candidate contains exactly 1260 template users and five distinct positive allowed IDs each. If the replacement gate is false, retain v2 as the working model and do not interpret this file as a recommended new submission.", "",
        f"Model: `{run.relative_to(ROOT)}/final/model.pkl`. Final candidate: `{run.relative_to(ROOT)}/final/submission.csv`; verified convenience copy: `submission_v3.csv`. SHA-256: **`{validation['sha256']}`**. Previous `submission_v2.csv` remains unchanged. Local models contain user information and are trusted-pickle artifacts, not publication assets.", "",
        "## Reproduction", "", "Run from `/Users/timur/Recsys/midterm-codex-package` with the original raw data in place. Use new output names; commands refuse to overwrite existing run/evidence directories. Dependencies are pinned in `requirements-v3.lock`.", "", "```sh",
        ".venv/bin/python -m pip install -r requirements-v3.lock",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/prepare_improvement_v3.py --output-root experiments/improvement-v3-data-reproduction",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_improvement_v3.py --destination evidence/improvement-v3-reproduction",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.improve_v3 --prepared experiments/improvement-v3-data-reproduction --run-dir experiments/improvement-v3-reproduction --config config/improvement-v3.json --template data/raw/extracted/submission.csv",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_improvement_v3.py --run-dir experiments/improvement-v3-reproduction --metadata-change-audit evidence/improvement-v3-01/protected-files-change-audit.json",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_improvement_v3_als.py --run-dir experiments/improvement-v3-reproduction",
        "```", "",
        "Traceability: `manifest.json` binds source/config/input hashes and environment; `selection.json` records exact per-slice and aggregate values, ordering and gate; each slice retains all trials, native scores, recommendations, per-user AP, RRF contributions, candidate unions, segment memberships, diagnostics, models and exact AP audits. `evidence/improvement-v3-01/` preserves prechange/freeze/preflight/preparation/test/console evidence. No private/public leaderboard score was used as an objective.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text))
    print(args.output)


if __name__ == "__main__":
    main()
