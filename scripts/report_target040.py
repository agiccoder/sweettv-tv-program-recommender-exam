#!/usr/bin/env python3
"""Write actual round evidence into an English repository report, never scores by guess."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def table(headings, rows):
    return "\n".join(["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run_dir.resolve()
    config, manifest, summary, selection, audit, final = [read(run / name) for name in [
        "frozen-config.json", "manifest.json", "summary.json", "selection.json", "independent-audit.json", "final/submission-validation.json"]]
    assert audit["passed"] and summary["status"] == "complete_local_target040_round"
    slices = {b["name"]: read(run / b["name"] / "summary.json") for b in config["slices"]}
    freeze = read(ROOT / config["freeze_evidence"])
    best = selection["selected_method"]
    text = [f"# {config['protocol_id']}: actual results", "",
        f"Run `{run.relative_to(ROOT)}`; protocol frozen at {freeze['frozen_utc']} with SHA-256 `{manifest['config_sha256']}`. {summary['completed_trials']} completed trials, {summary['failed_trials']} failures. All automated work/checks are attributed to Codex, not the student's personal verification.", "",
        f"Selected: **{best}**. Replaces the previous local best: **{selection['recommended_replacement_of_previous_best']}**. Raw aggregate winner: `{selection['raw_best_method']}`. Strict wins over the reference: {selection['raw_best_strict_slice_wins_over_reference']}/3.", "",
        "## Question and protocol", "",
        config.get("scope", config.get("count_nmf", {}).get("hypothesis", "See frozen config")), "",
        "The three cutoffs and targets are reused, overlapping and previously inspected; they are not independent untouched tests. Same frozen eligible template users, candidate pools, allowed repeats and training-viewer-popularity fallback across methods. User IDs are exact strings. Candidate availability uses deduplicated nonconflicting schedule seconds, never future viewing or content fitting. All trained profiles, graph edges, vocabularies, factor states and statistics end before the respective cutoff.", "",
        "S = Top5 accepted seconds; W = Top5 accepted-airing count; A = all observed programmes, diagnostic only. The hidden target formula is unknown. Selection maximizes the exact rational mean across folds of .5 S + .5 W; exact ties prefer the previous best, then frozen order. A replacement must improve the mean and win at least2/3 folds. No returned private score chooses local parameters.", "",
        table(["Fold", "Train end", "Target end", "Users", "Candidates"], [[b["name"],b["train_end"],b["target_end"],slices[b["name"]]["cohort_size"],slices[b["name"]]["candidate_count"]] for b in config["slices"]]), "",
        "## Components and parent families", "",
        "Content and PPR are evaluated as separate permitted families, with a separate popularity baseline. Personal histories and channel availability are auxiliary signals; neither counts as another family. The earlier v3 ALS32/64 evaluations remain preserved, including their negative results.", "",
        "Opportunity context retains the v3 historical unioned channel × weekday ×4h counts, including unlabelled airtime, and training-global21600s smoothing. Candidate score = v3 mean context affinity × (unique target broadcast hours)^power, power .5 or1. Exponent0 is the existing mean-affinity ablation. True empty-history users have no native affinity. Multiplicity after schedule deduplication is intentional broadcast opportunity; accidental row duplication cannot boost it. Raw scores are not calibrated probabilities.", "",
        "All PPR methods use the same training-only co-view graph: X=row-L1(log1p(seconds)); XᵀX without diagonal, row top100, row normalization. Dangling mass returns to the personal restart; L1 tolerance1e-8 and cap300. Native lists have at most50 positive finite scores. RRF uses fixed weight/(60+rank), missing=0, no per-item renormalization, and fallback only after fusion. Hybrid candidate unions include each history native list, PPR and the selected channel list, <=250 IDs.", ""]
    if "joint_mf" in config:
        text += ["Shared-user implicit MF: " + config["joint_mf"]["objective"], "", "Readouts are the count head, duration head, or an equal mean after separate full-candidate-pool L1 normalization. All views keep the same fitted shared user factors. Unknown programme IDs have no invented factors; common fallback follows ranking. Ranks32/64 and fixed fusion weights are frozen before metrics. The independent checker reconstructs both training blocks, the complete KL loss including unobserved cells, factor orientation and sampled full-pool readouts.", ""]
    if "frobenius_mf" in config:
        text += [config["frobenius_mf"]["hypothesis"], "", "Squared count-intensity NMF uses coordinate descent in pinned scikit-learn1.6.1, with the same fixed regularization, iteration limit and seed. Independent sparse reconstruction checks ||Q||²-2<Q,UVᵀ>+trace((UᵀU)(VᵀV)) against sklearn reconstruction error.", ""]
    if "external_feedback_disclosure" in config:
        text += [config["external_feedback_disclosure"], ""]
    if not config.get("reuse_cached_fits", True):
        text += ["All cutoff-specific states in this round are fitted fresh. The old cached-run manifest is provenance only; no old fold fit or training inputs are reused. Each new prepared interaction/context/recency/availability file is hashed in this run. Actual measured component/shared fitting and prediction fields and manifest reuse_cached_fits=false describe the fresh work; shared costs are not additive across trials. The12-week fold retains its old source periods for a reproducibility comparison; later cutoffs diagnose training-history length and remain overlapping previously observed periods.", ""]
    if "engagement" in config:
        definitions = read(ROOT/config["engagement"]["feature_definition"])["features"]
        text += ["Accepted-airing engagement features retain the original120s acceptance and canonical join. Thresholding happens before any transform. Each feature is training-only, with exact seconds/watches parity and bounded event endpoints. The following raw auxiliary histories are separately evaluated; they do not count as another qualifying family:", "", table(["Signal","Definition"],definitions.items()), "", "Each signal also forms a normalized PPR restart on the unchanged training graph, alpha=.1. Fixed availability power.5 and PPR/MF RRF weights.9/.1 are inherited without tuning. All frozen trials remain in the actual table.", ""]
    if "intensity_mixture" in config:
        text += ["Normalized intensity mixture: " + config["intensity_mixture"]["formula"], "", config["intensity_mixture"]["interpretation"], ""]
    if "mixed_restart" in config:
        text += ["Mixed PPR changes only restart: v=(1-c)L1(sqrt(seconds)) + cL1(watches or recent_watches), c in {.5,.75,1}, link-following alpha=.1. Recent watches sum one28-day decay per accepted canonical airing using the last clipped positive overlap end, bounded by cutoff. It does not decay one last programme timestamp. This is personalized PageRank, not a new unrelated architecture. Two additional PPR/history hybrids use .5/.5 and .25/.75 reciprocal-rank weights.", "",
                 config["mixed_restart"]["external_feedback_disclosure"], ""]
    if "exposure_rate" in config:
        text += ["This round's new availability correction is " + config["exposure_rate"]["formula"] + ". Exposure is either total nonconflicting broadcast seconds (3600s smoothing) or canonical airings with >=120 nonconflicting seconds (one-airing smoothing). Powers .5/1 and the [.25,4] gain cap were frozen. The full base score over eligible known items is transformed before native top50 retrieval; no padding enters this transformation. Target airtime is a fixed allowed availability feature, not future user behaviour or a learned future normalization.", "",
                 config["exposure_rate"]["external_feedback"], ""]
    if "count_nmf" in config:
        text += ["The new implicit MF fits q_ui=watches_ui/sum_j watches_uj with nonnegative factors under generalized KL(q||WH), plus scaled L2 penalties alpha_W=alpha_H=1e-6. This models relative recorded count intensity, not fabricated ratings. Zeros influence the objective and approximate equal observation opportunity; they are not verified dislikes. Fixed ranks16/32, random seed20261030, multiplicative updates, cap100, tol1e-4; no validation early stopping. Observed items remain eligible; unknown IDs get no invented factors. Official [scikit-learn1.6.1 NMF documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.NMF.html) and installed orientation/persistence were checked. Per-state diagnostics retain convergence warnings, actual iterations, reconstruction error and factor variance. The independent auditor reconstructs normalized input, all factor shapes/nonnegativity and the full KL loss including the installed solver's numerical epsilon convention, then checks sampled scores/retrieval without production scoring. Four fixed rank fusions per factor use either the previous PPR winner or personal watches, with MF weights .10/.25.", "",
                 config["count_nmf"]["external_feedback"], ""]
    text += ["Existing fitted states are reused only after exact cached-run input and source hashes are checked, with identical cutoff and pool. New-cutoff final refit constructs fresh states. Cached baseline fit time is recorded as zero new work; original fitting costs remain in v3. Raw source/database and prior run files are preserved.", "",
        "## All actual comparisons", "", "Each fold cell is S / W / A. Aggregate = mean(.5 S+.5 W).", ""]
    text.append(table(["Method", *slices, "Aggregate", "Δ previous best"], [[spec["name"],
        *[" / ".join(f"{s['methods'][spec['name']]['metrics'][k]['map_at_5']:.6f}" for k in ["primary","secondary","all_observed"]) for s in slices.values()],
        f"{selection['aggregate_criterion'][spec['name']]:.6f}",f"{selection['aggregate_criterion'][spec['name']]-selection['aggregate_criterion'][config['reference_method']]:+.6f}"] for spec in config["methods"]]))
    text += ["", "## Retrieval, segments and error evidence", "", "All methods retain per-user AP for all three definitions, native score exports, candidate unions/recall, fusion contributions, subset hits, segment membership and the same two fixed users as v3. The segment definitions are 1–20 /21–99 /100+ accepted training IDs, context-only and no-history; n<30 is flagged. Context-only and no-history cohorts are very small and cannot establish general cold-start performance.", ""]
    compared = list(dict.fromkeys([config["reference_method"], best, selection["ranked_complete_methods"][0]]))
    text.append(table(["Fold", "Method", "Native recall S/W", "Final MAP S/W", "Globally new S hits/positions"], [
        [fold,method," / ".join(f"{s['methods'][method]['candidate_recall'][k]['macro_native_union_recall']:.6f}" for k in ["primary","secondary"]),
         " / ".join(f"{s['methods'][method]['metrics'][k]['map_at_5']:.6f}" for k in ["primary","secondary"]),
         str(s["methods"][method]["metrics"]["primary"]["target_subsets"]["absent_from_all_train"]["hits_at_5"])+"/"+str(s["methods"][method]["metrics"]["primary"]["target_subsets"]["absent_from_all_train"]["target_positions"])] for fold,s in slices.items() for method in compared]))
    text += ["", f"Measured evaluation/refit wall time {manifest['wall_seconds']:.3f}s, peak process RSS {manifest['peak_rss_bytes']/1024**3:.3f}GiB. Per-component fit/predict and rank/export times are recorded; shared/cached costs are not additive isolated benchmarks. All environments, seeds and resource limits are in the manifest/config.", "",
        f"Independent actual-export audit checked {sum(v['AP_rows_checked'] for v in audit['slices'].values()):,} AP rows, {sum(v['fusion_contributions_checked'] for v in audit['slices'].values()):,} fusion terms and {sum(v.get('opportunity_scores_checked',0) for v in audit['slices'].values()):,} opportunity scores, with exact selection reproduction and {audit['legacy_prediction_slots_identical']:,} unchanged reference recommendation slots. It confirmed {audit['protected_original_files_unchanged']} protected pre-task files unchanged. Original states/runs/CSV files are not replaced.", "",
        f"Additional independent checks: {sum(v.get('mixed_restart_rows_checked',0) for v in audit['slices'].values()):,} full restart rows, {sum(v.get('mixed_native_scores_checked',0) for v in audit['slices'].values()):,} sampled PPR scores, {sum(v.get('exposure_rate_checks',{}).get('gains',0) for v in audit['slices'].values()):,} availability gains and {sum(v.get('exposure_rate_checks',{}).get('native_scores',0) for v in audit['slices'].values()):,} sampled adjusted scores/retrieval boundaries. The checker reconstructs score formulas from saved state and inputs without invoking production scoring.", "",
        "Bootstrap intervals, where shown in the run JSON, use2000 paired user resamples with identical user multiplicity across folds. They are post-selection descriptive intervals, do not correct selection bias and do not prove Kaggle improvement.", "",
        "## Designated local candidate", "",
        f"`{run.relative_to(ROOT)}/final/model.pkl`; `{run.relative_to(ROOT)}/final/submission.csv`. SHA-256 `{final['sha256']}`; 1260 users ×5 unique positive target-pool IDs, exact template headers/strings/order. Selection was saved before refit on [2020-03-09,2020-07-27), predicting [2020-07-27,2020-10-19).", "",
        "A locally valid candidate is not a scored Kaggle submission. Own-account API observations and exact downloaded/uploaded bytes are recorded separately in `evidence/target040-01/`. The target is a genuinely completed PRIVATE score>=.40000, not any local or public score. An already submitted identical file is not submitted again. The credentials remain outside the project and are never included in manifests or reports.", "",
        "## Reproduction", "", "Use new run/evidence destinations. Dependencies remain `requirements-v3.lock`; no training dependency was upgraded. The prepared data must match the frozen preparation hash. Cached states are used only when reuse_cached_fits is true; otherwise every cutoff is fitted fresh. Exact source snapshots preserve earlier round implementations before later compatible extensions.", "", "```sh",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_target040.py --destination evidence/target040-reproduction",
        f"PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m sweettv.target040 --config {freeze['config_path']} --prepared {config['prepared']} --run-dir experiments/target040-reproduction",
        "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_target040_round.py --run-dir experiments/target040-reproduction",
        "```", "",
        "Account workflow: `scripts/target040_kaggle.py` defaults to read-only inspection of an independently audited candidate. It verifies own-account entry/quota, refuses duplicate bytes, records submission ref/hash and genuine scores, and never accepts rules. `--mode submit` is used only for the locally designated candidate; `--mode poll --upload-receipt PATH` reads completion without resubmitting. The [official Kaggle command documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) and installed official1.7.4.5 implementation were checked. Closed competitions are located through authenticated entered competitions, not general search. Exact previous files were recovered using the official SDK0.1.38 download endpoint without upgrading the training environment.", ""]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(text))
    print(args.output)


if __name__ == "__main__":
    main()
