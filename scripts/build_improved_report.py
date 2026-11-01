#!/usr/bin/env python3
"""Build a separate, explicitly incomplete v2 exam report from actual outputs.

The original report/source remain untouched. No placeholder v2 scores are used:
source refresh requires the complete frozen real run and its exact AP audits.
The two unchanged user-score crops retain their limited account/file linkage;
neither is API-level submission verification. Prediction sources stay frozen.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

import fitz
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_report import escape, local_tag, p, paragraph_markup, read_json, register_fonts, table

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"report"
PDF_NAME = "timur_khayrislamov_exam_report_V2_DRAFT.pdf"
SOURCE_NAME = "report-v2-source.html"
EXAMPLES_NAME = "v2-two-user-examples.json"
FIXED_USERS = ["13571199046623737884", "10120801182216576239"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=3):
    return f"{float(value):,.{digits}f}"


def count(value):
    return f"{int(value):,}"


def method_label(spec):
    kind = spec["kind"]
    if kind == "popularity":
        return "Popularity"
    if kind == "content":
        return "Content TF-IDF"
    if kind == "original_ppr":
        return "Original PPR α=.5"
    if kind == "personal_history":
        return "History " + spec["signal"] + " (diag.)"
    signal = "sec" if spec["seed_signal"] == "seconds" else "watch"
    power = "sqrt" if spec["seed_power"] == .5 else "raw"
    return f"PPR α={spec['alpha']:g}, {signal}, {power}" + (", R28" if spec.get("recency") else "")


def read_csv(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def checked_inputs(run, original):
    manifest = read_json(run/"manifest.json")
    summary = read_json(run/"summary.json")
    if manifest.get("status") != "complete_local_v2_proxy_evaluation" or manifest.get("synthetic_fixture") is not False:
        raise ValueError("A complete actual v2 run is required; no metrics are inferred from partial or synthetic runs")
    if manifest.get("source_changed_during_run") is not False or manifest.get("source_hashes") != manifest.get("source_hashes_at_end"):
        raise ValueError("The v2 run did not preserve its frozen prediction sources")
    config = read_json(run/"frozen-config.json")
    development = read_json(run/"development/summary.json")
    confirmation = read_json(run/"confirmation/summary.json")
    selection = read_json(run/"selection.json")
    final = read_json(run/"final/submission-validation.json")
    if len(config.get("methods", [])) != 14 or len(development.get("methods", {})) != 14:
        raise ValueError("Report expects the final predeclared fourteen-method v2 development search")
    if selection.get("confirmation_metrics_available_at_selection") is not False or not summary.get("selection_frozen_before_confirmation"):
        raise ValueError("Development-only selection provenance is absent")
    selected = selection["selected_method"]
    if summary.get("selected_method") != selected or final.get("selected_method") != selected:
        raise ValueError("Selected strategy and final refit disagree")
    if final.get("passed") is not True or digest(run/"final/submission.csv") != final.get("sha256") or final.get("rows") != 1260:
        raise ValueError("Actual v2 CSV is absent, changed or not a validated 1260-user file")
    for block, values in [("development", development), ("confirmation", confirmation)]:
        for target in ["primary", "secondary", "all_observed"]:
            audit = read_json(run/block/f"independent-ap-audit-{target}.json")
            if audit.get("passed") is not True or set(audit.get("methods", {})) != set(values["methods"]):
                raise ValueError("The exact exported-AP audit is missing or disagrees with model names")
            for name, result in values["methods"].items():
                if abs(audit["methods"][name]["map_at_5"]-result["metrics"][target]["map_at_5"]) > 1e-12:
                    raise ValueError("Displayed metric differs from independently audited actual export")
    original_manifest = read_json(original/"manifest.json")
    if original_manifest.get("status") != "complete_local_competition_run" or original_manifest.get("synthetic_fixture") is not False:
        raise ValueError("Original actual C3 evidence is required")
    input_hashes = manifest.get("input_hashes", {})
    for name in ["train-interactions.parquet", "valid-interactions.parquet", "candidates.json", "metadata.json", "protocol.json"]:
        if input_hashes.get(f"confirmation/{name}") != digest(original/"prepared"/name):
            raise ValueError("The confirmation prepared inputs differ from their frozen v2 identities")
    return manifest, summary, config, development, confirmation, selection, final


def build_examples(run, original, selected):
    fixed = read_json(original/"example-users.json")
    if [row["user_id"] for row in fixed.get("users", [])] != FIXED_USERS:
        raise ValueError("The examples must retain the original heavy/light users")
    target_audit = read_json(run/"confirmation/cohort-and-targets.json")
    target_by_user = {row["user_id"]: row for row in target_audit["details"]}
    metadata = read_json(original/"prepared/metadata.json")
    frames = {window: pd.read_parquet(original/"prepared"/f"{window}-interactions.parquet") for window in ["train", "valid"]}
    records = read_csv(run/"confirmation/recommendations.csv")
    lookup = {(row["method"], row["user_id"], int(row["rank"])): row for row in records}
    roles = [("Popularity", "original_popularity"), ("Content", "original_content"), ("Selected", selected)]
    examples = []
    for descriptor in fixed["users"]:
        user = descriptor["user_id"]
        targets = target_by_user[user]
        if not targets["in_cohort"]:
            raise ValueError("An original fixed example lacks observable v2 confirmation targets")
        histories = {}
        for window, frame in frames.items():
            histories[window] = []
            for row in frame[frame.user_id == user].itertuples():
                meta = metadata[window].get(str(row.item_id), {})
                histories[window].append({"item_id": int(row.item_id), "seconds": float(row.seconds), "watches": int(row.watches),
                                          "title": meta.get("tv_show_title", ""), "category": meta.get("tv_show_category", "")})
        valid_pairs = {row["item_id"]: row for row in histories["valid"]}
        primary = set(targets["primary_ordered_ids"])
        all_observed = set(targets["all_observed_ids"])
        recommendation_rows = []
        for role, method in roles:
            for rank in range(1, 6):
                raw = lookup[method, user, rank]
                item = int(raw["item_id"])
                meta = metadata["valid"][str(item)]
                proxy_hit = item in primary
                broad_hit = item in all_observed
                if proxy_hit != (raw["primary_hit"] == "True") or broad_hit != (raw["all_observed_hit"] == "True"):
                    raise ValueError("Recommendation labels disagree with actual fixed-user targets")
                if broad_hit != (item in valid_pairs) or raw["title"] != meta["tv_show_title"] or raw["category"] != meta["tv_show_category"]:
                    raise ValueError("Recommendation metadata/hits disagree with observed heldout pairs")
                if not all([math_finite(float(raw["score"])), item > 0]):
                    raise ValueError("Invalid actual recommendation score or ID")
                recommendation_rows.append({"display_role": role, "method": method, "rank": rank, "item_id": item,
                                            "title": raw["title"], "category": raw["category"], "score": float(raw["score"]),
                                            "reason": raw["reason"], "primary_top5_seconds_hit": proxy_hit, "broad_observed_hit": broad_hit,
                                            "actual_heldout_seconds": valid_pairs[item]["seconds"] if broad_hit else None,
                                            "actual_heldout_watches": valid_pairs[item]["watches"] if broad_hit else None})
        examples.append({**descriptor, "actual_training_items": len(histories["train"]), "actual_heldout_items": len(histories["valid"]),
                         "histories": histories, "primary_ordered_ids": targets["primary_ordered_ids"],
                         "primary_programmes": [valid_pairs[item] for item in targets["primary_ordered_ids"]],
                         "recommendations": recommendation_rows})
    payload = {"scope": "Actual v2 confirmation illustrations, not hidden Kaggle test labels", "source_run": str(run),
               "fixed_selection_source": str(original/"example-users.json"), "selection": fixed["selection"],
               "selected_method": selected, "roles": roles, "users": examples, "recommendation_rows": sum(len(row["recommendations"]) for row in examples),
               "input_sha256": {"recommendations": digest(run/"confirmation/recommendations.csv"),
                                "targets": digest(run/"confirmation/cohort-and-targets.json"), "fixed_users": digest(original/"example-users.json")}}
    payload["metadata_label_policy"] = "Earliest nonempty schedule metadata in each respective window; an ID may have different displayed titles across windows."
    payload["input_sha256"].update({window+"_interactions": digest(original/"prepared"/f"{window}-interactions.parquet") for window in ["train", "valid"]})
    if payload["recommendation_rows"] != 30:
        raise ValueError("Expected thirty actual three-role recommendation rows")
    return payload


def math_finite(value):
    import math
    return math.isfinite(value)


def source_document(run, original, external_path):
    manifest, summary, config, development, confirmation, selection, final = checked_inputs(run, original)
    examples = build_examples(run, original, selection["selected_method"])
    selected = selection["selected_method"]
    spec_by_name = {row["name"]: row for row in config["methods"]}
    chosen_label = method_label(spec_by_name[selected])
    original_summary = read_json(original/"validation-summary.json")
    raw = read_json(ROOT/"evidence/real-input-independent-audit.json")
    old_audit = read_json(original/"audit/data-audit.json")
    joins = read_json(original/"prepared/join-audit.json")
    old_targets = read_json(original/"cohort.json")
    target_summaries = {block: read_json(run/block/"cohort-and-targets.json") for block in ["development", "confirmation"]}
    external = read_json(external_path)
    observation = read_json(ROOT/"evidence/kaggle/submission-01-observation.json")
    image_path = ROOT/observation["screenshot"]["path"]
    if digest(image_path) != observation["screenshot"]["sha256"] or not observation["screenshot"].get("original_bytes_preserved"):
        raise ValueError("The genuine first-score screenshot differs from its original provenance")
    current_observation = read_json(ROOT/"evidence/kaggle/submission-02-observation.json")
    if not current_observation:
        raise ValueError("The genuine second score observation must be preserved before this report update")
    current_image_path = ROOT/current_observation["screenshot"]["path"]
    if digest(current_image_path) != current_observation["screenshot"]["sha256"] or not current_observation["screenshot"].get("original_bytes_preserved"):
        raise ValueError("The genuine second-score screenshot differs from its original provenance")
    current_scores = current_observation["score_interpretation"]
    first_scores = observation["score_interpretation"]
    public_gain = current_scores["public_score"]-first_scores["public_score"]
    private_gain = current_scores["private_score"]-first_scores["private_score"]
    private_relative_gain = private_gain/first_scores["private_score"]
    kaggle = external.get("kaggle", {})
    if (kaggle.get("screenshot_sha256") != current_observation["screenshot"]["sha256"]
            or kaggle.get("private_score") != current_scores["private_score"] or kaggle.get("public_score") != current_scores["public_score"]):
        raise ValueError("Latest central score evidence must identify the genuine second observation before rendering")
    refs = read_json(ROOT/"evidence/references.json").get("references", [])
    if not 3 <= len(refs) <= 7:
        raise ValueError("Three to seven previously verified primary sources are required")
    if raw.get("status") != "complete":
        raise ValueError("Independent actual raw audit is missing")
    for role in ["train", "valid", "future", "views", "template"]:
        if raw["files"][role]["sha256"] != old_audit["files"][role]["sha256"]:
            raise ValueError("Raw input hashes differ between the original actual run and independent audit")
    confirm_selected = confirmation["methods"][selected]
    confirm_pop = confirmation["methods"]["original_popularity"]
    dev_selected = development["methods"][selected]
    primary = confirm_selected["metrics"]["primary"]["map_at_5"]
    pop_primary = confirm_pop["metrics"]["primary"]["map_at_5"]
    v1_map = original_summary["methods"]["ppr_0.5"]["map_at_5"]
    original_primary = confirmation["methods"]["original_ppr_0.5"]["metrics"]["primary"]["map_at_5"]
    local_relative_gain = primary/original_primary-1
    old_relevant_counts = [len(row["relevant_ids"]) for row in old_targets["details"] if row["in_cohort"]]
    abstract = (
        "Recommending five SWEET.TV programmes requires reconstructing preferences from viewing intervals despite incomplete schedule labels and uncertain hidden targets. "
        "A predeclared fourteen-method early-development comparison uses preferred-Top5 overlap-seconds proxy targets and later confirmation, previously inspected in the original solution. "
        f'Development selected {chosen_label}; confirmation MAP@5 is {primary:.6f} versus {pop_primary:.6f} for popularity, while the second user-reported private score is {current_scores["private_score"]:.5f}. '
        f'The local {100*local_relative_gain:.1f}% proxy gain over original PPR accompanies only a {100*private_relative_gain:.2f}% user-reported private-score difference, with evaluator equivalence, independent submission linkage, Kaggle username, personal verification and accepted native-export handling unresolved.'
    )
    if not 50 <= len(abstract.split()) <= 150:
        raise ValueError("Abstract must contain 50–150 words")
    identity = external.get("identity", {})
    author = "Timur Khayrislamov"
    if identity.get("confirmed_russian_full_name"):
        author += " (" + identity["confirmed_russian_full_name"] + ")"
    identity_text = "Kaggle: " + (identity.get("kaggle_username") or "unconfirmed")
    if not identity.get("confirmed_russian_full_name"):
        identity_text += "; full Russian name unconfirmed"
    schedule_rows = [[role, count(raw["files"][role]["rows"]), count(raw["files"][role]["labelled_rows"]),
                      f'{100*raw["files"][role]["labelled_fraction"]:.2f}%'] for role in ["train", "valid", "future"]]
    full = joins["full"]
    page1 = (
        "<h1>Preferred-Top5 Television Recommendation on SWEET.TV</h1>"
        +p("<b>V2 FOLLOW-UP DRAFT — author, submission and final evidence pending</b>", "notice")
        +p(f"<b>{escape(author)}</b> · Individual FKN LLM4Rec exam · 30 October 2026<br />{escape(identity_text)}", "author")
        +"<h3>Abstract</h3>"+p(escape(abstract))
        +"<h2>1. Introduction</h2>"
        +p("Rank five programmes per user for weeks 31–42. The first completed solution compared popularity, content profiles and personalized PageRank (PPR). "
           f'Its broad all-observed MAP@5 was {v1_map:.6f}; the first user-provided crop shows private 0.28587, with exact file linkage unconfirmed. '
           "Its relevance set contained every qualifying held-out programme. This follow-up asks whether stronger personal seeds better predict a narrower preferred-Top5 proxy. "
           "A viewing records observable use, not enjoyment, novelty or revenue.")
        +"<h2>2. Related Work</h2>"
        +p("Attribute profiles follow the content-recommendation framework [1]. Nonuniform restart gives personalized diffusion [2]; our programme co-view graph and seed choices are project decisions. "
           "Official documentation supports the actual TF-IDF [3] and sparse CSR [4] operations. "
           "Implicit MF was an alternative, but PPR keeps co-view links and restart directly inspectable. Association rules would require an additional basket/window definition, while content directly uses supplied attributes. "
           "These are design rationales, not measured inferiority claims.")
        +"<h2>3. Method</h2><h3>Actual data, join and labelling losses</h3>"
        +p(f'The immutable CSVs contain {count(raw["files"]["views"]["rows"])} views from {count(raw["files"]["views"]["distinct_raw_user_ids"])} users and 1,260 template users; '
           f'{count(raw["files"]["template"]["user_ids_above_signed_int64"])} template IDs exceed signed int64 and remain strings. All supplied row counts match. '
           "Schedules use DD.MM.YYYY and views ISO timestamps, without offsets. Independent endpoint tests establish schedule seconds; one zero-duration unlabelled row is excluded. "
           "The 117 viewing-duration discrepancies are all +3,600 s relative to duraton, consistent with a DST hypothesis; no timezone correction is assumed.")
        +table(["Schedule", "Rows", "Nonzero-ID rows", "Labelled"], schedule_rows, ".17,.29,.30,.24")
        +p("Join half-open intervals on the same channel using max(0,min(view_end,airing_end)−max(view_start,airing_start)). Union repeated/overlapping views and same-item overlapping airings; discard conflicting positive labels. "
           "Aggregate fragments per canonical airing, then require 120 seconds, fixed before comparison. ID 0 is discarded as preference evidence. "
           f'The raw-view first quartile/median are 133/396 s. Across full history, {count(full["labelled_seconds"])} of {count(full["view_seconds_after_union"])} unioned channel-view seconds are labelled; '
           f'{count(full["unlabelled_seconds"])} are ID-0 airtime, {count(full["outside_schedule_seconds"])} outside schedules. '
           f'{count(full["below_threshold_user_airings"])} positive user-airings fail threshold; {count(full["users_without_usable_interactions"])} viewing users lack usable preferences. '
           f'The final history has {count(full["usable_user_item_pairs"])} user–programme pairs. Raw hashes, loss denominators and real accepted/rejected joins remain exported.', "small")
    )
    page1 += ("<h3>Shared modelling and v2 proxy</h3>"
        +p("Popularity counts distinct training viewers. Content uses category, three genres, year, director and actor tokens, binary TF, training-only smooth IDF and L2 vectors; log(1+seconds) weights the user profile. "
           "PPR keeps the original log-seconds co-view graph: per-user L1-normalized X, XᵀX without self-edges, deterministic top-100 neighbours and row-normalized P. "
           "Iterate p′=α(Pᵀp+dangling_mass·v)+(1−α)v to L1≤1e−8 or 300 iterations. All models allow repeats and share training-popularity fallback, including cold users.", "small")
        +p("V2 changes restart v to normalized seconds or watch counts raised to power .5 or 1. The one fixed recency variant sums accepted-airing seconds decayed with a 28-day half-life before normalization. "
           "The graph and fallback are unchanged. Every method ranks the same positive-duration labelled target candidates, returns five distinct IDs, and breaks ties by ID. "
           "Training-only metadata/vocabulary/weights precede each cutoff; future schedules provide availability/identity, never future-user preferences.", "small"))

    dev_rows = []
    for spec in config["methods"]:
        values = development["methods"][spec["name"]]
        dev_rows.append([("* " if spec["name"] == selected else "")+method_label(spec),
                         f'{values["metrics"]["primary"]["map_at_5"]:.5f}', f'{values["metrics"]["secondary"]["map_at_5"]:.5f}',
                         f'{values["metrics"]["all_observed"]["map_at_5"]:.5f}', f'{values["fit_seconds"]:.2f}', f'{values["predict_seconds"]:.2f}'])
    conf_rows = [[method_label(spec_by_name[name]), f'{values["metrics"]["primary"]["map_at_5"]:.5f}',
                  f'{values["metrics"]["secondary"]["map_at_5"]:.5f}', f'{values["metrics"]["all_observed"]["map_at_5"]:.5f}',
                  f'{values["fit_seconds"]:.2f}', f'{values["predict_seconds"]:.2f}'] for name, values in confirmation["methods"].items()]
    dev_cohort, conf_cohort = target_summaries["development"], target_summaries["confirmation"]
    overlap = conf_cohort["target_overlap"]
    old_concentration = confirmation["methods"]["original_ppr_0.5"]["score_concentration"]
    selected_concentration = confirm_selected["score_concentration"]
    page2 = (
        "<h2>3. Method (continued)</h2><h3>Frozen development and confirmation</h3>"
        +p("Early development trains [9 March,13 April) and validates [13 April,18 May), 2020. Select the highest primary MAP only; exact ties follow the frozen method order. "
           "Confirmation trains [9 March,18 May), validates [18 May,27 July), and assesses the three original baselines plus the selected method once. "
           "This later block had already been inspected in v1; it is not an untouched test and does not select parameters. The final selected strategy is refitted on [9 March,27 July) for [27 July,19 October).", "small")
        +p("Primary targets are at most five unique eligible held-out programmes with largest total accepted seconds; secondary targets use accepted watch counts. Both ties use ID ascending. "
           "All-observed labels remain diagnostic. The 14 predeclared methods are three originals, two ineligible history diagnostics, eight PPR variants alpha in {.05,.25} × seconds/watches × sqrt/raw, and one fixed α=.15/raw-seconds/28-day-recency variant. "
           "The grid revision was recorded before actual v2 metrics; no confirmation retuning or Kaggle feedback search was performed.", "small")
        +p("AP@5=Σ(r=1…5) precision@r·first_distinct_hit(r)/min(5,|R|); MAP averages the fixed cohort. Duplicate predictions retain rank but earn no repeat credit. "
           "Empty-target AP is zero; users without eligible targets are excluded once for every method. Fixtures 11/12 and 8/15 average .725. "
           f'Development has {count(dev_cohort["evaluation_users"])} users ({count(dev_cohort["cold_users"])} cold), confirmation {count(conf_cohort["evaluation_users"])} ({count(conf_cohort["cold_users"])} cold), '
           f'with {count(dev_cohort["candidate_count"])} / {count(conf_cohort["candidate_count"])} candidates and {count(dev_cohort["excluded_no_eligible_targets"])} / {count(conf_cohort["excluded_no_eligible_targets"])} excluded users.', "small")
        +"<h2>4. Experiments</h2><h3>All frozen development trials; * selected</h3>"
        +table(["Method", "MAP S", "MAP W", "MAP all", "Fit s", "Rank s"], dev_rows, ".40,.13,.13,.13,.105,.105", "compact")
        +"<h3>One confirmation assessment</h3>"
        +table(["Method", "MAP S", "MAP W", "MAP all", "Fit s", "Rank s"], conf_rows, ".40,.13,.13,.13,.105,.105", "compact")
        +p("S=Top5 seconds (primary), W=Top5 watches, all=broad observed labels; these are local proxies, not interchangeable with the hidden competition metric. "
           f'Original broad targets had median {statistics.median(old_relevant_counts):g} items; {count(sum(value>5 for value in old_relevant_counts))} users exceeded five. '
           f'Confirmation Top5 seconds retains mean {100*overlap["top5_seconds_mass_fraction"]["mean"]:.1f}% of eligible viewing seconds; '
           f'seconds/watch target mean Jaccard is {overlap["primary_secondary_jaccard"]["mean"]:.3f}. '
           f'In confirmation, original PPR exposes {count(old_concentration["distinct_recommended_programmes"])} distinct programmes; selected exposes {count(selected_concentration["distinct_recommended_programmes"])} '
           f'(top-five programme exposure shares {100*old_concentration["top5_programmes_exposure_fraction"]:.1f}% / {100*selected_concentration["top5_programmes_exposure_fraction"]:.1f}%). '
           f'Selected primary AP has median {confirm_selected["metrics"]["primary"]["ap_distribution"]["median"]:.3f}, '
           f'{count(confirm_selected["metrics"]["primary"]["ap_zero_users"])} zero and {count(confirm_selected["metrics"]["primary"]["ap_one_users"])} perfect users. '
           "Concentration measures returned slots; scores are not probabilities. Full AP distributions and warm/cold means are exported.", "small")
    )
    user_pages = []
    example_analysis = []
    for user_index, user in enumerate(examples["users"]):
        histories = user["histories"]
        history_rows = []
        for row in sorted(histories["train"], key=lambda record: (-record["seconds"], record["item_id"]))[:3]:
            history_rows.append(["Train", str(row["item_id"]), row["title"], row["category"], f'{row["seconds"]:.0f}'])
        if not histories["train"]:
            history_rows.append(["Train", "—", "No usable training history", "—", "0"])
        for row in user["primary_programmes"]:
            history_rows.append(["Target", str(row["item_id"]), row["title"], row["category"], f'{row["seconds"]:.0f}'])
        rec_rows = [[row["display_role"][0], str(row["rank"]), str(row["item_id"]), row["title"], row["category"], f'{row["score"]:.5g}',
                     ("Y" if row["primary_top5_seconds_hit"] else "N")+"/"+("Y" if row["broad_observed_hit"] else "N")] for row in user["recommendations"]]
        role_results = []
        for role, method in examples["roles"]:
            records = [row for row in user["recommendations"] if row["display_role"] == role]
            role_results.append(f'{role}: {sum(row["primary_top5_seconds_hit"] for row in records)}/5 proxy, {sum(row["broad_observed_hit"] for row in records)}/5 broad hits')
        selected_rows = [row for row in user["recommendations"] if row["display_role"] == "Selected"]
        concrete = selected_rows[0]
        example_analysis.append(f'{user["role"]}: '+"; ".join(role_results)+".")
        page = (
            f'<h2>4. Experiments: fixed {escape(user["role"])} viewer</h2>'
            +p(f'<b>Exact user ID {escape(user["user_id"])}</b>. Retained from v1 before this search: '
               f'{"maximum" if user_index==0 else "minimum"} distinct usable training items, deterministic ID ties; light differs from heavy. '
               f'{count(user["actual_training_items"])} training and {count(user["actual_heldout_items"])} held-out programmes are observed. '
               "Training rows below are the largest three by seconds; target rows show the complete actual primary Top5. Full histories are exported. "
               "Titles/categories use each window’s earliest nonempty schedule metadata; one ID can have different displayed titles across windows.")
            +table(["Window", "ID", "Observed programme title", "Category", "Seconds"], history_rows, ".10,.12,.45,.19,.14", "examples")
            +p("P=Popularity; C=Content; S=development-selected strategy. Proxy/Broad flags distinguish membership in the five seconds-ranked targets from any observed eligible held-out programme. "
               "Scores have model-specific units; displayed values are rounded, full precision and fallback reasons are preserved.", "small")
            +table(["M", "#", "ID", "Programme title", "Category", "Score", "P/B"], rec_rows, ".045,.04,.12,.37,.155,.17,.10", "examples")
            +p(escape("; ".join(role_results)+". ")+f'The selected list starts with ID {concrete["item_id"]}, “{escape(concrete["title"])}” ({escape(concrete["category"])}), '
               f'with {"an actual preferred-Top5 hit" if concrete["primary_top5_seconds_hit"] else "no observed preferred-Top5 hit"}. '
               "A broad hit outside Top5 illustrates the protocol distinction, not a proven dislike. Absence from these logs does not establish a programme was never watched.", "small")
        )
        if user_index == 1:
            page += ("<h3>Genuine user-provided returned-score history</h3>"
                +f'<img src="{escape(str(image_path))}" width="499" alt="Unmodified user-provided first Kaggle submission row: public 0.29103, private 0.28587" />'
                +f'<img src="{escape(str(current_image_path))}" width="499" alt="Unmodified user-provided v2 Kaggle submission row: public 0.32589, private 0.28919" />'
                +p(f'First crop: submission.csv, public {first_scores["public_score"]:.5f}, private {first_scores["private_score"]:.5f}. '
                   f'Second: submission_v2.csv, Complete (after deadline), public {current_scores["public_score"]:.5f}, private {current_scores["private_score"]:.5f}. '
                   "The user’s v2 workflow and visible filename match the released v2 file; exact Kaggle ID/account/description, absolute upload time and submitted hash are not independently verified. "
                   "The displayed relative times are not converted into timestamps. Original pixels and the earlier report are preserved; observation numbering does not establish all attempted submissions.", "small"))
        user_pages.append(page)

    github = external.get("github", {})
    github_url = github.get("url")
    github_verified = bool(github.get("verified") is True and github.get("csv_sha256") == final["sha256"]
                           and github.get("anonymous_repo_http_status") == 200 and github.get("anonymous_csv_http_status") == 200
                           and re.fullmatch(r"[0-9a-f]{40}", str(github.get("commit", ""))))
    if github_url and re.fullmatch(r"https://github\.com/[^/]+/[^/]+/?", github_url):
        github_text = f'<a href="{escape(github_url)}">{escape(github_url)}</a>. '
        if github_verified:
            github_text += f'Verified v2 commit <a href="{escape(github_url.rstrip("/")+"/commit/"+github["commit"])}">{escape(github["commit"])}</a>; remote/local CSV hashes match. '
            if github.get("code_model_commit") and github["code_model_commit"] != github["commit"]:
                github_text += f'The latest commit updates score documentation only; model/code commit <a href="{escape(github_url.rstrip("/")+"/commit/"+github["code_model_commit"])}">{escape(github["code_model_commit"])}</a> is unchanged. '
        else:
            github_text += "Existing publication evidence identifies the earlier CSV; v2 commit/accessibility/CSV-hash verification remains pending. "
    else:
        github_text = "A verified accessible repository URL and exact v2 CSV linkage remain pending. "
    if primary >= pop_primary:
        decision = f'I would select {chosen_label} as the next offline SWEET.TV strategy: early-development primary MAP is {dev_selected["metrics"]["primary"]["map_at_5"]:.6f}, and confirmation is {primary:.6f} versus popularity {pop_primary:.6f}.'
    else:
        decision = f'I would retain popularity for a service decision because its confirmation primary MAP {pop_primary:.6f} exceeds the development-selected {chosen_label} at {primary:.6f}. The new CSV still follows the predeclared development selection; confirmation does not retrospectively retune it.'
    environment = manifest["environment"]
    native = external.get("native_packaging", {})
    native_name = native.get("course_handin_native_filename", "timur_khayrislamov_exam_session.jsonl")
    native_manifest_path = ROOT/native.get("latest_cutoff_manifest", "evidence/native-sessions-cutoff-04/export-manifest.json")
    native_label = "cutoff"+native_manifest_path.parent.name.rsplit("-", 1)[-1]
    native_snapshot = read_json(native_manifest_path)
    if not native_snapshot or len(native_snapshot.get("files", [])) != 5 or native_snapshot.get("all_originals_unchanged") is not True:
        raise ValueError("The actual five-file latest native snapshot is required")
    for entry in native_snapshot["files"]:
        copy = ROOT/entry["copy"]
        if (entry.get("byte_for_byte") is not True or entry.get("unmodified_native_jsonl") is not True
                or copy.stat().st_size != entry["bytes"] or digest(copy) != entry["sha256"]):
            raise ValueError("A latest native copy fails its exact-byte provenance check")
    native_root = next(entry for entry in native_snapshot["files"] if entry.get("parent_thread_id") is None)
    score_receipt = dt.datetime.fromisoformat(current_observation["image_received_in_native_record_utc"].replace("Z", "+00:00"))
    native_cutoff = dt.datetime.fromisoformat(native_snapshot["cutoff_started_utc"])
    score_in_snapshot = score_receipt < native_cutoff
    score_native_note = (f'The second score image received at {escape(current_observation["image_received_in_native_record_utc"])} is included before {native_label}. '
                         if score_in_snapshot else f'The second score image received at {escape(current_observation["image_received_in_native_record_utc"])} is after {native_label} and outside that snapshot. ')
    inventory = read_json(ROOT/"evidence/native-session-inventory.json")
    cli_version = next((str(row["cli_version"]) for row in inventory.get("sessions", [])
                        if row.get("session_id") == inventory.get("root_session_id") and row.get("cli_version")), "not recorded")
    selected_diag = read_json(run/"confirmation/diagnostics"/f"{selected}.json")
    verification = read_json(ROOT/"evidence/verification.json")
    test_note = ""
    if (verification.get("passed") is True and verification.get("source_hashes_stable") is True
            and verification.get("source_hashes") == manifest.get("source_hashes")):
        test_note = f' The matching source checkpoint passes {count(verification["tests_run"])} synthetic tests.'
    independent_path = ROOT/"evidence/improvement-v2-independent-audit.json"
    independent_v2 = read_json(independent_path)
    independent_note = " A separate raw-target/checkpoint audit remains pending."
    if independent_v2:
        if independent_v2.get("passed") is not True or independent_v2.get("final_comparison", {}).get("new_sha256") != final["sha256"]:
            raise ValueError("The additional independent v2 audit fails or identifies a different final CSV")
        independent_note = (f' The separate independent audit reconstructs targets and AP, reproduces {count(independent_v2["original_three_baselines_identical_confirmation_slots"])} '
                            f'original baseline slots, checks selected fitted state, and records {count(independent_v2["final_comparison"]["changed_rows_vs_v1"])} changed final rows '
                            '(evidence/improvement-v2-independent-audit.json).')
    convergence = ""
    if "walk_calls" in selected_diag:
        convergence = (f' All {count(selected_diag["converged_walks"])}/{count(selected_diag["walk_calls"])} selected validation walks converged; '
                       f'maximum recorded mass error {selected_diag["max_observed_mass_error"]:.2g}.')
    page5 = (
        "<h2>5. Discussion</h2>"
        +p("<b>Which recommendation strategy would you select for SWEET.TV, and why?</b> "+escape(decision)
           +f' The selected confirmation fit/ranking cost is {confirm_selected["fit_seconds"]:.3f}/{confirm_selected["predict_seconds"]:.3f} s. '
           "Personal seeds emphasize a viewer’s repeatedly watched items; graph diffusion transfers co-view signal; content compares attributes; popularity captures broadly watched programmes. "
           "The history-only diagnostics separate the seed change from graph contribution. "+escape(" ".join(example_analysis)))
        +p("The two deterministic extremes were fixed before v2 scoring and are illustrations, not representative user samples. In the cold example, common fallback can make methods identical; that does not establish personalization. "
           f'The selected broad-label MAP is {confirm_selected["metrics"]["all_observed"]["map_at_5"]:.6f}, versus original PPR {confirmation["methods"]["original_ppr_0.5"]["metrics"]["all_observed"]["map_at_5"]:.6f}; '
           "the preferred-Top5 gain is not an improvement under every objective. "
           "The original broad labels reward popular programmes even when they are outside the five most-watched items. V2 narrows that target and tests concentration explicitly, but total seconds can reflect background viewing; watches and seconds also disagree. "
           "Neither proxy identifies Kaggle’s hidden target construction. Unlabelled airtime, missing attributes, unknown timezone and the previously inspected confirmation block limit transfer. No enjoyment, online lift or revenue effect is measured.")
        +p(f'Against original PPR, primary local MAP rose {100*local_relative_gain:.2f}% ({original_primary:.6f} to {primary:.6f}); '
           f'the two user-provided score observations instead differ by public {public_gain:+.5f} and private {private_gain:+.5f} ({100*private_relative_gain:+.2f}% relative). '
           "The roughly 60% local proxy gain over original PPR accompanies only a 1.16% user-reported private-score difference, suggesting limited proxy transfer. Context supports a v2 association, but exact submitted bytes and account linkage are unverified; this is not established transfer, causal proof or hidden-evaluator equivalence. "
           "The configuration and predictions were frozen before the second observation, and no new grid or post-score model tuning was performed. Repeated score observations are not treated as a clean holdout.", "small")
        +"<h3>Execution, checks and observed failures</h3>"
        +p(f'Actual v2 run: {manifest["wall_seconds"]:.2f} s, peak RSS {manifest["peak_resident_memory_bytes"]/1024**3:.2f} GiB; '
           f'{escape(environment.get("cpu", "CPU not recorded"))}, Python {escape(environment["python"].split(" (")[0])}, '
           f'NumPy {escape(environment["packages"]["numpy"])}, pandas {escape(environment["packages"]["pandas"])}, '
           f'SciPy {escape(environment["packages"]["scipy"])}, sklearn {escape(environment["packages"]["scikit-learn"])}. '
           "All six exported-AP audits use exact Fraction arithmetic; prediction sources stayed frozen. "+escape(convergence+test_note+independent_note)
           +" The earlier real pipeline stopped on schedule-date parsing and file-date overlaps; DD.MM parsing and explicit clipped calendar windows fixed those observed defects. "
           "Independent raw/ID/candidate/metadata audits and synthetic boundary/cold/duplicate fixtures remain recorded; no new failure is invented.", "small")
        +p(f'New full-history fit/ranking takes {final["fit_seconds"]:.3f}/{final["predict_seconds"]:.3f} s and yields exactly 1,260 rows of five distinct IDs from {count(final["candidate_count"])} eligible future programmes. '
           f'Exact v2 CSV SHA-256: <b>{final["sha256"]}</b>. '+github_text
           +f'The user-reported v2 crop genuinely displays private {current_scores["private_score"]:.5f} and public {current_scores["public_score"]:.5f}. '
           +"Its filename/context match the released v2 artifact, while exact Kaggle ID/account/description/submitted-hash verification remains pending; the screenshot is not described as API-verified file linkage.", "small")
        +"<h2>6. AI Usage Disclosure</h2>"
        +p(f'I used Codex (CLI {escape(cli_version)}) and delegated agents to inspect supplied sources and data, implement and debug the joins/models, run checks, prepare the v2 proxy and render this draft. '
           "The agents performed the automated calculations, reference checks and PDF inspection. Pazzani–Billsus was available through a publisher preview; PageRank sections/appendix and official API documentation were reviewed by AI. "
           "My own hand calculation, personal source reading, rerun and interpretation have not been reported; no such actions are invented here. Author verification and a final rewrite remain pending.")
        +p(f'The intended native hand-in filename is <b>{escape(native_name)}</b>, a fresh unmodified root JSONL copy with an explicit cutoff. '
           f'Actual {native_label} began at {escape(native_snapshot["cutoff_started_utc"])}: root <b>{escape(Path(native_root["copy"]).name)}</b>, '
           f'{count(native_root["bytes"])} unchanged bytes, SHA-256 <b>{native_root["sha256"]}</b>. '
           "All five native copies pass exact-byte checks. Four non-root sessions (three direct children and a development_prep grandchild) are retained separately in private evidence, disclosed as work absent from the root tool trace, and never reconstructed or merged. "
           "Checkpoint02 predates the v2 work, first-score image and grandchild session, so it is not a complete final export. "
           "The unchanged root log now exceeds the guidelines’ 10 MB contact threshold (p. 11); accepted native-size handling remains unresolved. No instructor contact, compressed-format acceptance or upload exception is claimed, and no transcript is trimmed. "
           +score_native_note+"Final score-report corrections, rendering, inspection, evidence bookkeeping and handoff occur after this cutoff and are outside the snapshot. This report remains outside the two-file course hand-in until Kaggle identity, author checks, submission linkage and native-size handling are settled. "
           "FKN deadline: 1 November 2026, 19:00 Europe/Moscow.", "small")
    )
    reference_page = "<h2>References</h2>"+p("Verified primary sources, retrieved during this work; access date 30 October 2026. Exact retrieval hashes and passages: evidence/references.json.", "small")
    for ref in refs:
        prefix = ref.get("ieee", ref["title"]).partition("[Online]. Available:")[0]
        reference_page += p(f'[{ref["id"]}] {escape(prefix)} <a href="{escape(ref["url"])}">[Primary source]</a>. [Accessed: {escape(ref["accessed_client_date"])}].', "reference")
    reference_page += p("Reproduction and traceability: the original real run is experiments/competition-20261030-03; the new proxy run is "+escape(str(run.relative_to(ROOT)))
                        +". The frozen configuration, development and confirmation summaries, all per-user AP exports, target sets, full scores, fitted states and source hashes are retained. "
                        +"report/v2-two-user-examples.json preserves all thirty displayed recommendation rows and complete histories for the two original fixed users. "
                        +"Rebuild v2 preparation with scripts/prepare_improvement.py --base-run experiments/competition-20261030-03 --output-root experiments/improvement-v2-data; the runner is python -m sweettv.improve. "
                        +"The deterministic configuration records seed 20261030. Reported values are rounded; JSON/CSV exports retain full precision. The original report and predictions remain preserved.", "small")
    pages = [page1, page2, *user_pages, page5, reference_page]
    css = 'body{font:9.5pt/1.2 "Times New Roman",serif;color:#000}.page{page-break-after:always}table{border-collapse:collapse;width:100%;font-size:8pt}td,th{border:.35pt solid #777;padding:4pt;vertical-align:top}th{background:#eee}.small{font-size:8.5pt}.examples{font-size:7.7pt}.compact{font-size:8pt}a{color:#000}h1{text-align:center;font-size:18pt}h2{font-size:11.5pt}h3{font-size:10pt}.author,.notice{text-align:center}img{max-width:100%;height:auto}'
    document = '<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8" /><title>SWEET.TV v2 follow-up draft</title><style>'+css+'</style></head><body>'
    document += "".join('<div class="page">'+page+'</div>' for page in pages)+"</body></html>\n"
    return document, examples, {"abstract_words": len(abstract.split()), "abstract_sentences": 4, "github_v2_verified": github_verified,
                                "first_score_preserved": True, "v2_score_pending": False,
                                "current_score_observation": current_observation["observation_id"],
                                "user_reported_private_score": current_scores["private_score"], "user_reported_public_score": current_scores["public_score"],
                                "public_score_gain": public_gain, "private_score_gain": private_gain, "private_relative_gain": private_relative_gain,
                                "independent_exact_kaggle_file_linkage_verified": False,
                                "score_image_sha256": current_observation["screenshot"]["sha256"],
                                "prior_v2_report_preservation_manifest": "evidence/report-v2-before-submission-02/preservation-manifest.json",
                                "native_snapshot_manifest": str(native_manifest_path), "native_cutoff_utc": native_snapshot["cutoff_started_utc"],
                                "native_root_bytes": native_root["bytes"], "native_root_sha256": native_root["sha256"],
                                "native_oversize_handling_unresolved": True, "report_work_after_native_cutoff_disclosed": True,
                                "final_csv_sha256": final["sha256"], "source_run": str(run)}


def render(source, destination):
    normal, bold = register_fonts()
    styles = {
        "body": ParagraphStyle("V2Body", fontName=normal, fontSize=9.35, leading=11.1, spaceAfter=5.5, splitLongWords=True),
        "small": ParagraphStyle("V2Small", fontName=normal, fontSize=8.45, leading=10, spaceAfter=5, splitLongWords=True),
        "title": ParagraphStyle("V2Title", fontName=bold, fontSize=17, leading=19, alignment=TA_CENTER, spaceAfter=7),
        "h2": ParagraphStyle("V2H2", fontName=bold, fontSize=11.4, leading=13.2, spaceBefore=7, spaceAfter=4, keepWithNext=True),
        "h3": ParagraphStyle("V2H3", fontName=bold, fontSize=10, leading=11.7, spaceBefore=5, spaceAfter=3, keepWithNext=True),
        "author": ParagraphStyle("V2Author", fontName=normal, fontSize=8.8, leading=10.4, alignment=TA_CENTER, spaceAfter=6),
        "notice": ParagraphStyle("V2Notice", fontName=bold, fontSize=9.3, leading=11, alignment=TA_CENTER, spaceAfter=6),
        "cell": ParagraphStyle("V2Cell", fontName=normal, fontSize=8, leading=9.3, splitLongWords=True),
        "compact": ParagraphStyle("V2Compact", fontName=normal, fontSize=8, leading=9.2, splitLongWords=True),
        "examples": ParagraphStyle("V2Examples", fontName=normal, fontSize=7.7, leading=9.1, splitLongWords=True),
        "reference": ParagraphStyle("V2Reference", fontName=normal, fontSize=9.2, leading=11.2, spaceAfter=10, splitLongWords=True),
    }
    root = ET.parse(source).getroot()
    body = next(element for element in root if local_tag(element) == "body")
    width, height = A4
    margin = 46
    usable = width-2*margin
    story = []
    for index, page in enumerate(element for element in body if local_tag(element) == "div"):
        if index:
            story.append(PageBreak())
        for element in page:
            tag, cls = local_tag(element), element.attrib.get("class", "")
            if tag in {"h1", "h2", "h3", "p"}:
                style = styles[{"h1": "title", "h2": "h2", "h3": "h3"}.get(tag, cls if cls in styles else "body")]
                story.append(Paragraph(paragraph_markup(element), style))
            elif tag == "img":
                path = Path(element.attrib["src"])
                info = fitz.Pixmap(str(path))
                display_width = min(usable, float(element.attrib.get("width", usable)))
                story.extend([Spacer(1, 3), Image(str(path), width=display_width, height=display_width*info.height/info.width), Spacer(1, 4)])
            elif tag == "table":
                data = []
                for row in element:
                    cells = []
                    for cell in row:
                        markup = paragraph_markup(cell)
                        if local_tag(cell) == "th":
                            markup = "<b>"+markup+"</b>"
                        cells.append(Paragraph(markup, styles[cls] if cls in {"examples", "compact"} else styles["cell"]))
                    data.append(cells)
                ratios = [float(value) for value in element.attrib["data-widths"].split(",")]
                item = Table(data, colWidths=[usable*ratio/sum(ratios) for ratio in ratios], repeatRows=1, hAlign="LEFT")
                pad = 3.2 if cls in {"compact", "examples"} else 3.7
                item.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                                          ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#777777")),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                          ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
                story.extend([Spacer(1, 2), item, Spacer(1, 5)])
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(bold, 8)
        canvas.drawString(margin, height-28, "V2 FOLLOW-UP DRAFT — submission linkage, identity and personal verification pending")
        canvas.setLineWidth(.35)
        canvas.line(margin, height-33, width-margin, height-33)
        canvas.setFont(normal, 8)
        canvas.drawString(margin, 24, "Timur Khayrislamov · FKN LLM4Rec midterm · 30 October 2026")
        canvas.drawRightString(width-margin, 24, str(doc.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=45, bottomMargin=40,
                                 title="SWEET.TV preferred-Top5 recommendation — V2 DRAFT", author="Timur Khayrislamov")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    with fitz.open(destination) as pdf:
        refs = [index for index, page in enumerate(pdf) if re.search(r"^References\s*$", page.get_text(), re.MULTILINE)]
        if not refs or refs[0] > 5:
            raise ValueError(f"Expected at most five body pages; observed references start {refs[0] if refs else 'missing'}")
        if any("DRAFT" not in page.get_text() for page in pdf):
            raise ValueError("Every report page must identify draft status")
        return {"body_pages": refs[0], "references_pages": len(pdf)-refs[0], "total_pages": len(pdf),
                "pdf_links": sum(len(page.get_links()) for page in pdf),
                "embedded_images": sum(len(page.get_images(full=True)) for page in pdf)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"experiments/improvement-v2-01")
    parser.add_argument("--original-run", type=Path, default=ROOT/"experiments/competition-20261030-03")
    parser.add_argument("--external-evidence", type=Path, default=ROOT/"evidence/external-evidence.json")
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    source = REPORT/SOURCE_NAME
    context = {}
    if args.refresh_source or not source.exists():
        document, examples, context = source_document(args.run_dir.resolve(), args.original_run.resolve(), args.external_evidence)
        source.write_text(document, encoding="utf-8")
        (REPORT/EXAMPLES_NAME).write_text(json.dumps(examples, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    destination = REPORT/PDF_NAME
    checks = render(source, destination)
    evidence = {"status": "v2_incomplete_draft", "source": str(source), "pdf": str(destination), "examples": str(REPORT/EXAMPLES_NAME),
                "script_sha256": digest(Path(__file__)), "source_sha256": digest(source), "pdf_sha256": digest(destination),
                "examples_sha256": digest(REPORT/EXAMPLES_NAME), **checks, **context,
                "visual_inspection_pending": True, "old_report_overwritten": False}
    (ROOT/"evidence/report-v2-render-checks.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key: value for key, value in evidence.items() if key not in {"source_sha256", "examples_sha256"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
