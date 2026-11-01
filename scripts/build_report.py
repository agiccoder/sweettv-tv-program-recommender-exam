#!/usr/bin/env python3
"""Build an explicitly incomplete exam draft from one editable XHTML source.

Run with --refresh-source to refresh prose from current local evidence. Without
that flag, existing report/report-source.html edits are preserved and rendered.
This script intentionally has no final-report mode and writes only under report/.
It does not fabricate absent competition evidence or modify evidence/session logs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
CLIENT_DATE = "2026-10-30"
DRAFT_NAME = "timur_khayrislamov_exam_report_DRAFT.pdf"


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def escape(value):
    return html.escape(str(value), quote=True)


def p(value, cls=None):
    return f'<p{(" class=" + chr(34) + cls + chr(34)) if cls else ""}>{value}</p>'


def table(headers, rows, widths, cls=""):
    content = "<tr>" + "".join(f"<th>{escape(x)}</th>" for x in headers) + "</tr>"
    for row in rows:
        content += "<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>"
    return f'<table data-widths="{widths}" class="{escape(cls)}">{content}</table>'


def verification_sentence(verification, failure):
    """Prefer an explicit current manifest, without guessing its nested schema."""
    passed = None
    successful = False
    def visit(obj):
        nonlocal passed, successful
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if key in {"tests_passed", "tests_run", "test_count", "tests_count", "observed_tests_passed"} and isinstance(value, int) and not isinstance(value, bool):
                passed = max(passed or 0, value)
            if key in {"exit_code", "returncode", "observed_exit_code"} and value == 0:
                successful = True
            if key == "passed" and value is True:
                successful = True
            if key == "status" and str(value).lower() in {"passed", "pass", "ok", "success", "successful"}:
                successful = True
            if isinstance(value, dict):
                visit(value)
    visit(verification)
    if (verification and passed is not None and successful
            and verification.get("passed") is True
            and verification.get("failures") == 0 and verification.get("errors") == 0
            and verification.get("source_hashes_stable") is True):
        return f'The current verification manifest records {passed} passing synthetic tests; see <b>evidence/verification.json</b> for the exact command, scope, and artifacts.'
    if verification:
        return 'A current verification manifest is available in <b>evidence/verification.json</b>; its recorded status must be read alongside the exact test logs. No test total is inferred here.'
    resolution = failure.get("resolution", {})
    count = resolution.get("observed_tests_passed")
    if count is not None and resolution.get("observed_exit_code") == 0:
        return f'The recorded post-fix run passed {int(count)} synthetic tests (<b>evidence/synthetic-pipeline-after-fix.txt</b>). This repair checkpoint is not a claim about later edits or a final competition run.'
    return 'A current verification manifest is pending. No completed test count is asserted.'


def create_source(run_dir, session_manifest=None, external_evidence=None):
    """Use completed real artifacts only; absent external evidence keeps a draft."""
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir/'manifest.json')
    summary = read_json(run_dir/'validation-summary.json')
    if (manifest.get('status') != 'complete_local_competition_run'
            or manifest.get('synthetic_fixture') is not False
            or manifest.get('source_hashes_match') is not True
            or summary.get('status') != 'complete'):
        raise ValueError('A completed real competition run with stable sources is required to refresh this report; no real metrics are inferred from partial or synthetic runs.')
    audit = read_json(run_dir/'audit/data-audit.json')
    independent = read_json(ROOT/'evidence/real-input-independent-audit.json')
    joins = read_json(run_dir/'prepared/join-audit.json')
    protocol = read_json(run_dir/'prepared/protocol.json')
    candidates = read_json(run_dir/'prepared/candidates.json')
    cohort = read_json(run_dir/'cohort.json')
    demo = read_json(run_dir/'two-user-comparison.json')
    ap_audit = read_json(run_dir/'independent-ap-audit.json')
    final = read_json(run_dir/'final/submission-validation.json')
    verification = read_json(ROOT/'evidence/verification.json')
    failure = read_json(ROOT/'evidence/synthetic-pipeline-failure.json')
    refs = read_json(ROOT/'evidence/references.json').get('references', [])
    inventory = read_json(ROOT/'evidence/native-session-inventory.json')
    snapshot = read_json(session_manifest) if session_manifest else {}
    external = read_json(external_evidence) if external_evidence else {}
    if ap_audit.get('passed') is not True or final.get('passed') is not True:
        raise ValueError('Actual exported-AP audit and final CSV validation must both pass.')
    final_path = run_dir/'final/submission.csv'
    actual_csv_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    if actual_csv_hash != final.get('sha256'):
        raise ValueError('The actual final CSV differs from its recorded validation hash.')
    identity = external.get('identity', {})
    russian_name = str(identity.get('confirmed_russian_full_name') or '').strip()
    kaggle_name = str(identity.get('kaggle_username') or '').strip()
    email = str(identity.get('email') or '').strip()
    github = external.get('github', {})
    github_verified = github.get('verified') is True
    if github_verified:
        if github.get('csv_sha256') != actual_csv_hash:
            raise ValueError('Verified GitHub evidence must identify the exact actual final CSV hash.')
        if github.get('anonymous_repo_http_status') != 200 or github.get('anonymous_csv_http_status') != 200:
            raise ValueError('Verified public GitHub evidence needs successful anonymous repository and CSV checks.')
        if not re.fullmatch(r'https://github\.com/[^/]+/[^/]+/?', str(github.get('url', ''))):
            raise ValueError('Verified GitHub evidence needs an actual repository URL.')
        if not re.fullmatch(r'[0-9a-f]{40}', str(github.get('commit', ''))):
            raise ValueError('Verified GitHub evidence needs the exact full commit SHA.')
    author_name='Timur Khayrislamov'+(' ('+russian_name+')' if russian_name else '')
    identity_line=('Kaggle: '+escape(kaggle_name) if kaggle_name else 'Kaggle name unconfirmed')
    identity_line+=' · '+('Email: '+escape(email) if email else 'Email not supplied')
    if not russian_name:
        identity_line+=' · Russian full name unconfirmed'
    pending_items=[]
    if not russian_name:
        pending_items.append('confirmed full Russian name')
    if not kaggle_name:
        pending_items.append('confirmed Kaggle username')
    if not github_verified:
        pending_items.append('verified GitHub publication')
    pending_items.extend(['designated Kaggle returned private score and genuine screenshot', 'student verification and author-written interpretation'])
    pending_phrase=', '.join(pending_items[:-1])+', and '+pending_items[-1]

    if len(demo.get('recommendations', [])) != 30:
        raise ValueError('Expected exactly 30 actual two-user Top-5 rows for three methods.')
    methods = summary['methods']
    selected = summary['selected_method']
    best_ppr = summary['selected_ppr']
    if set(demo['methods']) != {'popularity', 'content', best_ppr}:
        raise ValueError('Two-user methods disagree with the selected PPR trial.')
    for name, result in methods.items():
        if abs(result['map_at_5'] - ap_audit['methods'][name]['map_at_5']) > 1e-12:
            raise ValueError('Metric summary and independent exported AP audit disagree.')
    files = audit['files']
    raw = independent.get('files', {})
    if independent.get('status') != 'complete' or not set(files).issubset(raw):
        raise ValueError('The complete independent raw audit is required; missing raw statistics must never become zero-valued claims.')
    if not 3 <= len(refs) <= 7:
        raise ValueError('Three to seven verified primary references are required.')
    if final.get('rows') != files['template']['rows'] or final.get('rows') != 1260:
        raise ValueError('The real final CSV must cover exactly 1260 actual template rows.')
    if final.get('template_sha256') != files['template']['sha256']:
        raise ValueError('Final CSV validation used a different input template.')
    for role, info in files.items():
        if raw.get(role) and info.get('sha256') != raw[role].get('sha256'):
            raise ValueError('Independent raw audit hashes do not match the actual model run.')
    counts = lambda x: f'{int(x):,}'
    number = lambda x, d=3: f'{float(x):,.{d}f}'
    pct = lambda x: f'{100*float(x):.4f}%'
    h = lambda seconds: number(float(seconds)/3600, 1)
    fmt_method = lambda name: {'popularity':'Popularity', 'content':'Content'}.get(name, 'PPR '+name.split('_')[-1])
    environment = manifest['environment']
    search = summary['search']
    source_run = str(run_dir.relative_to(ROOT))
    selected_map = methods[selected]['map_at_5']
    pop_map = methods['popularity']['map_at_5']
    map_change = selected_map-pop_map
    train, valid = joins['train'], joins['valid']
    view_info = raw.get('views', {})
    boundary = protocol.get('explicit_calendar_boundaries') or manifest.get('calendar_boundaries') or {}
    date_string = lambda role: str(boundary.get(role, protocol.get('timestamp_display', {}).get(role, 'not recorded'))).replace('T00:00:00','').replace('T00:00','')
    relevant_counts=[len(row['relevant_ids']) for row in cohort['details'] if row['in_cohort']]
    raw_rows = sum(int(info['rows']) for role,info in files.items() if role != 'template')
    schedule_rows = [[role, counts(files[role]['rows']), counts(files[role]['labelled_rows']), pct(files[role]['labelled_fraction']), counts(len(candidates[role]))] for role in ['train','valid','future']]
    join_rows = []
    for name in ['train','valid']:
        info=joins[name]
        join_rows.append([name,h(info.get('view_seconds_after_union',0)),h(info.get('labelled_seconds',0)),h(info.get('unlabelled_seconds',0)),h(info.get('ambiguous_labels_seconds',0)),h(info.get('outside_schedule_seconds',0)),counts(info['usable_user_item_pairs'])])
    result_rows = [[fmt_method(name),number(result['map_at_5'],6),number(result['fit_seconds'],3),number(result['predict_seconds'],3)] for name,result in methods.items()]
    metrics_text = '; '.join(f'{fmt_method(name)} {result["map_at_5"]:.6f}' for name,result in methods.items())
    abstract = (
        f'Top-5 television recommendation requires reconstructing preferences from viewing intervals and incompletely labelled broadcast schedules. '
        f'On {counts(files["views"]["rows"])} real viewing rows, this study compares popularity, metadata TF-IDF profiles, and personalized PageRank using a shared blocked temporal MAP@5 holdout. '
        f'{fmt_method(selected)} achieved MAP@5 {selected_map:.6f} across {counts(cohort["evaluation_users"])} eligible template users, compared with {pop_map:.6f} for popularity. '
        f'The validation-selected strategy is locally reproducible, while this draft remains incomplete until {pending_phrase} are supplied.'
    )
    if not 50 <= len(abstract.split()) <= 150:
        raise ValueError('Real-results abstract must contain 50–150 words.')

    page1 = (
        '<h1>Temporal Television Recommendation on SWEET.TV</h1>'
        +p('<b>REAL-RESULTS DRAFT — external and personal evidence pending</b>','notice')
        +p(f'<b>{escape(author_name)} · Individual · {CLIENT_DATE}</b><br />FKN LLM4Rec midterm exam · {identity_line}','author')
        +'<h3>Abstract</h3>'+p(escape(abstract))
        +'<h2>1. Introduction</h2>'
        +p('The task is to rank five programmes per template user for a later schedule. A view spanning two airings must be divided by same-channel overlap. '
           'This implementation contributes a common protocol, two personalized methods and a separate popularity reference, measured on actual competition files. '
           'The decision concerns observable platform viewing, not enjoyment or revenue.')
        +'<h2>2. Related Work</h2>'
        +p('Content profiles follow the item-attribute/user-profile framework of Pazzani and Billsus [1]. Personalized diffusion uses the nonuniform restart principle described by Haveliwala [2], adapted here to programme co-viewing. '
           'Official versioned documentation [3,4] supports the actual TF-IDF, normalization and sparse APIs. Association rules and implicit MF were considered but not implemented; no measured inferiority is claimed.')
        +'<h2>3. Method</h2><h3>Data and interaction audit</h3>'
        +p(f'The five UTF-8, comma-separated files contain {counts(raw_rows)} non-template rows and {counts(files["template"]["rows"])} unique template users. '
           f'The view log has {counts(audit["view_distinct_users"])} users and {counts(audit["view_duplicate_rows"])} exact duplicate rows. '
           f'{counts(raw.get("template",{}).get("user_ids_above_signed_int64",0))} template IDs exceed signed int64, so IDs remain exact strings. File hashes and missing-field counts are exported; supplied row-count targets match.')
        +table(['Schedule','Rows','Nonzero-ID rows','Labelled rows','Candidates'],schedule_rows,'.14,.23,.23,.19,.21')
        +p('Schedule dates use DD.MM.YYYY, views ISO dates; all lack time-zone offsets. One zero-duration airing is excluded; no duplicate airing keys occur within the individual raw files. Schedule duration is seconds: same-channel endpoint checks match at least 99.96% in each full independent test. '
           f'View duraton has exact median {number(view_info.get("duraton_exact_quantiles_seconds",{}).get("0.5",0),0)} s and maximum {number(view_info.get("duraton_exact_quantiles_seconds",{}).get("1",0),0)} s. '
           f'{counts(view_info.get("duraton_vs_endpoint_mismatches_gt_1s",0))} records differ from endpoint subtraction by +3,600 s; a cause is not established. Endpoints define overlap, and naive times retain wall-clock order.')
        +p('Use positive half-open overlap max(0,min(view_end,airing_end)−max(view_start,airing_start)) on the same channel. Union repeated views per user/channel and overlapping same-item airings; discard conflicting nonzero labels. '
           'ID 0 is unlabelled context only. Aggregate fragments per canonical airing, then require 120 seconds. The predeclared threshold is below the 133-second view-log first quartile and 396-second median; short relevant clips can still be lost. It was not tuned against MAP.')
        +table(['Window','View h','Labelled h','ID-0 h','Conflict h','Outside h','Usable pairs'],join_rows,'.12,.14,.15,.14,.14,.14,.17')
        +p(f'Training/validation labelled fragments: {counts(train.get("labelled_fragments",0))}/{counts(valid.get("labelled_fragments",0))}; below-threshold user-airings: {counts(train.get("below_threshold_user_airings",0))}/{counts(valid.get("below_threshold_user_airings",0))}. '
           f'Users with views but no labelled time: {counts(train["users_without_labelled_time"])}/{counts(valid["users_without_labelled_time"])}; no usable interactions: {counts(train["users_without_usable_interactions"])}/{counts(valid["users_without_usable_interactions"])}. '
           'Hours are unioned channel-viewing time, not unique clock time across channels. Full losses and genuine matched/rejected examples are exported.','small')
    )
    page2 = (
        '<h2>3. Method (continued)</h2><h3>Shared temporal and ranking protocol</h3>'
        +p(f'Train [{escape(date_string("train"))}, {escape(date_string("valid"))}); validate [{escape(date_string("valid"))}, {escape(date_string("future"))}); predict [{escape(date_string("future"))}, {escape(date_string("end"))}). '
           'These explicit nominal ISO-week midnights follow inspection of real file spillovers. Historical schedule rows from both historical files are clipped by actual overlap; future candidates use only future schedule rows. '
           'Crossing views/airings are clipped to each half-open window and thresholded independently. All learned preferences, vocabulary/IDF, popularity and graph weights use the training window only.')
        +p(f'The same {counts(len(candidates["valid"]))} labelled positive-duration validation candidates are used by all methods; previously watched programmes are allowed. '
           f'The fixed cohort contains {counts(cohort["evaluation_users"])} of {counts(cohort["template_users"])} template users, including {counts(cohort["eligible_zero_history_users"])} with no usable training history. '
           f'{counts(cohort["excluded_no_eligible_targets"])} users lack observable eligible targets and are excluded before every comparison. '
           f'Overlapping reasons: no observed holdout {counts(cohort["reason_counts_overlapping"]["no_observable_holdout_interactions"])}, no eligible targets {counts(cohort["reason_counts_overlapping"]["no_eligible_holdout_targets"])}, some targets outside candidates {counts(cohort["reason_counts_overlapping"]["some_targets_outside_candidates"])}, no training history {counts(cohort["reason_counts_overlapping"]["no_usable_training_history"])}. '
           f'Holdout users outside the template: {counts(cohort["holdout_users_outside_template"])}.')
        +p('A cold user, missing historical representation, or nonpositive/short score list uses training distinct-viewer popularity within the target pool. '
           'Every method returns five distinct positive IDs, ordered by score then ascending ID for ties. Scores are not probabilities. Future viewing never enters fitting; the future schedule supplies identity/availability only.')
        +'<h3>Actual models and frozen search</h3>'
        +p('<b>Popularity:</b> distinct qualifying training viewers per item. <b>Content:</b> binary field-prefixed category, three genres, production year, director and actor tokens, excluding titles; smoothed IDF log((1+n)/(1+df))+1 and L2 item vectors. '
           'The user profile is the L2-normalized sum weighted by log(1+overlap seconds); score by cosine. Unseen historical attributes/items get no learned representation.')
        +p('<b>PPR:</b> L1-normalize each user’s log(1+seconds) weights X; form X^T X in bounded sparse row blocks, remove self-edges, retain 100 strongest neighbours per row with ID ties, then row-normalize P. '
           'From history seed v, iterate p_next=alpha(P^T p+dangling_mass·v)+(1−alpha)v. Return dangling mass to v; stop at L1 difference ≤1e-8 or 300 iterations. '
           'The two predeclared alpha trials are 0.5 and 0.85 (link-following probabilities, complementing [2]’s restart notation). No hybrid is used. Highest MAP wins; ties prefer popularity, content, PPR 0.5, then PPR 0.85. Seed: 20261030.')
        +'<h3>MAP@5 and verification</h3>'
        +p('For unique eligible target set R_u, AP_u@5=sum_(r=1..5) P_u@r·rel_u(r)/min(5,|R_u|); MAP is the macro mean over the fixed cohort. Duplicate predictions retain rank but earn no repeat hit. '
           'Empty-target AP is zero if explicitly evaluated; the primary cohort excludes such users. Slides fixtures give 11/12 and 8/15 (mean 0.725); this is a synthetic arithmetic check, not a competition score. '
           f'Local target sets have median {statistics.median(relevant_counts):g} items (range {min(relevant_counts)}–{max(relevant_counts)}); {counts(sum(n>5 for n in relevant_counts))} users have more than five. '
           'The competition overview describes five most-preferred future programmes, whereas local relevance includes all qualifying held-out programmes. These broad local targets make the task easier; evaluator equivalence and a Kaggle score cannot be inferred.')
        +'<h2>4. Experiments</h2>'
        +table(['Method','Real MAP@5','Fit s','Predict cohort s'],result_rows,'.30,.24,.23,.23')
        +p(f'Actual hardware: {escape(environment.get("cpu","not recorded"))}, {number(environment.get("physical_memory_bytes",0)/1024**3,1)} GiB RAM, {escape(environment.get("platform",""))}; Python {escape(environment.get("python","").split(" (")[0])}. '
           f'NumPy {escape(environment["packages"]["numpy"])}, pandas {escape(environment["packages"]["pandas"])}, SciPy {escape(environment["packages"]["scipy"])}, sklearn {escape(environment["packages"]["scikit-learn"])}. '
           f'Import {number(audit["runtime_seconds"],2)} s; join {number(protocol["runtime_seconds"],2)} s; whole run {number(manifest["wall_seconds"],2)} s, peak RSS {number(manifest["peak_resident_memory_bytes"]/1024**3,2)} GiB. '
           f'The independent actual per-user AP audit passed (maximum error {ap_audit["max_per_user_ap_error"]:.2g}); source hashes were stable. '+verification_sentence(verification,failure),'small')
    )

    example_pages=[]
    example_summaries=[]
    for example in demo['selection']['users']:
        user=example['user_id'];role=example['role']
        train_history=[x for x in demo['histories'] if x['user_id']==user and x['window']=='train']
        watched=[x for x in demo['histories'] if x['user_id']==user and x['window']=='valid']
        history_rows=[]
        for window,records in [('train',train_history),('valid',watched)]:
            for row in sorted(records,key=lambda r:(-r['seconds'],r['item_id']))[:3]:
                history_rows.append([window,str(row['item_id']),row.get('title',''),row.get('category',''),number(row['seconds'],0)])
        if not train_history:
            history_rows.insert(0,['train','—','No usable training history','—','0'])
        recs=[x for x in demo['recommendations'] if x['user_id']==user]
        ordered=[r for method in demo['methods'] for r in sorted([x for x in recs if x['method']==method],key=lambda x:x['rank'])]
        if len(ordered)!=15:
            raise ValueError('Each example must have 15 actual recommendation rows.')
        rec_rows=[[fmt_method(r['method']),str(r['rank']),str(r['item_id']),r.get('title',''),r.get('category',''),f'{r["score"]:.6g}','Yes' if r['relevant'] else 'No'] for r in ordered]
        hit_counts={m:sum(bool(r['relevant']) for r in recs if r['method']==m) for m in demo['methods']}
        hit_ids=sorted({r['item_id'] for r in recs if r['relevant']})
        hit_sentence=', '.join(str(i) for i in hit_ids) if hit_ids else 'none'
        train_ids={r['item_id'] for r in train_history}
        repeat_counts={m:sum(r['item_id'] in train_ids for r in recs if r['method']==m) for m in demo['methods']}
        example_summaries.append(f'{role.capitalize()} user: hits '+', '.join(f'{fmt_method(m)} {hit_counts[m]}/5' for m in demo['methods'])+'.')
        content_list=[r for r in ordered if r['method']=='content']
        ppr_list=[r for r in ordered if r['method']==best_ppr]
        if [r['item_id'] for r in content_list] == [r['item_id'] for r in ppr_list]:
            concrete='Content and PPR return the same ordered IDs for this user. '
            if not train_history:
                concrete+='This is the common cold-user fallback, not evidence of learned personalization. '
        else:
            ppr_ids={r['item_id'] for r in ppr_list}
            content_ids={r['item_id'] for r in content_list}
            c=next((r for r in content_list if r['item_id'] not in ppr_ids),content_list[0])
            g=next((r for r in ppr_list if r['item_id'] not in content_ids),ppr_list[0])
            concrete=(f'A concrete difference is content ID {c["item_id"]} “{c.get("title", "")}” ({c.get("category", "")}; '
                      f'{"held-out hit" if c["relevant"] else "no observed held-out hit"}), compared with PPR ID {g["item_id"]} “{g.get("title", "")}” '
                      f'({g.get("category", "")}; {"held-out hit" if g["relevant"] else "no observed held-out hit"}). ')
        category_seconds={}
        for row in train_history:
            category=row.get('category') or 'missing category'
            category_seconds[category]=category_seconds.get(category,0)+row['seconds']
        if category_seconds:
            category=max(sorted(category_seconds),key=lambda cat:category_seconds[cat])
            concrete+=f'The largest observed training category by matched seconds is {category} ({h(category_seconds[category])} h). '
        label='4. Experiments: fixed heavy viewer' if role=='heavy' else '4. Experiments: fixed light viewer'
        page=(f'<h2>{label}</h2>'
          +p(f'<b>User ID {escape(user)}</b>. Chosen before scoring by {"maximum" if role=="heavy" else "minimum"} distinct usable training items among the fixed cohort (ties by exact ID; light differs from heavy). '
             f'Full histories contain {counts(len(train_history))} training and {counts(len(watched))} held-out programme IDs. '
             'Table below shows at most three programmes per window, selected by descending overlap seconds, then ID; these are bounded excerpts, not complete histories. Titles/categories use the earliest nonempty metadata in the respective window, so one ID may have different displayed titles across windows.')
          +table(['Window','ID','Observed programme title','Category','Seconds'],history_rows,'.09,.11,.48,.19,.13','examples')
          +p('All ordered Top-5 lists follow. Hit means membership in this user’s observed eligible validation set; scores are rounded to six significant digits and have method-specific units. '
             'Fallback reasons and full-precision scores are in validation-recommendations.csv.','small')
          +table(['Method','Rank','ID','Programme title','Category','Score','Hit'],rec_rows,'.13,.06,.10,.36,.15,.12,.08','examples')
          +p('Actual held-out hits appearing in the displayed lists: '+hit_sentence+'. '
             +'Observed-training repeats among five: '+', '.join(f'{fmt_method(m)} {repeat_counts[m]}' for m in demo['methods'])+'. '
             +'Hits: '+', '.join(f'{fmt_method(m)} {hit_counts[m]}/5' for m in demo['methods'])+'. '
             +escape(concrete)+'An item outside this training history is not asserted to be never watched. Full histories and all 30 recommendations are preserved in two-user-comparison.json.','small'))
        example_pages.append(page)

    reason = (f'{fmt_method(selected)} has the highest common-holdout MAP@5 ({selected_map:.6f}), '
              f'an absolute change of {map_change:+.6f} against popularity ({pop_map:.6f}).')
    if selected == 'popularity':
        rationale='The current evidence favors the simpler distinct-viewer reference. Personalized attributes and graph diffusion were implemented but did not improve this holdout; selecting them merely for personalization would contradict the measured comparison.'
    elif selected == 'content':
        rationale='Historical metadata similarity provides the strongest measured ranking here. The choice still needs comparison with its measured fit/prediction cost, sparse metadata coverage, and the fixed users; attribute similarity cannot establish enjoyment.'
    else:
        rationale='Historical co-view diffusion provides the strongest measured ranking here. The cost and graph diagnostics must be considered with the gain; sparse history and missing item identities still limit the inference.'
    diagnostic = read_json(run_dir/'diagnostics'/f'{selected}.json')
    ppr_diag=read_json(run_dir/'diagnostics'/f'{best_ppr}.json')
    content_diag=read_json(run_dir/'diagnostics/content.json')
    actual_join_examples=read_json(run_dir/'prepared/join-examples.json')
    accepted=next((x for x in actual_join_examples if x.get('watched') and x.get('item_id')),None)
    example_check=''
    if accepted:
        stamp=lambda value:dt.datetime.fromtimestamp(value,dt.timezone.utc).strftime('%m-%d %H:%M:%S')
        example_check=(f'A genuine channel-{accepted["channel_id"]} view [{stamp(accepted["view_start"])}, {stamp(accepted["view_end"])}) '
                       f'overlaps programme {accepted["item_id"]} [{stamp(accepted["airing_start"])}, {stamp(accepted["airing_end"])}) by '
                       f'{accepted["overlap_seconds"]:.0f} s; aggregate {accepted["aggregated_airing_overlap_seconds"]:.0f} s passes 120 s. ')
        rejected=next((x for x in actual_join_examples if not x.get('watched') and x.get('reason')=='unlabelled'),None)
        if rejected:
            example_check+=f'An ID-0 overlap of {rejected["overlap_seconds"]:.0f} s is discarded as unlabelled. Dates are 2020 source wall-clock coordinates, not an asserted UTC timezone. '
    cli_version=next((s.get('cli_version') for s in inventory.get('sessions',[]) if s.get('session_id')==inventory.get('root_session_id')),'not recorded')
    root_session_id=inventory.get('root_session_id')
    ordered_sessions=sorted(inventory.get('sessions',[]),key=lambda entry:entry.get('session_id')!=root_session_id)
    native_names=[Path(entry['path']).name for entry in ordered_sessions if entry.get('path')]
    root_original=next((Path(entry['path']).name for entry in ordered_sessions if entry.get('session_id')==root_session_id and entry.get('path')),'root source filename not recorded')
    handin_name=external.get('native_packaging',{}).get('course_handin_native_filename','timur_khayrislamov_exam_session.jsonl')
    native_note=(f'The chosen two-file course hand-in is the completed PDF plus {handin_name}, a full byte-for-byte copy of root source {root_original}; only the filename changes. '
                 'Three genuine child sessions are retained separately in private evidence and contain delegated tool calls/file operations absent from the parent export. '
                 'They are disclosed here and are not merged into the attached parent or added as extra course uploads. This follows the source-authorized native-format/two-file default; no instructor approval is claimed. ')
    if snapshot:
        saved=sorted(snapshot.get('files',[]),key=lambda entry:entry.get('session_id')!=root_session_id)
        native_names=[Path(entry['copy']).name for entry in saved if entry.get('copy')]
        root_saved=next((entry for entry in saved if entry.get('session_id')==root_session_id),None)
        if not root_saved:
            raise ValueError('The supplied native checkpoint does not include the identified root session.')
        root_copy=Path(root_saved['copy'])
        root_copy=root_copy if root_copy.is_absolute() else ROOT/root_copy
        native_bytes=root_copy.read_bytes()
        if (hashlib.sha256(native_bytes).hexdigest()!=root_saved.get('sha256')
                or len(native_bytes)!=root_saved.get('bytes')
                or root_saved.get('byte_for_byte') is not True
                or root_saved.get('unmodified_native_jsonl') is not True):
            raise ValueError('The selected root native checkpoint fails its byte-for-byte provenance checks.')
        native_note+=(f'The supplied private checkpoint was copied during {snapshot.get("cutoff_started_utc","unrecorded")} to {snapshot.get("copy_completed_utc","unrecorded")}. '
                     f'Root checkpoint: {root_saved["bytes"]:,} bytes, SHA-256 {root_saved["sha256"]}. '
                     'The manifest preserves original filenames and cutoff provenance. '
                     +snapshot.get('work_after_cutoff','Later report work is outside that checkpoint.')+' Report-builder/document revisions after that checkpoint are also outside its record. ')
    else:
        native_note+='A fresh final export cutoff and SHA-256 provenance will be recorded after the remaining work; this draft does not claim an earlier checkpoint covers later work. '
    if github_verified:
        repository_url=github['url'].rstrip('/')
        commit_url=repository_url+'/commit/'+github['commit']
        external_text=(f'Published repository: <a href="{escape(repository_url)}">{escape(repository_url)}</a>; '
                       f'commit <a href="{escape(commit_url)}">{escape(github["commit"])}</a>. '
                       'The recorded public-access check and remote CSV hash verification link this commit to the exact local CSV above. '
                       'The designated returned Kaggle private score and genuine screenshot are still missing; no substitute is shown.')
    else:
        external_text='No verified GitHub URL/remote commit or completed Kaggle submission/private-score screenshot is supplied; no substitute is shown.'
    page5=(
        '<h2>5. Discussion</h2>'
        +p('<b>Which strategy should SWEET.TV select?</b> '+escape(reason)+' '+rationale+' '
           f'Its measured validation fit/prediction times are {methods[selected]["fit_seconds"]:.3f}/{methods[selected]["predict_seconds"]:.3f} s. '
           f'The gain is {100*map_change/pop_map:.2f}% relative; popularity is already strong at fit/prediction {methods["popularity"]["fit_seconds"]:.3f}/{methods["popularity"]["predict_seconds"]:.3f} s and remains the cold-user fallback. '
           +' '.join(example_summaries)+' These deterministic extremes illustrate behavior; they are not a representative user sample.')
        +p('Content weights historical metadata similarity, while PPR propagates co-view connections and popularity gives the same broadly watched list to everyone. '
           f'Content learned {counts(content_diag.get("vocabulary_size",0))} feature tokens and {counts(content_diag.get("nonempty_profiles",0))} nonempty profiles. '
           f'The selected PPR graph has {counts(ppr_diag.get("graph_nodes",0))} nodes and {counts(ppr_diag.get("graph_edges",0))} retained edges; '
           f'{counts(ppr_diag.get("converged_walks",0))}/{counts(ppr_diag.get("walk_calls",0))} recorded walks converged, maximum mass error {ppr_diag.get("max_observed_mass_error",0):.2g}. '
           'For the heavy example, broad agreement with the dominant film category still yields only one content hit, while popularity and PPR recover five observed programmes; category similarity alone does not establish item relevance. '
           'Unlabelled airtime, broad local targets, missing attributes, excluded empty-target users and a single blocked split limit transfer. No live A/B, enjoyment or revenue effect is claimed. A next improvement is an additional earlier blocked split before changing the viewing threshold.')
        +'<h3>Observed failure and verified correction</h3>'
        +p('The first real attempt lacked inputs and remains archived. After the files arrived, schedule parsing required the observed DD.MM.YYYY format; the next run deliberately stopped with “Schedule files overlap in start dates.” '
           'The independent audit found tail spillovers after nominal week boundaries, so raw file membership was not a valid time split. '
           'The fix uses the explicitly documented week-start boundaries and clips candidates, metadata and interaction windows; the completed real run and boundary tests confirm execution. '
           'No DST root cause is asserted for the observed duration discrepancies. Earlier synthetic nan/inf validation and nonpositive-candidate defects, their guards and rerun logs remain separately recorded. '
           +escape(example_check))
        +'<h2>6. AI Usage Disclosure</h2>'
        +p(f'Codex in VS Code (native CLI metadata {escape(cli_version)}) and delegated agents read task sources, checked primary references, wrote/reviewed code, ran raw and exported-output audits, debugged failures and generated this report. '
           'ReportLab/PyMuPDF build and inspect the PDF. Pazzani–Billsus was read only through the publisher preview; PageRank key sections/appendix and official API sections were read by AI. '
           '<b>No personal hand calculation, source reading, rerun or interpretation is attributed to Timur.</b> Student verification and a rewrite in the author’s own voice remain pending.')
        +p(f'The selected strategy was refitted on the full historical window and produced {counts(final["rows"])} template-preserving rows of five distinct valid IDs over {counts(final["candidate_count"])} future candidates. '
           f'Final fit/prediction: {final["fit_seconds"]:.3f}/{final["predict_seconds"]:.3f} s. Exact CSV SHA-256: <b>{actual_csv_hash}</b>. '
           'This is a validated local prediction file, not proof of a returned Kaggle score. '+external_text,'small')
        +p('Native session filenames (root first; three private children follow): '+'; '.join(escape(x) for x in native_names)+'. '+escape(native_note),'small')
        +p(escape(pending_phrase[:1].upper()+pending_phrase[1:])+' remain outstanding. '
           'The course hand-in must contain exactly the completed PDF and unmodified root native session export; this draft stays outside submission/. FKN deadline: 1 November 2026, 19:00 Europe/Moscow.','small')
    )
    reference_page='<h2>References</h2>'+p('Verified primary sources; access date 2026-10-30. Scope, stable URLs and local retrieval hashes: evidence/references.json.','small')
    for ref in refs:
        text=ref.get('ieee',ref['title'])
        prefix,marker,suffix=text.partition('[Online]. Available:')
        content=escape(prefix)+f'<a href="{escape(ref["url"])}">[Online source]</a>. [Accessed: {escape(ref["accessed_client_date"])}].' if marker else escape(text)
        reference_page+=p(f'[{ref["id"]}] '+content,'reference')
    reference_page+=p('Official scikit-learn 1.6.1 DictVectorizer and normalize API pages were also consulted; they are linked in the reference audit. '
                      'The exact run directory is '+escape(source_run)+'. Primary exports are data-audit.json, prepared/join-audit.json and protocol.json, cohort.json, validation-summary.json, per-user-ap.csv, validation-recommendations.csv, two-user-comparison.json, independent-ap-audit.json and final/submission-validation.json. '
                      'Reported values are rounded for display; full precision and complete histories remain in those outputs. Cached reference PDFs and native logs are private evidence.','small')
    pages=[page1,page2,*example_pages,page5,reference_page]
    css='''@page { size:A4; margin:17mm; } body { font:9.7pt/1.22 "Times New Roman",Georgia,serif; color:#000; } .page { page-break-after:always; } .page:last-child { page-break-after:auto; } h1 {font-size:18pt;text-align:center;} h2 {font-size:11.7pt;} h3 {font-size:10.5pt;} p {margin:0 0 6pt;} table {border-collapse:collapse;width:100%;font-size:8.1pt;} td,th {border:0.35pt solid #777;padding:4pt;vertical-align:top;} th {background:#eee;} .examples {font-size:7.6pt;} .notice,.author {text-align:center;} .small {font-size:8.7pt;} a {color:#000;} .draft-banner {font-weight:bold;border-bottom:1px solid #000;margin-bottom:10pt;}'''
    body=''.join('<div class="page'+(' references-page' if i==len(pages)-1 else '')+'"><div class="draft-banner">REAL-RESULTS DRAFT — external evidence and personal verification pending</div>'+text+'</div>' for i,text in enumerate(pages))
    return '<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><head><meta charset="utf-8" /><title>SWEET.TV real-results draft</title><style>'+css+'</style></head><body>'+body+'</body></html>\n'


def register_fonts():
    directory = Path('/System/Library/Fonts/Supplemental')
    fonts = {'ExamSerif': 'Times New Roman.ttf', 'ExamSerif-Bold': 'Times New Roman Bold.ttf',
             'ExamSerif-Italic': 'Times New Roman Italic.ttf', 'ExamSerif-BoldItalic': 'Times New Roman Bold Italic.ttf'}
    if all((directory / file).exists() for file in fonts.values()):
        for name, file in fonts.items():
            pdfmetrics.registerFont(TTFont(name, str(directory / file)))
        pdfmetrics.registerFontFamily('ExamSerif', normal='ExamSerif', bold='ExamSerif-Bold', italic='ExamSerif-Italic', boldItalic='ExamSerif-BoldItalic')
        return 'ExamSerif', 'ExamSerif-Bold'
    return 'Times-Roman', 'Times-Bold'


def local_tag(element):
    return element.tag.rsplit('}', 1)[-1]


def paragraph_markup(element):
    """Render the same editable XHTML inline text through ReportLab."""
    out = escape(element.text or '')
    for child in element:
        tag = local_tag(child)
        if tag == 'br':
            out += '<br/>'
        elif tag in {'b', 'strong', 'i', 'em', 'sub', 'sup'}:
            target = {'strong':'b', 'em':'i'}.get(tag, tag)
            out += '<' + target + '>' + paragraph_markup(child) + '</' + target + '>'
        elif tag == 'a':
            out += '<a href="' + escape(child.attrib.get('href', '')) + '" color="black"><u>' + paragraph_markup(child) + '</u></a>'
        else:
            out += paragraph_markup(child)
        out += escape(child.tail or '')
    return out


def build_pdf(source, destination):
    normal, bold = register_fonts()
    styles = {
        'body': ParagraphStyle('Body', fontName=normal, fontSize=9.7, leading=11.7, spaceAfter=6.3, splitLongWords=True),
        'small': ParagraphStyle('Small', fontName=normal, fontSize=8.7, leading=10.5, spaceAfter=6),
        'title': ParagraphStyle('Title', fontName=bold, fontSize=18, leading=20.5, alignment=TA_CENTER, spaceAfter=9),
        'h2': ParagraphStyle('Heading2', fontName=bold, fontSize=11.7, leading=14, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'h3': ParagraphStyle('Heading3', fontName=bold, fontSize=10.5, leading=12.7, spaceBefore=6, spaceAfter=4, keepWithNext=True),
        'author': ParagraphStyle('Author', fontName=normal, fontSize=9, leading=11.2, alignment=TA_CENTER, spaceAfter=8),
        'notice': ParagraphStyle('Notice', fontName=bold, fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=8),
        'cell': ParagraphStyle('Cell', fontName=normal, fontSize=8.1, leading=9.7, spaceAfter=0, splitLongWords=True),
        'reference': ParagraphStyle('Reference', fontName=normal, fontSize=9.3, leading=11.8, spaceAfter=12, splitLongWords=True),
        'examples': ParagraphStyle('Examples', fontName=normal, fontSize=7.6, leading=9, spaceAfter=0, splitLongWords=True),
        'native': ParagraphStyle('Native', fontName=normal, fontSize=8.3, leading=10.3, leftIndent=8, firstLineIndent=0, spaceAfter=5, splitLongWords=True),
    }
    root = ET.parse(source).getroot()
    body = next(x for x in root if local_tag(x) == 'body')
    width, height = A4
    margin = 48
    usable_width = width - 2 * margin
    story = []
    for page_index, page in enumerate(x for x in body if local_tag(x) == 'div' and 'page' in x.attrib.get('class', '').split()):
        if page_index:
            story.append(PageBreak())
        for element in page:
            tag = local_tag(element)
            cls = element.attrib.get('class', '')
            if cls == 'draft-banner':
                continue  # Identical banner appears through the repeating footer callback.
            if tag in {'h1', 'h2', 'h3'}:
                if tag == 'h2' and ''.join(element.itertext()).strip() == 'References':
                    story.append(Paragraph('<a name="references"/>', styles['small']))
                story.append(Paragraph(paragraph_markup(element), styles[{'h1':'title','h2':'h2','h3':'h3'}[tag]]))
            elif tag == 'p':
                style = styles.get(cls, styles['body'])
                story.append(Paragraph(paragraph_markup(element), style))
            elif tag == 'ul':
                for item in element:
                    if local_tag(item) == 'li':
                        story.append(Paragraph(paragraph_markup(item), styles['native'] if cls == 'native' else styles['body'], bulletText='•'))
            elif tag == 'table':
                data = []
                for row in element:
                    if local_tag(row) != 'tr':
                        continue
                    cells = []
                    for cell in row:
                        text = paragraph_markup(cell)
                        if local_tag(cell) == 'th':
                            text = '<b>' + text + '</b>'
                        cells.append(Paragraph(text, styles['examples'] if cls == 'examples' else styles['cell']))
                    data.append(cells)
                ratios = [float(x) for x in element.attrib.get('data-widths', '').split(',') if x]
                widths = [usable_width * r / sum(ratios) for r in ratios] if ratios else None
                item = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
                item.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeeeee')),
                    ('GRID', (0,0), (-1,-1), .35, colors.HexColor('#777777')),
                    ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
                ]))
                story.extend([Spacer(1,3), item, Spacer(1,7)])
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(bold, 8)
        canvas.drawString(margin, height - 29, 'INCOMPLETE REAL-RESULTS DRAFT — external evidence and personal verification pending')
        canvas.setLineWidth(.35)
        canvas.line(margin, height - 34, width-margin, height-34)
        canvas.setFont(normal, 8)
        canvas.drawString(margin, 25, 'Timur Khayrislamov · FKN LLM4Rec midterm · 30 October 2026')
        canvas.drawRightString(width-margin, 25, f'{doc.page}')
        canvas.restoreState()
    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=margin, leftMargin=margin,
        topMargin=47, bottomMargin=42, title='Temporal Television Recommendation on SWEET.TV — INCOMPLETE DRAFT', author='Timur Khayrislamov',
        subject='Incomplete exam draft; no real-data result or external completion claimed')
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    import fitz
    with fitz.open(destination) as pdf:
        reference_pages = [i for i, page in enumerate(pdf) if re.search(r'^References\s*$', page.get_text(), re.MULTILINE)]
        body_count = reference_pages[0] if reference_pages else len(pdf)
        if not reference_pages:
            raise ValueError('References heading is missing from generated PDF')
        if body_count > 5:
            raise ValueError(f'Draft body has {body_count} pages; maximum target is five')
        if any('INCOMPLETE' not in page.get_text() or 'DRAFT' not in page.get_text() for page in pdf):
            raise ValueError('Every PDF page must have the incomplete-draft label')
        return {'total_pages':len(pdf), 'body_pages':body_count, 'references_pages':len(pdf)-body_count,
                'pdf_links':sum(len(page.get_links()) for page in pdf)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-source', action='store_true', help='Regenerate editable source from current evidence; overwrites prior source prose edits')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'experiments/competition-20261030-03', help='Actual local run evidence directory; absent results remain unavailable')
    parser.add_argument('--session-manifest', type=Path, default=None, help='Authentic snapshot manifest to disclose; use --refresh-source to incorporate it into existing prose')
    parser.add_argument('--external-evidence', type=Path, default=ROOT/'evidence/external-evidence.json', help='Actual external evidence manifest, if supplied; report remains a draft until all required checks are completed')
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    source = REPORT/'report-source.html'
    if args.refresh_source or not source.exists():
        source.write_text(create_source(args.run_dir, args.session_manifest, args.external_evidence), encoding='utf-8')
    destination = REPORT/DRAFT_NAME
    checks = build_pdf(source, destination)
    result = {'status':'incomplete_draft', 'source':str(source), 'pdf':str(destination), **checks,
              'pdf_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'warning':'No final report or course submission is created. Required external and personal evidence remains pending.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
