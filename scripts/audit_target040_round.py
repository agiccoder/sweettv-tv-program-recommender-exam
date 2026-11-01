#!/usr/bin/env python3
"""Independent target040 round export, opportunity, reference-parity and preservation checks.

This script does not import production metrics, models or selection functions.
It consumes the saved recommendation/native/contribution files and source inputs.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def exact_ap(predictions, relevant):
    first = {}
    for rank, item in enumerate(predictions[:5], 1):
        if item in relevant:
            first.setdefault(item, rank)
    hits = sorted(first.values())
    return sum((Fraction(i, rank) for i, rank in enumerate(hits, 1)), Fraction(0)) / min(5, len(relevant)) if relevant else Fraction(0)


def rank_map(records):
    return {item: (i, score) for i, (item, score) in enumerate(records, 1)}


def history_rank(native, user, constant):
    scores = defaultdict(float)
    for component in ["history_seconds", "history_watches", "history_recent"]:
        for rank, (item, _) in enumerate(native[user][component], 1):
            scores[item] += (1 / 3) / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def component_spec(spec, native, history, user):
    if spec["kind"] == "rank_fusion":
        return {name: (rank_map(native[user][name]), weight) for name, weight in zip(spec["components"], spec["weights"])}
    if spec["kind"] == "history_fusion":
        return {name: (rank_map(native[user][name]), 1 / 3) for name in ["history_seconds", "history_watches", "history_recent"]}
    if spec["kind"] not in {"hybrid", "hybrid_als"}:
        return None
    a, b, c = spec["weights"]
    channel = spec.get("channel_component", "channel_time")
    result = {"ppr_v2": (rank_map(native[user]["ppr_v2"]), a),
              "personal_history_fusion": (rank_map(history[user]), b),
              channel: (rank_map(native[user][channel]), c)}
    if spec["kind"] == "hybrid_als":
        result["ppr_v2"] = (result["ppr_v2"][0], a / 2)
        result["als32"] = (rank_map(native[user]["als32"]), a / 2)
    return result


def audit_rate_models(destination, prepared, block, config, train, native, cohort):
    """Independent gains/full-base-score retrieval, without production scoring."""
    specifications = [s for s in config["methods"] if s["kind"] == "exposure_adjusted"]
    if not specifications:
        return {"gains": 0, "native_scores": 0}
    with (destination / "fitted-components.pkl").open("rb") as stream:
        family = pickle.load(stream)
    directory = ROOT / config["rate_features_root"]
    historical = load(directory / (block["name"] + "-train.json"))
    future = load(directory / (block["name"] + "-target.json"))
    gains_checked = scores_checked = 0
    samples = list(dict.fromkeys(cohort[:3] + config["examples"]))
    candidates = load(prepared / block["name"] / "candidates.json")["valid"]
    for specification in specifications:
        model = family.models[specification["name"]]
        base = family.models[specification["base_component"]]
        field = specification["exposure_measure"]
        prior = config["exposure_rate"]["smoothing"][field]
        expected_gain = {}
        for item in candidates:
            old = historical[field].get(str(item), 0.)
            new = future[field].get(str(item), 0.)
            ratio = ((new+prior)/future["days"])/((old+prior)/historical["days"])
            low, high = config["exposure_rate"]["gain_clip"]
            expected_gain[item] = min(high,max(low,ratio)) ** specification["rate_power"]
            assert abs(expected_gain[item]-model.gain[item]) < 1e-12
            gains_checked += 1
        for user in samples:
            if specification["base_component"] == "history_watches":
                rows_for_user = train[train.user_id == user]
                raw = {int(row.item_id): float(row.watches) for row in rows_for_user.itertuples() if int(row.item_id) in expected_gain}
            elif hasattr(base, "transition"):
                index = family.data.user_index.get(user)
                raw = {}
                if index is not None:
                    restart = base.restart_rows.getrow(index).toarray().ravel()
                    values = restart.copy()
                    for iteration in range(config["graph"]["iteration_cap"]):
                        new = base.alpha * (base.transition.T @ values + values[base.dangling].sum()*restart) + (1-base.alpha)*restart
                        delta = np.abs(new-values).sum()
                        values = new
                        if delta <= config["graph"]["tolerance_l1"]:
                            break
                    assert delta <= config["graph"]["tolerance_l1"]
                    raw = {item:float(values[family.data.item_index[item]]) for item in candidates if item in family.data.item_index}
            else:
                raise AssertionError("Unimplemented independent exposure base")
            reference = {item:score*expected_gain[item] for item,score in raw.items()}
            exported = native[user][specification["name"]]
            for item,score in exported:
                assert abs(score-reference[item]) < 1e-10
                scores_checked += 1
            if len(exported) == 50:
                selected = {item for item,_ in exported}
                assert all(score <= exported[-1][1]+1e-10 for item,score in reference.items() if item not in selected)
            elif not raw:
                assert not exported
    return {"gains": gains_checked, "native_scores": scores_checked}


def audit_intensity_mixtures(destination, config, train, native, cohort, candidates):
    """Rebuild full candidate scores from factors/counts/stationary equations."""
    specifications = [s for s in config["methods"] if s["kind"] == "intensity_mixture"]
    if not specifications:
        return {"sampled_users":0,"native_scores":0}
    with (destination / "fitted-components.pkl").open("rb") as stream:
        family = pickle.load(stream)
    pool = set(candidates)

    def scores(model, user):
        index = family.data.user_index.get(user)
        if index is None:
            return {}
        if hasattr(model,"gain"):
            return {i:s*model.gain[i] for i,s in scores(model.base,user).items()}
        if hasattr(model,"transition"):
            restart = model.restart_rows.getrow(index).toarray().ravel()
            values = restart.copy()
            for _ in range(config["graph"]["iteration_cap"]):
                new = model.alpha*(model.transition.T@values+values[model.dangling].sum()*restart)+(1-model.alpha)*restart
                delta = np.abs(new-values).sum();values = new
                if delta <= config["graph"]["tolerance_l1"]:break
            assert delta <= config["graph"]["tolerance_l1"]
            result = {i:float(values[family.data.item_index[i]]) for i in pool if i in family.data.item_index}
        elif hasattr(model,"user_factors"):
            values = np.sum(model.item_factors*model.user_factors[index][None,:],axis=1)
            result = {i:float(values[family.data.item_index[i]]) for i in pool if i in family.data.item_index}
        elif getattr(model,"signal",None) == "watches":
            result = {int(r.item_id):float(r.watches) for r in train[train.user_id==user].itertuples() if int(r.item_id) in pool}
        else:
            raise AssertionError("Independent intensity component not implemented")
        return {i:s for i,s in result.items() if math.isfinite(s) and s>0}

    checked = users = 0
    samples = list(dict.fromkeys(cohort[:3]+config["examples"]))
    for spec in specifications:
        name = spec["name"]
        if name not in family.models:
            assert any(component not in family.models for component in spec["components"])
            continue
        model = family.models[name]
        assert model.pool == pool and model.weights == spec["weights"]
        assert all(component is family.models[n] for component,n in zip(model.components,spec["components"]))
        for user in samples:
            expected = defaultdict(float)
            for component,weight in zip(spec["components"],spec["weights"]):
                raw = scores(family.models[component],user)
                total = math.fsum(raw.values())
                if total:
                    for item,value in raw.items():expected[item] += weight*value/total
            ordered = sorted(((i,s) for i,s in expected.items() if s>0),key=lambda pair:(-pair[1],pair[0]))[:50]
            actual = native[user][name]
            assert [i for i,_ in ordered] == [i for i,_ in actual]
            for (_,wanted),(_,value) in zip(ordered,actual):
                assert abs(wanted-value)<1e-10
                checked += 1
            users += 1
    return {"sampled_users":users,"native_scores":checked,"full_pool_normalization_and_retrieval_verified":True}


def audit_inductive_mf(destination,prepared,block,config,native,cohort,candidates):
    specifications = [s for s in config["methods"] if s["kind"] == "inductive_mf"]
    if not specifications:return {}
    with (destination/"fitted-components.pkl").open("rb") as stream:family = pickle.load(stream)
    root = ROOT/config["inductive_mf"]["root"]
    history = sparse.load_npz(root/(block["name"]+"-history.npz"))
    record = load(root/(block["name"]+"-history.json"))
    available = load(prepared/block["name"]/"availability.json")
    days = (pd.Timestamp(block["target_end"])-pd.Timestamp(block["train_end"])).total_seconds()/86400.
    future = sparse.diags(np.asarray(available["airtime_seconds"])/(3600.*days))@sparse.load_npz(prepared/block["name"]/"availability.npz")
    output = {}
    for specification in specifications:
        name = specification["name"]
        if name not in family.models:
            assert specification["base_component"] not in family.models
            output[name] = {"available":False};continue
        model = family.models[name];base = family.models[specification["base_component"]]
        assert record["items"] == list(family.data.items)
        assert (model.history != history).nnz == 0 and (model.future != future).nnz == 0
        assert list(model.items) == available["items"]
        coefficient = model.ridge.coef_.T
        residual = history@coefficient-base.item_factors
        gradient = history.T@residual+specification["ridge_alpha"]*coefficient
        relative = float(np.linalg.norm(gradient))/max(float(np.linalg.norm(history.T@base.item_factors)),1.)
        assert relative<1e-3 and abs(relative-model.diagnostics["relative_normal_equation_residual"])<1e-12
        expected = np.maximum(future@coefficient,0)*specification["cold_scale"]
        known = cold = 0
        for index,item in enumerate(available["items"]):
            if item in family.data.item_index:
                wanted = base.item_factors[family.data.item_index[item]]
                np.testing.assert_array_equal(model.candidate_factors[index],wanted)
                expected[index] = wanted;known += 1
            else:cold += 1
        np.testing.assert_allclose(model.candidate_factors,expected,atol=1e-12,rtol=1e-12)
        checked = 0
        for user in list(dict.fromkeys(cohort[:3]+config["examples"])):
            index = family.data.user_index.get(user)
            if index is None:
                assert not native[user][name];continue
            values = np.sum(expected*base.user_factors[index][None,:],axis=1)
            ordered = sorted(((item,float(value)) for item,value in zip(available["items"],values) if value>0),key=lambda pair:(-pair[1],pair[0]))[:50]
            actual = native[user][name]
            assert [i for i,_ in actual] == [i for i,_ in ordered]
            for (_,a),(_,b) in zip(actual,ordered):assert abs(a-b)<1e-10
            checked += len(actual)
        output[name] = {"available":True,"known_factors_verified_unchanged":known,"cold_factors_verified":cold,
                       "normal_equation_relative_residual":relative,"native_scores_checked":checked,
                       "future_descriptor_uses_only_fixed_availability":True}
    return output


def audit_cold_completion(destination,config,native,cohort,candidates):
    specifications=[s for s in config["methods"] if s["kind"]=="cold_completion"]
    if not specifications:return {}
    with (destination/"fitted-components.pkl").open("rb") as stream:family=pickle.load(stream)
    def base_scores(model,user):
        if hasattr(model,"gain"):
            return {i:s*model.gain[i] for i,s in base_scores(model.base,user).items()}
        index=family.data.user_index.get(user)
        if index is None:return {}
        restart=model.restart_rows.getrow(index).toarray().ravel();values=restart.copy()
        for _ in range(config["graph"]["iteration_cap"]):
            new=model.alpha*(model.transition.T@values+values[model.dangling].sum()*restart)+(1-model.alpha)*restart
            delta=np.abs(new-values).sum();values=new
            if delta<=config["graph"]["tolerance_l1"]:break
        assert delta<=config["graph"]["tolerance_l1"]
        return {i:float(values[family.data.item_index[i]]) for i in candidates if i in family.data.item_index and values[family.data.item_index[i]]>0}
    output={}
    for spec in specifications:
        model=family.models[spec["name"]];factor=family.models[spec["factor_component"]]
        known_checked=native_checked=0
        for user in list(dict.fromkeys(cohort[:3]+config["examples"])):
            base=base_scores(family.models[spec["base_component"]],user);index=family.data.user_index.get(user)
            values=np.sum(factor.candidate_factors*factor.base.user_factors[index][None,:],axis=1) if index is not None else np.zeros(len(factor.items))
            mf={i:float(v) for i,v in zip(factor.items,values) if v>0 and math.isfinite(v)}
            denominator=math.fsum(s for i,s in mf.items() if i in family.data.item_index)
            scale=math.fsum(base.values())/denominator if denominator else 0.
            expected=dict(base)
            if scale:expected.update({i:s*scale for i,s in mf.items() if i not in family.data.item_index})
            known_checked+=len(base)
            ordered=sorted(expected.items(),key=lambda p:(-p[1],p[0]))[:50];actual=native[user][spec["name"]]
            assert [i for i,_ in ordered]==[i for i,_ in actual]
            for (_,a),(_,b) in zip(ordered,actual):assert abs(a-b)<1e-10
            native_checked+=len(actual)
        output[spec["name"]]={"known_scores_preserved_checked":known_checked,"native_scores_checked":native_checked,"full_pool_mass_alignment_and_retrieval_verified":True}
    return output


def audit_factor_loss(intensity,users,items,estimator):
    """Independent sparse full-matrix objective, including unobserved cells."""
    coo=intensity.tocoo();log_terms=retained_mass=cross=0.;stabilized=0
    epsilon=float(np.finfo(np.float32).eps)
    for begin in range(0,len(coo.data),8192):
        part=slice(begin,begin+8192)
        predictions=np.sum(users[coo.row[part]]*items[coo.col[part]],axis=1)
        assert np.all(predictions>=0)
        values=coo.data[part];cross+=float(np.dot(values,predictions))
        keep=values>epsilon;v=values[keep];pred=predictions[keep]
        stabilized+=int(np.count_nonzero(pred<epsilon))
        log_terms+=float(np.sum(v*np.log(v/np.maximum(pred,epsilon))))
        retained_mass+=float(v.sum())
    if estimator.beta_loss=='frobenius':
        gram_users=np.einsum("ij,ik->jk",users,users,optimize=False)
        gram_items=np.einsum("ij,ik->jk",items,items,optimize=False)
        squared=float(np.sum(coo.data*coo.data))-2*cross+float(np.sum(gram_users*gram_items))
        assert abs(squared-float(estimator.reconstruction_err_)**2)<1e-7*max(1.,abs(squared))
        return {"independent_squared_Frobenius":squared,"independent_sklearn_Frobenius_matches":True}
    assert estimator.beta_loss=='kullback-leibler'
    kl=log_terms-retained_mass+float(np.dot(users.sum(axis=0),items.sum(axis=0)))
    assert abs(kl-float(estimator.reconstruction_err_)**2/2)<1e-7*max(1.,abs(kl))
    return {"independent_epsilon_stabilized_generalized_KL":kl,"installed_solver_numerical_floor":epsilon,
            "stabilized_positive_predictions":stabilized,"independent_sklearn_sqrt_2KL_matches":True}


def audit_joint_mf(destination,config,train,native,cohort,candidates):
    specs=[s for s in config['methods'] if s['kind']=='joint_nmf']
    if not specs:return {}
    with (destination/'fitted-components.pkl').open('rb') as stream:family=pickle.load(stream)
    rows_=[family.data.user_index[u] for u in train.user_id]
    cols=[family.data.item_index[int(i)] for i in train.item_id];blocks=[]
    for signal in ['watches','seconds']:
        matrix=sparse.csr_matrix((train[signal].to_numpy(dtype=float),(rows_,cols)),shape=family.data.weights.shape)
        totals=matrix.sum(axis=1).A.ravel()
        blocks.append(sparse.diags(np.divide(.5,totals,out=np.zeros_like(totals),where=totals>0))@matrix)
    expected=sparse.hstack(blocks,format='csr');output={};losses={};n=len(family.data.items)
    for spec in specs:
        joint=family.models[spec['name']].joint;U,V=joint.user_factors,joint.item_factors
        assert U.shape==(len(family.data.users),spec['factors']) and V.shape==(2*n,spec['factors'])
        assert np.isfinite(U).all() and np.isfinite(V).all() and (U>=0).all() and (V>=0).all()
        difference=(expected-joint.training_intensity).tocsr()
        assert not difference.nnz or abs(difference.data).max()<1e-12
        if spec['factors'] not in losses:losses[spec['factors']]=audit_factor_loss(expected,U,V,joint.model)
        known=sorted(candidates&set(family.data.items));indices=[family.data.item_index[i] for i in known];checked=0
        for user in list(dict.fromkeys(cohort[:3]+config['examples'])):
            index=family.data.user_index.get(user);actual=native[user][spec['name']]
            if index is None:assert not actual;continue
            heads=[np.sum(V[np.array(indices)+offset]*U[index],axis=1) for offset in [0,n]]
            if spec['view']=='mean':
                values=sum((.5*h/h.sum() if h.sum()>0 else np.zeros_like(h)) for h in heads)
            else:values=heads[0 if spec['view']=='watches' else 1]
            reference=sorted(((i,float(v)) for i,v in zip(known,values) if v>0),key=lambda p:(-p[1],p[0]))[:50]
            assert [i for i,_ in reference]==[i for i,_ in actual]
            for (_,a),(_,b) in zip(reference,actual):assert abs(a-b)<1e-12
            checked+=len(actual)
        output[spec['name']]={'paired_input_reconstructed':True,'orientation_and_full_pool_view_verified':True,
            'native_scores_checked':checked,**losses[spec['factors']]}
    return output


def audit_count_mf(destination, config, train, native, cohort, candidates):
    """Reconstruct input intensity, factor scores and complete KL loss."""
    specifications = [s for s in config["methods"] if s["kind"] == "count_nmf"]
    if not specifications:
        return {}
    with (destination / "fitted-components.pkl").open("rb") as stream:
        family = pickle.load(stream)
    output = {}
    for specification in specifications:
        name = specification["name"]
        if name not in family.models:
            output[name] = {"available":False}
            continue
        model = family.models[name]
        rows_index = [family.data.user_index[u] for u in train.user_id]
        columns_index = [family.data.item_index[int(i)] for i in train.item_id]
        observed = sparse.csr_matrix((train[specification.get("signal","watches")].to_numpy(dtype=float)**specification.get("power",1.),(rows_index,columns_index)),shape=family.data.weights.shape)
        totals = observed.sum(axis=1).A.ravel()
        intensity = ((sparse.diags(np.divide(1.,totals,out=np.zeros_like(totals),where=totals>0)) @ observed) if specification.get("normalize_rows",True) else observed).tocsr()
        difference = (intensity-model.training_intensity).tocsr()
        assert not difference.nnz or abs(difference.data).max()<1e-12
        users,items = model.user_factors,model.item_factors
        assert users.shape==(len(family.data.users),specification["factors"])
        assert items.shape==(len(family.data.items),specification["factors"])
        assert np.isfinite(users).all() and np.isfinite(items).all() and (users>=0).all() and (items>=0).all()
        loss_checks = audit_factor_loss(intensity,users,items,model.model)
        checked = 0
        for user in list(dict.fromkeys(cohort[:3]+config["examples"])):
            index = family.data.user_index.get(user)
            exported = native[user][name]
            if index is None:
                assert not exported
                continue
            scores = np.sum(items*users[index],axis=1)
            reference = {item:float(scores[family.data.item_index[item]]) for item in candidates if item in family.data.item_index}
            for item,score in exported:
                assert abs(score-reference[item])<1e-12
                checked += 1
            if len(exported)==50:
                selected = {item for item,_ in exported}
                assert all(score<=exported[-1][1]+1e-12 for item,score in reference.items() if item not in selected)
        output[name] = {"available":True,"input_rows":int(intensity.shape[0]),"input_items":int(intensity.shape[1]),
                        "input_nonzeros":int(intensity.nnz),**loss_checks,"native_scores_checked":checked,
                        "factor_orientation_nonnegativity_and_repeat_eligibility_checked":True}
    return output


def audit_slice(run, prepared, block, config, template):
    source, destination = prepared / block["name"], run / block["name"]
    summary = load(destination / "summary.json")
    train = pd.read_parquet(source / "train-interactions.parquet")
    heldout = pd.read_parquet(source / "valid-interactions.parquet")
    candidates = set(load(source / "candidates.json")["valid"])
    quantities = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in heldout.itertuples(index=False):
        assert type(row.user_id) is str and row.item_id > 0
        if row.item_id in candidates:
            value = quantities[row.user_id][int(row.item_id)]
            value[0] += float(row.seconds)
            value[1] += int(row.watches)
    cohort = [user for user in template if quantities[user]]
    assert cohort == load(destination / "cohort-and-targets.json")["user_ids"]
    targets = {kind: {} for kind in ["primary", "secondary", "all_observed"]}
    for user in cohort:
        records = quantities[user]
        targets["primary"][user] = set(sorted(records, key=lambda item: (-records[item][0], item))[:5])
        targets["secondary"][user] = set(sorted(records, key=lambda item: (-records[item][1], item))[:5])
        targets["all_observed"][user] = set(records)
    predictions, details = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    metadata = {int(item): record for item, record in load(source / "metadata.json")["valid"].items()}
    for row in rows(destination / "recommendations.csv"):
        method, user, item = row["method"], row["user_id"], int(row["item_id"])
        assert user in template and item in candidates and math.isfinite(float(row["score"]))
        assert int(row["rank"]) == len(predictions[method][user]) + 1
        assert row["title"] == metadata.get(item, {}).get("tv_show_title", "")
        assert row["category"] == metadata.get(item, {}).get("tv_show_category", "")
        predictions[method][user].append(item)
        details[method][user].append(row)
    assert set(predictions) == set(summary["methods"])
    for method, users in predictions.items():
        assert [user for user in users if user in targets["primary"]] == cohort
        assert all(len(items) == len(set(items)) == 5 for items in users.values())
    exact_maps, count = {}, 0
    per_user = {}
    for kind in targets:
        values = defaultdict(list)
        per_user[kind] = defaultdict(dict)
        for row in rows(destination / ("per-user-ap-" + kind + ".csv")):
            user, method = row["user_id"], row["method"]
            assert user in targets[kind] and method in predictions
            assert set(json.loads(row["relevant_ids"])) == targets[kind][user]
            assert json.loads(row["predictions"]) == predictions[method][user]
            exact = exact_ap(predictions[method][user], targets[kind][user])
            assert abs(float(exact) - float(row["ap_at_5"])) < 1e-12
            assert user not in per_user[kind][method]
            per_user[kind][method][user] = exact
            values[method].append(exact)
            count += 1
        assert set(values) == set(summary["methods"])
        exact_maps[kind] = {}
        for method, scores in values.items():
            assert list(per_user[kind][method]) == cohort
            mean = sum(scores, Fraction(0)) / len(cohort)
            saved = summary["methods"][method]["metrics"][kind]
            assert abs(float(mean) - saved["map_at_5"]) < 1e-12
            assert str(mean) == saved["exact_macro_mean"]
            exact_maps[kind][method] = mean
    native = defaultdict(lambda: defaultdict(list))
    for row in rows(destination / "native-scores.csv.gz"):
        name, user, item, score = row["component"], row["user_id"], int(row["item_id"]), float(row["score"])
        assert row["popularity_padding"] == "False" and score > 0 and math.isfinite(score) and item in candidates
        assert int(row["rank"]) == len(native[user][name]) + 1
        native[user][name].append((item, score))
    scored_users = list(next(iter(predictions.values())))
    component_names = set(load(destination / "native-score-diagnostics.json"))
    for user in scored_users:
        for name in component_names:
            native[user].setdefault(name, [])
            records = native[user][name]
            assert len(records) <= 50 and len({i for i, _ in records}) == len(records)
            assert records == sorted(records, key=lambda pair: (-pair[1], pair[0]))
    history = {user: history_rank(native, user, config["fusion"]["rrf_constant"]) for user in scored_users}
    specs = {spec["name"]: spec for spec in config["methods"]}
    fusion_maps, expected_contributions = {}, 0
    for method in predictions:
        if specs[method]["kind"] in {"history_fusion", "hybrid", "hybrid_als", "rank_fusion"}:
            for user in scored_users:
                maps = component_spec(specs[method], native, history, user)
                fusion_maps[method, user] = maps
                expected_contributions += sum(len(ranks) for ranks, weight in maps.values())
    actual_contributions, group, seen = 0, None, set()
    fused_scores = defaultdict(lambda: defaultdict(float))
    for row in rows(destination / "fusion-contributions.csv.gz"):
        key = row["method"], row["user_id"]
        if key != group:
            group, seen = key, set()
        item, component = int(row["item_id"]), row["component"]
        assert (item, component) not in seen
        seen.add((item, component))
        ranks, weight = fusion_maps[key][component]
        rank, score = ranks[item]
        assert int(row["rank"]) == rank and abs(float(row["native_score"]) - score) < 1e-12
        assert abs(float(row["weight"]) - weight) < 1e-15
        contribution = weight / (config["fusion"]["rrf_constant"] + rank)
        assert abs(float(row["contribution"]) - contribution) < 1e-15
        fused_scores[key][item] += contribution
        actual_contributions += 1
    assert actual_contributions == expected_contributions
    popularity = train.groupby("item_id").user_id.nunique().to_dict()
    fallback_order = sorted(candidates, key=lambda item: (-popularity.get(item, 0), item))
    for method, users in predictions.items():
        for user, ids in users.items():
            if (method, user) in fusion_maps:
                scores = fused_scores[method, user]
            else:
                scores = dict(native[user][method])
            ordered = sorted(((item, value) for item, value in scores.items() if value > 0), key=lambda pair: (-pair[1], pair[0]))[:5]
            chosen = {item for item, _ in ordered}
            for item in fallback_order:
                if len(ordered) == 5:
                    break
                if item not in chosen:
                    ordered.append((item, float(popularity.get(item, 0))))
                    chosen.add(item)
            assert ids == [item for item, _ in ordered], (block["name"], method, user)
            assert all(abs(float(row["score"]) - expected[1]) < 1e-12 for row, expected in zip(details[method][user], ordered))
    union_recalls = {method: {kind: [0., 0., 0, 0, 0] for kind in ["primary", "secondary"]} for method in predictions}
    seen_union_users = defaultdict(list)
    for row in rows(destination / "candidate-unions.csv.gz"):
        method, user = row["method"], row["user_id"]
        assert user in cohort
        seen_union_users[method].append(user)
        kind = specs[method]["kind"]
        if kind in {"hybrid", "hybrid_als"}:
            names = ["ppr_v2", "history_seconds", "history_watches", "history_recent", specs[method].get("channel_component", "channel_time")]
            if kind == "hybrid_als":
                names.append("als32")
            expected = {item for name in names for item, _ in native[user][name]}
        elif kind == "rank_fusion":
            expected = {item for name in specs[method]["components"] for item, _ in native[user][name]}
        elif kind == "history_fusion":
            expected = {item for item, _ in history[user]}
        else:
            expected = {item for item, _ in native[user][method]}
        actual = set(json.loads(row["native_ids"]))
        assert actual == expected and actual <= candidates and len(actual) <= 300
        fallback = {int(r["item_id"]) for r in details[method][user] if "training_popularity" in r["reason"]}
        assert fallback == set(json.loads(row["fallback_ids"]))
        for label in ["primary", "secondary"]:
            relevant = targets[label][user]
            a, b = len(relevant & actual), len(relevant & (actual | fallback))
            assert abs(float(row[label + "_native_recall"]) - a / len(relevant)) < 1e-12
            assert abs(float(row[label + "_with_fallback_recall"]) - b / len(relevant)) < 1e-12
            assert float(per_user[label][method][user]) <= b / len(relevant) + 1e-12
            values = union_recalls[method][label]
            values[0] += a / len(relevant)
            values[1] += b / len(relevant)
            values[2] += len(relevant)
            values[3] += a
            values[4] += b
    for method in predictions:
        assert seen_union_users[method] == cohort
        for kind, values in union_recalls[method].items():
            saved = summary["methods"][method]["candidate_recall"][kind]
            assert abs(saved["macro_native_union_recall"] - values[0] / len(cohort)) < 1e-12
            assert abs(saved["macro_union_plus_actual_fallback_recall"] - values[1] / len(cohort)) < 1e-12
            assert [saved["target_positions"], saved["native_union_hits"], saved["union_plus_fallback_hits"]] == values[2:]
    lengths = train.groupby("user_id").item_id.nunique().to_dict()
    profile = load(prepared / "context" / block["name"] / "profile.json")
    context_totals = sparse.load_npz(prepared / "context" / block["name"] / "counts.npz").sum(axis=1).A.ravel()
    raw_users = {user for user, total in zip(profile["users"], context_totals) if total > 0}
    segments = defaultdict(list)
    for user in cohort:
        n = lengths.get(user, 0)
        segment = "channel_only" if n == 0 and user in raw_users else "no_history" if n == 0 else "short" if n <= 20 else "medium" if n < 100 else "long"
        segments[segment].append(user)
    frozen_segments = load(destination / "segments.json")["users"]
    assert all(segments[name] == users for name, users in frozen_segments.items())
    for method in predictions:
        for kind in targets:
            for segment, users in frozen_segments.items():
                saved = summary["methods"][method]["metrics"][kind]["segments"][segment]
                assert saved["users"] == len(users) and saved["small_denominator"] == (len(users) < 30)
                if users:
                    exact = sum((per_user[kind][method][u] for u in users), Fraction(0)) / len(users)
                    assert abs(float(exact) - saved["map_at_5"]) < 1e-12
    global_items = set(int(item) for item in train.item_id)
    train_pairs = set(zip(train.user_id, train.item_id))
    for user in scored_users:
        for component in ["ppr_v2", "original_ppr_0.5", "ppr_cosine", "ppr_lift", "als32", "als64", "history_seconds", "history_watches", "history_recent"]:
            if component in native[user]:
                assert all(item in global_items for item, _ in native[user][component])
    opportunity_checks = 0
    opportunity_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "opportunity"}
    if opportunity_specs:
        exposure = sparse.load_npz(source / "availability.npz")
        available = load(source / "availability.json")
        matrix = sparse.load_npz(prepared / "context" / block["name"] / "counts.npz").tocsr()
        profile_info = load(prepared / "context" / block["name"] / "profile.json")
        raw_index = {user: i for i, user in enumerate(profile_info["users"])}
        global_counts = matrix.sum(axis=0).A.ravel()
        global_profile = global_counts / global_counts.sum()
        smoothing = config["context"]["smoothing_seconds"]
        for user in scored_users:
            index = raw_index.get(user)
            total = float(matrix.getrow(index).sum()) if index is not None else 0.
            for name, specification in opportunity_specs.items():
                if not total:
                    assert not native[user][name]
                    continue
                posterior = (matrix.getrow(index).toarray().ravel() + smoothing * global_profile) / (total + smoothing)
                expected_scores = (exposure @ posterior) * (np.asarray(available["airtime_seconds"]) / 3600.) ** specification["airtime_power"]
                score_by_id = dict(zip(available["items"], expected_scores))
                for item, score in native[user][name]:
                    assert abs(score - score_by_id[item]) < 1e-10
                    opportunity_checks += 1
                if len(native[user][name]) == 50:
                    boundary = native[user][name][-1][1]
                    selected_ids = {item for item, _ in native[user][name]}
                    assert all(score <= boundary + 1e-10 for item, score in score_by_id.items() if item not in selected_ids)
    mixed_rows_checked = mixed_native_checked = 0
    mixed_specs = [s for s in config["methods"] if s["kind"] in {"mixed_ppr", "engagement_ppr"}]
    if mixed_specs:
        # Trusted state deserialization only; no production scoring methods.
        with (destination / "fitted-components.pkl").open("rb") as stream:
            fitted = pickle.load(stream)
        if config.get("reuse_cached_fits", True):
            with (ROOT / config["cache_run"] / block["name"] / "fitted-components.pkl").open("rb") as stream:
                previous = pickle.load(stream)
        else:
            previous = fitted
        data = fitted.data
        recency = pd.read_parquet(prepared / "recency" / (block["name"] + ".parquet"))
        assert recency.last_event_end.max() <= profile["cutoff"]
        matrix_rows = [data.user_index[u] for u in recency.user_id]
        matrix_cols = [data.item_index[int(i)] for i in recency.item_id]
        duration = sparse.csr_matrix((np.sqrt(recency.seconds), (matrix_rows, matrix_cols)), shape=data.weights.shape)
        row_sums = duration.sum(axis=1).A.ravel()
        duration = sparse.diags(np.divide(1., row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)) @ duration
        for specification in mixed_specs:
            model = fitted.models[specification["name"]]
            assert (model.transition != previous.models["ppr_v2"].transition).nnz == 0
            signal_frame = recency
            if specification["kind"] == "engagement_ppr":
                signal_frame = pd.read_parquet(ROOT / config["engagement"]["root"] / (block["name"] + ".parquet"))
                keys = ["user_id","item_id","seconds","watches"]
                pd.testing.assert_frame_equal(signal_frame[keys].sort_values(keys[:2]).reset_index(drop=True),
                    train[keys].sort_values(keys[:2]).reset_index(drop=True),check_dtype=False,check_exact=True)
                assert signal_frame.last_event_end.max() <= profile["cutoff"]
            count_rows = [data.user_index[u] for u in signal_frame.user_id]
            count_cols = [data.item_index[int(i)] for i in signal_frame.item_id]
            count_matrix = sparse.csr_matrix((signal_frame[specification["count_signal"]].to_numpy(dtype=float), (count_rows, count_cols)), shape=data.weights.shape)
            totals = count_matrix.sum(axis=1).A.ravel()
            count_matrix = sparse.diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ count_matrix
            expected = ((1-specification["count_share"]) * duration + specification["count_share"] * count_matrix).tocsr()
            totals = expected.sum(axis=1).A.ravel()
            expected = (sparse.diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ expected).tocsr()
            difference = (expected - model.restart_rows).tocsr()
            assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-12
            mixed_rows_checked += expected.shape[0]
            for user in list(dict.fromkeys(cohort[:3] + config["examples"])):
                index = data.user_index.get(user)
                if index is None:
                    assert not native[user][specification["name"]]
                    continue
                restart = expected.getrow(index).toarray().ravel()
                values = restart.copy()
                for iteration in range(config["graph"]["iteration_cap"]):
                    new = specification["alpha"] * (model.transition.T @ values + values[model.dangling].sum() * restart) + (1-specification["alpha"]) * restart
                    delta = np.abs(new-values).sum()
                    values = new
                    if delta <= config["graph"]["tolerance_l1"]:
                        break
                assert delta <= config["graph"]["tolerance_l1"] and abs(values.sum()-1) < 1e-10
                for item, score in native[user][specification["name"]]:
                    assert abs(score-values[data.item_index[item]]) < 1e-10
                    mixed_native_checked += 1
    rate_checks = audit_rate_models(destination, prepared, block, config, train, native, cohort)
    mf_checks = audit_count_mf(destination, config, train, native, cohort, candidates)
    joint_checks = audit_joint_mf(destination,config,train,native,cohort,candidates)
    intensity_checks = audit_intensity_mixtures(destination, config, train, native, cohort, candidates)
    inductive_checks = audit_inductive_mf(destination,prepared,block,config,native,cohort,candidates)
    cold_checks = audit_cold_completion(destination,config,native,cohort,candidates)
    engagement_history_checks = 0
    engagement_histories = [s for s in config["methods"] if s["kind"] == "engagement_history"]
    if engagement_histories:
        feature_frame = pd.read_parquet(ROOT/config["engagement"]["root"]/(block["name"]+".parquet"))
        grouped = {user:part for user,part in feature_frame.groupby("user_id",sort=False)}
        for specification in engagement_histories:
            for user in native:
                if user not in grouped:
                    assert not native[user][specification["name"]]
                    continue
                expected = sorted(((int(item),float(value)) for item,value in grouped[user][["item_id",specification["signal"]]].itertuples(index=False,name=None)
                    if int(item) in candidates and value>0),key=lambda pair:(-pair[1],pair[0]))[:50]
                actual = native[user][specification["name"]]
                assert expected == actual
                engagement_history_checks += len(actual)
    counts = {}
    for kind in ["primary", "secondary"]:
        counts[kind] = {"positions": sum(len(v) for v in targets[kind].values()),
                        "seen_by_user": sum((u, i) in train_pairs for u, values in targets[kind].items() for i in values),
                        "absent_from_train": sum(i not in global_items for values in targets[kind].values() for i in values)}
        for label, predicate in [
                ("seen_by_same_user", lambda user, item: (user, item) in train_pairs),
                ("absent_from_all_train", lambda user, item: item not in global_items),
                ("known_but_not_seen_by_user", lambda user, item: item in global_items and (user, item) not in train_pairs)]:
            for method in predictions:
                subsets = {user: {item for item in relevant if predicate(user, item)} for user, relevant in targets[kind].items()}
                positions = sum(map(len, subsets.values()))
                hits = sum(len(relevant & set(predictions[method][user])) for user, relevant in subsets.items())
                n_users = sum(bool(relevant) for relevant in subsets.values())
                saved = summary["methods"][method]["metrics"][kind]["target_subsets"][label]
                assert [saved["target_positions"], saved["hits_at_5"], saved["users_with_subset"]] == [positions, hits, n_users]
                assert saved["small_denominator"] == (n_users < 30)
                assert saved["micro_recall_at_5"] is None if not positions else abs(saved["micro_recall_at_5"] - hits / positions) < 1e-12
    return {"cohort_size": len(cohort), "candidate_count": len(candidates), "AP_rows_checked": count,
            "fusion_contributions_checked": actual_contributions, "opportunity_scores_checked": opportunity_checks,
            "mixed_restart_rows_checked": mixed_rows_checked, "mixed_native_scores_checked": mixed_native_checked, "target_positions": counts,
            "exposure_rate_checks": rate_checks,
            "count_mf_checks": mf_checks,
            "joint_mf_checks":joint_checks,
            "intensity_mixture_checks": intensity_checks,
            "engagement_history_native_scores_checked":engagement_history_checks,
            "inductive_mf_checks":inductive_checks,
            "cold_completion_checks":cold_checks,
            "exact_maps": {kind: {method: str(value) for method, value in values.items()} for kind, values in exact_maps.items()},
            "segments": {name: len(users) for name, users in frozen_segments.items()}, "all_native_rank_fusion_and_fallback_predictions_reproduced": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prechange", type=Path, default=ROOT / "evidence/target040-01/prechange-manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-change-audit", type=Path,
                        help="Explicit recorded .DS_Store-only hash exceptions; never permits data/model/source changes")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    destination = args.output or run / "independent-audit.json"
    if destination.exists():
        raise FileExistsError("Independent audit must be saved under a new name")
    manifest, config, selection = load(run / "manifest.json"), load(run / "frozen-config.json"), load(run / "selection.json")
    assert manifest["status"] == "complete_local_target040_round" and not manifest["source_changed_during_run"]
    freeze = load(ROOT / config["freeze_evidence"])
    original = run / "source-snapshot" / freeze["config_path"]
    if not original.exists():
        original = ROOT / freeze["config_path"]
    assert digest(original) == manifest["config_sha256"] == freeze["sha256"]
    assert load(original) == config
    prepared = Path(manifest["prepared"])
    for name, value in manifest["input_hashes"].items():
        assert digest(prepared / name) == value
    for name, value in manifest.get("extra_input_hashes", {}).items():
        assert digest(Path(name)) == value
    template = [row["user_id"] for row in rows(manifest["template"])]
    results = {block["name"]: audit_slice(run, prepared, block, config, template) for block in config["slices"]}
    complete = set.intersection(*(set(result["exact_maps"]["primary"]) for result in results.values()))
    criteria, per_slice = {}, {}
    for name in complete:
        per_slice[name] = {block: (Fraction(result["exact_maps"]["primary"][name]) + Fraction(result["exact_maps"]["secondary"][name])) / 2
                           for block, result in results.items()}
        criteria[name] = sum(per_slice[name].values(), Fraction(0)) / len(results)
    ranked = sorted(criteria, key=lambda name: (-criteria[name], config["selection"]["tie_order"].index(name)))
    best = ranked[0]
    wins = sum(per_slice[best][block] > per_slice[config["reference_method"]][block] for block in results)
    replacement = best != config["reference_method"] and criteria[best] > criteria[config["reference_method"]] and wins >= 2
    selected = best if replacement else config["reference_method"]
    assert selected == selection["selected_method"] and best == selection["raw_best_method"]
    assert replacement == selection["recommended_replacement_of_previous_best"]
    assert {name: str(value) for name, value in criteria.items()} == selection["exact_aggregate_criterion"]
    references = {"popularity", "content", "ppr_v2", "history_watches", "personal_history_fusion", "channel_time", "hybrid_70_20_10"}
    checked = 0
    for block in config["slices"]:
        name = block["name"]
        if not (ROOT / config["cache_run"] / name / "recommendations.csv").exists():
            assert not config.get("reuse_cached_fits", True)
            continue
        old = {(r["method"], r["user_id"], r["rank"]): (int(r["item_id"]), float(r["score"]))
               for r in rows(ROOT / config["cache_run"] / name / "recommendations.csv") if r["method"] in references}
        for row in rows(run / name / "recommendations.csv"):
            if row["method"] in references:
                expected = old[row["method"], row["user_id"], row["rank"]]
                assert expected[0] == int(row["item_id"]) and abs(expected[1] - float(row["score"])) < 1e-12
                checked += 1
    final = list(rows(run / "final/submission.csv"))
    candidates = set(load(prepared / "final/candidates.json")["valid"])
    assert [row["user_id"] for row in final] == template and len(template) == manifest["expected_users"]
    for row in final:
        text = row["tv_show_id"]
        tokens = text.split(" ")
        assert len(tokens) == 5 and all(token.isascii() and token.isdecimal() for token in tokens)
        ids = list(map(int, tokens))
        assert len(set(ids)) == 5 and set(ids) <= candidates and min(ids) > 0
        assert text == " ".join(map(str, ids))
    validation = load(run / "final/submission-validation.json")
    assert digest(run / "final/submission.csv") == validation["sha256"]
    assert (run / "selection.json").stat().st_mtime_ns <= (run / "final/model.pkl").stat().st_mtime_ns
    protected = load(args.prechange)
    exceptions = {}
    if args.metadata_change_audit:
        recorded = load(args.metadata_change_audit)
        for change in recorded["changed"]:
            assert Path(change["path"]).name == ".DS_Store"
            assert change["expected"] == protected["files"][change["path"]]
            exceptions[change["path"]] = change
    unchanged = 0
    for name, record in protected["files"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        actual = {"bytes": path.stat().st_size, "sha256": digest(path)}
        if name in exceptions:
            assert actual == exceptions[name]["actual"], name
        else:
            assert actual == record, name
            unchanged += 1
    result = {"passed": True, "performed_by": "Codex independent actual-export audit, not author personal verification",
              "original_frozen_config_sha256_verified":digest(original),
              "exported_config_serialization_sha256":digest(run/"frozen-config.json"),
              "exported_config_parsed_exactly_equals_original":True,
              "script_sha256": digest(__file__), "slices": results, "legacy_prediction_slots_identical": checked,
              "selected_method": selected, "selection_reproduced_exactly": True, "strict_slice_wins": wins,
              "protected_original_files_checked": len(protected["files"]),
              "protected_original_files_unchanged": unchanged,
              "protected_metadata_changes": list(exceptions.values()), "final_rows": len(final),
              "final_csv_sha256": validation["sha256"], "kaggle_uploaded": False, "kaggle_score": None}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "selected": selected, "AP_rows": sum(v["AP_rows_checked"] for v in results.values()),
                      "fusion_rows": sum(v["fusion_contributions_checked"] for v in results.values()),
                      "legacy_slots": checked, "protected_files_unchanged": unchanged,
                      "explicit_metadata_exceptions": len(exceptions), "output": str(destination)}))


if __name__ == "__main__":
    main()
