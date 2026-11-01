"""Bounded availability-opportunity ablations grounded in the existing PPR hybrid.

The only new signal is an explicit exposure multiplier, not future preference.
Existing trained states may be reused only for exactly matching prepared inputs.
"""
import copy
import json
from pathlib import Path
import pickle
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from .data import sha256, timestamp
from .models import PPRModel, _BaseModel
from .target040_intensity import IntensityMixture
from .target040_inductive import InductiveMF
from .target040_cold import ColdCompletion
from .target040_joint import JointIntensityFit, JointMFView
from .v3_models import (Family as PreviousFamily, ChannelModel, SignalModel, native_scores, ordered_native, rrf,
                       required_components as previous_dependencies)

ROOT = Path(__file__).resolve().parents[2]


def required_components(specs, definitions=None):
    result = set()
    for spec in specs:
        names = set(spec["components"]) if spec["kind"] == "rank_fusion" else previous_dependencies([spec])
        if spec["kind"] in {"exposure_adjusted", "inductive_mf"}:
            names.add(spec["base_component"])
        if spec["kind"] == "intensity_mixture":
            names.update(spec["components"])
        if spec["kind"] == "cold_completion":
            names.update([spec["base_component"],spec["factor_component"]])
        if spec.get("channel_component"):
            names.discard("channel_time")
            names.add(spec["channel_component"])
        result.update(names)
    if definitions is not None:
        lookup = {s["name"]: s for s in definitions}
        while True:
            previous = set(result)
            for name in previous:
                specification = lookup.get(name, {})
                if specification.get("kind") in {"exposure_adjusted", "inductive_mf"}:
                    result.add(specification["base_component"])
                elif specification.get("kind") in {"rank_fusion", "intensity_mixture"}:
                    result.update(specification["components"])
                elif specification.get("kind") == "cold_completion":
                    result.update([specification["base_component"],specification["factor_component"]])
            if result == previous:
                break
    return result


def mixed_restart(data, duration_restart, recency, signal, share):
    """Mix normalized training signals before the unchanged PPR transition."""
    if signal not in {"watches", "recent_watches", "capped600", "capped1800", "fractional_watches", "active_days", "recent_capped600"} or not 0 <= share <= 1:
        raise ValueError("Invalid count restart specification")
    values = recency[signal].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Count restart must be finite and nonnegative")
    rows = [data.user_index[u] for u in recency.user_id]
    columns = [data.item_index[int(i)] for i in recency.item_id]
    counts = sparse.csr_matrix((values, (rows, columns)), shape=data.weights.shape)
    totals = counts.sum(axis=1).A.ravel()
    count_restart = sparse.diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ counts
    result = ((1 - share) * duration_restart + share * count_restart).tocsr()
    totals = result.sum(axis=1).A.ravel()
    result = (sparse.diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ result).tocsr()
    result.sort_indices()
    return result


class OpportunityModel(ChannelModel):
    def __init__(self, data, state, exposure, items, airtime_seconds, power):
        super().__init__(data, state, exposure, items)
        self.power = float(power)
        self.hours = np.asarray(airtime_seconds, dtype=float) / 3600.
        if self.hours.shape != (len(items),) or not np.isfinite(self.hours).all() or np.any(self.hours < 0):
            raise ValueError("Invalid deduplicated schedule duration")
        if self.power not in {.5, 1.}:
            raise ValueError("Power outside frozen opportunity grid")
        self.multiplier = dict(zip(items, self.hours ** self.power))
        self.diagnostics.update({"method": "channel_opportunity", "airtime_power": self.power,
            "formula": "historical smoothed context affinity times deduplicated candidate broadcast hours**power",
            "qualifying_family": None, "future_viewing_used": False, "future_content_used": False})

    def native(self, user, candidates):
        return {item: score * self.multiplier[item] for item, score in super().native(user, candidates).items()}


class ExposureAdjustedModel(_BaseModel):
    """Fixed availability transform after base scoring, not a future fit."""
    def __init__(self, base, historical, future, items, specification, config):
        super().__init__(base.data)
        self.base = base
        measure = specification["exposure_measure"]
        prior = config["smoothing"][measure]
        if historical["days"] <= 0 or future["days"] <= 0 or historical["upper"] != future["lower"]:
            raise ValueError("Exposure windows must be positive and adjacent")
        self.gain = {}
        for item in items:
            old = float(historical[measure].get(str(item), 0.))
            new = float(future[measure].get(str(item), 0.))
            if not np.isfinite(old+new) or min(old,new) < 0:
                raise ValueError("Invalid schedule exposure")
            ratio = ((new + prior)/future["days"])/((old + prior)/historical["days"])
            self.gain[item] = float(np.clip(ratio, *config["gain_clip"]) ** specification["rate_power"])
        self.diagnostics.update({"method": specification["name"], "base_component": specification["base_component"],
            "qualifying_family": specification["qualifying_family"], "exposure_measure": measure,
            "rate_power": specification["rate_power"], "formula": config["formula"],
            "minimum_gain": min(self.gain.values()) if self.gain else None,
            "maximum_gain": max(self.gain.values()) if self.gain else None,
            "historical_bounds": [historical["lower"], historical["upper"]],
            "target_bounds": [future["lower"],future["upper"]], "future_viewing_used": False,
            "unknown_base_items": "No invented base score; common fallback applies after native ranking"})

    def native(self, user, candidates):
        return {item: score*self.gain[item] for item,score in native_scores(self.base, user, candidates).items()}


class CountIntensityMF(_BaseModel):
    """Low-rank reconstruction of nonnegative relative implicit count intensity."""
    def __init__(self, data, frame, factors, config):
        from sklearn.decomposition import NMF
        super().__init__(data)
        rows = [data.user_index[u] for u in frame.user_id]
        columns = [data.item_index[int(i)] for i in frame.item_id]
        signal=config.get("signal","watches");power=float(config.get("power",1.))
        values = frame[signal].to_numpy(dtype=float)**power
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Implicit counts must be finite nonnegative")
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=data.weights.shape)
        totals = matrix.sum(axis=1).A.ravel()
        self.training_intensity = ((sparse.diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ matrix) if config.get("normalize_rows",True) else matrix).tocsr()
        self.model = NMF(n_components=factors, init=config["init"], solver=config["solver"], beta_loss=config["beta_loss"],
            max_iter=config["max_iter"], tol=config["tol"], random_state=config["seed"],
            alpha_W=config["alpha_W"], alpha_H=config["alpha_H"], l1_ratio=config["l1_ratio"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.user_factors = self.model.fit_transform(self.training_intensity)
        self.item_factors = self.model.components_.T.copy()
        if self.user_factors.shape != (len(data.users), factors) or self.item_factors.shape != (len(data.items), factors):
            raise ValueError("Count MF user/item orientation mismatch")
        if any(not np.isfinite(a).all() or np.any(a < 0) for a in [self.user_factors, self.item_factors]):
            raise ValueError("Invalid nonnegative count MF factors")
        self.diagnostics.update({"method":"count_intensity_nmf", "qualifying_family":"implicit matrix factorization",
            "objective":config["objective"], "matrix":config["matrix"], "factors":factors,
            "input_signal":signal,"input_power":power,"rows_L1_normalized":config.get("normalize_rows",True),
            "input_matrix_shape":list(self.training_intensity.shape), "actual_iterations":int(self.model.n_iter_),
            "sklearn_reconstruction_err":float(self.model.reconstruction_err_),
            "reached_iteration_cap":bool(self.model.n_iter_ >= config["max_iter"]),
            "warnings":[str(w.message) for w in caught],
            "user_factor_variance":float(np.var(self.user_factors)), "item_factor_variance":float(np.var(self.item_factors)),
            "unobserved_interpretation":config["unobserved"], "previously_viewed_items_filtered":False,
            "unknown_ID_policy":"No invented factor; common fallback or independently fitted context"})

    def native(self, user, candidates):
        index = self.data.user_index.get(user)
        if index is None:
            return {}
        known = [(item, self.data.item_index[item]) for item in candidates if item in self.data.item_index]
        if not known:
            return {}
        values = np.einsum("ij,j->i", self.item_factors[[j for _,j in known]], self.user_factors[index], optimize=False)
        return {item:float(value) for (item,_),value in zip(known,values)}


class Family(PreviousFamily):
    def __init__(self, train, metadata, recency, state, exposure, items, config, specs):
        needed = required_components(specs, config["methods"])
        channel_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "opportunity"}
        mixed_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "mixed_ppr"}
        recent_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "recent_history"}
        adjusted_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "exposure_adjusted"}
        nmf_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "count_nmf"}
        mixture_specs = {s["name"]: s for s in config["methods"] if s["kind"] == "intensity_mixture"}
        engagement_specs = {s["name"]: s for s in config["methods"] if s["kind"] in {"engagement_history", "engagement_ppr"}}
        inductive_specs = {s["name"]:s for s in config["methods"] if s["kind"] == "inductive_mf"}
        cold_specs = {s["name"]:s for s in config["methods"] if s["kind"] == "cold_completion"}
        joint_specs = {s["name"]:s for s in config["methods"] if s["kind"] == "joint_nmf"}
        if needed & (set(mixed_specs) | set(recent_specs)):
            if (recency.last_event_end > state.cutoff).any() or not np.isfinite(recency.recent_watches).all():
                raise ValueError("Recent counts cross cutoff or contain nonfinite values")
            if (recency.recent_watches < 0).any() or (recency.recent_watches > recency.watches * (1 + 1e-14)).any():
                raise ValueError("Recent count exceeds its accepted-airing count")
        base_needed = needed - set(channel_specs) - set(mixed_specs) - set(recent_specs) - set(adjusted_specs) - set(nmf_specs) - set(mixture_specs) - set(engagement_specs) - set(inductive_specs) - set(cold_specs) - set(joint_specs)
        if needed & set(channel_specs):
            base_needed.add("channel_time")
        if needed & (set(mixed_specs) | {n for n,s in engagement_specs.items() if s["kind"] == "engagement_ppr"}):
            base_needed.add("ppr_v2")
        matching = [block for block in config["slices"] if timestamp(block["train_end"]) == state.cutoff]
        if matching and config.get("reuse_cached_fits", True):
            block = matching[0]["name"]
            cache = ROOT / config["cache_run"] / block / "fitted-components.pkl"
            with cache.open("rb") as stream:
                base = pickle.load(stream)
            if base.pool != set(items) or len(base.data.users) != train.user_id.nunique() or len(base.data.items) != train.item_id.nunique():
                raise ValueError("Cached state dimensions/pool differ")
            self.cache_source = str(cache)
            self.__dict__.update(base.__dict__)
            self.models = {name: model for name, model in base.models.items() if name in base_needed}
            self.fit_seconds = {name: 0. for name in self.models}
            self.shared_data_seconds = self.shared_graph_seconds = 0.
            self.cache_source = str(cache)
        else:
            base_config = json.loads((ROOT / "config/improvement-v3.json").read_text())
            base_specs = [s for s in base_config["methods"] if s["name"] in base_needed]
            if "history_recent" in base_needed and not any(s["kind"] == "history_fusion" for s in base_specs):
                base_specs.append({"name": "personal_history_fusion", "kind": "history_fusion"})
            super().__init__(train, metadata, recency, state, exposure, items, base_config, base_specs)
            self.cache_source = None
        if not base_needed <= set(self.models):
            raise ValueError("Required shared fitted components unavailable")
        self.config = config
        self.predict_seconds, self.native_diagnostics, self.failures = {}, {}, {}
        for model in self.models.values():
            if hasattr(model, "_cache"):
                model._cache.clear()
        block_name = matching[0]["name"] if matching else "final"
        availability = json.loads((ROOT / config["prepared"] / block_name / "availability.json").read_text())
        if availability["items"] != list(items):
            raise ValueError("Schedule duration ID ordering mismatch")
        for name in sorted(needed & set(channel_specs)):
            started = time.perf_counter()
            self.models[name] = OpportunityModel(self.data, state, exposure, items, availability["airtime_seconds"], channel_specs[name]["airtime_power"])
            self.fit_seconds[name] = time.perf_counter() - started
        for name in sorted(needed & set(recent_specs)):
            started = time.perf_counter()
            self.models[name] = SignalModel(self.data, recency, recent_specs[name]["signal"])
            self.fit_seconds[name] = time.perf_counter() - started
        for name in sorted(needed & set(mixed_specs)):
            started = time.perf_counter()
            specification = mixed_specs[name]
            base = self.models["ppr_v2"]
            restart = mixed_restart(self.data, base.restart_rows, recency, specification["count_signal"], specification["count_share"])
            graph_state = (base.transition, base.dangling, restart, base.diagnostics["graph_edges_before_pruning"],
                           {"graph_reuse": "Exact same cutoff's original co-view transition; no new edge fitting"})
            model = PPRModel(self.data, alpha=specification["alpha"], graph_state=graph_state)
            model.diagnostics.update({"method": name, "qualifying_family": "personalized PageRank",
                "restart_formula": config["mixed_restart"]["formula"], "count_signal": specification["count_signal"],
                "count_share": specification["count_share"], "cutoff": state.cutoff,
                "maximum_training_event_end": float(recency.last_event_end.max()) if len(recency) else None})
            self.models[name] = model
            self.fit_seconds[name] = time.perf_counter() - started
        if needed & set(engagement_specs):
            directory = ROOT / config["engagement"]["root"]
            for filename, expected in config["engagement"]["hashes"].items():
                if sha256(directory / filename) != expected:
                    raise ValueError("Frozen engagement feature changed: " + filename)
            window = "full" if block_name == "final" else block_name
            audit = json.loads((directory / "engagement-feature-audit.json").read_text())["windows"][window]
            if audit["start_inclusive"] != state.lower or audit["cutoff_exclusive"] != state.cutoff:
                raise ValueError("Engagement window differs from the fitting cutoff")
            frame = pd.read_parquet(directory / (window + ".parquet"))
            keys = ["user_id", "item_id", "seconds", "watches"]
            pd.testing.assert_frame_equal(frame[keys].sort_values(keys[:2]).reset_index(drop=True),
                train[keys].sort_values(keys[:2]).reset_index(drop=True), check_dtype=False, check_exact=True)
            if frame.duplicated(keys[:2]).any() or (frame.last_event_end > state.cutoff).any():
                raise ValueError("Duplicated or future engagement event")
            for signal in ["capped600", "capped1800", "fractional_watches", "active_days", "recent_capped600"]:
                if not np.isfinite(frame[signal]).all() or (frame[signal] < 0).any() or (frame[signal] > frame.watches*(1+1e-12)).any():
                    raise ValueError("Engagement signal outside accepted count bounds")
            for name in sorted(needed & set(engagement_specs)):
                started = time.perf_counter()
                specification = engagement_specs[name]
                if specification["kind"] == "engagement_history":
                    model = SignalModel(self.data, frame, specification["signal"])
                else:
                    base = self.models["ppr_v2"]
                    restart = mixed_restart(self.data, base.restart_rows, frame, specification["count_signal"], specification["count_share"])
                    model = PPRModel(self.data, alpha=specification["alpha"], graph_state=(base.transition,base.dangling,restart,
                        base.diagnostics["graph_edges_before_pruning"], {"graph_reuse":"Same cutoff's original co-view transition"}))
                    model.diagnostics.update({"method":name,"qualifying_family":"personalized PageRank",
                        "count_signal":specification["count_signal"],"count_share":specification["count_share"],
                        "restart_formula":config["engagement"]["PPR"],"cutoff":state.cutoff})
                self.models[name] = model
                self.fit_seconds[name] = time.perf_counter()-started
        for name in sorted(needed & set(nmf_specs)):
            started = time.perf_counter()
            print("Fitting count-intensity MF: " + name, flush=True)
            try:
                parameters=dict(config["count_nmf"])
                for key in ["beta_loss","solver","objective","signal","power","normalize_rows","matrix"]:
                    if key in nmf_specs[name]:parameters[key]=nmf_specs[name][key]
                model = CountIntensityMF(self.data, train, nmf_specs[name]["factors"], parameters)
                if time.perf_counter()-started > config["resources"]["max_method_wall_seconds"]:
                    raise TimeoutError("Count MF exceeded frozen component wall-time bound")
                self.models[name] = model
            except (ImportError, MemoryError, TimeoutError) as error:
                self.failures[name] = {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
            self.fit_seconds[name] = time.perf_counter()-started
        joint_cache={}
        for name in sorted(needed & set(joint_specs)):
            started=time.perf_counter();specification=joint_specs[name];rank=specification["factors"]
            if rank not in joint_cache:
                print("Fitting shared count/duration MF rank: "+str(rank),flush=True)
                joint_cache[rank]=JointIntensityFit(self.data,train,rank,config["count_nmf"])
            self.models[name]=JointMFView(joint_cache[rank],specification["view"],items)
            self.fit_seconds[name]=time.perf_counter()-started
        if needed & set(inductive_specs):
            directory = ROOT/config["inductive_mf"]["root"]
            for filename,expected in config["inductive_mf"]["hashes"].items():
                if sha256(directory/filename) != expected:
                    raise ValueError("Frozen schedule descriptor changed: " + filename)
            record = json.loads((directory/(block_name+"-history.json")).read_text())
            if (record["lower"],record["upper"]) != (state.lower,state.cutoff) or record["channels"] != state.channels:
                raise ValueError("Historical factor descriptors differ from the training window/vocabulary")
            history = sparse.load_npz(directory/(block_name+"-history.npz"))
            target_end = matching[0]["target_end"] if matching else config["final"]["target_end"]
            target_days = (timestamp(target_end)-state.cutoff)/86400.
            future = (sparse.diags(np.asarray(availability["airtime_seconds"])/(3600.*target_days))@exposure).tocsr()
            ridge_cache = {}
            for name in sorted(needed & set(inductive_specs)):
                started = time.perf_counter();specification = inductive_specs[name]
                base_name = specification["base_component"]
                if base_name not in self.models:
                    self.failures[name] = {"type":"UnavailableComponent","message":"Missing implicit-MF base: "+base_name}
                    continue
                key = (base_name,specification["ridge_alpha"])
                model = InductiveMF(self.models[base_name],history,record["items"],future,items,specification,
                                    config["inductive_mf"],fitted_ridge=ridge_cache.get(key))
                ridge_cache[key] = model.ridge
                self.models[name] = model
                self.fit_seconds[name] = time.perf_counter()-started
        if needed & set(adjusted_specs):
            directory = ROOT / config["rate_features_root"]
            for filename, expected in config["rate_features_hashes"].items():
                if sha256(directory / filename) != expected:
                    raise ValueError("Frozen schedule exposure feature changed: " + filename)
            historical = json.loads((directory / (block_name + "-train.json")).read_text())
            future = json.loads((directory / (block_name + "-target.json")).read_text())
            if historical["lower"] != state.lower or historical["upper"] != state.cutoff:
                raise ValueError("Schedule exposure history crosses the model cutoff")
            target_end = matching[0]["target_end"] if matching else config["final"]["target_end"]
            if future["upper"] != timestamp(target_end):
                raise ValueError("Schedule target exposure differs from the frozen prediction window")
            for name in sorted(needed & set(adjusted_specs)):
                started = time.perf_counter()
                specification = adjusted_specs[name]
                self.models[name] = ExposureAdjustedModel(self.models[specification["base_component"]], historical, future,
                                                          items, specification, config["exposure_rate"])
                self.fit_seconds[name] = time.perf_counter() - started

        for name in sorted(needed & set(mixture_specs)):
            started = time.perf_counter()
            specification = mixture_specs[name]
            missing = [component for component in specification["components"] if component not in self.models]
            if missing:
                self.failures[name] = {"type":"UnavailableComponent", "message":"Missing components: " + ", ".join(missing)}
            else:
                self.models[name] = IntensityMixture([self.models[n] for n in specification["components"]], items,
                                                     specification["weights"], specification)
            self.fit_seconds[name] = time.perf_counter() - started

        for name in sorted(needed & set(cold_specs)):
            started=time.perf_counter();specification=cold_specs[name]
            self.models[name]=ColdCompletion(self.models[specification["base_component"]],self.models[specification["factor_component"]],items,specification)
            self.fit_seconds[name]=time.perf_counter()-started

    def compose(self, spec, lists):
        if spec["kind"] == "rank_fusion":
            components = {name: lists[name] for name in spec["components"]}
            scores, contributions = rrf(components, dict(zip(spec["components"], spec["weights"])),
                                        self.config["fusion"]["rrf_constant"], self.pool)
            return scores, contributions, set(scores)
        channel = spec.get("channel_component")
        if not channel:
            return super().compose(spec, lists)
        adjusted = dict(lists)
        adjusted["channel_time"] = lists[channel]
        scores, contributions, union = super().compose(spec, adjusted)
        for values in contributions.values():
            if "channel_time" in values:
                values[channel] = values.pop("channel_time")
        return scores, contributions, union
