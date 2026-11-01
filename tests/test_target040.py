"""Opportunity ablation and complete finite-round correctness on synthetic data."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv.data import build_interactions, import_data, sha256, timestamp
from sweettv.target040 import read_slice, run, select_strategy
from sweettv.target040_models import CountIntensityMF, ExposureAdjustedModel, Family, OpportunityModel, mixed_restart, required_components
from sweettv.v3_context import ContextState
from sweettv.v3_models import Family as PreviousFamily
from test_pipeline_data import build_fixture, HUGE_USER, COLD_USER

ROOT = Path(__file__).resolve().parents[1]


class OpportunityTests(unittest.TestCase):
    def test_hand_exposure_scores_cold_and_shape_checks(self):
        from sweettv.models import _TrainingData
        data = _TrainingData([{"user_id": "u", "item_id": 1, "seconds": 120, "watches": 1}])
        counts = sparse.csr_matrix(([10.], ([0], [0])), shape=(1, 42))
        state = ContextState(counts, ["u"], [5], 0, 10, smoothing_seconds=0)
        exposure = sparse.csr_matrix(([.5, .25], ([0, 1], [0, 0])), shape=(2, 42))
        model = OpportunityModel(data, state, exposure, [1, 2], [3600., 14400.], .5)
        self.assertEqual(model.native("u", {1, 2}), {1: .5, 2: .5})
        total = OpportunityModel(data, state, exposure, [1, 2], [3600., 14400.], 1.)
        self.assertEqual(total.native("u", {1, 2}), {1: .5, 2: 1.})
        self.assertEqual(total.native("cold", {1, 2}), {})
        with self.assertRaises(ValueError):
            OpportunityModel(data, state, exposure, [1, 2], [3600.], 1.)

    def test_fixed_fusion_substitutes_channel_without_normalizing_missing_weights(self):
        family = object.__new__(Family)
        family.config = json.loads((ROOT / "config/target040-round01.json").read_text())
        family.pool = {1, 2, 3, 4, 5}
        spec = {"name": "new", "kind": "hybrid", "weights": [.8, .15, .05], "channel_component": "channel_opportunity_total"}
        lists = {name: [] for name in ["ppr_v2", "history_seconds", "history_watches", "history_recent", "channel_time", "channel_opportunity_total"]}
        lists["channel_time"] = [(1, 1.)]
        lists["channel_opportunity_total"] = [(5, 10.)]
        scores, contributions, union = family.compose(spec, lists)
        self.assertEqual(scores, {5: .05 / 61})
        self.assertEqual(union, {5})
        self.assertEqual(contributions[5]["channel_opportunity_total"]["weight"], .05)
        self.assertNotIn("channel_time", required_components([spec]))

    def test_selection_references_previous_best_history_not_v2(self):
        config = json.loads((ROOT / "config/target040-round01.json").read_text())
        config["methods"] = [s for s in config["methods"] if s["name"] in {"ppr_v2", "history_watches"}]
        values = {"ppr_v2": [.5, .39, .39], "history_watches": [.4] * 3}
        summaries = {b["name"]: {"methods": {name: {"metrics": {k: {"map_at_5": scores[i]} for k in ["primary", "secondary"]}} for name, scores in values.items()}} for i, b in enumerate(config["slices"])}
        selected = select_strategy(summaries, config)
        self.assertEqual(selected["raw_best_method"], "ppr_v2")
        self.assertEqual(selected["selected_method"], "history_watches")
        self.assertFalse(selected["recommended_replacement_of_previous_best"])

    def test_all13_synthetic_cutoffs_cached_parity_refit_and_roundtrip(self):
        self._complete_round("target040-round01.json", 39)

    def test_all16_mixed_restart_round_cutoffs_refit_and_roundtrip(self):
        self._complete_round("target040-round02.json", 48)

    def test_all16_exposure_rate_round_cutoffs_refit_and_roundtrip(self):
        self._complete_round("target040-round03.json", 48)

    def test_all16_count_mf_round_and_optional_resource_failure(self):
        self._complete_round("target040-round04.json", 48)

    def test_new_cutoffs_fit_fresh_without_cached_fold_files(self):
        self._complete_round("target040-round03.json", 48, fresh=True)

    def test_all16_full_pool_intensity_mixtures_fresh_and_independently_audited(self):
        self._complete_round("target040-round06.json", 48, fresh=True)

    def test_all28_engagement_methods_fresh_cutoffs_and_independent_audit(self):
        self._complete_round("target040-round07.json", 84, fresh=True)

    def test_all23_inductive_mf_methods_and_known_factors_at_fresh_cutoffs(self):
        self._complete_round("target040-round08.json", 69, fresh=True)

    def test_all23_cold_completion_methods_reused_graph_and_independent_exports(self):
        self._complete_round("target040-round09.json",69)

    def test_all37_joint_and_squared_mf_objectives_independent_audit(self):
        self._complete_round("target040-round10.json",111)

    def test_all31_transformed_implicit_inputs_independently_audited(self):
        self._complete_round("target040-round11.json",93)

    def test_joint_paired_views_full_pool_normalization_orientation_and_pickle(self):
        from sweettv.models import _TrainingData
        from sweettv.target040_joint import JointIntensityFit,JointMFView
        records=[{"user_id":"a","item_id":1,"seconds":100.,"watches":10},
                 {"user_id":"a","item_id":2,"seconds":1000.,"watches":1},
                 {"user_id":"b","item_id":2,"seconds":400.,"watches":2},
                 {"user_id":"b","item_id":3,"seconds":200.,"watches":4}]
        data=_TrainingData(records);frame=pd.DataFrame(records)
        config=json.loads((ROOT/'config/target040-round10.json').read_text())['count_nmf']
        fit=JointIntensityFit(data,frame,2,config)
        self.assertEqual(fit.training_intensity.shape,(2,6))
        np.testing.assert_allclose(fit.training_intensity.sum(axis=1).A.ravel(),[1,1])
        self.assertAlmostEqual(fit.training_intensity[0,0],.5*10/11)
        self.assertAlmostEqual(fit.training_intensity[0,4],.5*1000/1100)
        count=JointMFView(fit,'watches',[1,2,3,99]);duration=JointMFView(fit,'seconds',[1,2,3,99])
        self.assertGreater(count.native('a',{1,2})[1],count.native('a',{1,2})[2])
        self.assertGreater(duration.native('a',{1,2})[2],duration.native('a',{1,2})[1])
        mean=JointMFView(fit,'mean',[1,2,3,99]);values=mean.native('a',{1,2,3,99})
        self.assertAlmostEqual(sum(values.values()),1.)
        self.assertEqual(mean.native('a',{1}),{1:values[1]})
        self.assertNotIn(99,values);self.assertEqual(mean.native('cold',{1,2}),{})
        self.assertEqual(pickle.loads(pickle.dumps(mean)).native('a',{1,2,3,99}),values)

    def test_count_mf_nonsquare_orientation_intensity_repeats_and_pickle(self):
        from sweettv.models import _TrainingData
        records = [{"user_id":u,"item_id":i,"seconds":float(w*120),"watches":w}
                   for u,values in [("a",[100,1,0,0]),("b",[0,0,1,100]),("c",[50,10,10,0])]
                   for i,w in enumerate(values,1) if w]
        frame = pd.DataFrame(records)
        data = _TrainingData(records)
        config = json.loads((ROOT / "config/target040-round04.json").read_text())["count_nmf"]
        model = CountIntensityMF(data,frame,2,config)
        self.assertEqual(model.user_factors.shape,(3,2))
        self.assertEqual(model.item_factors.shape,(4,2))
        np.testing.assert_allclose(model.training_intensity.toarray().sum(axis=1),np.ones(3))
        self.assertAlmostEqual(model.training_intensity[0,0],100/101)
        first = model.native("a",{1,2,3,4,999})
        self.assertNotIn(999,first)
        self.assertGreater(first[1],first[4])
        self.assertGreater(model.native("b",{1,2,3,4})[4],model.native("b",{1,2,3,4})[1])
        self.assertEqual(model.native("unknown",{1,2}),{})
        for item,score in first.items():
            expected = float(np.sum(model.user_factors[0]*model.item_factors[data.item_index[item]]))
            self.assertAlmostEqual(score,expected,places=12)
        self.assertEqual(pickle.loads(pickle.dumps(model)).native("a",{1,2,3,4,999}),first)

    def test_exposure_rate_by_hand_clipping_unknown_items_and_duplicate_airings(self):
        from types import SimpleNamespace
        from sweettv.models import _TrainingData
        data = _TrainingData([{"user_id": "u", "item_id": item, "seconds": 120., "watches": 1} for item in [1,2]])
        base = SimpleNamespace(data=data, native=lambda user, candidates: {i:s for i,s in {1:2.,2:1.}.items() if i in candidates} if user=="u" else {})
        history = {"lower":0,"upper":86400,"days":1.,"seconds":{"1":3600.,"2":7200.},"airings":{"1":1,"2":1}}
        future = {"lower":86400,"upper":172800,"days":1.,"seconds":{"1":3600.,"2":28800.},"airings":{"1":1,"2":9}}
        config = {"smoothing":{"seconds":3600.,"airings":1.},"gain_clip":[.25,4.],"formula":"fixed test rate"}
        spec = {"name":"rate","base_component":"base","exposure_measure":"seconds","rate_power":1.,"qualifying_family":"PPR hybrid"}
        model = ExposureAdjustedModel(base,history,future,[1,2,3],spec,config)
        self.assertEqual(model.native("u",{1,2,3}),{1:2.,2:3.})
        self.assertEqual(model.native("cold",{1,2,3}),{})
        spec.update(exposure_measure="airings")
        self.assertEqual(ExposureAdjustedModel(base,history,future,[1,2,3],spec,config).native("u",{1,2,3}),{1:2.,2:4.})
        module_spec = importlib.util.spec_from_file_location("exposure_fixture", ROOT / "scripts/prepare_target040_exposure.py")
        exposure = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(exposure)
        rows = [(0,200,1),(0,200,1),(200,400,1),(50,100,2),(0,400,0)]
        self.assertEqual(exposure.exposure(rows,0,400),({1:350.},{1:2}))
        self.assertEqual(exposure.exposure(list(reversed(rows)),0,400),({1:350.},{1:2}))
        self.assertEqual(exposure.exposure(rows,100,300),({1:200.},{}))

    def test_mixed_restart_hand_values_and_dense_dangling_stationary_solution(self):
        from sweettv.models import PPRModel, _TrainingData
        data = _TrainingData([{"user_id": "u", "item_id": i, "seconds": s, "watches": w}
                              for i, s, w in [(1, 100., 9), (2, 900., 1)]])
        frame = pd.DataFrame([{"user_id": "u", "item_id": 1, "watches": 9., "recent_watches": 1.},
                              {"user_id": "u", "item_id": 2, "watches": 1., "recent_watches": 1.}])
        duration = sparse.csr_matrix([[.25, .75]])
        restart = mixed_restart(data, duration, frame, "watches", .75)
        np.testing.assert_allclose(restart.toarray(), [[.7375, .2625]], atol=1e-15)
        np.testing.assert_allclose(mixed_restart(data, duration, frame, "recent_watches", 1.).toarray(), [[.5, .5]])
        transition = sparse.csr_matrix([[0., 1.], [0., 0.]])
        dangling = np.array([False, True])
        model = PPRModel(data, alpha=.1, graph_state=(transition, dangling, restart, 1, {}))
        v = restart.toarray()[0]
        effective = transition.toarray() + dangling[:, None] * v
        expected = np.linalg.solve(np.eye(2) - .1 * effective.T, .9 * v)
        np.testing.assert_allclose(model._walk("u"), expected, atol=1e-8)
        self.assertAlmostEqual(float(model._walk("u").sum()), 1.)
        with self.assertRaises(ValueError):
            mixed_restart(data, duration, frame, "watches", 1.5)

    def test_late_recency_event_is_rejected_before_cached_model_reuse(self):
        state = ContextState(sparse.csr_matrix((1, 42)), ["u"], [5], 0, 10)
        frame = pd.DataFrame([{"user_id": "u", "item_id": 1, "watches": 2.,
                              "recent_watches": 1., "last_event_end": 11.}])
        spec = {"name": "mixed", "kind": "mixed_ppr", "count_signal": "recent_watches", "count_share": .5}
        with self.assertRaisesRegex(ValueError, "cross cutoff"):
            Family(None, None, frame, state, None, [], {"methods": [spec]}, [spec])

    def _complete_round(self, round_config, expected_trials, fresh=False):
        spec = importlib.util.spec_from_file_location("prepare_target_fixture", ROOT / "scripts/prepare_improvement_v3.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, base, prepared, cached = root / "raw", root / "base", root / "prepared", root / "cache"
            base.mkdir();cached.mkdir()
            build_fixture(raw)
            if round_config in {"target040-round06.json", "target040-round07.json", "target040-round08.json", "target040-round09.json", "target040-round10.json", "target040-round11.json"}:
                import csv
                from sweettv.data import FILES
                from test_pipeline_data import _schedule
                for role,day in [("valid",2),("future",3)]:
                    with (raw/FILES[role]).open("a",newline="") as stream:
                        csv.writer(stream).writerow(_schedule(day,7,0,10,101))
            audit = import_data(raw, base / "raw.sqlite", base / "audit")
            build_interactions(base / "raw.sqlite", audit, base / "prepared")
            previous = json.loads((ROOT / "config/improvement-v3.json").read_text())
            for block in previous["slices"]:
                block.update(train_start="2020-01-01T00:00:00", train_end="2020-01-02T00:00:00", target_end="2020-01-03T00:00:00")
            previous["final"] = {"train_start": "2020-01-01T00:00:00", "train_end": "2020-01-03T00:00:00", "target_end": "2020-01-04T00:00:00"}
            previous["examples"] = [HUGE_USER, COLD_USER]
            previous_path = root / "previous.json";previous_path.write_text(json.dumps(previous))
            prepare.prepare(base, prepared, previous_path, raw / "submission.csv", expected_users=3)
            config = json.loads((ROOT / "config" / round_config).read_text())
            config["unicode_serialization_fixture"] = "A Unicode en dash: \u2013"
            config.update(slices=previous["slices"], final=previous["final"], examples=previous["examples"], cache_run=str(cached), prepared=str(prepared), previous_config_sha256=sha256(previous_path))
            if "engagement" in config:
                module_spec = importlib.util.spec_from_file_location("engagement_fixture",ROOT/"scripts/prepare_target040_engagement.py")
                engagement = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(engagement)
                windows = {b["name"]:(b["train_start"],b["train_end"]) for b in config["slices"]}
                windows["full"] = (config["final"]["train_start"],config["final"]["train_end"])
                references = {n:prepared/("final" if n=="full" else n)/"train-interactions.parquet" for n in windows}
                features = root/"engagement"
                engagement.build_engagement_features(base/"raw.sqlite",features,audit,windows=windows,references=references)
                config["engagement"]["root"] = str(features)
                config["engagement"]["hashes"] = {p.name:sha256(p) for p in features.iterdir() if p.is_file()}
            if "inductive_mf" in config:
                from sweettv.v3_context import schedule_features
                descriptors = root/"descriptors";descriptors.mkdir()
                for block in config["slices"]+[{"name":"final",**config["final"]}]:
                    train_frame = pd.read_parquet(prepared/block["name"]/"train-interactions.parquet")
                    ids = sorted(int(i) for i in train_frame.item_id.unique())
                    state = ContextState.load(prepared/"context"/("full" if block["name"]=="final" else block["name"]))
                    matrix,items,totals,_ = schedule_features(base/"raw.sqlite",audit,state,ids,state.lower,state.cutoff)
                    days = (state.cutoff-state.lower)/86400.
                    matrix = sparse.diags(totals/(3600.*days))@matrix
                    sparse.save_npz(descriptors/(block["name"]+"-history.npz"),matrix)
                    (descriptors/(block["name"]+"-history.json")).write_text(json.dumps({"items":items,"lower":state.lower,"upper":state.cutoff,"channels":state.channels}))
                config["inductive_mf"]["root"] = str(descriptors)
                config["inductive_mf"]["hashes"] = {p.name:sha256(p) for p in descriptors.iterdir()}
            if "rate_features_root" in config:
                rates = root / "rates";rates.mkdir()
                for block in config["slices"]+[{"name":"final",**config["final"]}]:
                    ids = json.loads((prepared / block["name"] / "candidates.json").read_text())["valid"]
                    for label,start,end in [("train",block["train_start"],block["train_end"]),("target",block["train_end"],block["target_end"])]:
                        days = (timestamp(end)-timestamp(start))/86400.
                        record = {"lower":timestamp(start),"upper":timestamp(end),"days":days,
                                  "seconds":{str(i):3600.*days for i in ids},"airings":{str(i):days for i in ids}}
                        (rates/(block["name"]+"-"+label+".json")).write_text(json.dumps(record))
                config["rate_features_root"] = str(rates)
                config["rate_features_hashes"] = {p.name:sha256(p) for p in rates.glob("*.json")}
            cached.joinpath("manifest.json").write_text(json.dumps({"status": "fixture"}))
            expected = {}
            for block in config["slices"]:
                _, train, recency, state, exposure, items, metadata = read_slice(prepared, block, config)
                family = PreviousFamily(train, metadata["train"], recency, state, exposure, items, previous, previous["methods"])
                destination = cached / block["name"];destination.mkdir()
                with (destination / "fitted-components.pkl").open("wb") as stream:pickle.dump(family, stream)
                expected[block["name"]] = family.recommend_from(HUGE_USER, next(s for s in config["methods"] if s["name"] == "ppr_v2"), family.lists(HUGE_USER))[0]
            if fresh:
                import shutil
                config["reuse_cached_fits"] = False
                for block in config["slices"]:
                    shutil.rmtree(cached / block["name"])
            config_path = root / "config.json";config_path.write_text(json.dumps(config))
            args = argparse.Namespace(config=config_path, prepared=prepared, run_dir=root / "run", template=raw / "submission.csv", expected_users=3, synthetic_fixture=True)
            before = sha256(base / "raw.sqlite")
            run(args)
            self.assertEqual((args.run_dir/"frozen-config.json").read_bytes(),config_path.read_bytes())
            result = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(result["completed_trials"], expected_trials)
            self.assertEqual(result["failed_trials"], 0)
            self.assertEqual(sha256(base / "raw.sqlite"), before)
            for block in config["slices"]:
                with (args.run_dir / block["name"] / "fitted-components.pkl").open("rb") as stream:family = pickle.load(stream)
                if fresh:
                    self.assertIsNone(family.cache_source)
                actual = family.recommend_from(HUGE_USER, next(s for s in config["methods"] if s["name"] == "ppr_v2"), family.lists(HUGE_USER))[0]
                self.assertEqual(actual, expected[block["name"]])
                self.assertTrue(json.loads((args.run_dir / block["name"] / "summary.json").read_text())["exact_AP_audits_passed"])
            final = json.loads((args.run_dir / "final/submission-validation.json").read_text())
            self.assertTrue(final["passed"])
            self.assertEqual(final["rows"], 3)
            self.assertFalse(json.loads((args.run_dir / "manifest.json").read_text())["source_changed_during_run"])
            if round_config in {"target040-round04.json", "target040-round06.json"}:
                _,train,recency,state,exposure,items,metadata = read_slice(prepared,config["final"],config,final=True)
                hybrid = next(s for s in config["methods"] if s["kind"]=="rank_fusion" and s["components"][0]=="rate_mix_seconds_50")
                deep = Family(train,metadata["train"],recency,state,exposure,items,config,[hybrid])
                self.assertTrue({"ppr_mix_rw_75","rate_mix_seconds_50",hybrid["components"][1]}<=set(deep.models))
                self.assertEqual(len(deep.recommend_from(HUGE_USER,hybrid,deep.lists(HUGE_USER))[0]),5)
                unavailable = argparse.Namespace(**vars(args));unavailable.run_dir=root/"unavailable"
                with patch("sweettv.target040_models.CountIntensityMF",side_effect=MemoryError("synthetic MF resource failure")):
                    if round_config == "target040-round06.json":
                        with self.assertRaisesRegex(RuntimeError,"reference must finish"):
                            run(unavailable)
                        self.assertEqual(json.loads((unavailable.run_dir/"manifest.json").read_text())["status"],"failed")
                    else:
                        run(unavailable)
                if round_config != "target040-round06.json":
                    failed = json.loads((unavailable.run_dir/"summary.json").read_text())
                    self.assertEqual(failed["completed_trials"],18)
                    self.assertEqual(failed["failed_trials"],30)
            if round_config in {"target040-round02.json", "target040-round03.json", "target040-round04.json", "target040-round06.json", "target040-round07.json", "target040-round08.json", "target040-round09.json", "target040-round10.json", "target040-round11.json"}:
                audit_spec = importlib.util.spec_from_file_location("target_independent_audit", ROOT / "scripts/audit_target040_round.py")
                independent = importlib.util.module_from_spec(audit_spec)
                audit_spec.loader.exec_module(independent)
                template_users = [r["user_id"] for r in independent.rows(raw / "submission.csv")]
                checked = independent.audit_slice(args.run_dir, prepared, config["slices"][0], config, template_users)
                self.assertIsInstance(checked["AP_rows_checked"], int)
                self.assertGreater(checked["mixed_restart_rows_checked"], 0)
                if round_config == "target040-round03.json":
                    self.assertGreater(checked["exposure_rate_checks"]["gains"], 0)
                if round_config == "target040-round04.json":
                    self.assertTrue(checked["count_mf_checks"]["count_nmf16"]["independent_sklearn_sqrt_2KL_matches"])
                if round_config == "target040-round06.json":
                    self.assertGreater(checked["intensity_mixture_checks"]["native_scores"],0)
                    self.assertTrue(checked["intensity_mixture_checks"]["full_pool_normalization_and_retrieval_verified"])
                if round_config == "target040-round08.json":
                    check = checked["inductive_mf_checks"]["ind_mf_a1_c25"]
                    self.assertGreater(check["known_factors_verified_unchanged"],0)
                    self.assertGreater(check["cold_factors_verified"],0)
                    self.assertLess(check["normal_equation_relative_residual"],1e-3)
                if round_config == "target040-round09.json":
                    self.assertGreater(checked["cold_completion_checks"]["cold_original_a1_c25"]["known_scores_preserved_checked"],0)
                if round_config == "target040-round10.json":
                    self.assertTrue(checked['count_mf_checks']['count_nmf_frob32']['independent_sklearn_Frobenius_matches'])
                    self.assertTrue(checked['joint_mf_checks']['joint_nmf32_mean']['orientation_and_full_pool_view_verified'])
                json.dumps(checked)  # Exact audit output must also serialize.


if __name__ == "__main__":
    unittest.main()
