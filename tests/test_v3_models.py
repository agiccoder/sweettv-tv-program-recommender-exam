"""Exact fusion/graph/API/legacy fixtures; all data in this file is invented."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from sweettv.improved_models import fit_improved_ppr, fit_personal_history, _prepare
from sweettv.models import fit_content, fit_popularity, fit_ppr
from sweettv.v3_context import ContextState
from sweettv.v3_models import ALSModel, Family, SelectedModel, normalized_graph, ordered_native, rrf

ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    def test_hand_ranks_fixed_weights_missing_components_and_id_ties(self):
        scores, detail = rrf({"a": [(1, 100), (2, 1)], "b": [(2, 50), (3, 10)]}, {"a": .5, "b": .5})
        self.assertAlmostEqual(scores[2], .5 / 62 + .5 / 61)
        self.assertEqual([i for i, _ in ordered_native(scores)], [2, 1, 3])
        self.assertEqual(detail[2]["a"]["native_score"], 1)
        missing, _ = rrf({"a": [(1, 100)], "b": []}, {"a": .5, "b": .5})
        self.assertEqual(missing[1], .5 / 61)  # No candidate-specific renormalization.
        ties, _ = rrf({"a": [(3, 100)], "b": [(1, 1)]}, {"a": .5, "b": .5})
        self.assertEqual([i for i, _ in ordered_native(ties)], [1, 3])
        with self.assertRaises(ValueError):
            rrf({"a": [(1, 1), (1, 2)]}, {"a": 1})
        restricted, _ = rrf({"a": [(7, 9), (1, 8)]}, {"a": 1}, allowed={1})
        self.assertEqual(restricted, {1: 1 / 62})


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/improvement-v3.json").read_text())
        self.train = pd.DataFrame([
            {"user_id": user, "item_id": item, "seconds": float(seconds), "watches": watches}
            for user, item, seconds, watches in [
                ("0001", 1, 600, 1), ("0001", 2, 1200, 5), ("0001", 3, 300, 2),
                ("u2", 1, 100, 1), ("u2", 2, 400, 2), ("u3", 2, 200, 1), ("u3", 3, 500, 4)]])
        self.recency = self.train.copy()
        self.recency["recent_seconds"] = self.recency.seconds * .5
        self.state = ContextState(sparse.csr_matrix(([500.], ([0], [0])), shape=(1, 42)), ["channel-only"], [1], 0, 10000)
        self.items = list(range(1, 9))
        self.exposure = sparse.csr_matrix((np.ones(8), (np.arange(8), np.zeros(8, dtype=int))), shape=(8, 42))
        self.metadata = {i: {"tv_show_category": "one" if i % 2 else "two"} for i in self.items[:3]}

    def family(self, specs):
        return Family(self.train, self.metadata, self.recency, self.state, self.exposure, self.items, self.config, specs)

    def test_two_distinct_graph_formulas_match_dense_matrices(self):
        data, seeds, restart, _ = _prepare(self.train, "seconds", .5)
        w = data.weights.toarray()
        x = w / w.sum(axis=1)[:, None]
        c = x.T @ x
        support = (w > 0).astype(float).T @ (w > 0).astype(float)
        observed = []
        for kind in ["cosine", "lift"]:
            if kind == "cosine":
                expected = c / np.sqrt(np.outer(c.diagonal(), c.diagonal()))
            else:
                mass = x.sum(axis=0)
                expected = len(data.users) * c / np.outer(mass + 1, mass + 1) * support / (support + 10)
            np.fill_diagonal(expected, 0)
            expected[support < 2] = 0
            expected /= expected.sum(axis=1)[:, None]
            graph = normalized_graph(data, restart, kind, self.config["graph"])
            np.testing.assert_allclose(graph[0].toarray(), expected, atol=1e-14)
            self.assertEqual(graph[0].diagonal().sum(), 0)
            observed.append(graph[0].toarray())
        self.assertFalse(np.allclose(observed[0], observed[1]))

    def test_original_models_are_identical_with_shared_training_graph_and_native_padding(self):
        names = {"popularity", "content", "original_ppr_0.5", "ppr_v2", "history_seconds", "history_watches"}
        specs = [spec for spec in self.config["methods"] if spec["name"] in names]
        family = self.family(specs)
        references = {"popularity": fit_popularity(self.train), "content": fit_content(self.train, self.metadata),
                      "original_ppr_0.5": fit_ppr(self.train, alpha=.5),
                      "ppr_v2": fit_improved_ppr(self.train, alpha=.25, seed_signal="seconds", seed_power=.5),
                      "history_seconds": fit_personal_history(self.train, signal="seconds", power=1),
                      "history_watches": fit_personal_history(self.train, signal="watches", power=1)}
        for user in ["0001", "missing"]:
            lists = family.lists(user)
            for spec in specs:
                actual = family.recommend_from(user, spec, lists)[0]
                expected = references[spec["name"]].recommend(user, self.items)
                self.assertEqual([r["item_id"] for r in actual], [r["item_id"] for r in expected])
                np.testing.assert_array_equal([r["score"] for r in actual], [r["score"] for r in expected])

    def test_context_only_user_has_new_programmes_but_true_cold_fallback_matches_all16(self):
        family = self.family(self.config["methods"])
        self.assertFalse(family.failures)
        lists = family.lists("missing")
        expected = [row["item_id"] for row in fit_popularity(self.train).recommend("missing", self.items)]
        for spec in self.config["methods"]:
            recs, _, union = family.recommend_from("missing", spec, lists)
            self.assertEqual([row["item_id"] for row in recs], expected, spec["name"])
            self.assertEqual(len(set(row["item_id"] for row in recs)), 5)
        lists = family.lists("channel-only")
        spec = next(s for s in self.config["methods"] if s["name"] == "hybrid_70_20_10")
        recs, contributions, union = family.recommend_from("channel-only", spec, lists)
        self.assertTrue({4, 5, 6, 7, 8} <= union)  # Candidates need not be graph nodes.
        self.assertEqual(contributions[1]["channel_time"]["weight"], .1)
        self.assertEqual(set(contributions[1]), {"channel_time"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.pkl"
            SelectedModel(family, spec).save(path)
            restored = pickle.loads(path.read_bytes())
            self.assertEqual(restored.recommend("channel-only"), recs)

    def test_als_actual_orientation_repeat_flag_confidence_and_official_save_roundtrip(self):
        nonsquare = pd.concat([self.train, pd.DataFrame([{"user_id": "0001", "item_id": 4,
                                      "seconds": 600., "watches": 1}])], ignore_index=True)
        data, _, _, _ = _prepare(nonsquare, "seconds", .5)
        model = ALSModel(data, factors=4, config=self.config["als"])
        self.assertEqual(model.confidence.shape, (3, 4))
        self.assertEqual(model.model.user_factors.shape, (3, 4))
        self.assertEqual(model.model.item_factors.shape, (4, 4))
        self.assertAlmostEqual(float(model.confidence[0, 0]), 1 + 40 * np.log(2), places=5)
        index = data.user_index["0001"]
        ids, scores = model.model.recommend(index, model.confidence[index], N=4, filter_already_liked_items=False)
        direct = model.model.item_factors @ model.model.user_factors[index]
        self.assertTrue(set(ids.tolist()) & set(model.confidence[index].indices.tolist()))
        np.testing.assert_allclose(scores, direct[ids], rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(np.var(direct), 0)
        self.assertEqual(model.native("missing", set(self.items)), {})
        self.assertNotIn(8, model.native("0001", set(self.items)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "als.npz"
            model.model.save(path)
            restored = type(model.model).load(path)
            np.testing.assert_array_equal(restored.item_factors @ restored.user_factors[index], direct)


if __name__ == "__main__":
    unittest.main()
