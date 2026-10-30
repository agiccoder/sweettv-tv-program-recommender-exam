"""Hand-checkable SYNTHETIC model fixtures, not competition measurements."""

from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from sweettv.models import fit_models, fit_ppr, fit_content, fit_popularity


def interaction(user, item, seconds=100):
    return {"user_id": user, "item_id": item, "seconds": seconds, "watches": 1}


class SyntheticModelsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [interaction("001", 1), interaction("001", 2),
                     interaction("002", 2), interaction("002", 3),
                     interaction("003", 4)]
        self.metadata = {
            1: {"tv_show_category": "sport", "tv_show_title": "unused title"},
            2: {"tv_show_category": "sport"},
            3: {"tv_show_category": "news"},
            4: {"tv_show_category": "children"},
            5: {"tv_show_category": "sport"},
            6: {"tv_show_category": "news"},
        }
        self.pool = [7, 6, 5, 4, 3, 2, 1]

    def test_popularity_unique_viewers_and_deterministic_ties(self):
        model = fit_popularity(self.rows+[interaction("001", 1)])
        result = model.recommend("unknown", self.pool)
        self.assertEqual([r["item_id"] for r in result], [2, 1, 3, 4, 5])
        self.assertEqual([r["score"] for r in result], [2., 1., 1., 1., 0.])
        self.assertEqual(model.data.popularity[1], 1)

    def test_content_known_attribute_and_no_title_signal(self):
        model = fit_content(self.rows, self.metadata)
        result = model.recommend("001", self.pool)
        self.assertEqual([r["item_id"] for r in result[:3]], [1, 2, 5])
        self.assertTrue(all(abs(r["score"]-1) < 1e-12 for r in result[:3]))
        self.assertEqual(result[3]["reason"], "short_model_list:training_popularity")
        self.assertTrue(all("title" not in name for name in model.vectorizer.get_feature_names_out()))
        self.assertNotIn(7, model.metadata_index)
        with self.assertRaises(ValueError):
            fit_content(self.rows, self.metadata, content_fields=["tv_show_title"])

    def test_person_tokens_split_before_whitespace_normalization(self):
        model = fit_content(self.rows, {1: {"actors": "Alice Smith\nBob Jones", "director": "Some Person; Other Person"}})
        names = set(model.vectorizer.get_feature_names_out())
        self.assertEqual(names, {"actors=alice smith", "actors=bob jones", "director=some person", "director=other person"})

    def test_cold_fallback_identical_for_all_families(self):
        models = fit_models(self.rows, self.metadata)
        expected = [2, 1, 3, 4, 5]
        for name, model in models.items():
            result = model.recommend("cold", self.pool)
            self.assertEqual([r["item_id"] for r in result], expected, name)
            self.assertEqual(len(set(r["item_id"] for r in result)), 5)
        for name in ("content", "ppr"):
            self.assertTrue(all(r["reason"] == "cold_user:training_popularity" for r in models[name].recommend("cold", self.pool)))

    def test_missing_metadata_and_no_positive_candidate_fallback(self):
        content = fit_content(self.rows, {})
        result = content.recommend("001", self.pool)
        self.assertEqual([r["item_id"] for r in result], [2, 1, 3, 4, 5])
        self.assertTrue(all(r["reason"].startswith("cold_user") for r in result))
        for model in (fit_content(self.rows, self.metadata), fit_ppr(self.rows)):
            result = model.recommend("001", [10, 11, 12, 13, 14])
            self.assertEqual([r["item_id"] for r in result], [10, 11, 12, 13, 14])
            self.assertTrue(all(r["reason"] == "no_positive_model_score:training_popularity" for r in result))

    def test_ppr_conservation_convergence_and_training_graph(self):
        for alpha in (.5, .85):
            model = fit_ppr(self.rows, alpha=alpha)
            self.assertEqual(list(model.data.items), [1, 2, 3, 4])
            self.assertTrue(model.dangling[model.data.item_index[4]])
            result = model.recommend("001", self.pool)
            self.assertEqual(len(result), 5)
            audit = model.diagnostics["per_user_walk"]["001"]
            self.assertTrue(audit["converged"])
            self.assertLessEqual(audit["l1_residual"], 1e-8)
            self.assertAlmostEqual(audit["mass"], 1., places=12)
            self.assertGreater(audit["score_variance"], 0)
            model.recommend("003", self.pool)
            dangling_audit = model.diagnostics["per_user_walk"]["003"]
            self.assertAlmostEqual(dangling_audit["mass"], 1., places=12)
            self.assertEqual(dangling_audit["iterations"], 1)
            # Repeat viewing is deliberately eligible for every method.
            self.assertEqual(model.recommend("003", self.pool)[0]["item_id"], 4)

    def test_ppr_sparse_pruning_and_tie_break(self):
        rows = [interaction("u", item) for item in range(1, 8)]
        model = fit_ppr(rows, max_neighbors=2)
        self.assertLessEqual(max(np.diff(model.transition.indptr)), 2)
        self.assertEqual(list(model.transition.getrow(0).indices), [1, 2])
        self.assertTrue(np.allclose(model.transition.sum(axis=1), 1))

    def test_ppr_prunes_across_product_block_boundary(self):
        # SYNTHETIC complete co-view fixture crosses the 128-item product block.
        model = fit_ppr([interaction("u", item) for item in range(1, 133)], max_neighbors=3)
        self.assertEqual(model.diagnostics["graph_product_block_rows"], 128)
        self.assertEqual(model.diagnostics["graph_edges_before_pruning"], 132*131)
        self.assertEqual(model.diagnostics["graph_edges"], 132*3)
        self.assertEqual(list(model.transition.getrow(0).indices), [1, 2, 3])
        self.assertEqual(list(model.transition.getrow(130).indices), [0, 1, 2])
        self.assertTrue(np.allclose(model.transition.data, 1/3))

    def test_saved_fitted_state_reproduces_lists(self):
        for name, model in fit_models(self.rows, self.metadata).items():
            before = model.recommend("001", self.pool)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/name/"model.pkl"
                model.save(path)
                with path.open("rb") as stream:
                    restored = pickle.load(stream)
                self.assertEqual(restored.recommend("001", self.pool), before)
                if name == "ppr":
                    self.assertTrue(restored.diagnostics["per_user_walk"]["001"]["converged"])
                    self.assertAlmostEqual(restored.diagnostics["per_user_walk"]["001"]["mass"], 1.)

    def test_input_rejection_and_exact_string_users(self):
        model = fit_popularity(self.rows)
        self.assertIn("001", model.data.user_index)
        for pool in ([1, 2, 3, 4], [1, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]):
            with self.assertRaises(ValueError):
                model.recommend("001", pool)
        for row in (interaction(1, 1), interaction("001", 0), interaction("001", 1, -1), interaction("001", 1, float("nan"))):
            with self.assertRaises(ValueError):
                fit_popularity([row])

    def test_empty_histories_are_supported(self):
        for model in fit_models([], {}).values():
            result = model.recommend("cold", [1, 2, 3, 4, 5])
            self.assertEqual([r["item_id"] for r in result], [1, 2, 3, 4, 5])
        content = fit_content([], self.metadata)
        self.assertEqual(len(content.recommend("cold", self.pool)), 5)


if __name__ == "__main__":
    unittest.main()
