"""Hand-checkable SYNTHETIC fixtures for versioned personal-restart models."""

from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from sweettv.improved_models import fit_improved_ppr, fit_personal_history, fit_history
from sweettv.models import fit_ppr, fit_popularity


def row(user, item, seconds, watches=1, **extra):
    return dict(user_id=user, item_id=item, seconds=seconds, watches=watches, **extra)


class SyntheticImprovedModelsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row("001", 1, 100, 5), row("001", 2, 10000, 1),
                     row("002", 1, 300), row("002", 3, 600), row("solo", 4, 100)]
        self.pool = [1, 2, 3, 4, 5, 6]

    def test_graph_unchanged_and_duration_seed_less_compressed(self):
        original = fit_ppr(self.rows, alpha=.15)
        revised = fit_improved_ppr(self.rows, alpha=.15, seed_signal="seconds", seed_power=.5)
        self.assertEqual((original.transition != revised.transition).nnz, 0)
        index = revised.data.user_index["001"]
        actual = revised.restart_rows.getrow(index).toarray().ravel()
        self.assertTrue(np.allclose(actual[:2], [1/11, 10/11]))
        before = original.restart_rows.getrow(index).toarray().ravel()
        self.assertGreater(actual[1]/actual[0], before[1]/before[0])

    def test_walk_matches_independent_dense_stationary_solution(self):
        for alpha in (.15, .5):
            model = fit_improved_ppr(self.rows, alpha=alpha)
            model.recommend("001", self.pool)
            v = model.restart_rows.getrow(model.data.user_index["001"]).toarray().ravel()
            q_transpose = model.transition.toarray().T + np.outer(v, model.dangling.astype(float))
            expected = np.linalg.solve(np.eye(len(v))-alpha*q_transpose, (1-alpha)*v)
            self.assertTrue(np.allclose(model._walk("001"), expected, atol=1e-8))
            audit = model.diagnostics["per_user_walk"]["001"]
            self.assertTrue(audit["converged"])
            self.assertLessEqual(audit["l1_residual"], 1e-8)
            self.assertAlmostEqual(audit["mass"], 1., places=12)
            model.recommend("solo", self.pool)
            self.assertEqual(model.diagnostics["per_user_walk"]["solo"]["iterations"], 1)
            self.assertAlmostEqual(model.diagnostics["per_user_walk"]["solo"]["mass"], 1.)

    def test_seconds_and_watches_preserve_different_personal_preference(self):
        duration = fit_personal_history(self.rows, signal="seconds", power=1)
        count = fit_personal_history(self.rows, signal="watches", power=1)
        self.assertEqual(duration.recommend("001", self.pool)[0]["item_id"], 2)
        self.assertEqual(duration.recommend("001", self.pool)[0]["score"], 10000)
        self.assertEqual(count.recommend("001", self.pool)[0]["item_id"], 1)
        self.assertEqual(count.recommend("001", self.pool)[0]["score"], 5)
        self.assertIsNone(count.diagnostics["qualifying_family"])
        self.assertEqual(fit_history(self.rows, seed_kind="watches", power=1).recommend("001", self.pool), count.recommend("001", self.pool))

    def test_scale_before_aggregation_and_power_never_changes_graph(self):
        rows = [row("u", 1, 100, recency=.25), row("u", 1, 100, recency=1), row("u", 2, 100, recency=1)]
        model = fit_improved_ppr(rows, history_scale="recency")
        seeds = model.seed_values.getrow(0).toarray().ravel()
        self.assertTrue(np.allclose(seeds, np.sqrt([125, 100])))
        array = fit_improved_ppr(rows, history_scale=[.25, 1, 1])
        self.assertTrue(np.allclose(model.restart_rows.toarray(), array.restart_rows.toarray()))
        unscaled = fit_improved_ppr(rows)
        self.assertEqual((model.transition != unscaled.transition).nnz, 0)
        mapping = fit_improved_ppr(self.rows, history_scale={("001", 2): 0})
        self.assertEqual(mapping.restart_rows.getrow(mapping.data.user_index["001"]).nnz, 1)
        self.assertEqual(mapping.diagnostics["history_scale_zero_rows"], 1)

    def test_zero_seeds_and_cold_users_share_original_fallback(self):
        no_watches = [row("u", 1, 100, 0), row("u", 2, 100, 0)]
        reference = fit_popularity(no_watches)
        expected = [r["item_id"] for r in reference.recommend("u", self.pool)]
        for model in (fit_improved_ppr(no_watches, seed_signal="watches"), fit_personal_history(no_watches, signal="watches")):
            for user in ("u", "cold"):
                recs = model.recommend(user, self.pool)
                self.assertEqual([r["item_id"] for r in recs], expected)
                self.assertTrue(all(r["reason"] == "cold_user:training_popularity" for r in recs))
            if "walk_calls" in model.diagnostics:
                self.assertEqual(model.diagnostics["walk_calls"], 0)

    def test_state_independent_of_dataframe_and_readonly_after_save(self):
        for factory in (fit_improved_ppr, fit_personal_history):
            frame = pd.DataFrame(self.rows)
            model = factory(frame)
            before = model.recommend("001", self.pool)
            frame.loc[:, "seconds"] = 0
            self.assertEqual(model.recommend("001", self.pool), before)
            with self.assertRaises(ValueError):
                model.seed_values.data[0] = 0
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"model.pkl"
                model.save(path)
                with path.open("rb") as stream:
                    restored = pickle.load(stream)
                self.assertEqual(restored.recommend("001", self.pool), before)
                with self.assertRaises(ValueError):
                    restored.restart_rows.data[0] = 0

    def test_ties_unique_valid_output_and_no_candidate_fitting(self):
        model = fit_personal_history([row("u", 2, 100), row("u", 1, 100)])
        result = model.recommend("u", list(reversed(self.pool)))
        self.assertEqual([r["item_id"] for r in result], [1, 2, 3, 4, 5])
        self.assertEqual(len(set(r["item_id"] for r in result)), 5)
        self.assertEqual(model.data.items.tolist(), [1, 2])
        self.assertTrue(all(r["score"] == 0 for r in result[2:]))
        with self.assertRaises(ValueError):
            model.recommend("u", [1, 2, 3, 4])

    def test_empty_and_invalid_configuration(self):
        for model in (fit_improved_ppr([]), fit_personal_history([])):
            self.assertEqual(len(model.recommend("cold", self.pool)), 5)
        for kwargs in ({"seed_signal": "rating"}, {"seed_power": 0}, {"seed_power": float("nan")},
                       {"history_scale": [-1]*len(self.rows)}, {"history_scale": [1]},
                       {"history_scale": [float("inf")]*len(self.rows)}, {"history_scale": "absent"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_improved_ppr(self.rows, **kwargs)


if __name__ == "__main__":
    unittest.main()
