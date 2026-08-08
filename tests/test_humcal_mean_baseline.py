import unittest

import numpy as np

from run.aggreated.run_humcal_mean_baseline import (
    fit_humcal_mean,
    significance_stars,
)


class HuMCalMeanBaselineTest(unittest.TestCase):
    def test_fit_returns_simplex_weights_and_improves_uniform_loss(self):
        llm = np.array(
            [
                [0.0, 0.2, 1.0],
                [0.2, 0.4, 0.9],
                [0.8, 0.6, 0.1],
                [1.0, 0.8, 0.0],
            ],
            dtype=float,
        )
        target_weights = np.array([0.65, 0.25, 0.10], dtype=float)
        human = llm @ target_weights

        weights, diagnostics = fit_humcal_mean(llm, human)

        self.assertTrue(diagnostics["success"])
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertLessEqual(
            diagnostics["train_mse_norm"],
            diagnostics["uniform_train_mse_norm"],
        )
        np.testing.assert_allclose(llm @ weights, human, atol=1e-6, rtol=0.0)

    def test_fit_is_deterministic(self):
        rng = np.random.default_rng(17)
        llm = rng.uniform(size=(30, 8))
        human = llm @ np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.08, 0.07, 0.05])

        first, _ = fit_humcal_mean(llm, human)
        second, _ = fit_humcal_mean(llm, human)

        np.testing.assert_allclose(first, second, atol=1e-12, rtol=0.0)

    def test_significance_stars(self):
        self.assertEqual(significance_stars(0.2), "")
        self.assertEqual(significance_stars(0.08), "*")
        self.assertEqual(significance_stars(0.02), "**")
        self.assertEqual(significance_stars(0.005), "***")


if __name__ == "__main__":
    unittest.main()
