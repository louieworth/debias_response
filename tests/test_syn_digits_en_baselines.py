import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from run.syn_digits.run_syn_digits_en_baselines import (
    elastic_net_fit_transfer,
    hard_impute_svd,
    summarize_seed_rows,
    significance_stars,
)


class SynDigitsElasticNetTest(unittest.TestCase):
    def test_transfer_matches_reference_preprocessing(self):
        synthetic_donors = np.array(
            [
                [1.0, 2.0, 5.0],
                [2.0, 1.0, 4.0],
                [3.0, 4.0, 2.0],
                [4.0, 3.0, 1.0],
            ]
        )
        synthetic_target = np.array([1.0, 2.0, 4.0, 3.0])
        human_donors = np.array(
            [
                [2.0, 2.0, 4.0],
                [3.0, 1.0, 3.0],
                [4.0, 4.0, 1.0],
                [5.0, 3.0, 2.0],
            ]
        )
        alpha = 0.01
        l1_ratio = 0.3
        fit = elastic_net_fit_transfer(
            synthetic_donors,
            synthetic_target,
            human_donors,
            alpha=alpha,
            l1_ratio=l1_ratio,
            min_column_std=1.0,
            human_normalization="separate",
        )

        syn_mean = synthetic_donors.mean(axis=0)
        syn_std = synthetic_donors.std(axis=0)
        syn_std[syn_std < 1.0] = 1.0
        human_mean = human_donors.mean(axis=0)
        human_std = human_donors.std(axis=0)
        human_std[human_std < 1.0] = 1.0
        target_mean = synthetic_target.mean()
        target_std = synthetic_target.std()
        if target_std < 1.0:
            target_std = 1.0
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=10_000,
            tol=1e-4,
            selection="cyclic",
        ).fit(
            (synthetic_donors - syn_mean) / syn_std,
            (synthetic_target - target_mean) / target_std,
        )
        expected = (
            model.predict((human_donors - human_mean) / human_std)
            * target_std
            + target_mean
        )
        np.testing.assert_allclose(fit.predictions, expected, atol=1e-12, rtol=0.0)

    def test_single_digital_twin_collapses_to_raw_target(self):
        synthetic_donors = np.array([[1.0, 4.0, 2.0]])
        human_means = np.array([[2.5, 3.0, 1.5]])
        synthetic_target = np.array([5.0])
        fit = elastic_net_fit_transfer(
            synthetic_donors,
            synthetic_target,
            human_means,
            human_normalization="synthetic",
        )
        self.assertEqual(fit.active_coefficients, 0)
        self.assertAlmostEqual(float(fit.predictions[0]), 5.0, places=12)

    def test_multiple_digital_twins_can_learn_nonzero_transfer(self):
        synthetic_donors = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
                [5.0, 0.0],
            ]
        )
        synthetic_target = synthetic_donors[:, 0] + 0.5
        human_means = np.array([[4.0, 1.0]])
        fit = elastic_net_fit_transfer(
            synthetic_donors,
            synthetic_target,
            human_means,
            human_normalization="synthetic",
        )
        self.assertGreater(fit.active_coefficients, 0)
        self.assertTrue(np.all(np.isfinite(fit.predictions)))

    def test_summary_preserves_level_specific_k(self):
        rows = []
        for level, method, k in [
            ("individual", "SYN-DIGITS-EN", 1),
            ("population", "SYN-DIGITS-EN-Mean", 50),
        ]:
            for seed in range(5):
                rows.append(
                    {
                        "level": level,
                        "dataset": "example",
                        "method": method,
                        "seed": seed,
                        "deterministic": True,
                        "k": k,
                        "MAE": 1.0,
                        "Acc": 2.0,
                        "HA": 3.0,
                        "SA": 4.0,
                    }
                )
        summary = summarize_seed_rows(pd.DataFrame(rows)).set_index("level")
        self.assertEqual(int(summary.loc["individual", "k"]), 1)
        self.assertEqual(int(summary.loc["population", "k"]), 50)

    def test_hard_imputation_is_deterministic_and_preserves_observed(self):
        matrix = np.array(
            [
                [1.0, np.nan, 3.0],
                [2.0, 2.0, np.nan],
                [3.0, 3.0, 1.0],
                [4.0, np.nan, 0.0],
            ]
        )
        first, first_diagnostics = hard_impute_svd(matrix, rank=2)
        second, second_diagnostics = hard_impute_svd(matrix, rank=2)
        observed = np.isfinite(matrix)
        np.testing.assert_array_equal(first[observed], matrix[observed])
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_significance_stars(self):
        self.assertEqual(significance_stars(0.2), "")
        self.assertEqual(significance_stars(0.08), "*")
        self.assertEqual(significance_stars(0.02), "**")
        self.assertEqual(significance_stars(0.005), "***")


if __name__ == "__main__":
    unittest.main()
