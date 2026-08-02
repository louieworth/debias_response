import unittest

import numpy as np

from debias.debias_variants import PyTorchMLPRegressor


class PyTorchMLPHeadTest(unittest.TestCase):
    def test_all_heads_fit_and_predict_finite_values(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(40, 5)).astype(np.float32)
        target = np.clip(
            0.5 + 0.15 * features[:, 0] - 0.1 * features[:, 1],
            0.0,
            1.0,
        ).astype(np.float32)

        for head in ("mse", "gaussian", "beta"):
            with self.subTest(head=head):
                model = PyTorchMLPRegressor(
                    hidden_layers=(12, 6),
                    learning_rate=1e-2,
                    alpha=0.0,
                    batch_size=16,
                    max_iter=4,
                    early_stopping=False,
                    dropout=0.0,
                    standardize=True,
                    device="cpu",
                    random_state=3,
                    prediction_head=head,
                )
                predictions = model.fit(features, target).predict(features)
                self.assertEqual(predictions.shape, target.shape)
                self.assertTrue(np.all(np.isfinite(predictions)))
                if head == "beta":
                    self.assertTrue(np.all(predictions > 0.0))
                    self.assertTrue(np.all(predictions < 1.0))

    def test_beta_head_handles_endpoint_targets(self):
        features = np.eye(4, dtype=np.float32)
        target = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        model = PyTorchMLPRegressor(
            hidden_layers=(4,),
            learning_rate=1e-2,
            alpha=0.0,
            batch_size=4,
            max_iter=2,
            early_stopping=False,
            device="cpu",
            random_state=5,
            prediction_head="beta",
        )
        predictions = model.fit(features, target).predict(features)
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
