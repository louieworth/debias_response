import unittest

import numpy as np

from debias.debias_variants import prepare_features


class PopulationLlmBudgetTest(unittest.TestCase):
    @staticmethod
    def _row():
        return {
            "Variable_Name": "q1",
            "Question_Embedding": [0.25, 0.75],
            "Average_Human_Response": 3.0,
            "Average_Human_Response_norm": 0.5,
            "Average_LLM_Response_norm": None,
            "score_range": [1, 5],
            "gpt-4o_norm": [0.0, 0.2, 0.8, 1.0],
        }

    def test_mean_uses_only_budgeted_responses(self):
        X, _, _, _, baseline, _ = prepare_features(
            [self._row()],
            "x_avg_llm",
            llm_field="gpt-4o_norm",
            llm_dim=2,
        )
        self.assertAlmostEqual(float(X[0, -1]), 0.1)
        self.assertAlmostEqual(float(baseline[0]), 0.1)

    def test_vector_and_base_share_the_same_budget(self):
        X, _, _, _, baseline, _ = prepare_features(
            [self._row()],
            "x_all_llm",
            llm_field="gpt-4o_norm",
            llm_dim=2,
        )
        np.testing.assert_allclose(X[0, -2:], [0.0, 0.2])
        self.assertAlmostEqual(float(baseline[0]), 0.1)


if __name__ == "__main__":
    unittest.main()
