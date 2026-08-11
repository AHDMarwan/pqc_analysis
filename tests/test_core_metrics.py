import unittest

import numpy as np

from pqc_analysis.core.sampling import sample_parameters
from pqc_analysis.geometry.redundancy import redundant_parameter_ratio
from pqc_analysis.geometry.spectrum import condition_score, effective_dimension, metric_rank
from pqc_analysis.trainability.gradients import gradient_statistics


class TestCoreMetrics(unittest.TestCase):
    def test_sampling_is_reproducible(self):
        a = sample_parameters(3, 5, seed=7)
        b = sample_parameters(3, 5, seed=7)
        np.testing.assert_allclose(a, b)

    def test_spectrum_diagnostics(self):
        metric = np.diag([2.0, 1.0, 0.0])
        self.assertEqual(metric_rank(metric), 2)
        self.assertAlmostEqual(redundant_parameter_ratio(metric), 1.0 / 3.0)
        self.assertAlmostEqual(condition_score(metric), 0.5)
        self.assertAlmostEqual(effective_dimension(metric), 2.0 / 3.0 + 1.0 / 2.0)

    def test_gradient_statistics(self):
        samples = np.array([[1.0, 2.0], [3.0, 4.0]])

        def gradient_fn(theta):
            return 2.0 * theta

        result = gradient_statistics(gradient_fn, samples)
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        self.assertAlmostEqual(result.mean_abs_gradient, float(np.mean(np.abs(expected))))
        self.assertAlmostEqual(result.gradient_variance, float(np.var(expected)))
        self.assertAlmostEqual(result.near_zero_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
