import numpy as np

from pqc_analysis.trainability import barren_plateau_scan


def test_barren_plateau_scan_recovers_exponential_decay():
    def gradient_factory(n_qubits):
        scale = np.exp(-0.25 * n_qubits)

        def gradient(theta):
            return scale * np.asarray(theta, dtype=float)

        return gradient

    result = barren_plateau_scan(
        gradient_factory,
        qubit_counts=[2, 4, 6, 8],
        n_params=3,
        samples=500,
        seed=7,
        min_r_squared=0.7,
        min_decay_rate=0.1,
    )

    assert result.log_variance_slope < -0.35
    assert result.r_squared > 0.7
    assert result.shows_exponential_suppression
    assert result.gradient_variances.shape == (4,)


def test_barren_plateau_scan_supports_parameter_count_callable():
    def gradient_factory(_n_qubits):
        return lambda theta: np.ones_like(theta, dtype=float) + 0.01 * np.asarray(theta)

    result = barren_plateau_scan(
        gradient_factory,
        qubit_counts=[2, 3],
        n_params=lambda n: 2 * n,
        samples=20,
        seed=3,
    )

    assert result.qubit_counts.tolist() == [2, 3]
