import numpy as np
import pytest

import pqc_analysis as pqa


def test_gradient_profile_per_parameter_and_layers():
    samples = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])

    def grad(theta):
        return np.array([theta[0], 0.0, 2.0 * theta[2], 0.0])

    profile = pqa.gradient_profile(
        grad,
        samples,
        layer_groups={"early": [0, 1], "late": [2, 3]},
        near_zero_tol=1e-12,
    )

    assert profile.n_params == 4
    assert profile.near_zero_fraction[1] == pytest.approx(1.0)
    assert profile.near_zero_fraction[3] == pytest.approx(1.0)
    assert set(profile.layer_statistics) == {"early", "late"}
    assert profile.weakest_parameters(k=2) == (1, 3)


def test_gradient_profile_rejects_overlapping_layers():
    samples = np.ones((2, 3))
    with pytest.raises(ValueError, match="must not overlap"):
        pqa.gradient_profile(lambda x: x, samples, layer_groups={"a": [0, 1], "b": [1, 2]})


def test_geometry_pruning_plan_identifies_null_coordinate():
    metric = np.diag([2.0, 1.0, 0.0])
    plan = pqa.geometry_pruning_plan(metric)
    assert plan.estimated_rank == 2
    assert plan.candidate_indices == (2,)
    assert plan.redundancy_scores[2] == pytest.approx(1.0)


def test_aggregate_pruning_plan_uses_persistent_nullity():
    metrics = [np.diag([2.0, 1.0, 0.0]), np.diag([3.0, 0.5, 0.0])]
    plan = pqa.aggregate_pruning_plan(metrics)
    assert plan.candidate_indices == (2,)
    assert plan.metadata["minimum_nullity"] == 1


def test_parameter_shift_resource_accounting():
    estimate = pqa.estimate_training_resources(
        10,
        100,
        gradient_method="parameter-shift",
        shots_per_circuit=1000,
        include_objective_evaluation=True,
    )
    assert estimate.circuit_evaluations_per_step == 21
    assert estimate.shots_per_step == 21_000
    assert estimate.total_circuit_evaluations == 2_100
    assert estimate.total_shots == 2_100_000


def test_spsa_and_adjoint_resource_accounting():
    spsa = pqa.estimate_gradient_resources(100, gradient_method="spsa", shots_per_circuit=500)
    assert spsa.circuit_evaluations_per_step == 2
    assert spsa.shots_per_step == 1000

    adjoint = pqa.estimate_gradient_resources(100, gradient_method="adjoint", shots_per_circuit=500)
    assert adjoint.circuit_evaluations_per_step is None
    assert adjoint.shots_per_step is None
