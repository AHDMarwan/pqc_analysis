import numpy as np

from pqc_analysis.experimental import BenchmarkMatrixConfig, build_benchmark_case, build_benchmark_matrix, parameter_count


def test_parameter_counts():
    assert parameter_count("hardware_efficient", 4, 3) == 24
    assert parameter_count("alternating", 4, 3) == 12
    assert parameter_count("tree", 4, 3) == 12


def test_small_matrix_factorial_size_and_names():
    config = BenchmarkMatrixConfig(
        qubit_counts=(2, 4),
        depths=(1, 2),
        ansatz_families=("hardware_efficient", "tree"),
        cost_types=("local", "global"),
    )
    cases = build_benchmark_matrix(config)
    assert len(cases) == 2 * 2 * 2 * 2
    assert len({case.name for case in cases}) == len(cases)
    assert {case.cost_type for case in cases} == {"local", "global"}
    assert {case.ansatz_family for case in cases} == {"hardware_efficient", "tree"}


def test_benchmark_case_gradient_shape():
    case = build_benchmark_case("alternating", 2, 1, "local")
    theta = np.array([0.2, -0.4])
    gradient = case.gradient_fn(theta)
    assert gradient.shape == (2,)
    assert np.all(np.isfinite(gradient))


def test_global_cost_case_builds():
    case = build_benchmark_case("tree", 3, 2, "global")
    assert case.n_params == 6
    assert case.name == "tree__n3__d2__global"
