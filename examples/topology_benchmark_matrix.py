"""Pilot and full topology/geometry/trainability benchmark matrices.

Run with the TDA extra installed:
    python -m pip install -e ".[tda]"
"""

from pqc_analysis.experimental import BenchmarkMatrixConfig, run_benchmark_matrix_experiment


# Pilot configuration for development and CI-scale experimentation.
pilot = BenchmarkMatrixConfig(
    qubit_counts=(2, 4),
    depths=(1, 2),
    ansatz_families=("hardware_efficient", "tree"),
    cost_types=("local", "global"),
)

result = run_benchmark_matrix_experiment(
    pilot,
    seeds=(0, 1, 2),
    geometry_samples=12,
    topology_samples=24,
    topology_max_dim=1,
    permutations=500,
)

print("Cases:", result.metadata["n_cases"])
print("Records:", result.metadata["n_records"])
print("\nFirst records")
for row in result.records()[:3]:
    print(row)

print("\nStrongest absolute correlations")
correlations = sorted(
    result.correlation_records(),
    key=lambda row: abs(float(row["correlation"])),
    reverse=True,
)
for row in correlations[:10]:
    print(row)


# Full exploratory matrix. Keep this commented unless you intentionally want
# the larger experiment: 3 families x 3 sizes x 3 depths x 2 costs = 54 cases
# before multiplying by seeds.
# full = BenchmarkMatrixConfig()
# full_result = run_benchmark_matrix_experiment(
#     full,
#     seeds=range(20),
#     geometry_samples=50,
#     topology_samples=100,
#     topology_max_dim=1,
#     permutations=5000,
# )
