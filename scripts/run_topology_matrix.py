import argparse
import csv
import json
from pathlib import Path

from pqc_analysis.experimental import BenchmarkMatrixConfig, run_benchmark_matrix_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", default="artifacts/topology_matrix")
    return parser.parse_args()


def config_for_profile(profile):
    if profile == "pilot":
        return {
            "config": BenchmarkMatrixConfig(
                qubit_counts=(2, 4),
                depths=(1, 2),
                ansatz_families=("hardware_efficient", "tree"),
                cost_types=("local", "global"),
            ),
            "seeds": range(3),
            "geometry_samples": 12,
            "topology_samples": 24,
            "permutations": 500,
        }
    return {
        "config": BenchmarkMatrixConfig(),
        "seeds": range(20),
        "geometry_samples": 50,
        "topology_samples": 100,
        "permutations": 5000,
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    options = config_for_profile(args.profile)
    result = run_benchmark_matrix_experiment(
        options["config"],
        seeds=options["seeds"],
        geometry_samples=options["geometry_samples"],
        topology_samples=options["topology_samples"],
        topology_max_dim=1,
        permutations=options["permutations"],
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = result.records()
    correlations = result.correlation_records()
    write_csv(output / "records.csv", records)
    write_csv(output / "correlations.csv", correlations)
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, default=list), encoding="utf-8"
    )
    print(f"Wrote {len(records)} records and {len(correlations)} correlation rows to {output}")


if __name__ == "__main__":
    main()
