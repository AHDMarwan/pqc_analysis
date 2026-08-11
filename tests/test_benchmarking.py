import pennylane as qml

import pqc_analysis as pqa


def test_benchmark_produces_flat_records_and_aggregates():
    def circuit(theta):
        qml.RY(theta[0], wires=0)
        return qml.state()

    result = pqa.benchmark(
        [pqa.PQCSpec("single_ry", circuit, n_qubits=1, n_params=1)],
        seeds=[1, 2],
        samples=4,
    )

    records = result.to_records()
    aggregate = result.aggregate()

    assert len(records) == 2
    assert {row["architecture"] for row in records} == {"single_ry"}
    assert "single_ry" in aggregate
    assert aggregate["single_ry"]["runs"] == 2.0
    assert "metric_rank_mean" in aggregate["single_ry"]


def test_benchmark_rejects_duplicate_names():
    def circuit(theta):
        qml.RY(theta[0], wires=0)
        return qml.state()

    specs = [
        pqa.PQCSpec("same", circuit, 1, 1),
        pqa.PQCSpec("same", circuit, 1, 1),
    ]

    try:
        pqa.benchmark(specs, seeds=[0], samples=2)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate benchmark names should fail")
