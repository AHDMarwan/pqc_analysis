from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pennylane as qml

from ..adapters import make_pennylane_gradient_fn
from .study import TopologyStudySpec


@dataclass(frozen=True)
class BenchmarkMatrixConfig:
    qubit_counts: Sequence[int] = (2, 4, 6)
    depths: Sequence[int] = (1, 2, 4)
    ansatz_families: Sequence[str] = ("hardware_efficient", "alternating", "tree")
    cost_types: Sequence[str] = ("local", "global")


@dataclass(frozen=True)
class BenchmarkMatrixCase:
    name: str
    ansatz_family: str
    n_qubits: int
    depth: int
    cost_type: str
    n_params: int
    state_circuit: Callable
    cost_circuit: Callable
    gradient_fn: Callable

    def to_study_spec(self) -> TopologyStudySpec:
        return TopologyStudySpec(
            name=self.name,
            circuit=self.state_circuit,
            n_qubits=self.n_qubits,
            n_params=self.n_params,
            gradient_fn=self.gradient_fn,
            metadata={
                "depth": self.depth,
                "ansatz_family_id": float({"hardware_efficient": 0, "alternating": 1, "tree": 2}[self.ansatz_family]),
                "cost_type_id": float({"local": 0, "global": 1}[self.cost_type]),
            },
        )


def _validate_config(config: BenchmarkMatrixConfig) -> None:
    if not config.qubit_counts or not config.depths or not config.ansatz_families or not config.cost_types:
        raise ValueError("all benchmark matrix dimensions must be non-empty")
    if any(int(n) < 2 for n in config.qubit_counts):
        raise ValueError("qubit_counts must contain values >= 2")
    if any(int(depth) <= 0 for depth in config.depths):
        raise ValueError("depths must be positive")
    supported_ansatz = {"hardware_efficient", "alternating", "tree"}
    supported_costs = {"local", "global"}
    if not set(config.ansatz_families).issubset(supported_ansatz):
        raise ValueError(f"ansatz_families must be chosen from {sorted(supported_ansatz)}")
    if not set(config.cost_types).issubset(supported_costs):
        raise ValueError(f"cost_types must be chosen from {sorted(supported_costs)}")


def parameter_count(ansatz_family: str, n_qubits: int, depth: int) -> int:
    if ansatz_family == "hardware_efficient":
        return 2 * n_qubits * depth
    if ansatz_family == "alternating":
        return n_qubits * depth
    if ansatz_family == "tree":
        # One trainable single-qubit rotation per wire and layer. The entangler
        # topology differs from the alternating ansatz and follows a binary tree.
        return n_qubits * depth
    raise ValueError(f"unsupported ansatz_family: {ansatz_family}")


def _apply_ansatz(theta, *, family: str, n_qubits: int, depth: int) -> None:
    cursor = 0
    for layer in range(depth):
        if family == "hardware_efficient":
            for wire in range(n_qubits):
                qml.RY(theta[cursor], wires=wire)
                qml.RZ(theta[cursor + 1], wires=wire)
                cursor += 2
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        elif family == "alternating":
            for wire in range(n_qubits):
                qml.RY(theta[cursor], wires=wire)
                cursor += 1
            parity = layer % 2
            for wire in range(parity, n_qubits - 1, 2):
                qml.CZ(wires=[wire, wire + 1])

        elif family == "tree":
            for wire in range(n_qubits):
                qml.RY(theta[cursor], wires=wire)
                cursor += 1
            stride = 1
            while stride < n_qubits:
                for control in range(0, n_qubits - stride, 2 * stride):
                    target = control + stride
                    if target < n_qubits:
                        qml.CNOT(wires=[control, target])
                stride *= 2
        else:
            raise ValueError(f"unsupported ansatz family: {family}")


def _global_observable(n_qubits: int):
    observable = qml.PauliZ(0)
    for wire in range(1, n_qubits):
        observable = observable @ qml.PauliZ(wire)
    return observable


def build_benchmark_case(
    ansatz_family: str,
    n_qubits: int,
    depth: int,
    cost_type: str,
    *,
    device_name: str = "default.qubit",
) -> BenchmarkMatrixCase:
    n_params = parameter_count(ansatz_family, n_qubits, depth)

    def state_circuit(theta):
        _apply_ansatz(theta, family=ansatz_family, n_qubits=n_qubits, depth=depth)
        return qml.state()

    def cost_circuit(theta):
        _apply_ansatz(theta, family=ansatz_family, n_qubits=n_qubits, depth=depth)
        if cost_type == "local":
            return qml.expval(qml.PauliZ(0))
        if cost_type == "global":
            return qml.expval(_global_observable(n_qubits))
        raise ValueError(f"unsupported cost_type: {cost_type}")

    gradient_fn = make_pennylane_gradient_fn(
        cost_circuit,
        n_qubits=n_qubits,
        device_name=device_name,
    )
    name = f"{ansatz_family}__n{n_qubits}__d{depth}__{cost_type}"
    return BenchmarkMatrixCase(
        name=name,
        ansatz_family=ansatz_family,
        n_qubits=n_qubits,
        depth=depth,
        cost_type=cost_type,
        n_params=n_params,
        state_circuit=state_circuit,
        cost_circuit=cost_circuit,
        gradient_fn=gradient_fn,
    )


def build_benchmark_matrix(
    config: BenchmarkMatrixConfig = BenchmarkMatrixConfig(),
    *,
    device_name: str = "default.qubit",
) -> Tuple[BenchmarkMatrixCase, ...]:
    """Build a reproducible factorial PQC benchmark matrix.

    The default grid spans three circuit families, three system sizes, three
    depths, and local/global observables. It is intentionally modest enough for
    exploratory simulation while preserving the factors needed to test whether
    topology contributes signal beyond architecture size and cost locality.
    """
    _validate_config(config)
    cases: List[BenchmarkMatrixCase] = []
    for family in config.ansatz_families:
        for n_qubits in config.qubit_counts:
            for depth in config.depths:
                for cost_type in config.cost_types:
                    cases.append(
                        build_benchmark_case(
                            family,
                            int(n_qubits),
                            int(depth),
                            cost_type,
                            device_name=device_name,
                        )
                    )
    return tuple(cases)


def matrix_to_study_specs(cases: Iterable[BenchmarkMatrixCase]) -> Tuple[TopologyStudySpec, ...]:
    return tuple(case.to_study_spec() for case in cases)
