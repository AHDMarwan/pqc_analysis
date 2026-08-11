from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .topology_metrics import flatten_persistence_summaries


@dataclass(frozen=True)
class TopologyStateSpaceResult:
    """Persistent-homology analysis of a sampled pure-state PQC manifold."""

    diagrams: Sequence[np.ndarray]
    metrics: Dict[str, float]
    distance_matrix: np.ndarray
    parameter_samples: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


def pure_state_bures_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Bures distance for normalized pure states.

    For pure states, sqrt fidelity equals ``|<left|right>|``, avoiding density
    matrices and matrix square roots.
    """
    left = np.asarray(left, dtype=complex).reshape(-1)
    right = np.asarray(right, dtype=complex).reshape(-1)
    if left.size != right.size or left.size == 0:
        raise ValueError("states must be non-empty vectors with equal dimension")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("state vectors must have non-zero norm")
    overlap = abs(np.vdot(left / left_norm, right / right_norm))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - overlap))))


def pairwise_pure_state_bures(states: Sequence[np.ndarray]) -> np.ndarray:
    states = [np.asarray(state, dtype=complex).reshape(-1) for state in states]
    if len(states) < 2:
        raise ValueError("at least two states are required")
    dimension = states[0].size
    if any(state.size != dimension for state in states):
        raise ValueError("all state vectors must have equal dimension")
    distances = np.zeros((len(states), len(states)), dtype=float)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            distance = pure_state_bures_distance(states[i], states[j])
            distances[i, j] = distances[j, i] = distance
    return distances


def analyze_topological_state_space(
    circuit: Callable,
    *,
    n_qubits: int,
    n_params: int,
    samples: int = 100,
    max_dim: int = 2,
    seed: int = 0,
    parameter_samples: Optional[np.ndarray] = None,
    init_strategy: str = "uniform",
    device_name: str = "default.qubit",
) -> TopologyStateSpaceResult:
    """Sample a PQC pure-state manifold and compute persistent homology.

    ``ripser`` is imported lazily; install ``pqc_analysis[tda]`` to use this
    function. The circuit must return ``qml.state()`` and accept one parameter
    vector.
    """
    if n_qubits <= 0 or n_params <= 0 or samples <= 1:
        raise ValueError("n_qubits/n_params must be positive and samples must exceed one")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")
    if init_strategy not in {"uniform", "normal"}:
        raise ValueError("init_strategy must be 'uniform' or 'normal'")

    try:
        import pennylane as qml
        from ripser import ripser
    except ImportError as exc:
        raise ImportError("Topological state-space analysis requires the 'tda' extra") from exc

    if parameter_samples is None:
        rng = np.random.default_rng(seed)
        if init_strategy == "uniform":
            theta_samples = rng.uniform(-np.pi, np.pi, size=(samples, n_params))
        else:
            theta_samples = rng.normal(0.0, np.pi, size=(samples, n_params))
    else:
        theta_samples = np.asarray(parameter_samples, dtype=float)
        if theta_samples.ndim != 2 or theta_samples.shape[1] != n_params:
            raise ValueError("parameter_samples must have shape (n_samples, n_params)")
        if theta_samples.shape[0] < 2:
            raise ValueError("parameter_samples must contain at least two rows")

    device = qml.device(device_name, wires=n_qubits)
    qnode = qml.QNode(circuit, device, interface=None)
    states = [np.asarray(qnode(theta), dtype=complex) for theta in theta_samples]
    distances = pairwise_pure_state_bures(states)
    diagrams = ripser(distances, distance_matrix=True, maxdim=max_dim)["dgms"]
    metrics = flatten_persistence_summaries(diagrams)

    return TopologyStateSpaceResult(
        diagrams=tuple(np.asarray(diagram, dtype=float) for diagram in diagrams),
        metrics=metrics,
        distance_matrix=distances,
        parameter_samples=theta_samples,
        metadata={
            "n_qubits": int(n_qubits),
            "n_params": int(n_params),
            "n_samples": int(theta_samples.shape[0]),
            "max_dim": int(max_dim),
            "seed": int(seed),
            "init_strategy": init_strategy,
            "distance": "pure-state Bures",
            "experimental": True,
        },
    )
