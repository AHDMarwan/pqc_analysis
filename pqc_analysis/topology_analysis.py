import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import scipy.linalg
import torch
from persim import plot_diagrams
from ripser import ripser
from tqdm import tqdm


def bures_distance(rho, sigma):
    """Compute the Bures distance between two density matrices."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = scipy.linalg.sqrtm(product)
    fidelity = np.real(np.trace(sqrt_product))
    fidelity = np.clip(fidelity, 0, 1)
    return np.sqrt(2 * (1 - fidelity))


def vectorized_bures_distances(density_matrices):
    """Compute a symmetric matrix of pairwise Bures distances."""
    n = len(density_matrices)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = bures_distance(density_matrices[i], density_matrices[j])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def filter_diagram(diagram, threshold=0.05):
    """Remove persistence features with lifetime below ``threshold``."""
    return diagram[(diagram[:, 1] - diagram[:, 0]) > threshold]


def compute_entropy(diagrams):
    """Compute persistence entropy for each homology dimension."""
    entropy = []
    for homology_diagram in diagrams:
        lifetimes = homology_diagram[:, 1] - homology_diagram[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
        if len(lifetimes) == 0:
            entropy.append(0.0)
        else:
            probs = lifetimes / np.sum(lifetimes)
            entropy.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-18)))))
    return entropy


def pqc_topology_analysis(
    pqc,
    n_params,
    n_qubits,
    n_samples=100,
    max_dim=2,
    init_strategy="normal",
    init_thetas=None,
    entropy_threshold=0.05,
    show_plot=False,
    seed=42,
):
    """Analyze a PQC state space using Bures distances and persistent homology."""
    if n_params <= 0 or n_qubits <= 0:
        raise ValueError("n_params and n_qubits must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative")
    if init_strategy not in {"normal", "uniform"}:
        raise ValueError("init_strategy must be 'normal' or 'uniform'")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dev = qml.device("default.qubit", wires=n_qubits)
    pqc_qnode = qml.QNode(pqc, dev, interface="torch")

    density_matrices = []
    print(f"Generating {n_samples} quantum states for topological analysis...")
    for i in tqdm(range(n_samples)):
        if init_thetas is not None:
            theta = torch.as_tensor(init_thetas[i], dtype=torch.float64)
        elif init_strategy == "normal":
            theta = np.pi * torch.randn(n_params, dtype=torch.float64)
        else:
            theta = torch.tensor(
                np.random.uniform(-np.pi, np.pi, n_params), dtype=torch.float64
            )

        try:
            state = pqc_qnode(theta).detach().cpu().numpy()
            density_matrices.append(np.outer(state, state.conj()))
        except (RuntimeError, ValueError, TypeError):
            continue

    if not density_matrices:
        print("No states generated for topological analysis. Aborting.")
        return [0.0] * (max_dim + 1)

    print("Computing pairwise Bures distances...")
    distance_matrix = vectorized_bures_distances(np.asarray(density_matrices))

    print(f"Computing persistent homology up to dimension {max_dim}...")
    raw_diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=max_dim)["dgms"]

    print("Filtering persistence diagrams and computing topological entropy...")
    filtered = [filter_diagram(diagram, threshold=entropy_threshold) for diagram in raw_diagrams]
    entropy = compute_entropy(filtered)

    print("\n--- PQC Topology Analysis Summary ---")
    for dimension, value in enumerate(entropy):
        label = "Rich topology" if value > 1.0 else "Simple topology"
        print(f"Entropy H{dimension}: {value:.4f} - {label}")

    if show_plot:
        print("\nDisplaying persistence diagrams (close plot to continue)...")
        plot_diagrams(raw_diagrams, show=True)
        print("Interpretation notes:")
        print("- Longer bars = more robust topological features.")
        print("- H0 bars relate to connected components.")
        print("- H1 bars relate to loops/holes.")
        print("- H2 bars relate to voids.")

    print(
        "\nBenchmarking suggestion: compare persistence entropy across PQC "
        "architectures under the same sampling protocol."
    )
    return entropy
