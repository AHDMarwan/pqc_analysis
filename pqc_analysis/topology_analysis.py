
# topology_analysis.py
import numpy as np
import torch
import pennylane as qml
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import scipy.linalg

def bures_distance(rho, sigma):
    """
    Compute the Bures distance between two density matrices.

    Args:
        rho (np.ndarray): Density matrix 1.
        sigma (np.ndarray): Density matrix 2.

    Returns:
        float: The Bures distance between rho and sigma.
    """
    sqrt_rho = scipy.linalg.sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = scipy.linalg.sqrtm(product)
    fidelity = np.real(np.trace(sqrt_product))
    fidelity = np.clip(fidelity, 0, 1)
    return np.sqrt(2 * (1 - fidelity))

def vectorized_bures_distances(density_matrices):
    """
    Compute the pairwise Bures distances between a set of density matrices.

    Args:
        density_matrices (np.ndarray): Array of shape (N, d, d) representing N density matrices.

    Returns:
        np.ndarray: Symmetric matrix (N x N) of pairwise Bures distances.
    """
    n = len(density_matrices)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = bures_distance(density_matrices[i], density_matrices[j])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def filter_diagram(diagram, threshold=0.05):
    """
    Filters a persistence diagram to remove topological features with small lifetimes.

    Args:
        diagram (np.ndarray): A persistence diagram (birth, death) pairs.
        threshold (float): Minimum lifetime (death - birth) for a feature to be considered.

    Returns:
        np.ndarray: Filtered persistence diagram.
    """
    return diagram[(diagram[:, 1] - diagram[:, 0]) > threshold]

def compute_entropy(diagrams):
    """
    Computes the topological entropy from persistence diagrams.
    Higher entropy suggests a richer and more complex topology.

    Args:
        diagrams (list): A list of persistence diagrams for different homology dimensions.

    Returns:
        list: A list of entropy values for each homology dimension.
    """
    entropy = []
    for H_k in diagrams:
        lifetimes = H_k[:, 1] - H_k[:, 0]
        lifetimes = lifetimes[lifetimes > 0]  # Consider only positive lifetimes
        if len(lifetimes) == 0:
            entropy.append(0.0)
        else:
            probs = lifetimes / np.sum(lifetimes)
            entropy.append(-np.sum(probs * np.log(np.maximum(probs, 1e-18))))
    return entropy

def pqc_topology_analysis(pqc, n_params, n_qubits, n_samples=100, max_dim=2,
                          init_strategy='normal', init_thetas=None,
                          entropy_threshold=0.05, show_plot=False, seed=42):
    """
    Performs topological analysis of a PQC's state space using persistent homology
    and Bures distance between density matrices.

    Args:
        pqc (callable): PennyLane quantum circuit function returning a state vector.
        n_params (int): Number of parameters in the PQC.
        n_qubits (int): Number of qubits.
        n_samples (int): Number of parameter samples for analysis.
        max_dim (int): Max homology dimension for persistent homology.
        init_strategy (str): Parameter initialization strategy ('normal' or 'uniform').
        init_thetas (np.ndarray, optional): Pre-defined parameter samples.
        entropy_threshold (float): Threshold for filtering persistence diagrams.
        show_plot (bool): If True, show persistence diagrams.
        seed (int): Random seed for reproducibility.

    Returns:
        list: Topological entropy values per homology dimension.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dev = qml.device("default.qubit", wires=n_qubits)
    pqc_qnode = qml.QNode(pqc, dev, interface="torch")

    density_matrices = []
    print(f"Generating {n_samples} quantum states (density matrices) for topological analysis...")
    for i in tqdm(range(n_samples)):
        theta = init_thetas[i] if init_thetas is not None else (
            np.pi * torch.randn(n_params, dtype=torch.float32) if init_strategy == 'normal'
            else torch.tensor(np.random.uniform(-np.pi, np.pi, n_params), dtype=torch.float32))
        try:
            state = pqc_qnode(theta).detach().numpy()
            # Convert pure state vector to density matrix: ρ = |ψ><ψ|
            rho = np.outer(state, state.conj())
            density_matrices.append(rho)
        except Exception as e:
            # print(f"[Error] Could not compute state for sample {i}: {e}")
            continue

    if not density_matrices:
        print("No states generated for topological analysis. Aborting.")
        return [0.0] * (max_dim + 1)

    print("Computing pairwise Bures distances...")
    D = vectorized_bures_distances(np.array(density_matrices))

    print(f"Computing persistent homology up to dimension {max_dim}...")
    raw_diagrams = ripser(D, distance_matrix=True, maxdim=max_dim)['dgms']

    print("Filtering persistence diagrams and computing topological entropy...")
    filtered = [filter_diagram(d, threshold=entropy_threshold) for d in raw_diagrams]
    entropy = compute_entropy(filtered)

    print("
--- PQC Topology Analysis Summary ---")
    for i, h in enumerate(entropy):
        print(f"Entropy H{i}: {h:.4f} - {'Rich topology' if h > 1.0 else 'Simple topology'}")

    if show_plot:
        print("
Displaying persistence diagrams (close plot to continue)...")
        plot_diagrams(raw_diagrams, show=True)
        print("Interpretation notes:")
        print("- Longer bars = more robust topological features.")
        print("- H0 bars relate to connected components.")
        print("- H1 bars relate to loops/holes.")
        print("- H2 bars relate to voids.")

    print("
Benchmarking Suggestion: Compare topological entropy across PQC architectures to assess their intrinsic topological complexity.")
    return entropy

