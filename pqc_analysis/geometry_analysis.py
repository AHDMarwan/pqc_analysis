
# geometry_analysis.py
import numpy as np
import torch
import pennylane as qml
from tqdm import tqdm
from tabulate import tabulate

def compute_fs_metric(pqc_state_fn, theta, regularization_eps=1e-10):
    """
    Computes the Fubini-Study metric tensor for a given PQC state function.

    Args:
        pqc_state_fn (qml.QNode): PennyLane QNode representing the PQC state preparation.
        theta (torch.Tensor or np.ndarray): Parameters of the PQC.
        regularization_eps (float): Small epsilon for numerical stability to avoid
                                   division by zero or issues with near-zero eigenvalues/determinants.

    Returns:
        np.ndarray: The Fubini-Study metric tensor (shape: n_params x n_params).
    """
    theta_tensor = torch.tensor(theta, requires_grad=True, dtype=torch.float32)
    metric_fn = qml.metric_tensor(pqc_state_fn, approx=None)
    g = metric_fn(theta_tensor)
    g_np = g.detach().numpy()

    # Regularize small values for numerical stability
    g_np[g_np < regularization_eps] = 0.0
    return g_np


def pqc_geometry_analysis(pqc, n_params, n_qubits, n_samples=100,
                          init_strategy='uniform', init_thetas=None,
                          seed=42, noisy=True, use_log_volume=False, **kwargs):
    """
    Performs geometric analysis of a PQC using the Fubini-Study metric.

    Calculates metrics such as quantum volume, metric rank, and inverse condition number (gamma)
    to assess expressibility, trainability, and parameter redundancy.

    Args:
        pqc (callable): PennyLane quantum circuit function returning qml.state().
        n_params (int): Number of parameters in the PQC.
        n_qubits (int): Number of qubits in the PQC.
        n_samples (int): Number of random parameter samples to analyze. Defaults to 100.
        init_strategy (str): Parameter initialization strategy ('uniform' or 'normal'). Defaults to 'uniform'.
        init_thetas (np.ndarray, optional): Predefined array of parameters for sampling.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
        noisy (bool): Use 'default.mixed' device (mixed states) if True; else 'default.qubit' (pure states).
        use_log_volume (bool): If True, compute and report log quantum volume. Defaults to False.
        **kwargs: Additional args for future extensions.

    Returns:
        dict: Dictionary containing computed geometric analysis summary.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Use 'default.mixed' device for noisy/mixed states simulation, else pure states device
    dev_type = "default.mixed" if noisy else "default.qubit"
    dev = qml.device(dev_type, wires=n_qubits + 1)
    pqc_qnode = qml.QNode(pqc, dev, interface="torch", diff_method="parameter-shift")

    volumes, log_volumes, ranks, gammas = [], [], [], []
    singular_count, redundant_count = 0, 0

    print(f"Starting geometry analysis with {n_samples} samples on {dev_type} device...")
    for i in tqdm(range(n_samples)):
        # Initialize parameters
        if init_thetas is not None:
            theta = init_thetas[i]
        else:
            if init_strategy == 'uniform':
                theta = np.random.uniform(-np.pi, np.pi, n_params)
            else:  # normal initialization
                theta = np.random.normal(0, np.pi, n_params)

        try:
            # Compute Fubini-Study metric tensor g(theta)
            g = compute_fs_metric(pqc_qnode, theta)

            # Matrix rank with a tolerance for numerical noise
            rank = np.linalg.matrix_rank(g, tol=1e-7)

            # Eigenvalues clipped to zero lower bound to avoid small negatives
            eigvals = np.clip(np.linalg.eigvalsh(g), 0, None)

            # Inverse condition number gamma: min(eig)/max(eig)
            gamma = np.min(eigvals) / np.max(eigvals) if np.max(eigvals) > 1e-10 else 0.0

            det = np.linalg.det(g)
            # Use log-eigenvalues for numerical stability (log-det)
            log_det = np.sum(np.log(np.maximum(eigvals, 1e-12)))

            volumes.append(np.sqrt(det) if det > 0 else 0)
            log_volumes.append(0.5 * log_det if det > 0 else -np.inf)
            ranks.append(rank)
            gammas.append(gamma)

            singular_count += int(det <= 0)
            redundant_count += int(rank < n_params)

        except Exception as e:
            # If an error occurs, treat metric as singular and redundant
            ranks.append(0)
            gammas.append(0.0)
            volumes.append(0)
            log_volumes.append(-np.inf)
            singular_count += 1
            redundant_count += 1

    # Normalize volume by parameter space size (2*pi)^n_params, assuming parameter domain [-pi, pi]
    mean_volume = np.mean(volumes) * (2 * np.pi) ** n_params
    mean_log_volume = np.mean(log_volumes) + n_params * np.log(2 * np.pi)

    summary = {
        "Mean Quantum Volume": mean_volume,
        "Mean Log Quantum Volume": mean_log_volume,
        "Avg Metric Rank": np.mean(ranks),
        "Redundant Param Ratio": redundant_count / n_samples,
        "Avg Gamma (Cond # Inv)": np.mean(gammas),
        "Num Params": n_params,
        "Num Qubits": n_qubits,
        "Singular Metric Ratio": singular_count / n_samples,
    }

    print("
--- PQC Geometry Analysis Summary ---")
    print(tabulate(summary.items(), headers=["Metric", "Value"], tablefmt="fancy_grid"))
    print("
Interpretation Notes:")
    print("- 'Mean Quantum Volume': Expressibility measure of PQC state manifold volume.")
    print("- 'Avg Metric Rank': Effective number of independent parameters.")
    print("- 'Redundant Param Ratio': Fraction of samples with redundant parameters (rank < n_params).")
    print("- 'Avg Gamma (Cond # Inv)': Trainability proxy, higher values indicate less barren plateaus.")
    print("- 'Singular Metric Ratio': Fraction of samples with singular metric tensors (potential barren plateaus).")
    print("
Suggestion: Compare these metrics across PQC designs to evaluate architecture quality.")

    return summary

