
import numpy as np
import torch
import pennylane as qml
from tqdm import tqdm
from tabulate import tabulate

def compute_fs_metric(pqc_state_fn, theta):
    theta_tensor = torch.tensor(theta, requires_grad=True, dtype=torch.float32)
    metric_fn = qml.metric_tensor(pqc_state_fn, approx=None)
    g = metric_fn(theta_tensor)
    return g.detach().numpy()

def pqc_geometry_analysis(pqc, n_params, n_qubits, n_samples=100,
                          init_strategy='uniform', init_thetas=None, **kwargs):
    if not callable(pqc):
        raise TypeError("pqc must be callable.")
    if not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("n_params must be a positive integer.")
    if not isinstance(n_qubits, int) or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer.")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if init_thetas is not None:
        if not (isinstance(init_thetas, np.ndarray) and init_thetas.shape == (n_samples, n_params)):
            raise ValueError(f"init_thetas must be a numpy array of shape ({n_samples}, {n_params})")

    dev = qml.device("default.qubit", wires=n_qubits + 1)
    pqc_qnode = qml.QNode(pqc, dev, interface="torch", diff_method="parameter-shift")

    volumes, ranks, gammas = [], [], []
    singular_count, redundant_count = 0, 0

    for i in tqdm(range(n_samples), desc=f"PQC Geometry (Q={n_qubits}, P={n_params})"):
        if init_thetas is not None:
            theta = init_thetas[i]
        elif init_strategy == "uniform":
            theta = np.random.uniform(-np.pi, np.pi, n_params)
        elif init_strategy == "normal":
            theta = np.random.normal(0, np.pi, n_params)
        elif callable(init_strategy):
            theta = init_strategy(n_params)
        else:
            raise ValueError("init_strategy must be 'uniform', 'normal', or a callable")

        try:
            g = compute_fs_metric(pqc_qnode, theta)
            rank = np.linalg.matrix_rank(g, tol=1e-7)
            eigenvalues = np.clip(np.linalg.eigvalsh(g), 0, None)
            gamma = np.min(eigenvalues) / np.max(eigenvalues) if np.max(eigenvalues) > 1e-10 else 0.0

            ranks.append(rank)
            gammas.append(gamma)
            redundant_count += int(rank < n_params)

            det = np.linalg.det(g)
            volumes.append(np.sqrt(det) if det > 0 else 0)
            singular_count += int(det <= 0)

        except Exception as e:
            print(f"[Error] θ={theta.round(3)}: {e}")
            ranks.append(0)
            gammas.append(0.0)
            singular_count += 1
            redundant_count += 1

    avg_local_vol = np.mean(volumes)
    total_volume = avg_local_vol * (2 * np.pi) ** n_params
    avg_rank = np.mean(ranks)
    avg_gamma = np.mean(gammas)
    prop_redundant = redundant_count / n_samples

    summary = {
        "Total Volume": total_volume,
        "Avg Local Volume": avg_local_vol,
        "Avg Metric Rank": avg_rank,
        "Redundant Param Ratio": prop_redundant,
        "Avg Gamma (Cond # Inv)": avg_gamma,
        "Num Params": n_params,
        "Num Qubits": n_qubits,
    }

    print(f"\n[Geometry Summary]")
    print(tabulate(summary.items(), headers=["Metric", "Value"], tablefmt="fancy_grid"))
    return summary
