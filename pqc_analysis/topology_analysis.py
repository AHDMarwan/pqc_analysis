
import numpy as np
import torch
import pennylane as qml
from tqdm import tqdm
from ripser import ripser

def vectorized_fubini_study_distances(states):
    inner_products = states @ states.conj().T
    clipped = np.clip(np.abs(inner_products), 0.0, 1.0)
    distances = np.arccos(clipped)
    np.fill_diagonal(distances, 0.0)
    return distances

def compute_entropy(diagrams):
    entropy = []
    for H_k in diagrams:
        mask = ~np.isinf(H_k[:, 1])
        lifetimes = H_k[mask, 1] - H_k[mask, 0]
        lifetimes = lifetimes[lifetimes > 0]
        if len(lifetimes) == 0:
            entropy.append(0.0)
        else:
            probs = lifetimes / np.sum(lifetimes)
            entropy.append(-np.sum(probs * np.log(np.maximum(probs, 1e-18))))
    return entropy

def _generate_states_for_tda(pqc_qnode, n_params, num_samples, init_strategy='normal', init_thetas=None, **kwargs):
    states = []
    for i in tqdm(range(num_samples), desc="Generating PQC states"):
        if init_thetas is not None:
            theta = torch.tensor(init_thetas[i], dtype=torch.float32)
        elif init_strategy == 'uniform':
            theta = torch.tensor(np.random.uniform(-np.pi, np.pi, n_params), dtype=torch.float32)
        elif init_strategy == 'normal':
            theta = np.pi * torch.randn(n_params)
        elif callable(init_strategy):
            theta = torch.tensor(init_strategy(n_params), dtype=torch.float32)
        else:
            raise ValueError("init_strategy must be 'uniform', 'normal', or a callable")

        state = pqc_qnode(theta, **kwargs).detach().numpy()
        states.append(state)
    return np.array(states)

def pqc_topology_analysis(pqc, n_params, n_qubits, n_samples=100,
                          max_dim=2, init_strategy='normal', init_thetas=None, **kwargs):
    if not callable(pqc):
        raise TypeError("pqc must be callable.")
    if not isinstance(n_params, int) or n_params <= 0:
        raise ValueError("n_params must be a positive integer.")

    dev = qml.device("default.qubit", wires=n_qubits)
    pqc_qnode = qml.QNode(pqc, dev, interface="torch")

    print(f"\n--- Persistent Homology for {n_qubits} Qubits ---")

    states = _generate_states_for_tda(pqc_qnode, n_params, n_samples,
                                      init_strategy=init_strategy,
                                      init_thetas=init_thetas,
                                      **kwargs)

    D = vectorized_fubini_study_distances(states)
    diagrams = ripser(D, distance_matrix=True, maxdim=max_dim)['dgms']
    entropy = compute_entropy(diagrams)

    for i, h in enumerate(entropy):
        print(f"\n  Entropy H{i}: {h:.6f}")

    return entropy
