# pqc_analysis_/geometry_analysis.py

import numpy as np
import torch
import pennylane as qml
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh # Explicitly import eigvalsh for clarity


# Helper function to compute von Neumann entanglement entropy for a bipartite split
def compute_entanglement_entropy(state_vector, n_qubits):
    """
    Computes the von Neumann entanglement entropy for a bipartite split (e.g., qubit 0 vs rest).
    This is calculated for the reduced density matrix of the first qubit.
    """
    if n_qubits < 2:
        return 0.0 # Entanglement entropy is not well-defined for single qubit

    # Ensure state_vector is a numpy array (if it came as a torch.Tensor from qnode)
    if isinstance(state_vector, torch.Tensor):
        state_vector = state_vector.detach().numpy()
    
    # Check for trivial state (e.g., all zeros if circuit failed or state is |0...0>)
    if np.allclose(state_vector, 0):
        return 0.0

    density_matrix = np.outer(state_vector, state_vector.conj())

    # Check if the density_matrix is valid (e.g., not all zeros)
    if np.allclose(density_matrix, 0):
        return 0.0

    try:
        # qml.math.partial_trace expects a density matrix, wires to trace out, and full subsystem dimensions
        # Here, we trace out all qubits except the first one (wire 0)
        wires_to_trace_out = list(range(1, n_qubits)) 
        rho_A = qml.math.partial_trace(density_matrix, wires=wires_to_trace_out, dim=[2]*n_qubits)
    except Exception as e:
        # print(f"DEBUG: Error during partial trace in EE: {e}") # Uncomment for debugging
        return 0.0 # Return 0.0 if partial trace fails

    # Calculate eigenvalues of the reduced density matrix
    # eigvalsh is for Hermitian matrices
    eigenvalues = eigvalsh(rho_A)
    
    # Filter out zero or negative eigenvalues due to numerical precision
    eigenvalues = eigenvalues[eigenvalues > 1e-12] # Small epsilon to handle numerical zeros

    if len(eigenvalues) == 0:
        return 0.0 # No positive eigenvalues, entropy is 0

    # Compute von Neumann entropy: S = -sum(lambda * log2(lambda))
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    # Ensure entropy is real and non-negative
    return np.real(entropy) if np.real(entropy) >= 0 else 0.0


def compute_fs_metric(pqc_state_fn, theta, regularization_eps=1e-10):
    """
    Computes the Fubini-Study metric tensor for a given PQC state function.
    This function uses PennyLane's built-in `qml.metric_tensor` with `approx="block-diag"`.

    Args:
        pqc_state_fn (qml.QNode): PennyLane QNode representing the PQC state preparation.
        theta (torch.Tensor or np.ndarray): Parameters of the PQC.
        regularization_eps (float): Small epsilon for numerical stability.

    Returns:
        np.ndarray: The Fubini-Study metric tensor (shape: n_params x n_params).
    """
    theta_tensor = torch.tensor(theta, requires_grad=True, dtype=torch.float32) # Keep float32 as per your original
    metric_fn = qml.metric_tensor(pqc_state_fn, approx="block-diag") # Keep block-diag as per your original
    g = metric_fn(theta_tensor)
    g_np = g.detach().numpy()

    # Regularize small values for numerical stability as per your original
    g_np[g_np < regularization_eps] = 0.0 
    return g_np


def pqc_geometry_analysis(pqc_circuit_fn, n_params, n_qubits, n_samples=500,
                          init_strategy='uniform', init_thetas=None,
                          seed=42, noisy=True, show_volume_density_plot=False, **kwargs):
    """
    Performs comprehensive geometric analysis of a PQC's state manifold,
    including Quantum Volume, Effective Dimension, Entanglement Entropy,
    Metric Rank, Inverse Condition Number (Gamma), and Singular Metric Ratio.
    Also provides a distribution of local volume densities.

    Args:
        pqc_circuit_fn (callable): A Python function that defines the PQC circuit.
                                    It should take `params` and `wires` as arguments.
                                    Example: `def my_circuit(params, wires): ...`
        n_params (int): Total number of parameters in the PQC.
        n_qubits (int): Number of active qubits in the PQC's core circuit (e.g., 0 to n_qubits-1).
                        Note: The device will be created with `n_qubits + 1` wires.
        n_samples (int): Number of random parameter samples to analyze.
        init_strategy (str): Parameter initialization strategy ('uniform' or 'normal').
        init_thetas (np.ndarray, optional): Predefined array of parameters for sampling.
        seed (int): Random seed for reproducibility.
        noisy (bool): Use 'default.mixed' device if True; else 'default.qubit'.
        use_log_volume (bool): If True, compute and report log quantum volume.
        **kwargs: Additional arguments for future extensions (e.g., regularization_eps).

    Returns:
        dict: A dictionary containing the average geometric and entanglement metrics over all samples.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed) # Ensure torch is also seeded for parameter generation

    # --- QNode and Device Definition (as per your working code) ---
    # The device and QNode are defined internally in this function.
    dev_type = "default.mixed" if noisy else "default.qubit"
    # Use n_qubits + 1 wires as per your working setup
    dev = qml.device(dev_type, wires=n_qubits + 1) 

    # The QNode is created from the user-provided circuit function
    pqc_qnode = qml.QNode(pqc_circuit_fn, dev, interface="torch", diff_method="parameter-shift") 
    # --- End QNode Definition ---


    # Lists to collect metrics from each sample
    volumes, log_volumes, ranks, gammas = [], [], [], []
    singular_count, redundant_count = 0, 0
    effective_dimensions = [] # Added metric
    entanglement_entropies = [] # Added metric
    all_volume_densities = [] # For the histogram

    # Extract regularization_eps from kwargs or use a default
    regularization_eps = kwargs.get('regularization_eps', 1e-12) # Use 1e-12 as default for calculations


    print(f"Starting geometry analysis with {n_samples} samples on {dev_type} device...")
    for i in tqdm(range(n_samples)):
        # Initialize parameters
        if init_thetas is not None:
            theta = init_thetas[i]
        else:
            if init_strategy == 'uniform':
                theta = np.random.uniform(-np.pi, np.pi, n_params)
            else: # normal initialization
                theta = np.random.normal(0, np.pi, n_params)

        try:
            # Compute Fubini-Study metric tensor g(theta) using the dedicated function
            # Pass the internal QNode and the current parameters
            g = compute_fs_metric(pqc_qnode, theta, regularization_eps) # Pass regularization_eps to compute_fs_metric

            # --- Derived Metrics from G ---
            # Matrix rank with a tolerance for numerical noise
            rank = np.linalg.matrix_rank(g, tol=regularization_eps * 10) # Use a slightly larger tolerance for rank

            # Eigenvalues clipped to zero lower bound to avoid small negatives, critical for log
            eigvals = np.clip(np.linalg.eigvalsh(g), 0, None) # Corrected typo: eigvalsh
            
            # Filter out zero eigenvalues for robust calculations
            positive_eigvals = eigvals[eigvals > regularization_eps]

            # Inverse condition number gamma: min(eig)/max(eig)
            if len(positive_eigvals) > 0:
                gamma = np.min(positive_eigvals) / np.max(positive_eigvals)
            else:
                gamma = 0.0 # All eigenvalues are zero or near-zero

            # Quantum Volume (from determinant of G)
            if rank == n_params and np.linalg.det(g) > 0: # Only full rank matrices have non-zero determinant
                # Use log-determinant for numerical stability
                log_det_g = np.linalg.slogdet(g)[1] 
                volume = np.exp(0.5 * log_det_g)
                log_volume = 0.5 * log_det_g
            else: # If singular or det <= 0, volume is 0
                volume = 0.0
                log_volume = -np.inf

            # Effective Dimension
            if len(positive_eigvals) > 0:
                effective_dim = np.sum(positive_eigvals / (positive_eigvals + 1.0))
            else:
                effective_dim = 0.0

            # Entanglement Entropy
            # Get the state vector (torch.Tensor) for the current parameters
            # This requires running the QNode again to get the state, as compute_fs_metric only returns G
            psi_torch = pqc_qnode(torch.tensor(theta, dtype=torch.float64)) # Use float64 for state vector computation
            ee = compute_entanglement_entropy(psi_torch, n_qubits) # n_qubits here refers to active qubits for EE

            # --- Collect Results ---
            volumes.append(volume)
            log_volumes.append(log_volume)
            ranks.append(rank)
            gammas.append(gamma)
            effective_dimensions.append(effective_dim)
            entanglement_entropies.append(ee)
            all_volume_densities.append(volume) # For histogram

            singular_count += int(rank < n_params or volume <= 0) # Count as singular if not full rank or zero volume
            redundant_count += int(rank < n_params)

        except Exception as e:
            # If an error occurs, treat all metrics as failed/zero
            # print(f"DEBUG: Error processing sample {i}: {e}") # Uncomment for debugging
            ranks.append(0)
            gammas.append(0.0)
            volumes.append(0)
            log_volumes.append(-np.inf)
            effective_dimensions.append(0.0)
            entanglement_entropies.append(0.0)
            all_volume_densities.append(0.0)
            singular_count += 1
            redundant_count += 1

    # --- Calculate Average Metrics ---
    if n_samples == 0:
        avg_metrics = {
            'Mean Quantum Volume': 0.0,
            'Mean Log Quantum Volume': -np.inf,
            'Avg Metric Rank': 0.0,
            'Redundant Param Ratio': 0.0,
            'Avg Gamma (Cond # Inv)': 0.0,
            'Singular Metric Ratio': 0.0,
            'Avg Effective Dimension': 0.0,
            'Avg Entanglement Entropy': 0.0,
            'Num Params': n_params,
            'Num Qubits': n_qubits
        }
    else:
        # Only average finite log volumes, otherwise set to -inf
        finite_log_volumes = [lv for lv in log_volumes if np.isfinite(lv)]
        mean_log_volume = np.mean(finite_log_volumes) if finite_log_volumes else -np.inf

        avg_metrics = {
            # Note: The mean quantum volume is currently the average of local volume densities.
            # Your original code applies a normalization: * (2 * np.pi) ** n_params
            # I will keep this for consistency with your provided working reference,
            # but usually, the direct mean of volumes is the average local density.
            'Mean Quantum Volume': np.mean(volumes) * (2 * np.pi) ** n_params, 
            'Mean Log Quantum Volume': mean_log_volume,
            'Avg Metric Rank': np.mean(ranks),
            'Redundant Param Ratio': redundant_count / n_samples,
            'Avg Gamma (Cond # Inv)': np.mean(gammas),
            'Singular Metric Ratio': singular_count / n_samples,
            'Avg Effective Dimension': np.mean(effective_dimensions),
            'Avg Entanglement Entropy': np.mean(entanglement_entropies),
            'Num Params': n_params,
            'Num Qubits': n_qubits
        }

    # Volume density histogram plot
    if show_volume_density_plot and len(all_volume_densities) > 0:
        # Filter out zero volumes for better visualization if many are zero
        positive_volume_densities = [v for v in all_volume_densities if v > 0]
        
        if len(positive_volume_densities) > 0:
            plt.figure(figsize=(8,5))
            plt.hist(positive_volume_densities, bins=30, color='navy', alpha=0.75, edgecolor='k')
            plt.title(f"Volume Density Distribution (Q={n_qubits}, P={n_params})")
            plt.xlabel(r"Volume Density $\sqrt{\det g(\theta)}$")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            # plt.savefig(f"volume_density_Q{n_qubits}_P{n_params}.png") # Example save
            plt.show()
        else:
            print("No positive volume densities to plot for the histogram.")
    def _pqc_geometry_analysis_notes():
        print("\nInterpretation Notes:")
        print("- 'Mean Quantum Volume': Expressibility measure of PQC state manifold volume (normalized).")
        print("- 'Avg Effective Dimension': Effective number of parameters contributing to state changes.")
        print("- 'Avg Entanglement Entropy': Average entanglement generated by the PQC (for qubit 0 vs rest).")
        print("- 'Avg Metric Rank': Effective number of independent parameters (similar to effective dimension).")
        print("- 'Redundant Param Ratio': Fraction of samples with redundant parameters (rank < n_params).")
        print("- 'Avg Gamma (Cond # Inv)': Trainability proxy, higher values indicate less barren plateaus.")
        print("- 'Singular Metric Ratio': Fraction of samples with singular metric tensors (potential barren plateaus).")
        print("- Volume Density Distribution: Shows local sensitivity variation in parameter space.")
        print("Suggestion: Compare these metrics across PQC designs to evaluate architecture quality.")

    # Attach it as a function attribute
    pqc_geometry_analysis.note = _pqc_geometry_analysis_notes
    return avg_metrics
