from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ResourceEstimate:
    gradient_method: str
    n_params: int
    circuit_evaluations_per_step: Optional[int]
    shots_per_step: Optional[int]
    total_circuit_evaluations: Optional[int] = None
    total_shots: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def _evaluations_per_gradient(n_params: int, method: str) -> Optional[int]:
    method = method.lower().replace("_", "-")
    if method in {"parameter-shift", "param-shift"}:
        return 2 * n_params
    if method in {"central-finite-diff", "central-difference"}:
        return 2 * n_params
    if method in {"forward-finite-diff", "forward-difference"}:
        return n_params + 1
    if method == "spsa":
        return 2
    if method in {"adjoint", "backprop", "backpropagation"}:
        # These are simulator/backpropagation strategies rather than a fixed
        # hardware circuit-evaluation count. Reporting a fake shot count would
        # be misleading, so the count is intentionally unspecified.
        return None
    raise ValueError(
        "gradient_method must be one of: parameter-shift, central-finite-diff, "
        "forward-finite-diff, spsa, adjoint, backprop"
    )


def estimate_gradient_resources(
    n_params: int,
    *,
    gradient_method: str = "parameter-shift",
    shots_per_circuit: Optional[int] = None,
    include_objective_evaluation: bool = False,
) -> ResourceEstimate:
    """Estimate per-optimization-step circuit and shot requirements.

    The estimate is deliberately algorithmic rather than backend-specific. It
    counts the standard number of shifted/perturbed circuit evaluations used by
    a gradient estimator. Hardware batching, commuting-observable grouping,
    parameter broadcasting, and backend caching can change wall-clock cost and
    are recorded as out-of-scope rather than hidden in the estimate.
    """
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if shots_per_circuit is not None and shots_per_circuit <= 0:
        raise ValueError("shots_per_circuit must be positive when provided")

    method = gradient_method.lower().replace("_", "-")
    gradient_evals = _evaluations_per_gradient(n_params, method)
    evaluations = None if gradient_evals is None else gradient_evals + int(include_objective_evaluation)
    shots = None
    if evaluations is not None and shots_per_circuit is not None:
        shots = int(evaluations * shots_per_circuit)

    return ResourceEstimate(
        gradient_method=method,
        n_params=int(n_params),
        circuit_evaluations_per_step=evaluations,
        shots_per_step=shots,
        metadata={
            "shots_per_circuit": shots_per_circuit,
            "include_objective_evaluation": bool(include_objective_evaluation),
            "gradient_evaluations_only": gradient_evals,
        },
    )


def estimate_training_resources(
    n_params: int,
    steps: int,
    *,
    gradient_method: str = "parameter-shift",
    shots_per_circuit: Optional[int] = None,
    include_objective_evaluation: bool = False,
) -> ResourceEstimate:
    """Scale a per-step resource estimate to a fixed number of training steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    per_step = estimate_gradient_resources(
        n_params,
        gradient_method=gradient_method,
        shots_per_circuit=shots_per_circuit,
        include_objective_evaluation=include_objective_evaluation,
    )
    total_evals = (
        None
        if per_step.circuit_evaluations_per_step is None
        else int(per_step.circuit_evaluations_per_step * steps)
    )
    total_shots = None if per_step.shots_per_step is None else int(per_step.shots_per_step * steps)
    return ResourceEstimate(
        gradient_method=per_step.gradient_method,
        n_params=per_step.n_params,
        circuit_evaluations_per_step=per_step.circuit_evaluations_per_step,
        shots_per_step=per_step.shots_per_step,
        total_circuit_evaluations=total_evals,
        total_shots=total_shots,
        metadata={**per_step.metadata, "steps": int(steps)},
    )
