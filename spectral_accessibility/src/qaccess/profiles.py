from __future__ import annotations

from dataclasses import dataclass, asdict

from .circuits import CORE_FAMILIES, GENERIC_EXTENDED_FAMILIES, U1_EXTENDED_FAMILIES

EXTENDED = GENERIC_EXTENDED_FAMILIES + U1_EXTENDED_FAMILIES


@dataclass(frozen=True)
class Profile:
    name: str
    families: tuple[str, ...]
    n_values: tuple[int, ...]
    depth_factors: tuple[float, ...]
    generic_instances: int
    u1_instances: int
    tangents: int
    parameter_distributions: tuple[str, ...] = ("uniform_pi",)
    measurement_bases: tuple[str, ...] = ("computational",)
    bitflip_rates: tuple[float, ...] = (0.0,)
    readout_orders: tuple[int, ...] = (1, 2)
    spectrum: bool = False
    spectrum_random_projectors: int = 0
    tangent_prefixes: tuple[int, ...] = ()
    shard_size: int = 2
    save_tangent_rows: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, Profile] = {
    "smoke": Profile(name="smoke", families=CORE_FAMILIES, n_values=(4,), depth_factors=(1.0,), generic_instances=1, u1_instances=1, tangents=8, shard_size=1),
    "reproduce": Profile(name="reproduce", families=CORE_FAMILIES, n_values=(6, 8, 10), depth_factors=(6.0,), generic_instances=6, u1_instances=12, tangents=48, shard_size=2, save_tangent_rows=True),
    "pra_core": Profile(name="pra_core", families=EXTENDED, n_values=(6, 8, 10, 12), depth_factors=(6.0,), generic_instances=20, u1_instances=20, tangents=128, shard_size=4, save_tangent_rows=True),
    "pra_depth": Profile(name="pra_depth", families=EXTENDED, n_values=(6, 8, 10), depth_factors=(0.5, 1.0, 2.0, 4.0, 6.0, 8.0), generic_instances=8, u1_instances=8, tangents=96, shard_size=4),
    "pra_spectrum": Profile(name="pra_spectrum", families=("SU2-CNOT-line", "SU2-HaarU4-brickwork", "U1-RZ-XY-line", "U1-RZ-XY-ring"), n_values=(6, 8), depth_factors=(2.0, 6.0), generic_instances=8, u1_instances=8, tangents=1024, spectrum=True, spectrum_random_projectors=64, shard_size=1),
    "pra_robustness": Profile(name="pra_robustness", families=("RY-RZ-CZ-line", "SU2-HaarU4-brickwork", "U1-RZ-XY-line", "U1-RZ-XY-ring"), n_values=(8, 10), depth_factors=(6.0,), generic_instances=12, u1_instances=12, tangents=128, parameter_distributions=("uniform_pi", "normal_1", "normal_0p1"), measurement_bases=("computational", "x", "local_haar"), bitflip_rates=(0.0, 0.01, 0.05), shard_size=2),
    "pra_convergence": Profile(name="pra_convergence", families=("SU2-CNOT-line", "SU2-HaarU4-brickwork", "U1-RZ-XY-line", "U1-RZ-XY-ring"), n_values=(8, 10), depth_factors=(2.0, 6.0), generic_instances=10, u1_instances=10, tangents=256, tangent_prefixes=(32, 64, 128, 256), shard_size=1),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from exc


def all_profile_names() -> list[str]:
    return sorted(PROFILES)
