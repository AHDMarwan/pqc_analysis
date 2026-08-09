from __future__ import annotations

import hashlib


def stable_seed(label: str, master_seed: int = 20260809) -> int:
    """Deterministic 64-bit seed, stable across Python processes/platforms."""
    payload = f"{master_seed}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
