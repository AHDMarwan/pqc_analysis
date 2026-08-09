from __future__ import annotations

import json
from dataclasses import dataclass, asdict

from .circuits import family_group
from .profiles import Profile


@dataclass(frozen=True)
class Task:
    task_id: str
    profile: str
    family: str
    n: int
    depth_factor: float
    instance_start: int
    instance_stop: int
    tangents: int
    parameter_distribution: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_tasks(profile: Profile) -> list[Task]:
    tasks: list[Task] = []
    for family in profile.families:
        instances = profile.u1_instances if family_group(family) == "u1" else profile.generic_instances
        for n in profile.n_values:
            for depth_factor in profile.depth_factors:
                for init in profile.parameter_distributions:
                    for start in range(0, instances, profile.shard_size):
                        stop = min(instances, start + profile.shard_size)
                        slug = family.replace("/", "-")
                        task_id = f"{profile.name}__{slug}__n{n}__f{depth_factor:g}__{init}__i{start}-{stop-1}"
                        tasks.append(Task(task_id=task_id, profile=profile.name, family=family, n=n, depth_factor=float(depth_factor), instance_start=start, instance_stop=stop, tangents=profile.tangents, parameter_distribution=init))
    return tasks


def github_matrix(profile: Profile) -> str:
    return json.dumps({"include": [t.to_dict() for t in build_tasks(profile)]}, separators=(",", ":"))
