from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_results
from .controls import anisotropic_relative_orientation_control, isotropic_controls
from .experiment import run_task
from .planning import build_tasks, github_matrix
from .profiles import all_profile_names, get_profile
from .reproduction import compare_reproduction


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qaccess")
    sp = p.add_subparsers(dest="cmd", required=True)
    plan = sp.add_parser("plan")
    plan.add_argument("--profile", required=True, choices=all_profile_names())
    plan.add_argument("--github-matrix", action="store_true")
    plan.add_argument("--output")
    run = sp.add_parser("run-task")
    run.add_argument("--profile", required=True, choices=all_profile_names())
    run.add_argument("--family", required=True)
    run.add_argument("--n", type=int, required=True)
    run.add_argument("--depth-factor", type=float, required=True)
    run.add_argument("--instance-start", type=int, required=True)
    run.add_argument("--instance-stop", type=int, required=True)
    run.add_argument("--tangents", type=int, required=True)
    run.add_argument("--parameter-distribution", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--master-seed", type=int, default=20260809)
    ana = sp.add_parser("analyze")
    ana.add_argument("--input", type=Path, required=True)
    ana.add_argument("--output", type=Path, required=True)
    ctr = sp.add_parser("controls")
    ctr.add_argument("--output", type=Path, required=True)
    rep = sp.add_parser("check-reproduction")
    rep.add_argument("--observed", type=Path, required=True)
    rep.add_argument("--reference", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument("--strict", action="store_true")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "plan":
        profile = get_profile(args.profile)
        text = github_matrix(profile) if args.github_matrix else json.dumps([x.to_dict() for x in build_tasks(profile)], indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    if args.cmd == "run-task":
        run_task(
            profile=get_profile(args.profile), family=args.family, n=args.n,
            depth_factor=args.depth_factor, instance_start=args.instance_start,
            instance_stop=args.instance_stop, tangents=args.tangents,
            parameter_distribution=args.parameter_distribution,
            output_dir=args.output, master_seed=args.master_seed,
        )
        return 0
    if args.cmd == "analyze":
        analyze_results(args.input, args.output)
        return 0
    if args.cmd == "controls":
        args.output.mkdir(parents=True, exist_ok=True)
        isotropic_controls().to_csv(args.output / "isotropic_controls.csv", index=False)
        anisotropic_relative_orientation_control().to_csv(args.output / "anisotropic_relative_orientation.csv", index=False)
        return 0
    if args.cmd == "check-reproduction":
        out = compare_reproduction(args.observed, args.reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(out.to_string(index=False))
        if args.strict and not bool((out.core_means_match & out.deff_matches).all()):
            return 2
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
