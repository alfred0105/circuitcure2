"""Command line entry point for registering and executing PCB validation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pcb_validation import (
    evaluate_runs, production_qwen_reviewer, production_yolo_detector,
    register_cases, run_case, save_metrics_csv, save_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="circuitcare-validate")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="validate and import a JSON or CSV case file")
    register.add_argument("source")
    register.add_argument("destination")
    run = commands.add_parser("run", help="execute registered cases and save immutable run outputs")
    run.add_argument("cases")
    run.add_argument("results")
    run.add_argument("--real-yolo", action="store_true", help="use local production YOLO weights")
    run.add_argument("--mode", choices=("REAL_MODEL", "CACHED", "MOCK", "SYNTHETIC"),
                     help="explicit execution mode; REAL_MODEL requires local weights")
    run.add_argument("--strategy", choices=("baseline", "evidence_v1", "active_inspection_v1"), default="baseline",
                     help="baseline ranking, experimental evidence reranking, or ranking-neutral active inspection planning")
    args = parser.parse_args()
    if args.command == "register":
        print(register_cases(args.source, args.destination))
        return 0
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.real_yolo and args.mode not in {None, "REAL_MODEL"}:
        parser.error("--real-yolo can only be combined with --mode REAL_MODEL")
    execution_mode = "REAL_MODEL" if args.real_yolo else args.mode
    detector = production_yolo_detector if execution_mode == "REAL_MODEL" else None
    reviewer = production_qwen_reviewer if execution_mode == "REAL_MODEL" else None
    result_paths = []
    outputs = []
    for case in payload.get("cases", []):
        output = run_case(case, detector=detector, reviewer=reviewer,
                          strategy=args.strategy, execution_mode=execution_mode)
        outputs.append(output)
        target = Path(args.results) / f"{case['case_id']}-{output['run_id'][:8]}.json"
        result_paths.append(str(save_run(target, output)))
    summary = evaluate_runs(payload.get("cases", []), outputs)
    summary_path = Path(args.results) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = save_metrics_csv(Path(args.results) / "summary.csv", summary)
    print(json.dumps({"saved_runs": result_paths, "metrics_report": str(summary_path),
                      "metrics_csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
