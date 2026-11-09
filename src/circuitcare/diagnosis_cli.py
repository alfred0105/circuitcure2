"""Command line entry point for local Bayesian circuit diagnosis."""

import argparse
import json

from .diagnosis_ai import run_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="CircuitCare AI offline Bayesian diagnosis")
    parser.add_argument("--hypotheses", required=True, help="JSON hypothesis list referencing saved SPICE and risk results")
    parser.add_argument("--measurements", help="Optional JSON user/simulated-demo measurements")
    parser.add_argument("--output", required=True, help="Output directory for diagnosis artifacts")
    args = parser.parse_args()
    try:
        result = run_diagnosis(args.hypotheses, args.measurements, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["diagnosis"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
