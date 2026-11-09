# Phase 10 final evaluation record

Generated from `data/evaluations/phase09_summary.json` on 2026-11-03 with `python scripts/evaluate.py`. Re-run the script to refresh local timings and observed environment details. Values below describe this one Mac run; they are not general performance guarantees.

## Environment

| Item | Observed value |
|---|---|
| Python | 3.12.14 |
| Platform | macOS 27.0.1, ARM64 |
| ngspice | 47 |
| Streamlit | 1.66.1 |
| Local Ollama model available during preflight | `gemma4:12b`, MLX runner; model weights are external to the repository |
| Evaluation process peak RSS | 26.69 MB; excludes ngspice and Ollama child-process memory |

## Circuit results

| Example | Fault setting | Saved ngspice value | End-to-end CLI time |
|---|---|---:|---:|
| Resistor divider | R1 1 kΩ→10 kΩ | VOUT 2.5000→0.4545 V | 0.099875 s |
| RC charging | RCHARGE 10 kΩ→100 kΩ | VC at final saved sample: 0.24325→0.02488 V; 545 samples over 0–50 ms | 0.083346 s |
| U1 rating demo | output setpoint 5→8 V | C1 8 V / 6.3 V = 1.2698 (`EXCEEDED`); RLOAD 16 mW / 125 mW = 0.128 (`WITHIN_LIMIT`) | Uses committed scenario result; not counted as an independent circuit topology |

The U1 limits are explicitly marked demonstration assumptions. `WITHIN_LIMIT` means the configured and measured metric was under its configured limit; it does not prove that a part is undamaged or safe.

## Diagnosis and local explanation

- Saved U1 Bayesian scenario set: equal prior weights; a saved `simulated_demo` VOUT observation of 7.9 V ± 0.2 V updates the 8 V scenario to conditional posterior 1.0. It is not a real-world failure probability or a physical measurement.
- Bayesian calculation time in this run: 0.004179 s.
- Previously recorded Ollama `gemma4:12b` runs: 60.303 s and 29.700 s according to the stored Ollama `total_duration_ns` fields. These are earlier local observations, not a latency promise.
- Current UI preflight found local Ollama and the deterministic fallback. A live answer may be rejected by evidence validation; in that case the UI reports the reason and shows deterministic advice instead.

## Scope and limitations

- Two independent small circuit topologies were evaluated: a resistor divider and a first-order RC charging network.
- The U1 Bayesian scenario collection is a separate configured demo, not a third held-out topology.
- No independently labeled damaged components or field measurements were available. Damage classification accuracy, Top-k damage coverage, and avoided-inspection rates were not calculated.
- Several limits are demonstration assumptions, not verified manufacturer datasheet ratings.
- Stored RC CSV time range is 0–50 ms (not 5 ms); the last values in the table above are from that committed range.
- Evaluation-script RSS excludes ngspice and Ollama child processes.

## Reproduction

```bash
python -m pip install -e ".[dev,ui]"
python scripts/evaluate.py
python scripts/demo_check.py
python -m pytest -q
```
