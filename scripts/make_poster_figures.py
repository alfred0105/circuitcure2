"""Render standalone poster figures from checked-in simulation and diagnosis data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/phase10_demo_results.png"
OVER = ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise"
DIVIDER = ROOT / "data/evaluations/divider/simulation_results/fault_r1_resistance_drift"
RC = ROOT / "data/evaluations/rc_charge/simulation_results/fault_rcharge_resistance_drift"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    comparison = load_json(OVER / "comparison.json")
    risk = load_json(OVER / "risk_assessment.json")
    prior = load_json(ROOT / "data/diagnosis/prior_results/diagnosis.json")
    posterior = load_json(ROOT / "data/diagnosis/posterior_results/diagnosis.json")
    if comparison.get("simulation_success") is not True or risk.get("simulation_status") != "success":
        raise SystemExit("Saved overvoltage result is not successful; no figure generated.")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    figure.suptitle("CircuitCare AI · Stored simulation evidence (demo assumptions)", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    nodes = ["VOUT", "LOAD"]
    baseline = [comparison["voltage_nodes"][node]["baseline"]["last"] for node in nodes]
    fault = [comparison["voltage_nodes"][node]["fault"]["last"] for node in nodes]
    x = range(len(nodes))
    ax.bar([i - 0.18 for i in x], baseline, width=0.36, label="Baseline", color="#3478b8")
    ax.bar([i + 0.18 for i in x], fault, width=0.36, label="U1 output set to 8 V", color="#e47732")
    ax.set_xticks(list(x), nodes)
    ax.set_ylabel("Voltage (V)")
    ax.set_title("A · U1 overvoltage: baseline vs fault")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    candidates = {item["component_ref"]: item for item in risk.get("secondary_candidates", [])}
    c1 = candidates["C1"]["maximum_utilization_ratio"]
    rload = candidates["RLOAD"]["maximum_utilization_ratio"]
    ax.bar(["C1 voltage", "RLOAD power"], [c1, rload], color=["#bd3b35", "#4b9c70"])
    ax.axhline(1.0, color="#333333", linestyle="--", label="Configured rating = 100%")
    ax.set_ylim(0, max(1.4, c1 * 1.1))
    ax.set_ylabel("Stress / configured rating")
    ax.set_title("B · Rating utilization (demo limits)")
    ax.legend(frameon=False)
    ax.text(0, c1 + 0.025, f"{c1:.3f} · EXCEEDED", ha="center", fontsize=8)
    ax.text(1, rload + 0.025, f"{rload:.3f} · WITHIN_LIMIT", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    prior_by_id = {row["hypothesis_id"]: row["posterior"] for row in prior["posterior"]}
    post_by_id = {row["hypothesis_id"]: row["posterior"] for row in posterior["posterior"]}
    ids = [row["hypothesis_id"] for row in prior["posterior"]]
    labels = [item.replace("U1_OUTPUT_", "") for item in ids]
    ax.bar([i - 0.18 for i in range(len(ids))], [prior_by_id[item] for item in ids], width=0.36, label="Prior")
    ax.bar([i + 0.18 for i in range(len(ids))], [post_by_id[item] for item in ids], width=0.36, label="After VOUT 7.9 V ± 0.2 V")
    ax.set_xticks(range(len(ids)), labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Conditional posterior weight")
    ax.set_title("C · Bayesian saved demo (simulated measurement)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    div = load_json(DIVIDER / "comparison.json")
    div_values = [div["voltage_nodes"]["VOUT"][condition]["last"] for condition in ("baseline", "fault")]
    ax.bar(["R1 = 1 kΩ", "R1 = 10 kΩ"], div_values, color=["#3478b8", "#e47732"])
    ax.set_ylabel("VOUT (V)")
    ax.set_title("D · Resistor divider · saved ngspice result")
    for i, value in enumerate(div_values):
        ax.text(i, value + 0.04, f"{value:.4f} V", ha="center")
    ax.set_ylim(0, max(div_values) * 1.25)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2, 0]
    rc_rows = load_csv(RC / "comparison.csv")
    times_ms = [float(row["time_s"]) * 1000 for row in rc_rows]
    base_v = [float(row["baseline_V(VC)_V"]) for row in rc_rows]
    fault_v = [float(row["fault_V(VC)_V"]) for row in rc_rows]
    ax.plot(times_ms, base_v, label="RCHARGE = 10 kΩ", color="#3478b8")
    ax.plot(times_ms, fault_v, label="RCHARGE = 100 kΩ", color="#e47732")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("VC (V)")
    ax.set_title("E · RC transient · saved ngspice time range")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    ax.text(0.99, 0.03, f"samples: {len(rc_rows)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    ax = axes[2, 1]
    ax.axis("off")
    ax.set_title("F · Interpretation limits", loc="left")
    ax.text(0.02, 0.92,
            "• Rating values are demo assumptions, not datasheet-verified.\n"
            "• Posterior weights are conditional, not failure probabilities.\n"
            "• No physical damage labels or confirmed failures were available.\n"
            "• Only two independent circuit topologies were evaluated.\n"
            "• RC panel uses the committed transient sample time range.\n\n"
            "Sources: data/fault_scenarios/overvoltage_demo,\n"
            "data/evaluations/divider, data/evaluations/rc_charge,\n"
            "data/diagnosis/prior_results and posterior_results.",
            va="top", ha="left", linespacing=1.5, fontsize=9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
