from circuitcare.risk_engine import assess_component_risk


def test_reduced_resistor_power_does_not_get_stress_increase_points():
    rating_results = [{
        "component_ref": "RLOAD",
        "metrics": {
            "max_power_w": {
                "status": "evaluated",
                "utilization_ratio": 0.01,
                "rating_exceeded": False,
                "baseline_stress": 0.00625,
                "measured_stress": 0.0002,
                "stress_increase": -0.00605,
                "rating_source": {"type": "demo_assumption"},
            },
        },
    }]
    stress = {"components": [{"component_ref": "RLOAD", "delta_fault_minus_baseline": {"power_w": {"maximum": -0.00605}}}]}
    result = assess_component_risk(
        rating_results,
        stress,
        {"RLOAD": {"model_type": "resistor"}},
        failed_component="U1",
    )
    assert result[0]["priority_factors"]["stress_change_contribution"] == 0.0
