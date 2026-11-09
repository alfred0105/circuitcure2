from pathlib import Path

import pytest

from circuitcare.graph import direct_candidates
from circuitcare.netlist import load_netlist


FIXTURE = Path(__file__).parent / "fixtures" / "simple_kicad_netlist.xml"


def test_kicad_netlist_parses_components_and_connections():
    circuit = load_netlist(FIXTURE)
    assert len(circuit.components) == 5
    assert circuit.components["U1"].value == "Regulator"
    assert circuit.components["C3"].library_id == "Device:C"
    assert circuit.components["C3"].footprint == "Capacitor_SMD:C_0805_2012Metric"
    assert circuit.components["C3"].datasheet == "https://example.invalid/capacitor.pdf"
    assert any(x.ref == "U2" and x.pin == "1" and x.net == "VOUT" for x in circuit.connections)
    assert circuit.component_nets["U1"] == ("GND", "VIN", "VOUT")


def test_finding_candidates_does_not_claim_damage():
    circuit = load_netlist(FIXTURE)
    results = direct_candidates(circuit, "U1")
    by_ref = {x.target_ref: x for x in results}
    assert set(by_ref) == {"J1", "U2", "C3"}
    assert "VOUT" in by_ref["U2"].shared_nets
    assert by_ref["C3"].evidence_level == "connectivity_only"
    assert "R4" not in by_ref  # not electrically directly connected


def test_unknown_failure_ref_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        direct_candidates(load_netlist(FIXTURE), "U99")


def test_unnamed_nets_get_unique_names_and_duplicate_pins_are_removed(tmp_path):
    netlist = tmp_path / "unnamed.xml"
    netlist.write_text("""<export><components>
      <comp ref="R1"><value>1k</value></comp><comp ref="R2"><value>2k</value></comp>
    </components><nets>
      <net code="1"><node ref="R1" pin="1"/><node ref="R1" pin="1"/></net>
      <net code="1"><node ref="R2" pin="1"/></net>
      <net><node ref="R1" pin="2"/></net>
      <net code="2" name="__code_1"><node ref="R2" pin="2"/></net>
    </nets></export>""", encoding="utf-8")

    circuit = load_netlist(netlist)
    assert len(circuit.connections) == 4
    assert len({connection.net for connection in circuit.connections}) == 4
    assert circuit.component_nets["R1"] == ("__code_1_2", "__unnamed_1")


def test_conflicting_nets_for_one_pin_are_rejected(tmp_path):
    netlist = tmp_path / "conflict.xml"
    netlist.write_text("""<export><components><comp ref="U1"><value>IC</value></comp></components>
    <nets><net code="1" name="VIN"><node ref="U1" pin="1"/></net>
    <net code="2" name="VOUT"><node ref="U1" pin="1"/></net></nets></export>""", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Conflicting net connections for U1 pin 1"):
        load_netlist(netlist)
