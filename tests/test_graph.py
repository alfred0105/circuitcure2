from circuitcare.graph import build_graph, direct_candidates, find_paths, to_dot
from circuitcare.models import Circuit, Component, Connection
from circuitcare.netlist import load_netlist


def test_bipartite_graph_keeps_node_kinds_and_multiple_pins():
    circuit = Circuit(
        components={"U1": Component("U1", "IC"), "VOUT": Component("VOUT", "R")},
        connections=[Connection("U1", "1", "VOUT"), Connection("U1", "2", "VOUT")],
    )

    graph = build_graph(circuit)
    assert "component:VOUT" in graph["nodes"]
    assert "net:VOUT" in graph["nodes"]
    edge = next(item for item in graph["edges"] if item["component"] == "component:U1")
    assert edge["pins"] == ("1", "2")


def test_paths_include_indirect_candidate_and_keep_connectivity_evidence():
    circuit = load_netlist("tests/fixtures/simple_kicad_netlist.xml")
    paths = find_paths(circuit, "U1", max_hops=2)
    resistor_paths = [item for item in paths if item.target_ref == "R4"]

    assert resistor_paths
    assert all(item.connection_type == "indirect" and item.hops == 2 for item in resistor_paths)
    assert any("net:SIGNAL" in item.path and "component:U2" in item.path for item in resistor_paths)
    assert all(item.evidence_level == "connectivity_only" for item in paths)
    assert all(item.ground_only_path is None for item in paths)


def test_ground_type_is_reported_and_can_be_excluded_without_mutating_graph():
    circuit = Circuit(
        components={"U1": Component("U1", "IC"), "R1": Component("R1", "1k")},
        connections=[Connection("U1", "1", "GND"), Connection("R1", "2", "GND")],
    )
    graph = build_graph(circuit, {"GND": "ground"})
    paths = find_paths(circuit, "U1", net_types={"GND": "ground"})

    assert paths[0].ground_only_path
    assert paths[0].net_types == (("GND", "ground"),)
    assert find_paths(circuit, "U1", net_types={"GND": "ground"}, exclude_net_types={"ground"}) == []
    assert direct_candidates(circuit, "U1")[0].target_ref == "R1"
    dot = to_dot(graph, "U1", paths)
    assert '"component:U1"' in dot and '"net:GND"' in dot
    assert "Connectivity only" in dot
