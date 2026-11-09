# Fault influence map guide

Open the **분석 개요** tab in the local Streamlit dashboard. The map is generated from the selected KiCad Component–Net graph, the current scenario's saved/live risk result, and (only for the supported U1 Bayesian demo) the current simulated What-if inspection ranking.

## Views and colors

- **Connection View:** selected fault source is purple, components in its connectivity candidate paths are blue, and unrelated components are muted. Net diamonds retain names and declared/unknown types.
- **Electrical Stress View:** purple is the user-selected fault source, red is `EXCEEDED`, orange is `NEAR_LIMIT`, blue is `WITHIN_LIMIT`, and gray is `UNKNOWN`. The failed part's own assigned risk level is not a damage finding.
- **Inspection Priority View:** component fill shade shows a 0–100 posterior-weighted inspection index band (0–19, 20–39, 40–59, 60–79, 80–100) and `#rank` shows its ordering. This is separate from stress color and is not a probability. Bayesian updates can change the score while a component's rank remains the same.
- The currently selected component gets a dark border. Selecting it emphasizes its connected candidate route and opens the supporting details below the map.

The graph is bipartite and represents component-to-net connectivity, not the original KiCad schematic layout, directional current flow, or proof of fault propagation. Direct/indirect path type, hop count, and net names appear in the selected component's detail. GND links are preserved and do not by themselves imply risk.

The detail panel keeps connection path evidence, physical stress/rating evidence, and Bayesian inspection ranking in separate fields. Missing rating, stress, or matching scenario data remains `UNKNOWN` or unavailable; damage is never marked confirmed by this visualization.

Download `fault_map_data.json` and `fault_influence_map.dot` from the overview. **현재 지도와 What-if 결과를 로컬 저장** explicitly writes derived outputs to ignored `data/visualizations/`; no files are written automatically.
