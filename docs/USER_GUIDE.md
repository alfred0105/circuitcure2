# CircuitCare AI user guide

## Stability and diagnostics

Choose one of the 11 analysis areas from **분석 영역**; only that area's renderer runs. Graph results include a search status. A `PARTIAL_*` result means the displayed routes are incomplete and must not be treated as the full candidate set. PDF and PCB processing reuse recent content-matched results within bounded caches. For crash reporting and isolated stress checks, follow [CRASH_DIAGNOSTICS_GUIDE.md](CRASH_DIAGNOSTICS_GUIDE.md).

## Install and start

Use Python 3.11 or newer. Install the dashboard and optional local readers, then start Streamlit:

```bash
python -m pip install -e ".[dev,ui,vision,datasheet]"
python -m pytest -q
streamlit run src/circuitcare/ui.py
```

ngspice is optional for saved-result review and required for live circuit simulation (`brew install ngspice` on macOS). Ollama and model downloads are optional. See the [PCB vision guide](PCB_VISION_GUIDE.md) and [Datasheet AI guide](DATASHEET_AI_GUIDE.md) before using those features.

## Run a first analysis

1. Choose the cached regulator example in the sidebar, or choose **KiCad XML 업로드** for connectivity-only review of a local XML netlist.
2. Confirm the displayed reference and fault type. For the default cached scenario, U1 uses an 8 V output-rise demo.
3. Review the component/net map and its connectivity-only interpretation.
4. Open **SPICE 파형** and **2차 손상 위험**. Saved legacy results without an input fingerprint are labeled `PROVENANCE_UNVERIFIED` and withheld from the active analysis until a live run records provenance.
5. To run a fresh simulation, install ngspice and explicitly press **ngspice 실행**.
6. Use **AI 고장 가설 분석** to enter observations and generate rule candidates. LLM proposal and SPICE verification each require a separate explicit action.
7. For a continuous/off-grid value, generate the rule catalog first. The report then compares the entered value with candidates generated for the same reference and fault mode. If absent, it labels `EXACT_FAULT_PARAMETER_NOT_IDENTIFIED`; catalog coverage is not the same as a system-selected diagnosis.
8. Review the Workspace and analysis states on **분석 개요**.
9. Download the JSON, Markdown, HTML, or Evidence Bundle report from that overview.

In the Bayesian measurement form, submitting a reading records the entered value including `0.0 V`. The source `user_measurement` means user reported; it is not independently instrument verified. The What-if controls are a separate simulated preview. They do not replace the active diagnosis and appear in the main report only when the separate preview checkbox is selected; the preview remains `PREVIEW_ONLY`.

The sample can also be run from the command line with the commands in the README. For a guided description of all existing workflow tabs, see [Diagnostic workflow](DIAGNOSTIC_WORKFLOW.md).

## Optional features

- **PCB photo review:** provide a local image and manually confirm board registration and each Reference match. Detection alone does not confirm identity or damage.
- **Datasheet review:** verify page evidence, MPN-to-Reference mapping, conditions, and approval before applying a rating.
- **Repair history:** record actual repair outcomes only after the work. Local repair records are not uploaded automatically.
- **Bayesian diagnosis:** inspect its configured hypothesis set, prior, and measurement source. The posterior is conditional and is not a real-world failure probability.
- **Benchmark results:** the evaluation dashboard reads saved records and does not start benchmark runs.

## Common problems

| Message or condition | What it means | Recovery |
|---|---|---|
| Netlist parse error | The input is not a supported KiCad XML export or has malformed pins | Re-export the XML netlist from KiCad and select it again |
| Mapping/model unavailable | Circuit connectivity is readable but the component lacks supported SPICE mapping | Continue graph review or configure a supported mapping/model |
| ngspice unavailable | Live simulation cannot start | Install ngspice; saved results and report export remain available |
| Simulation failed | The selected model run did not produce a valid result | Review the command details and deck; do not treat failure as a safe-circuit result |
| Ollama unavailable | Local LLM generation is unavailable | Use deterministic rule candidates and template report text |
| Vision model unavailable | Learned image detection cannot run | Use manual registration/review where supported, or skip PCB analysis |
| Datasheet parsing/approval incomplete | Rating evidence is not approved for use | Review source pages, part mapping, conditions, and approval state |
| Prior unavailable | No eligible approved learned prior is active | Continue with the default configured prior |
| Report validation failure | The current circuit context or referenced evidence does not match | Refresh the current analysis and retry export |

## Limits

The system supports only configured circuit models and ratings. A connectivity path is not proof of electrical damage. SPICE results depend on the deck and model assumptions. Synthetic demo data and cached results are labeled separately from physical measurements. Real repair accuracy and physical damage prediction remain `NOT_EVALUATED` without an independent, verified repair dataset.
