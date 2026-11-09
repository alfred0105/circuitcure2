# AI fault hypothesis workflow

## What it does

The AI 고장 가설 분석 tab generates candidates using the existing supported fault injector models and explicit KiCad-to-SPICE mappings. Deterministic rules propose the catalog space. The optional local Ollama model can suggest candidates and short evidence-based reasons; its proposals are data only and must match the preflighted reference, mode, and parameter set. Unsupported proposals remain visible with a reason and are never simulated.

The supported modes are resistor open, short, and resistance drift; mapped regulator output rise, drop, and loss; and mapped switch stuck-open/stuck-on. Mapping a model type is not enough: the mapped SPICE element and required parameters must also pass the existing fault injector.

## User flow

1. Choose one of the bundled regulator, divider, or RC example circuits.
2. Enter an optional symptom and observed node voltage. Actual and demo/synthetic observations carry distinct source labels. Add positive measurement uncertainty if standardized residual or Bayesian comparison is needed.
3. Explicitly request candidate generation. Rule proposals are available offline; local LLM proposals are opt-in and run through the configured local Ollama model.
4. Review each candidate, source, and preflight reason.
5. Select up to five executable hypotheses and explicitly start ngspice.
6. Compare predicted and observed values, inspect residuals, rank the model fit, and view the highlighted component in the Fault Map.

The SPICE bridge only uses the approved base deck, an explicit mapping, and the current bounded fault injector. It does not accept LLM-authored netlist text or paths. Runs use temporary output and a timeout. A successful run means the configured deck executed; consistency means the prediction agrees with the supplied observation within the configured threshold. Neither establishes a real fault.

## Ranking and Bayesian behavior

Candidates with predictions for the same observed node set can reuse the existing Bayesian engine when each observation has an explicit positive uncertainty. The engine applies equal priors. LLM language confidence is never converted into a prior. Different probe coverage, missing predictions, absent uncertainty, or incompatible observations disable the Bayesian view and retain evidence ranking only.

## Evaluation

Run `python scripts/evaluate_fault_hypotheses.py --run-llm --model gemma4:12b --simulate` to execute the local benchmark. It scores three existing synthetic fault injections and writes separate candidate and simulation artifacts under `evaluation/fault_hypothesis/results/`. These cases are development examples, not independent repair records. Use `--simulate` only when ngspice is installed; the UI and script cap per-case simulation runs. Result values are not field accuracy or damage probabilities. Full results and limitations are in [Phase 15 evaluation](PHASE15_EVALUATION.md).

## Current coverage limits

Only components with an explicit compatible SPICE mapping can be executable. The demo covers a simple ideal regulator source, resistors, and one mapped switch abstraction. It does not model semiconductor internal damage, regulator protection behavior, or physical failure mechanisms. KiCad connectivity provides context and is not interpreted as causal evidence.
