# Bayesian What-if diagnosis

The **분석 개요** tab provides a simulated measurement preview for the saved U1 3 V / 6 V / 8 V hypothesis set. It reuses the existing `run_diagnosis`, Bayesian update, measurement recommendation, and posterior-weighted inspection ranking engines. No new model is trained.

## Use it

1. Choose a supported U1 regulator rise scenario in the sidebar.
2. Adjust the measurement node, voltage, and positive uncertainty σ. VOUT defaults to 7.9 V and σ 0.2 V with a 2–9 V range. Other supported nodes use bounds derived from the saved hypothesis predictions.
3. The conditional posterior chart, model-mismatch status, next-measurement suggestion, and posterior-weighted inspection score/rank table recalculate locally as the controls change.
4. Reset restores the default 7.9 V preview and clears only the What-if state.

Every preview measurement is tagged `simulated_demo`. It is held separately from `st.session_state.measurements`, the existing Bayesian tab's actual/saved measurement flow, and `diagnosis_result`. It cannot be promoted to a real measurement. Changing What-if inputs does not rerun ngspice, alter cached/live stress or ratings, invoke the LLM, or create an LLM answer. Circuit/fault context changes reset the preview. A new SPICE run invalidates it.

Posterior weights are conditional on the configured hypotheses, their priors and simulations, and the stated Gaussian uncertainty. They are not real failure probabilities. If every configured hypothesis poorly explains the input, the existing model-mismatch rule is surfaced and the top hypothesis is not highlighted as a diagnosis. A missing next-measurement recommendation means only that no configured unused demo node has meaningful expected information gain; it does not mean inspection is unnecessary or safe.

The posterior-weighted inspection score can change even when its order does not. In the included U1 example, the 7.9 V / 0.2 V input concentrates weight on `U1_OUTPUT_8V` and raises C1's weighted score from about 67.33 to 98.93, while C1 stays inspection rank 1 and the other components retain their order. A 2.0 V / 0.2 V what-if concentrates weight on `U1_OUTPUT_3V`, triggers model mismatch (>4σ from the nearest configured prediction), and lowers C1's weighted score to about 33.35; rank ordering remains unchanged. These are scenario-conditional demo results.

Download `what_if_diagnosis.json` from the overview. Local persistence occurs only when the explicit save button is clicked; generated visualization files are ignored by Git.
