# Prior Learning Guide

## What is learned

CircuitCare does not fine-tune an LLM, edit SPICE models, or learn from its own predictions. It may propose conditional Bayesian prior weights from reviewed, eligible physical repair cases. The existing Bayesian likelihood calculation is reused. If no compatible approved prior is supplied, the existing default prior behavior remains unchanged.

## Comparable case group

The estimator uses the exact circuit-family and hypothesis-set identifiers. It rejects duplicate source case IDs and cases that fail physical evidence eligibility. It requires a minimum number of eligible cases (default: 5). It does not infer compatibility from an LLM similarity description. Multi-fault and ambiguous-cause records are excluded.

## Smoothing

For mutually exclusive, consistently labeled hypotheses, the proposal uses Dirichlet-style smoothing:

`p_h = (n_h + alpha * pi0_h) / (N + alpha)`

Here `n_h` is the count of independently confirmed eligible cases assigned to hypothesis `h`, `N` is the eligible case count, `pi0_h` is the base prior, and `alpha` is the prior strength. The implementation default is `alpha = 5`; this is an explicit conservative configuration, not an experimentally optimized value. The estimator normalizes the proposal and prevents zero weights. If the minimum evidence threshold is not met, it returns `INSUFFICIENT_VERIFIED_REPAIR_DATA` without learned weights.

This method is valid only for the configured mutually exclusive hypothesis set. It is not applied to incompatible hypothesis catalogs or treated as a general field-failure probability.

## Version and approval lifecycle

A proposal records its model version, hypothesis set, circuit family, source case IDs/count, base prior, smoothing parameter, and estimated weights. A proposal is not automatically applied. A user reviews and approves it, then activates a version. The registry preserves versions and activation events and supports deactivation and rollback. If a source case is deleted or changed, the derived prior is invalidated and removed from active use.

Operational safeguard: proposal registration and activation both require at least five distinct reviewed real-repair cases, even if a development estimator was called with a lower test threshold. The UI does not allow lowering this operating minimum.

The Fault AI screen offers `Default Prior` and an approved compatible learned prior. Unapproved versions are preview-only in the learning screen. A missing or incompatible model falls back to the default prior and reports the reason. The Bayesian integration changes only optional prior weights; it continues to use the existing observation likelihood calculation.

## Comparison limits

Default and learned priors can be run against the same hypotheses and same evidence to inspect posterior and ranking differences. This is a mathematical comparison, not proof of better diagnosis, a new simulation, or altered electrical stress. Conditional posterior mass within a candidate set is not the probability that a real component is damaged.

See [Repair Feedback Guide](REPAIR_FEEDBACK_GUIDE.md) for record quality and [Phase 16 Evaluation](PHASE16_EVALUATION.md) for the current evidence state.
