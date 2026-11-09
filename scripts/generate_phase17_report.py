#!/usr/bin/env python3
"""Generate Phase 17 Markdown reports from one immutable result record."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='evaluation/phase17/results')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--output', default='docs')
    args = parser.parse_args()
    result_dir = ROOT / args.results
    if args.run_id:
        target = result_dir / f'{args.run_id}.json'
        record = json.loads(target.read_text(encoding='utf-8'))
    else:
        paths = sorted(path for path in result_dir.glob('*.json') if path.name != 'summary.json')
        if not paths:
            raise SystemExit('No saved Phase 17 result exists; refusing to invent performance values.')
        target = paths[-1]
        record = json.loads(target.read_text(encoding='utf-8'))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'BASELINE_COMPARISON.md').write_text(baseline_report(record), encoding='utf-8')
    (output / 'PHASE17_EVALUATION.md').write_text(evaluation_report(record), encoding='utf-8')
    print(json.dumps({'run_id': record.get('run_id'), 'status': record.get('status'),
                      'reports': ['docs/PHASE17_EVALUATION.md', 'docs/BASELINE_COMPARISON.md']}, indent=2))
    return 0


def baseline_report(record):
    lines = ['# Phase 17 Baseline Comparison', '',
        f"Selected immutable run: `{record.get('run_id')}` ({record.get('status')}).",
        'Synthetic results do not establish real repair accuracy.', '',
        '| Method | Component P / R | Fault mode P / R | Exact P / R | E2E Top-1 / Top-3 | Conditional Top-1 | Runtime | SPICE runs / success |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, method in record.get('methods', {}).items():
        generation = method.get('generation_metrics', {})
        levels = generation.get('recall', {})
        ranking = method.get('ranking', {}).get('top_k_accuracy', {})
        conditional_ranking = method.get('conditional_ranking', {}).get('top_k_accuracy', {})
        runtime = method.get('candidate_seconds', {})
        runtime_value = runtime.get('mean')
        total_runs = sum(run.get('simulation_runs', 0) for run in method.get('runs', []))
        successes = sum(run.get('simulation_successes', 0) for run in method.get('runs', []))
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name,
            _pair(levels.get('component', {})), _pair(levels.get('fault_mode', {})),
            _pair(levels.get('exact_hypothesis', {})),
            f"{_fmt(ranking.get('1'))} / {_fmt(ranking.get('3'))}",
            _fmt(conditional_ranking.get('1')),
            _fmt(runtime_value), f"{total_runs} / {successes}" if total_runs else 'NOT_EVALUATED'))
    lines += ['', '## Reading the comparison', '',
        '- Candidate generation and SPICE residual ranking are separate stages.',
        '- A missing true candidate remains a miss in end-to-end ranking; conditional ranking is reported separately when available.',
        '- LLM repeats are stability runs on the same cases, not new independent faults.',
        '- Runtime is the summed proposal-stage wall time per repeated batch. SPICE timing is stored separately in each run record.',
        '- SPICE total wall-time means: ' + '; '.join(
            f"{method_name}={_fmt(_mean([run.get('seconds') for run in method.get('runs', [])]))} s"
            for method_name, method in record.get('methods', {}).items() if method_name.endswith('+ SPICE')) + '.',
        '- See [benchmark methodology](BENCHMARK_METHODOLOGY.md) for input allowlists, parameter matching, and uncertainty limits.', '']
    return '\n'.join(lines)


def evaluation_report(record):
    meta = record.get('metadata', {})
    stats = record.get('statistics', {})
    lines = ['# Phase 17 Evaluation', '', '## Status', '',
        f"- Benchmark infrastructure: `IMPLEMENTED`",
        f"- Evaluation status: `{record.get('status', 'NOT_EVALUATED')}`",
        f"- Dataset: `{record.get('dataset_id', 'NOT_EVALUATED')}` / `{record.get('split', 'NOT_EVALUATED')}`",
        f"- Cases / topology groups: {record.get('case_count', 0)} / {record.get('topology_group_count', 0)}",
        f"- Independent synthetic holdout: `{record.get('holdout_status', 'NOT_EVALUATED')}`",
        f"- Supported / unsupported ground-truth cases: {record.get('supported_case_count', 0)} / {record.get('unsupported_ground_truth_case_count', 0)}",
        '- Real repair accuracy: `NOT_EVALUATED` (no verified physical repair benchmark labels).',
        f"- Phase 16 learned-prior comparison: `{record.get('phase16_prior_status', 'REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED')}`",
        f"- New datasheet holdout: `{record.get('datasheet_independent_holdout_status', 'NOT_EVALUATED')}`",
        f"- PCB physical Reference matching: `{record.get('pcb_vision_reference_match_status', 'NOT_EVALUATED')}`", '',
        '## Leakage and reproducibility audit', '',
        f"- Public inference payload audit: `{record.get('input_audit', {}).get('status', 'NOT_EVALUATED')}`",
        f"- Private truth passed to candidate generation: `{record.get('input_audit', {}).get('private_ground_truth_passed_to_proposal_method', 'NOT_EVALUATED')}`",
        f"- Unique public prompt payload hashes: {record.get('input_audit', {}).get('prompt_hash_count', 'NOT_EVALUATED')}",
        f"- Git SHA: `{meta.get('git_sha')}`",
        f"- Python / OS: `{meta.get('python')}` / `{meta.get('platform')}`",
        f"- LLM: `{meta.get('model_name') or 'NOT_RUN'}` · digest `{meta.get('model_digest') or 'NOT_RUN'}` · prompt `{meta.get('prompt_version') or 'NOT_RUN'}`",
        f"- ngspice: `{meta.get('simulator_version') or 'NOT_RUN'}`",
        f"- Process peak RSS bytes: `{record.get('performance_environment', {}).get('process_peak_rss_bytes', 'NOT_EVALUATED')}` (whole command, not method-isolated).",
        f"- LLM token counts: `{record.get('performance_environment', {}).get('llm_token_counts', 'NOT_EVALUATED')}`.",
        f"- Generation settings: `{json.dumps(record.get('generation_settings', {}), ensure_ascii=False)}`.",
        f"- Public dataset SHA-256: `{meta.get('dataset_hash')}`",
        f"- Ground-truth SHA-256: `{meta.get('ground_truth_hash')}`", '',
        '## Measured methods', '',
        'See [baseline comparison](BASELINE_COMPARISON.md) for component, mode, exact hypothesis, Top-K, runtime, and SPICE counts.', '',
        '### Statistics', '']
    for method, interval in stats.get('exact_recall_cluster_bootstrap_95ci', {}).items():
        lines.append(f"- {method}: `{interval.get('status')}`, clusters={interval.get('cluster_count')}, 95% CI [{_fmt(interval.get('lower'))}, {_fmt(interval.get('upper'))}]. Exploratory only; small cluster counts make the interval unstable.")
    if not stats.get('exact_recall_cluster_bootstrap_95ci'):
        lines.append('- Cluster-bootstrap confidence intervals: `NOT_EVALUATED`.')
    lines += ['', '### Measurement nodes', '',
        f"Node-count strata: `{json.dumps(record.get('measurement_node_count_comparison', {}), ensure_ascii=False)}`.",
        'Measurement selection was not compared against an independent fixed-versus-information-gain policy.', '',
        '### LLM repetition stability', '']
    llm = record.get('methods', {}).get('LLM Only', {})
    for key, summary in llm.get('repeat_stability', {}).get('metrics', {}).items():
        lines.append(f"- {key}: n={summary.get('count')}, mean={_fmt(summary.get('mean'))}, SD={_fmt(summary.get('standard_deviation'))}, min/max={_fmt(summary.get('min'))}/{_fmt(summary.get('max'))}.")
    if not llm.get('repeat_stability', {}).get('metrics'):
        lines.append('- LLM repetition stability: `NOT_EVALUATED`.')
    lines += ['', '### SPICE ranking interpretation', '',
        'Saved results separate end-to-end ranking (a generated-candidate miss counts as a miss) from conditional ranking after successful generation and simulation. Conditional ranking cannot stand alone as overall diagnostic accuracy.', '',
        '### Failure and open-set assessment', '',
        f"Simulation cases failed: {sum(run.get('simulation_failures', 0) for method in record.get('methods', {}).values() for run in method.get('runs', []))}.",
        'Unsupported-fault rejection, false-confidence, abstention, and open-set detection are `NOT_EVALUATED` unless a dedicated unsupported-label split is present.', '',
        '## Limitations', '']
    lines.extend(f'- {item}' for item in record.get('limitations', []))
    lines += ['', 'The complete per-run output, candidate counts, runtimes, and failure details are preserved in `evaluation/phase17/results/{}.json`.'.format(record.get('run_id')), '']
    return '\n'.join(lines)


def _pair(values):
    if not values:
        return 'NOT_EVALUATED'
    return f"{_fmt(values.get('precision'))} / {_fmt(values.get('recall'))}"


def _fmt(value):
    return 'NOT_EVALUATED' if value is None else f'{value:.4f}' if isinstance(value, (int, float)) else str(value)


def _mean(values):
    kept = [value for value in values if isinstance(value, (int, float))]
    return sum(kept) / len(kept) if kept else None


if __name__ == '__main__':
    raise SystemExit(main())
