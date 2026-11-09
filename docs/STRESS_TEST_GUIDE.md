# Stability and stress test guide

Tests are split by cost and environment:

```bash
python -m pytest -q -m "not integration and not stress"  # fast deterministic suite
python -m pytest -q -m integration                         # ngspice/native integrations
python -m pytest -q -m stress                              # bounded synthetic stress inputs
python scripts/run_crash_diagnostics.py --suite all        # isolated child-process suites
```

The diagnostic runner defaults to `data/private/diagnostics/` (Git ignored) and records `runtime_environment.json`, per-suite JSON, stdout/stderr logs, and `crash_summary.json`. Use `--output-dir /tmp/circuitcare-diagnostics` to keep a run elsewhere. `--timeout` controls each isolated pytest child. The process group is terminated on POSIX timeout. RSS is sampled with `ps` for the child and descendants where supported; `tracemalloc` is not used as a proxy for native memory.

The stress cases use synthetic PDFs/images and generated graphs; no repair database or evaluation ground truth is used. They impose explicit algorithmic limits and do not attempt to exhaust system memory. Native crash tests send a signal only to an isolated child process.

Generated simulation output cleanup is opt-in and previews first:

```bash
python scripts/cleanup_runtime_artifacts.py --older-than-days 30
python scripts/cleanup_runtime_artifacts.py --older-than-days 30 --apply
```

The cleanup only considers old folders under `data/ui_runs/`, skips folders carrying `.circuitcare-active`, and never targets repair cases, evaluation results, or source input files. Inspect the preview before choosing `--apply`.
