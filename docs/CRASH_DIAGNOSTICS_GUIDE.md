# Crash diagnostics guide

Start the dashboard with Python's fatal-signal traceback support:

```bash
python -X faulthandler -m streamlit run src/circuitcare/ui.py
```

Run the isolated diagnostic suites:

```bash
python scripts/run_crash_diagnostics.py --suite stress
python scripts/run_crash_diagnostics.py --suite integration
python scripts/run_crash_diagnostics.py --suite all
```

Reports are written to ignored `data/private/diagnostics/` unless `--output-dir` is supplied. They include Python, OS/architecture, Git SHA, optional runtime versions, available RAM when reported by the OS, PID, elapsed time, sampled process-tree RSS, exit code, signal classification, and stdout/stderr log names. The runner distinguishes a failed child from its still-running parent. A negative POSIX return code is mapped to its signal name; other platform codes stay numeric unless their meaning is known. Unknown process termination is not labelled OOM.

When the main Streamlit process itself exits unexpectedly, retain its console output and any macOS DiagnosticReports entry for the timestamp. Reports can contain local paths or project details; review them before sharing. No report should contain uploaded PCB or datasheet bytes by default.

RSS is sampled and can miss short peaks. It includes native allocations visible to the OS, while Python object tracing would not account for all OpenCV, ONNX Runtime, or child-process memory. A missing `peak_rss_mb` means the platform sampler was unavailable, not zero memory use.
