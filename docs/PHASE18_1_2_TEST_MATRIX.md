# Phase 18.1.2 — Test Matrix

Environment: macOS 27.0.1, arm64, Python 3.12.14 at `.venv/bin/python`. The bare `python` command is absent in this shell; its exact requested environment checks exited 127. All subsequent Python checks used the project virtual environment. Full stdout/stderr, metadata and diagnostic JSON are retained under ignored `data/private/diagnostics/phase18_1_2/`.

| Check | Command | Result |
|---|---|---|
| Environment | `python -VV` | unavailable: command not found (127); venv Python 3.12.14 confirmed |
| Platform | `python -c 'import sys,platform; ...'` | passed, macOS arm64 |
| Dependencies | `python -m pip check` | passed, no broken requirements |
| Compile | `python -m compileall -q src scripts tests` | passed |
| Diff whitespace | `git diff --check` | passed |
| Fast | `pytest -q -m 'not integration and not stress and not crash_injection'` | 170 passed, 17 deselected |
| Integration | `pytest -q -m integration -rs` | 6 passed; includes isolated real ONNX/OCR inference and live ngspice scenarios |
| Stress | `pytest -q -m stress -rs` | 10 passed; includes bounded PDF and 20-switch AppTest soak |
| Crash injection | `pytest -q -m crash_injection -rs` | 1 passed; child `SIGSEGV`, parent survives; deliberately isolated |
| Full regression | `pytest -q -m 'not crash_injection' -rs` | 186 passed, 1 deselected |
| Supervised diagnostics | `run_crash_diagnostics.py --suite all --timeout 300 ...` | Fast, integration and stress subprocesses all passed |
| Native inference CI guard | OCR integration uses `CIRCUITCARE_REQUIRE_NATIVE_INFERENCE=1` | Missing runtime/model becomes a failure instead of a skip |
| Streamlit runtime | `python -X faulthandler -m streamlit run ... --server.port 8502` | started; app and health endpoints returned HTTP 200; UI AppTest passed; browser-wide control flow not verified |
| Coverage | full non-crash suite with `pytest-cov` | 185 passed; 62% overall |
| Ruff | `ruff check src scripts tests` | exit 1, 184 findings |
| Mypy | `mypy src/circuitcare` | exit 1, existing type diagnostics (full log retained) |
| pip-audit | `pip-audit` | no known dependency vulnerabilities; local project skipped as non-PyPI package |
| Ollama API | local `/api/tags`, then generation | tags returned `gemma4:12b`; generation returned HTTP 501. CLI attempt separately logged; no inference pass claimed |
| Ollama CLI | `ollama run gemma4:12b 'Reply with the word OK only.'` | exit 0; real local inference returned `OK` |
| Concurrent native + LLM | Ollama CLI generation ran concurrently with `pytest -q tests/test_native_integration.py` | both exit 0; 5 native tests passed, including ONNX/OCR and ngspice cases |

All pytest runs were performed after SIGSEGV isolation. No tests were converted to skips to hide failures. The integration fixture uses blank synthetic images: it verifies inference execution, not detector accuracy. The constrained direct JSON API request failed (501), while the separate CLI generation and its concurrent native test run succeeded. The entire requested circuit-to-report workflow was not exercised as one Streamlit session.

The first pushed CI run (`41783595771`) passed Python 3.11/3.12, ngspice, stress, and required OCR, but the isolated crash job failed during test collection because it installed only `.[dev]` while unrelated collected modules import NumPy. The workflow was corrected to install the same collection-time runtime extras as the fast suite. The corrected run (`41783784532`) passed all six jobs: Python 3.11, Python 3.12, ngspice integration, bounded stress, required OCR inference, and isolated crash injection. The original failure remains visible in the earlier GitHub Actions run logs.
