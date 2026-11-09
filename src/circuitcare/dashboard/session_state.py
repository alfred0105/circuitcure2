"""Pure invalidation rules for the Streamlit dashboard session adapter."""

from __future__ import annotations

ANALYSIS_CONTEXT_KEYS = (
    "live_result_dir", "diagnosis_result", "measurements", "last_advisor", "live_error",
    "what_if_result", "what_if_result_key", "what_if_measurements", "what_if_context_key",
    "what_if_node", "what_if_voltage", "what_if_sigma",
)

WHAT_IF_KEYS = (
    "what_if_result", "what_if_result_key", "what_if_measurements", "what_if_context_key",
    "what_if_voltage", "what_if_node", "what_if_sigma",
)


def analysis_context_keys_to_clear(existing_keys: list[str] | tuple[str, ...]) -> set[str]:
    """Return legacy analysis keys invalidated when the analysis context changes."""
    keys = set(existing_keys)
    return {key for key in ANALYSIS_CONTEXT_KEYS if key in keys} | {
        key for key in keys if key.startswith("what_if_voltage_")
    }


def simulation_context_keys_to_clear(existing_keys: list[str] | tuple[str, ...]) -> set[str]:
    """Return preview/advisor keys invalidated before a new simulation starts."""
    keys = set(existing_keys)
    return {key for key in WHAT_IF_KEYS if key in keys} | {
        key for key in keys if key.startswith("what_if_voltage_")
    } | ({"live_result_dir", "live_context_key", "last_advisor"} & keys)


def reconcile_analysis_context(session_state, context_key) -> bool:
    """Apply existing context invalidation policy through a dict-like state adapter."""
    if session_state.get("analysis_context_key") == context_key:
        return False
    for key in analysis_context_keys_to_clear(list(session_state)):
        session_state.pop(key, None)
    session_state["analysis_context_key"] = context_key
    session_state["measurements"] = []
    return True
