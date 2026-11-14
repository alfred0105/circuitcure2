from pathlib import Path

from circuitcare.vision import weight_paths


def test_resolver_prefers_explicit_environment_path(tmp_path, monkeypatch):
    configured = tmp_path / "from-env"
    monkeypatch.setenv(weight_paths.WEIGHT_ENV, str(configured))
    monkeypatch.setattr(weight_paths, "WEIGHT_CONFIG_PATH", tmp_path / "user-config")
    assert weight_paths.resolve_pcb_weights_root(tmp_path / "checkout") == configured.resolve()


def test_resolver_uses_per_user_path_when_checkout_is_temporary(tmp_path, monkeypatch):
    configured = tmp_path / "shared" / "weights"
    configured.mkdir(parents=True)
    config = tmp_path / "pcb_weights_path"
    config.write_text(str(configured), encoding="utf-8")
    monkeypatch.delenv(weight_paths.WEIGHT_ENV, raising=False)
    monkeypatch.setattr(weight_paths, "WEIGHT_CONFIG_PATH", config)
    assert weight_paths.resolve_pcb_weights_root(tmp_path / "temporary-checkout") == configured.resolve()


def test_resolver_falls_back_to_weights_next_to_app(tmp_path, monkeypatch):
    app_dir = tmp_path / "checkout" / "apps" / "pcb-component-recognition"
    (app_dir / "weights").mkdir(parents=True)
    monkeypatch.delenv(weight_paths.WEIGHT_ENV, raising=False)
    monkeypatch.setattr(weight_paths, "WEIGHT_CONFIG_PATH", tmp_path / "missing-config")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing-local-models"))
    assert weight_paths.resolve_pcb_weights_root(app_dir) == app_dir.resolve() / "weights"
