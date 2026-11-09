import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuitcare.schematic import exporter


def _schematic(tmp_path: Path) -> Path:
    path = tmp_path / "source.kicad_sch"
    path.write_text('(kicad_sch (version 20250114) (generator "eeschema") '
                    '(uuid "root") (paper "A4") (lib_symbols) '
                    '(sheet_instances (path "/" (page "1"))))', encoding="utf-8")
    return path


def _mock_cli(monkeypatch, behavior=None):
    calls = []
    monkeypatch.setattr(exporter, "find_kicad_cli", lambda: "/fake/kicad-cli")

    def run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "version":
            return SimpleNamespace(stdout="10.0.6", stderr="", returncode=0)
        if behavior:
            return behavior(args, kwargs)
        output = Path(args[args.index("--output") + 1])
        (output / "source.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>',
            encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(exporter.subprocess, "run", run)
    return calls


def test_cli_unavailable_and_project_escape_are_rejected(tmp_path, monkeypatch):
    source = _schematic(tmp_path)
    monkeypatch.setattr(exporter, "find_kicad_cli", lambda: None)
    assert exporter.export_schematic(source, project_root=tmp_path,
                                     cache_root=tmp_path / "cache").status == "CLI_UNAVAILABLE"
    assert exporter.export_schematic(source, project_root=tmp_path,
                                     cache_root=tmp_path.parent / "outside").status == "REJECTED"


def test_successful_export_and_content_cache_invalidation(tmp_path, monkeypatch):
    source = _schematic(tmp_path)
    calls = _mock_cli(monkeypatch)
    first = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    second = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert first.status == second.status == "EXPORTED"
    assert first.cache_key == second.cache_key
    assert len([call for call in calls if call[1] == "sch"]) == 1
    source.write_text(source.read_text(encoding="utf-8").replace("20250114", "20250115"), encoding="utf-8")
    third = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert third.cache_key != first.cache_key
    assert len([call for call in calls if call[1] == "sch"]) == 2


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "missing", "malformed", "script", "external", "oversized"])
def test_export_failure_and_untrusted_svg_are_rejected(tmp_path, monkeypatch, failure):
    source = _schematic(tmp_path)

    def behavior(args, kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 0.01)
        if failure == "nonzero":
            return SimpleNamespace(stdout="", stderr="failed", returncode=1)
        if failure == "missing":
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        output = Path(args[args.index("--output") + 1])
        body = {
            "malformed": "<svg>",
            "script": '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            "external": '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.test/x"/></svg>',
            "oversized": '<svg xmlns="http://www.w3.org/2000/svg">' + (" " * 100) + "</svg>",
        }[failure]
        (output / "source.svg").write_text(body, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    _mock_cli(monkeypatch, behavior)
    if failure == "oversized":
        monkeypatch.setattr(exporter, "MAX_SVG_BYTES", 64)
    result = exporter.export_schematic(source, project_root=tmp_path,
                                       cache_root=tmp_path / "cache", timeout=0.01)
    expected = "TIMEOUT" if failure == "timeout" else "EXPORT_FAILED" if failure == "nonzero" else "INVALID_OUTPUT"
    assert result.status == expected


def test_child_sheet_bytes_invalidate_cache_and_external_child_is_blocked(tmp_path, monkeypatch):
    child = tmp_path / "child.kicad_sch"
    child.write_text('(kicad_sch (uuid "child") (paper "A4") (lib_symbols))', encoding="utf-8")
    source = tmp_path / "root.kicad_sch"
    source.write_text('(kicad_sch (uuid "root") (paper "A4") (lib_symbols) '
                      '(sheet (property "Sheet file" "child.kicad_sch") (uuid "sheet")))', encoding="utf-8")
    calls = _mock_cli(monkeypatch)
    first = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    child.write_text(child.read_text(encoding="utf-8").replace('"child"', '"changed"'), encoding="utf-8")
    second = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert first.cache_key != second.cache_key
    assert len([call for call in calls if call[1] == "sch"]) == 2
    source.write_text(source.read_text(encoding="utf-8").replace("child.kicad_sch", "../foreign.kicad_sch"), encoding="utf-8")
    assert exporter.export_schematic(source, project_root=tmp_path,
                                     cache_root=tmp_path / "cache").status == "INCOMPLETE_PROJECT"


def test_multiple_svg_pages_are_returned(tmp_path, monkeypatch):
    source = _schematic(tmp_path)

    def behavior(args, kwargs):
        output = Path(args[args.index("--output") + 1])
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
        (output / "sheet-1.svg").write_text(svg, encoding="utf-8")
        (output / "sheet-2.svg").write_text(svg, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    _mock_cli(monkeypatch, behavior)
    result = exporter.export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert result.status == "EXPORTED"
    assert len(result.svg_paths) == 2
