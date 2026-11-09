from pathlib import Path
from types import SimpleNamespace

import circuitcare.dashboard.views.unified_workspace as workbench


def test_kicad_schematic_reuses_bounded_export_with_ephemeral_project(monkeypatch):
    seen = {}

    def fake_export(source, *, project_root, cache_root):
        source = Path(source)
        project_root = Path(project_root)
        cache_root = Path(cache_root)
        seen["source"] = source
        seen["root"] = project_root
        seen["cache"] = cache_root
        svg = project_root / "page.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60"><path d="M0 0"/></svg>',
            encoding="utf-8",
        )
        return SimpleNamespace(status="EXPORTED", svg_paths=(svg,))

    monkeypatch.setattr(workbench, "export_schematic", fake_export)
    pages = workbench._export_schematic_pages(b"(kicad_sch (version 1))", "board.kicad_sch")

    assert len(pages) == 1
    assert pages[0][2] == (0.0, 0.0, 120.0, 60.0)
    assert "<svg" in pages[0][1]
    assert seen["source"].suffix == ".kicad_sch"
    assert seen["source"].is_relative_to(seen["root"])
    assert seen["cache"].is_relative_to(seen["root"])
    assert not seen["root"].exists()
