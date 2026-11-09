"""Bounded KiCad CLI SVG export with content-addressed cache entries."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from circuitcare.schematic.identity import discover_schematic_files

MAX_SOURCE_BYTES = 20_000_000
MAX_SVG_BYTES = 12_000_000
MAX_PAGES = 20
RENDERER_VERSION = "circuitcare-kicad-svg-v1"


@dataclass(frozen=True)
class ExportResult:
    status: str
    message: str
    svg_paths: tuple[Path, ...] = ()
    cache_key: str | None = None


def find_kicad_cli() -> str | None:
    candidate = shutil.which("kicad-cli")
    if candidate:
        return candidate
    for path in (Path("/Applications/KiCad.app/Contents/MacOS/kicad-cli"),
                 Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _svg_safe(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > MAX_SVG_BYTES:
        return False
    try:
        root = ET.fromstring(path.read_bytes())
    except (ET.ParseError, OSError):
        return False
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() in {"script", "foreignobject", "iframe", "object", "embed"}:
            return False
        if element.tag.rsplit("}", 1)[-1].lower() == "style":
            style_text = "".join(element.itertext()).lower()
            if "@import" in style_text or re.search(r"url\(\s*['\"]?(?!#)", style_text):
                return False
        for key, value in element.attrib.items():
            name = key.rsplit("}", 1)[-1].lower()
            if name.startswith("on") or (name in {"href", "src"} and not value.startswith("#")):
                return False
            if name == "style" and ("@import" in value.lower() or re.search(r"url\(\s*['\"]?(?!#)", value.lower())):
                return False
            if "javascript:" in value.lower():
                return False
    return True


def export_schematic(path: str | Path, *, project_root: str | Path, cache_root: str | Path,
                     page: str | None = None, timeout: float = 30.0) -> ExportResult:
    """Export a project-contained schematic; all paths and generated SVG are bounded."""
    source = Path(path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    cache = Path(cache_root).resolve()
    if source.suffix != ".kicad_sch" or not source.is_relative_to(root):
        return ExportResult("REJECTED", "회로도는 프로젝트 내부 .kicad_sch 경로여야 합니다.")
    if not cache.is_relative_to(root):
        return ExportResult("REJECTED", "SVG 캐시는 프로젝트 전용 경로 안에 있어야 합니다.")
    if source.stat().st_size > MAX_SOURCE_BYTES:
        return ExportResult("REJECTED", "회로도 파일이 크기 제한을 초과했습니다.")
    cli = find_kicad_cli()
    if not cli:
        return ExportResult("CLI_UNAVAILABLE", "KiCad CLI를 찾지 못했습니다.")
    try:
        version = subprocess.run([cli, "version"], capture_output=True, text=True,
                                 timeout=5, check=True, close_fds=False).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return ExportResult("CLI_ERROR", f"KiCad CLI 버전 확인 실패: {exc}")
    try:
        related, incomplete = discover_schematic_files(source, project_root=root)
    except (OSError, ValueError) as exc:
        return ExportResult("REJECTED", str(exc))
    if incomplete:
        return ExportResult("INCOMPLETE_PROJECT", "; ".join(incomplete))
    digest = hashlib.sha256()
    for item in sorted(set(related)):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(item.read_bytes())
    digest.update(f"{version}|svg|{page or 'all'}|{RENDERER_VERSION}".encode())
    key = digest.hexdigest()
    destination = cache / key
    existing = tuple(sorted(destination.glob("*.svg"))) if destination.is_dir() else ()
    if existing and len(existing) <= MAX_PAGES and all(_svg_safe(item) for item in existing):
        return ExportResult("EXPORTED", "캐시된 KiCad SVG", existing, key)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kicad-svg-") as temporary:
        args = [cli, "sch", "export", "svg", "--output", temporary]
        if page:
            args.extend(["--pages", page])
        args.append(str(source))
        try:
            proc = subprocess.run(args, cwd=str(root), capture_output=True, text=True,
                                  timeout=timeout, check=False, close_fds=False)
        except subprocess.TimeoutExpired:
            return ExportResult("TIMEOUT", "KiCad SVG Export 시간 제한 초과", cache_key=key)
        except OSError as exc:
            return ExportResult("CLI_ERROR", f"KiCad SVG Export 실행 실패: {exc}", cache_key=key)
        if proc.returncode:
            return ExportResult("EXPORT_FAILED", (proc.stderr or proc.stdout)[-2000:], cache_key=key)
        outputs = tuple(sorted(Path(temporary).glob("*.svg")))
        if not outputs or len(outputs) > MAX_PAGES or not all(_svg_safe(item) for item in outputs):
            return ExportResult("INVALID_OUTPUT", "SVG가 없거나 페이지 수·콘텐츠 검증에 실패했습니다.", cache_key=key)
        staging = cache / f".{key}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        copied = []
        for item in outputs:
            target = staging / item.name
            shutil.copyfile(item, target)
            copied.append(target)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return ExportResult("EXPORTED", "KiCad CLI SVG Export 완료", tuple(destination / x.name for x in copied), key)
