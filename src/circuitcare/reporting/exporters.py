"""JSON, Markdown, standalone HTML, and allowlisted evidence ZIP export."""

from __future__ import annotations

import hashlib
import html
import json
import copy
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from circuitcare.reporting.report_validator import validate_report, verify_bundle


def export_json(report: dict[str, Any]) -> str:
    errors = validate_report(report)
    if errors:
        raise ValueError("Report validation failed: " + ", ".join(errors))
    serializable = copy.deepcopy(report)
    pcb = serializable.get("pcb_context")
    if isinstance(pcb, dict):
        pcb.pop("photo_artifacts", None)
    return json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def export_markdown(report: dict[str, Any]) -> str:
    circuit = report.get("circuit", {})
    fault = report.get("fault_or_symptom", {})
    spice = report.get("spice_verification", {})
    risk = report.get("secondary_damage_risk", {})
    lines = ["# CircuitCare AI 진단 보고서", "",
             f"- 회로: {circuit.get('name', 'Unknown')}",
             f"- Circuit Hash: `{circuit.get('hash', 'NOT_EVALUATED')}`",
             f"- 생성 시각: {report.get('generated_at', 'NOT_EVALUATED')}",
             f"- 부품 / Net: {circuit.get('component_count', 0)} / {circuit.get('net_count', 0)}", "",
             "## 고장 및 관측", "",
             f"- 모드: {fault.get('mode', 'NOT_EVALUATED')}",
             f"- 대상: {fault.get('component_ref', 'NOT_EVALUATED')} / {fault.get('fault_type', 'NOT_EVALUATED')}",
             f"- 관측 분류: {fault.get('observation_classification', 'NOT_EVALUATED')}", "",
             "## 파라미터 후보 Grid", "",
             f"- 상태: {report.get('parameter_identification', {}).get('status', 'NOT_EVALUATED')}",
             f"- 해석: {report.get('parameter_identification', {}).get('interpretation', '후보 Grid 비교를 수행하지 않았습니다.')}", "",
             "## SPICE 검증", "",
             f"- 상태: {spice.get('status', 'NOT_EVALUATED')}",
             f"- 출처: {spice.get('source', 'NOT_EVALUATED')}",
             f"- Scenario ID: {spice.get('scenario_id') or 'NOT_EVALUATED'}", "",
             "## 정격 스트레스 검토", "",
             f"- 상태: {risk.get('status', 'RECORDED')}",
             "- 손상 확정 여부: 아니요", "",
             "## 근거", ""]
    for evidence in report.get("evidence", []):
        lines.append(f"- {evidence.get('evidence_id')}: {evidence.get('evidence_type')} · {evidence.get('verification_status')} · {evidence.get('source_artifact') or '원본 파일 없음'}")
    lines.extend(["", "## 측정값", "", "| 노드 | 값 | 단위 | 불확실도 | 출처 | 검증 상태 |",
                  "|---|---:|---|---:|---|---|"])
    for row in fault.get("observations", []):
        lines.append(f"| {row.get('node', '—')} | {row.get('value', '—')} | {row.get('unit', '—')} | {row.get('uncertainty', '—')} | {row.get('source', 'unknown')} | {row.get('verification_status', 'UNVERIFIED')} |")
    hypotheses = report.get("fault_influence", {}).get("hypotheses", [])
    lines.extend(["", "## 고장 가설", "", "| 부품 | 고장 유형 | 파라미터 | 생성 출처 | 검증 | 시뮬레이션 |",
                  "|---|---|---|---|---|---|"])
    for row in hypotheses:
        lines.append(f"| {row.get('component_ref', '—')} | {row.get('fault_type', '—')} | {json.dumps(row.get('parameters', {}), ensure_ascii=False)} | {row.get('generation_method', '—')} | {row.get('validation_status', '—')} | {row.get('simulation_status', '—')} |")
    nodes = spice.get("comparison", {}).get("voltage_nodes", {}) if isinstance(spice.get("comparison"), dict) else {}
    lines.extend(["", "## SPICE 비교 및 잔차", "", "| 노드 | Baseline | Fault | 변화 |", "|---|---:|---:|---:|"])
    for node, value in nodes.items():
        baseline = value.get("baseline", {}).get("last") if isinstance(value, dict) else None
        fault_value = value.get("fault", {}).get("last") if isinstance(value, dict) else None
        delta = value.get("delta_fault_minus_baseline", {}).get("last") if isinstance(value, dict) else None
        lines.append(f"| {node} | {baseline if baseline is not None else '—'} | {fault_value if fault_value is not None else '—'} | {delta if delta is not None else '—'} |")
    posterior = report.get("diagnosis", {}).get("posterior", [])
    lines.extend(["", "## 진단 순위 (설정 가설에 조건부)", "", "| 가설 ID | 부품 | Posterior | 잔차 점수 | 상태 |", "|---|---|---:|---:|---|"])
    for row in posterior:
        lines.append(f"| {row.get('hypothesis_id', '—')} | {row.get('component_ref', '—')} | {row.get('posterior', '—')} | {row.get('residual_score', '—')} | {row.get('status', '—')} |")
    candidates = risk.get("secondary_candidates", [])
    lines.extend(["", "## 위험 평가", "", "| 부품 | 위험 등급 | 스트레스/정격 | 이용률 | 근거 |", "|---|---|---|---:|---|"])
    for row in candidates:
        lines.append(f"| {row.get('component_ref', '—')} | {row.get('risk_level', row.get('status', '—'))} | {row.get('stress_description', '—')} | {row.get('utilization_ratio', '—')} | {row.get('rating_source', '—')} |")
    ranking = report.get("recommended_inspection", {}).get("ranking", [])
    lines.extend(["", "## 점검 우선순위", ""])
    lines.extend(f"- {row.get('rank', '—')}. {row.get('component_ref', '—')} · {row.get('reason', '근거 정보 없음')}" for row in ranking)
    preview = report.get("preview_context")
    if preview:
        lines.extend(["", "## What-if Preview (PREVIEW_ONLY)", "", f"- 상태: {preview.get('status')}",
                      f"- 측정 출처: {preview.get('measurement_source', 'unknown')}",
                      f"- 가상 측정값: {json.dumps(report.get('preview_measurements', []), ensure_ascii=False)}",
                      f"- Preview Posterior: {json.dumps(report.get('preview_diagnosis', {}).get('posterior', []), ensure_ascii=False)}"])
    lines.extend(["", "## 한계", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(["", "## 재현 정보", "", f"- Git SHA: `{report.get('reproducibility', {}).get('git_sha') or 'NOT_EVALUATED'}`", ""])
    pcb = report.get("pcb_context")
    if isinstance(pcb, dict):
        lines.extend(["## PCB 진단 Snapshot", "",
                      f"- 보드: {pcb.get('board_id') or '미지정'}",
                      f"- Registry: {pcb.get('registry_version') or '미구성'}",
                      f"- 분석 상태: {pcb.get('analysis_status', 'NOT_RUN')}",
                      f"- 소자 / 사진: {len(pcb.get('components', []))} / {len(pcb.get('photos', []))}",
                      f"- 선택 소자: {pcb.get('selected_board_component_id') or '없음'}", "",
                      "### 사진 참조", "", "| 사진 ID | 면 | 촬영 유형 | 파일 | SHA-256 |",
                      "|---|---|---|---|---|"])
        lines.extend(f"| {row.get('photo_id', '—')} | {row.get('side', '—')} | {row.get('capture_type', '—')} | {row.get('filename') or '—'} | {row.get('sha256') or 'NOT_EVALUATED'} |"
                     for row in pcb.get("photos", []))
        lines.extend(["", "### 현재 Top 3 후보", "", "| 순위 | 보드 소자 ID | 고장 가설 | 검증 상태 |",
                      "|---:|---|---|---|"])
        lines.extend(f"| {index} | {row.get('board_component_id') or row.get('component_id') or '미특정'} | {row.get('fault_title') or row.get('fault_hypothesis') or row.get('fault_id') or '미확정'} | {row.get('visual_status') or row.get('status') or '미확정'} |"
                     for index, row in enumerate(pcb.get("top3_candidates", []), start=1))
        lines.extend(["", "### PCB 증상·측정 Snapshot", "", "```json",
                      json.dumps({"symptoms": pcb.get("symptoms", []), "measurements": pcb.get("measurements", [])},
                                 ensure_ascii=False, indent=2, default=str), "```", ""])
    return "\n".join(lines)


def export_html(report: dict[str, Any]) -> str:
    circuit = report.get("circuit", {})
    fault = report.get("fault_or_symptom", {})
    spice = report.get("spice_verification", {})
    risk = report.get("secondary_damage_risk", {})
    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(item if item is not None else '—'))}</td>" for item in row) + "</tr>" for row in rows)
        return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    measurements = [[row.get(k, "—") for k in ("node", "value", "unit", "uncertainty", "source", "verification_status")]
                    for row in fault.get("observations", [])]
    hypotheses = [[row.get(k, "—") for k in ("component_ref", "fault_type", "parameters", "generation_method", "validation_status", "simulation_status")]
                  for row in report.get("fault_influence", {}).get("hypotheses", [])]
    risk_rows = [[row.get("component_ref", "—"), row.get("risk_level", row.get("status", "—")),
                  row.get("stress_description", "—"), row.get("utilization_ratio", "—"), row.get("rating_source", "—")]
                 for row in risk.get("secondary_candidates", [])]
    pcb = report.get("pcb_context") if isinstance(report.get("pcb_context"), dict) else None
    pcb_html = ""
    if pcb:
        photo_rows = [[row.get("photo_id"), row.get("side"), row.get("capture_type"),
                       row.get("filename"), row.get("sha256")] for row in pcb.get("photos", [])]
        candidate_rows = [[index, row.get("board_component_id") or row.get("component_id"),
                           row.get("fault_title") or row.get("fault_hypothesis") or row.get("fault_id"),
                           row.get("visual_status") or row.get("status")] for index, row in enumerate(pcb.get("top3_candidates", []), start=1)]
        pcb_html = (f"<section><h2>PCB 진단 Snapshot</h2><p>보드 {html.escape(str(pcb.get('board_id') or '미지정'))} · "
                    f"Registry {html.escape(str(pcb.get('registry_version') or '미구성'))} · "
                    f"분석 {html.escape(str(pcb.get('analysis_status', 'NOT_RUN')))}</p>"
                    f"{table(['사진 ID','면','촬영 유형','파일','SHA-256'], photo_rows)}"
                    f"{table(['순위','보드 소자 ID','고장 가설','검증 상태'], candidate_rows)}"
                    f"<h3>증상·측정 Snapshot</h3><pre>{html.escape(json.dumps({'symptoms': pcb.get('symptoms', []), 'measurements': pcb.get('measurements', [])}, ensure_ascii=False, indent=2, default=str))}</pre></section>")
    return ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>CircuitCare AI 진단 보고서</title><style>body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f8fafc}"
            "section{background:white;border:1px solid #d8e0e8;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;border-bottom:1px solid #e2e8f0;padding:.45rem;overflow-wrap:anywhere}"
            ".table-wrap{overflow:auto}.badge{font-weight:600;color:#174ea6}</style></head><body><h1>CircuitCare AI 진단 보고서</h1>"
            f"<section><h2>진단 개요</h2><p>회로 {html.escape(str(circuit.get('name', 'Unknown')))} · 부품 {circuit.get('component_count', 0)} · Net {circuit.get('net_count', 0)}</p>"
            f"<p>고장 후보 {html.escape(str(fault.get('component_ref', 'NOT_EVALUATED')))} / {html.escape(str(fault.get('fault_type', 'NOT_EVALUATED')))} · Circuit Hash <code>{html.escape(str(circuit.get('hash', 'NOT_EVALUATED')))}</code></p></section>"
            f"<section><h2>측정값</h2>{table(['노드','값','단위','불확실도','출처','검증'], measurements)}</section>"
            f"<section><h2>가설 및 검증 상태</h2>{table(['부품','고장 유형','파라미터','출처','검증','시뮬레이션'], hypotheses)}</section>"
            f"<section><h2>SPICE 비교</h2><p>상태: {html.escape(str(spice.get('status', 'NOT_EVALUATED')))} · 출처: {html.escape(str(spice.get('source', 'NOT_EVALUATED')))}</p></section>"
            f"<section><h2>위험 평가</h2><p>손상 확정은 보고하지 않습니다.</p>{table(['부품','등급','스트레스/정격','이용률','정격 근거'], risk_rows)}</section>"
            f"<section><h2>점검 우선순위</h2>{html.escape(json.dumps(report.get('recommended_inspection', {}), ensure_ascii=False, indent=2))}</section>"
            f"{pcb_html}<section><h2>근거 및 한계</h2>{table(['ID','Evidence Type','검증 상태','출처','ZIP 포함'], [[e.get('evidence_id'),e.get('evidence_type'),e.get('verification_status'),e.get('source_artifact'),e.get('included_in_bundle',False)] for e in report.get('evidence', [])])}"
            f"<ul>{''.join('<li>'+html.escape(str(x))+'</li>' for x in report.get('limitations', []))}</ul></section></body></html>")


def export_bundle(report: dict[str, Any], *, root: Path,
                  selected_artifacts: dict[str, Path] | None = None) -> bytes:
    """Bundle report outputs and only explicitly selected, in-root artifacts."""
    bundled_report = copy.deepcopy(report)
    files: dict[str, bytes] = {}
    photo_artifacts = (bundled_report.get("pcb_context", {}).pop("photo_artifacts", [])
                       if isinstance(bundled_report.get("pcb_context"), dict) else [])
    photo_count = 0
    total_photo_bytes = 0
    for item in photo_artifacts:
        archive_path = str(item.get("archive_path", ""))
        safe_path = PurePosixPath(archive_path)
        data = item.get("data")
        if (not archive_path.startswith("photos/") or safe_path.is_absolute()
                or ".." in safe_path.parts or "\\" in archive_path or not isinstance(data, bytes)):
            raise ValueError("Invalid PCB photo evidence reference.")
        if len(data) > 25 * 1024 * 1024:
            raise ValueError("PCB photo evidence exceeds the 25 MiB item limit.")
        photo_count += 1
        total_photo_bytes += len(data)
        if photo_count > 32 or total_photo_bytes > 50 * 1024 * 1024:
            raise ValueError("PCB photo evidence exceeds the bundle size limit.")
        files[archive_path] = data
        pcb = bundled_report.get("pcb_context", {})
        for row in pcb.get("photos", []):
            if row.get("photo_id") == item.get("photo_id"):
                row["bundle_included"] = True
                row["bundle_sha256"] = hashlib.sha256(data).hexdigest()
    allowed = {"measurements.json", "fault_hypotheses.json", "spice_results.json",
               "risk_assessment.json", "rating_evidence.json"}
    root = root.resolve()
    for archive_name, path in (selected_artifacts or {}).items():
        if archive_name not in allowed:
            raise ValueError(f"Artifact is not an approved bundle item: {archive_name}")
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence artifact must be inside the workspace root.") from exc
        if not resolved.is_file():
            raise ValueError(f"Evidence artifact does not exist: {archive_name}")
        artifact = resolved.read_bytes()
        if len(artifact) > 25 * 1024 * 1024:
            raise ValueError("Evidence artifact exceeds the 25 MiB bundle item limit.")
        files[f"evidence/{archive_name}"] = artifact
    included_by_name = {name.split("/", 1)[1]: hashlib.sha256(value).hexdigest()
                        for name, value in files.items() if name.startswith("evidence/")}
    artifact_aliases = {"comparison.json": "spice_results.json", "component_stress.json": "spice_results.json",
                        "risk_assessment.json": "risk_assessment.json", "measurements.json": "measurements.json",
                        "fault_hypotheses": "fault_hypotheses.json", "rating_evidence.json": "rating_evidence.json"}
    for evidence in bundled_report.get("evidence", []):
        source = evidence.get("source_artifact")
        if source is None:
            evidence.update(included_in_bundle=False, bundle_verification_status="SOURCE_NOT_BUNDLED")
            continue
        alias = artifact_aliases.get(Path(str(source)).name)
        digest = included_by_name.get(alias) if alias else None
        evidence["included_in_bundle"] = digest is not None
        if digest:
            evidence["source_artifact"] = f"evidence/{alias}"
            evidence["bundled_artifact_sha256"] = digest
            expected_digest = evidence.get("artifact_sha256")
            if not expected_digest:
                evidence["bundle_verification_status"] = "SOURCE_HASH_UNAVAILABLE"
            elif expected_digest == digest:
                evidence["bundle_verification_status"] = "VERIFIED_ARTIFACT"
            else:
                evidence["bundle_verification_status"] = "ARTIFACT_HASH_MISMATCH"
        else:
            evidence["bundle_verification_status"] = "SOURCE_NOT_BUNDLED"
    files["diagnosis.json"] = export_json(bundled_report).encode("utf-8")
    files["report.md"] = export_markdown(bundled_report).encode("utf-8")
    files["report.html"] = export_html(bundled_report).encode("utf-8")
    manifest = {"bundle_version": "circuitcare-evidence-bundle-v1",
                "workspace_id": report.get("workspace", {}).get("workspace_id"),
                "circuit_hash": report.get("circuit", {}).get("hash"),
                "scenario_id": report.get("spice_verification", {}).get("scenario_id"),
                "analysis_timestamp": report.get("generated_at"),
                "evidence_sources": bundled_report.get("evidence", []),
                "missing_artifacts": (["measurement raw file not selected"]
                                      + ([] if photo_count else ["PCB original image unavailable"])),
                "model_versions": report.get("reproducibility", {}),
                "artifacts": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}}
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    result = buffer.getvalue()
    verify_bundle(result)
    return result
