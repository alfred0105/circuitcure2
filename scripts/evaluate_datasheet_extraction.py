#!/usr/bin/env python3
"""Evaluate rules-only and rules-plus-local-LLM extraction on local official PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from circuitcare.datasheet.evaluation import score_extractions
from circuitcare.datasheet.evidence_validator import validate_candidate
from circuitcare.datasheet.llm_extractor import extract_with_local_llm
from circuitcare.datasheet.pdf_loader import extract_pdf_tables, load_pdf
from circuitcare.datasheet.rating_extractor import find_rating_candidates
from circuitcare.local_llm import LocalLLMError, OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/datasheet_extraction/manifest.json")
    parser.add_argument("--ground-truth", default="evaluation/datasheet_extraction/ground_truth/labels.json")
    parser.add_argument("--documents-dir", default="data/datasheets/evaluation")
    parser.add_argument("--output", default="evaluation/datasheet_extraction/results")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--model", default="gemma4:12b")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    labels = json.loads((ROOT / args.ground_truth).read_text(encoding="utf-8"))["records"]
    docs = {item["document_id"]: item for item in manifest.get("documents", [])}
    labels_by_doc = {}
    for label in labels:
        if label.get("verification_status") == "verified":
            labels_by_doc.setdefault(label["document_id"], []).append(label)
    rule_all, combined_all, documents = [], [], []
    local = OllamaClient() if args.run_llm else None
    model_metadata = None
    for document_id, doc in docs.items():
        path = ROOT / args.documents_dir / f"{document_id}.pdf"
        if not path.is_file():
            documents.append({"document_id": document_id, "status": "DOCUMENT_NOT_PROVIDED", "ground_truth_count": len(labels_by_doc.get(document_id, []))})
            continue
        data = path.read_bytes()
        parsed = load_pdf(data, path.name)
        tables = extract_pdf_tables(data)
        rule_start = time.perf_counter()
        rules = find_rating_candidates(parsed["pages"], tables)
        rule_latency = time.perf_counter() - rule_start
        llm_rows, model_info, llm_latency, llm_error = [], None, None, None
        if args.run_llm:
            try:
                raw = extract_with_local_llm(local, args.model, manufacturer=doc["manufacturer"], mpn=doc["mpn"],
                                             candidates=rules, pages=parsed["pages"], tables=tables)
                llm_rows = raw.get("ratings", [])
                model_info = raw.get("model_info")
                model_metadata = model_info
                llm_latency = raw.get("total_duration_ns") / 1e9 if isinstance(raw.get("total_duration_ns"), (int, float)) else None
            except (LocalLLMError, ValueError) as exc:
                llm_error = str(exc)[:500]
        for candidate in [*rules, *llm_rows]:
            candidate["manufacturer"] = doc["manufacturer"]
            candidate["mpn"] = doc["mpn"]
            candidate["source_document_sha256"] = parsed["sha256"]
        # Apply the same page quote / value / unit validator to both methods.
        rules_checked = [{**validate_candidate(dict(row), parsed["pages"], manufacturer=doc["manufacturer"], mpn=doc["mpn"]),
                          "document_id": document_id} for row in rules]
        llm_checked = [{**validate_candidate(dict(row), parsed["pages"], manufacturer=doc["manufacturer"], mpn=doc["mpn"]),
                        "document_id": document_id} for row in llm_rows]
        combined = _dedupe([*rules_checked, *llm_checked])
        truth = labels_by_doc.get(document_id, [])
        rule_metrics = score_extractions(truth, rules_checked, latency_seconds=rule_latency,
            full_document_ground_truth_complete=manifest.get("ground_truth_complete") is True)
        combined_metrics = score_extractions(truth, combined, latency_seconds=rule_latency + (llm_latency or 0),
            full_document_ground_truth_complete=manifest.get("ground_truth_complete") is True)
        rule_all.extend({**row, "document_id": document_id} for row in rules_checked)
        combined_all.extend({**row, "document_id": document_id} for row in combined)
        documents.append({"document_id": document_id, "status": "EVALUATED",
            "source_sha256": parsed["sha256"], "manifest_sha256_match": parsed["sha256"] == doc["source_sha256"],
            "page_count": parsed["page_count"], "table_count": len(tables),
            "ground_truth_count": len(truth), "rule_candidate_count": len(rules_checked),
            "llm_latency_seconds": llm_latency,
            "llm_candidate_count": len(llm_checked), "llm_status": "FAILED" if llm_error else ("SUCCESS" if args.run_llm else "NOT_RUN"),
            "llm_error": llm_error, "rule_only": rule_metrics,
            "rule_plus_llm": combined_metrics, "llm_model": model_info})
    all_truth = [row for group in labels_by_doc.values() for row in group]
    # Per-document metrics are the aggregate unit because a PDF can have many candidates.
    included_truth = [row for row in all_truth if any(doc.get("document_id") == row["document_id"] and doc.get("status") == "EVALUATED" for doc in documents)]
    measured_llm_latencies = [item["llm_latency_seconds"] for item in documents
                              if isinstance(item.get("llm_latency_seconds"), (int, float))]
    result = {
        "evaluation_status": "DEVELOPMENT_PILOT" if included_truth else "NOT_EVALUATED",
        "status": "PARTIALLY_EVALUATED" if included_truth else "NOT_EVALUATED",
        "dataset_id": manifest.get("dataset_id"), "dataset_version": manifest.get("version"),
        "split": manifest.get("split"), "independent_holdout": manifest.get("independent_holdout", False),
        "ground_truth_complete": manifest.get("ground_truth_complete", False),
        "run_metadata": {"git_commit_sha": _git_sha(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "os": platform.platform(), "llm": model_metadata,
            "model_requested": args.model if args.run_llm else None, "prompt_version": "datasheet-rating-v1" if args.run_llm else None,
            "documents_dir": Path(args.documents_dir).name, "llm_enabled": args.run_llm,
            "ngspice": "NOT_USED"},
        "document_count": len(docs), "evaluated_document_count": sum(item.get("status") == "EVALUATED" for item in documents),
        "manufacturer_count": len({docs[item["document_id"]]["manufacturer"] for item in documents if item.get("status") == "EVALUATED"}),
        "component_families": {family: sum(1 for item in documents if item.get("status") == "EVALUATED" and docs[item["document_id"]]["component_family"] == family)
                                for family in sorted({d["component_family"] for d in docs.values()})},
        "ground_truth_rating_count": len(included_truth),
        "candidate_counts": {"rule_only": len(rule_all), "rule_plus_llm": len(combined_all),
            "scored_in_target_scope_rule_only": sum(item.get("rule_only", {}).get("scored_candidate_count", 0) for item in documents),
            "scored_in_target_scope_rule_plus_llm": sum(item.get("rule_plus_llm", {}).get("scored_candidate_count", 0) for item in documents)},
        "llm_processing_seconds": {"total": sum(measured_llm_latencies) if measured_llm_latencies else None,
            "mean_per_document": sum(measured_llm_latencies) / len(measured_llm_latencies) if measured_llm_latencies else None,
            "evaluated_document_count": len(measured_llm_latencies), "status": "MEASURED" if measured_llm_latencies else "NOT_EVALUATED"},
        "rule_only": score_extractions(included_truth, rule_all,
            full_document_ground_truth_complete=manifest.get("ground_truth_complete") is True),
        "rule_plus_llm": score_extractions(included_truth, combined_all,
            full_document_ground_truth_complete=manifest.get("ground_truth_complete") is True), "documents": documents,
        "excluded_documents": manifest.get("not_acquired", []),
        "integrity_audit": {"status": "PASSED_TARGET_SCOPE_LABELS" if all(row.get("evaluation_scope", {}).get("complete_target_scope") is True for row in included_truth) else "PARTIALLY_EVALUATED",
            "full_document_inventory_complete": manifest.get("ground_truth_complete", False),
            "full_document_precision_evaluated": False,
            "unsupported_claim_and_hallucination_require_human_adjudication": True},
        "limitations": ["Target-rating metrics cover only a manually defined row/variant/cell, not all ratings in each PDF.",
                        "No candidate-level independent human error adjudications are present; hallucination and unsupported-claim rates are NOT_EVALUATED.",
                        "Small pilot, one document per scored family.", "PDFs are not stored in Git; local runs require exact SHA-matching files.",
                        "The Phase 14 YAGEO RC0805FR-071KL development smoke document is excluded.",
                        "No result generalizes to unseen manufacturers, merged tables, or scan-only documents."],
    }
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    _archive_legacy_results(out)
    path = out / "corrected_scope_development_metrics.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "corrected_scope_rule_only.json").write_text(json.dumps({"evaluation_status": result["evaluation_status"], "candidates": rule_all}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "corrected_scope_rule_plus_llm.json").write_text(json.dumps({"evaluation_status": result["evaluation_status"], "candidates": combined_all}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marked_old = out / "comparison.json"
    if marked_old.is_file():
        previous = json.loads(marked_old.read_text(encoding="utf-8"))
        if previous.get("evaluation_status") != "INVALID_EVALUATION":
            previous["evaluation_status"] = "INVALID_EVALUATION"
            previous["integrity_note"] = "Pre-hotfix full-candidate precision/hallucination interpretation is invalid because labels were not exhaustive; original numeric results are preserved for audit only."
            marked_old.write_text(json.dumps(previous, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "status": result["status"], "rule_only": result["rule_only"], "rule_plus_llm": result["rule_plus_llm"], "documents": documents}, ensure_ascii=False, indent=2))
    return 0


def _dedupe(rows):
    output, seen = [], set()
    for row in rows:
        key = (row.get("rating_key"), row.get("normalized_value"), row.get("normalized_unit"), row.get("pdf_page_index"), row.get("source_text"))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _archive_legacy_results(output: Path) -> None:
    archive = output / "pre_integrity_hotfix"
    archive.mkdir(exist_ok=True)
    for name in ("comparison.json", "rule_only.json", "rule_plus_llm.json"):
        source = output / name
        target = archive / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
            if name == "comparison.json":
                old = json.loads(target.read_text(encoding="utf-8"))
                old["evaluation_status"] = "INVALID_EVALUATION"
                old["integrity_note"] = "Preserved pre-hotfix run; incomplete ground truth made full-candidate precision and hallucination interpretation invalid."
                target.write_text(json.dumps(old, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
