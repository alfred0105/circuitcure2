import json

from circuitcare.benchmark import pcb_validation_cli


def test_live_dashboard_snapshot_is_written_after_each_case_with_source_provenance(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [
        {"case_id": "real-board-1", "board_id": "board-1", "case_kind": "REAL_CONFIRMED",
         "dataset_name": "Reviewed PCB set", "dataset_url": "https://example.test/dataset",
         "photo_source": "user-reviewed reference image", "label_source": "manual adjudication",
         "photo_paths": [{"photo_id": "front", "path": "/tmp/front.png"}],
         "inference_input": {},
         "reference": {"adjudication_status": "CONFIRMED", "evidence_sources": ["service log"]}},
    ]}), encoding="utf-8")

    monkeypatch.setattr(pcb_validation_cli, "run_case", lambda *_args, **_kwargs: {
        "run_id": "run-1", "case_id": "real-board-1", "board_id": "board-1",
        "case_kind": "REAL_CONFIRMED", "execution_mode": "REAL_MODEL", "status": "OK",
        "duration_ms": 123.0, "stage_trace": [], "detections": [],
        "hypotheses": [], "all_hypotheses": [], "top_candidates": [],
        "all_component_candidates": [], "active_inspection_plan": {},
    })
    monkeypatch.setattr(pcb_validation_cli, "production_yolo_detector", lambda _input: {})
    monkeypatch.setattr(pcb_validation_cli, "production_qwen_reviewer", lambda *_args: {})
    monkeypatch.setattr(pcb_validation_cli, "save_run", lambda path, data: path)
    monkeypatch.setattr("sys.argv", ["circuitcare-validate", "run", str(cases_path),
                                     str(tmp_path / "results"), "--mode", "REAL_MODEL", "--dashboard"])

    assert pcb_validation_cli.main() == 0

    snapshot = json.loads((tmp_path / "results" / "dashboard-live.json").read_text(encoding="utf-8"))
    assert snapshot["completed_cases"] == snapshot["total_cases"] == 1
    assert snapshot["runs"][0]["execution_mode"] == "REAL_MODEL"
    assert "detections" not in snapshot["runs"][0]
    assert snapshot["case_sources"][0]["dataset_url"] == "https://example.test/dataset"
    assert snapshot["case_sources"][0]["label_source"] == "manual adjudication"
    assert snapshot["case_sources"][0]["photo_paths"] == [{"photo_id": "front", "path": "/tmp/front.png"}]


def test_source_label_proxy_is_separate_from_adjudicated_detection_metrics():
    cases = [{"case_id": "source-proxy", "reference": {
        "source_label_proxy_components": [{"component_type": "resistor", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "source-proxy", "execution_mode": "REAL_MODEL", "detections": [
        {"class_name": "resistor", "bbox_xyxy": [0, 0, 10, 10]},
    ]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["status"] == "SOURCE_LABEL_PROXY"
    assert proxy["tp"] == 1 and proxy["fp"] == 0 and proxy["fn"] == 0
    assert proxy["f1"] == 1.0
    assert proxy["component_detection_rate"]["value"] == 1.0
    assert proxy["component_recognition_rate"]["value"] == 1.0
    assert "독립 검수 정답이 아니므로" in proxy["note"]


def test_source_label_proxy_does_not_score_failed_yolo_as_zero_recall():
    cases = [{"case_id": "failed-source-proxy", "reference": {
        "source_label_proxy_components": [{"component_type": "ic", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "failed-source-proxy", "execution_mode": "REAL_MODEL",
             "status": "PIPELINE_FAILED", "detections": [],
             "stage_trace": [{"stage": "yolo_detection:photo-1", "status": "ERROR"}]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["status"] == "NOT_EVALUATED"
    assert proxy["cases"] == 0
    assert proxy["fn"] == 0


def test_source_label_proxy_ignores_predictions_for_unlabeled_classes():
    cases = [{"case_id": "partial-labels", "reference": {
        "source_label_proxy_components": [{"component_type": "ic", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "partial-labels", "execution_mode": "REAL_MODEL", "detections": [
        {"class_name": "ic", "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "resistor", "bbox_xyxy": [20, 20, 30, 30]},
    ]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["tp"] == 1 and proxy["fp"] == 0 and proxy["fn"] == 0
    assert proxy["excluded_unlabeled_class_predictions"] == 1


def test_detection_rate_and_type_recognition_rate_are_separate():
    cases = [{"case_id": "wrong-type", "reference": {
        "source_label_proxy_components": [{"component_type": "ic", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "wrong-type", "execution_mode": "REAL_MODEL", "detections": [
        {"class_name": "capacitor", "bbox_xyxy": [0, 0, 10, 10]},
    ]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["component_detection_rate"]["value"] == 1.0
    assert proxy["component_recognition_rate"]["value"] == 0.0


def test_source_label_proxy_recognizes_production_korean_component_type_field():
    cases = [{"case_id": "app-row", "reference": {
        "source_label_proxy_components": [{"component_type": "ic", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "app-row", "execution_mode": "REAL_MODEL", "detections": [
        {"소자 종류": "ic", "bbox_xyxy": [0, 0, 10, 10]},
    ]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["component_detection_rate"]["value"] == 1.0
    assert proxy["component_recognition_rate"]["value"] == 1.0


def test_duplicate_ic_boxes_are_separated_from_wrong_location_boxes():
    cases = [{"case_id": "duplicates", "reference": {
        "source_label_proxy_components": [{"component_type": "ic", "bbox_xyxy": [0, 0, 10, 10]}],
    }}]
    runs = [{"case_id": "duplicates", "execution_mode": "REAL_MODEL", "detections": [
        {"소자 종류": "ic", "bbox_xyxy": [0, 0, 10, 10]},
        {"소자 종류": "ic", "bbox_xyxy": [1, 0, 11, 10]},
        {"소자 종류": "ic", "bbox_xyxy": [20, 20, 30, 30]},
    ]}]

    proxy = pcb_validation_cli._source_label_proxy(cases, runs)

    assert proxy["component_detection_rate"]["value"] == 1.0
    assert proxy["duplicate_detection_rate"]["duplicate_boxes"] == 1
    assert proxy["duplicate_detection_rate"]["value"] == 0.5
    assert proxy["component_location_false_positive_rate"]["false_positive"] == 1
    assert proxy["component_location_false_positive_rate"]["duplicate_ic_boxes_excluded"] == 1
