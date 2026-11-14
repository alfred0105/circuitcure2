from __future__ import annotations

from types import SimpleNamespace

import base64
import io
import json

from PIL import Image
import pytest

from circuitcare.local_llm import OllamaClient
from circuitcare.vision.ai_review_service import (
    AIReviewStageError, _review_images, review_components_with_qwen,
)
from circuitcare.vision.box_reconciliation import associate_ocr_with_components
from circuitcare.vision.box_reconciliation import (
    board_visible_review_rows, boxes_overlap_candidate, deduplicate_component_boxes,
    exclude_review_row, find_group_box_candidates, preserve_review_rows,
    remove_unreviewed_legacy_secondary_rows, same_component_box_pair,
)
from circuitcare.vision.component_recognition_panel import (
    CLASS_OPTIONS, _legacy_secondary_scope, _migrate_legacy_review_state, _review_fingerprint,
    _run_ai_review_workflow, invalidate_qwen_review, review_aux_keys,
    review_keys, review_scope, _review_detection_batches, _marking_ocr_targets,
    _missing_class_review_ids,
    _make_group_child_row, _apply_qwen_proposal_to_row,
    record_box_exclusion,
)
from circuitcare.component_taxonomy import (
    COMPONENT_CLASSES, COMPONENT_LABELS, QWEN_REVIEW_CLASSES, QWEN_SUBTYPE_DEFINITIONS, QWEN_SUBTYPES,
    normalize_component_class, normalize_qwen_class, normalize_qwen_subtype,
)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(output, format="PNG")
    return output.getvalue()


def test_ollama_chat_json_attaches_base64_image_to_user_message(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "model_info", lambda _name: {"name": "qwen3.5:9b"})
    calls = []

    def request(method, path, body=None):
        calls.append((method, path, body))
        return {"model": "qwen3.5:9b", "message": {"content": "{\"ok\":true}"}, "done": True}

    monkeypatch.setattr(client, "_request", request)
    image_bytes = _png()
    result = client.chat_json("qwen3.5:9b", [{"role": "user", "content": "Review this."}], images=[image_bytes])

    user_message = calls[0][2]["messages"][0]
    assert user_message["images"] == [base64.b64encode(image_bytes).decode("ascii")]
    assert result["content"] == {"ok": True}


def test_qwen_review_validates_detection_ids_and_unverified_boxes():
    class FakeClient:
        def __init__(self):
            self.images = None
            self.schema = None

        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            self.images = images
            self.schema = output_schema
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [
                    {"detection_id": "r1", "decision": "suggest", "proposed_class": "capacitor",
                     "confidence": .76, "evidence": "Two terminals and nearby C12 marking."},
                    {"detection_id": "missing-id", "decision": "suggest", "proposed_class": "IC",
                     "confidence": .99, "evidence": "Unknown target."},
                ],
                "missed_component_candidates": [
                    {"image_index": 0, "bbox_norm_999": [600, 400, 800, 700], "proposed_class": "capacitor",
                     "confidence": .66, "evidence": "Visible unboxed body."},
                    {"image_index": 0, "bbox_norm_999": [300, 300, 300, 400], "proposed_class": "resistor",
                     "confidence": .9, "evidence": "Degenerate box."},
                ],
                "notes": ["Review each suggestion manually."],
            }}

    client = FakeClient()
    result = review_components_with_qwen(
        client, "qwen3.5:9b", _png(), [
            {"component_id": "r1", "class_name": "resistor", "confidence": .42, "bbox_xyxy": [5, 5, 35, 25]},
            {"component_id": "r2", "class_name": "resistor", "confidence": .31, "bbox_xyxy": [8, 7, 37, 27]},
        ], [{"raw_text": "C12", "bbox_xyxy": [38, 5, 48, 12], "confidence": .9,
             "reference_candidates": ["C12"]}],
    )

    assert len(client.images) == 3  # annotated photo and two detection crops
    assert client.schema is not None
    assert len(result["class_reviews"]) == 1
    assert "duplicate_reviews" not in result
    assert len(result["missed_component_candidates"]) == 1
    assert result["verification_status"] == "AI_SUGGESTIONS_ONLY"
    candidate = result["missed_component_candidates"][0]
    assert candidate["review_status"] == "검수 필요"
    assert candidate["bbox_xyxy"][0] > 0


def test_qwen_unknown_class_is_reversibly_excluded_after_detection_review():
    row = {"component_id": "x", "소자 종류": "unknown", "상태": "검수 필요"}
    proposal = {"proposed_class": "unknown", "decision": "uncertain", "confidence": .4}
    assert _apply_qwen_proposal_to_row(row, proposal)
    record_box_exclusion({}, scope="test", row=row, reason="qwen_unknown", review=proposal)
    assert row["상태"] == "제외" and row["exclusion_reason"] == "qwen_unknown"
    assert board_visible_review_rows([row]) == []


def test_initial_yolo_detection_queues_all_active_boxes_for_qwen_9b(monkeypatch):
    from contextlib import nullcontext
    import circuitcare.vision.component_recognition_panel as panel

    queued = []

    class Detector:
        def load_component_locator(self):
            return None

        def run_prediction(self, _image, _options):
            return {"image_width": 100, "image_height": 80, "detections": [
                {"class_name": "resistor", "confidence": .8, "bbox_xyxy": [3, 4, 20, 18]},
                {"class_name": "capacitor", "confidence": .6, "bbox_xyxy": [22, 4, 39, 18]},
            ]}

    monkeypatch.setattr(panel, "_app_server", lambda: Detector())
    monkeypatch.setattr(panel, "_queue_initial_ai_review", lambda *args, **kwargs: queued.append(kwargs))
    api = SimpleNamespace(session_state={}, spinner=lambda *_args, **_kwargs: nullcontext())
    result, error = panel.ensure_component_detection(
        api, image_bytes=b"same-pcb", project_id="project", photo_id="photo",
    )
    assert error is None
    assert len(result["detections"]) == 2
    assert queued and len(queued[0]["rows"]) == 2
    assert all(row["상태"] != "제외" for row in queued[0]["rows"])


def test_silkscreen_reference_is_attached_as_class_hint_not_applied_as_fact():
    rows = associate_ocr_with_components(
        [{"component_id": "part-1", "class_name": "resistor", "bbox_xyxy": [0, 0, 20, 20]}],
        [{"text": "C12", "raw_text": "C12", "bbox_xyxy": [22, 4, 34, 11],
          "confidence": .93, "reference_candidates": ["C12"]}],
    )
    assert rows[0]["ocr_candidates"] == ["C12"]
    assert rows[0]["ocr_class_hints"] == ["capacitor"]
    assert rows[0]["reference_designators"] == ["C12"]
    assert rows[0]["class_name"] == "resistor"
    assert rows[0]["ocr_association_status"] == "ASSOCIATED"


def test_selected_box_review_runs_qwen_without_repeating_ocr_or_yolo(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import circuitcare.vision.component_recognition_panel as panel

    calls = []
    state = {}
    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "신뢰도": .8, "x1": 2, "y1": 2, "x2": 20, "y2": 18}]
    monkeypatch.setattr(panel, "_app_server", lambda: (_ for _ in ()).throw(AssertionError("YOLO called")))
    def run_qwen(*_args, **_kwargs):
        calls.append("qwen")
        return {"class_reviews": [], "review_image_count": 1,
                "ollama_metrics": {"total_duration_ns": 0}}

    monkeypatch.setattr("circuitcare.vision.ocr_backend.run_local_ocr",
                        lambda _image: (_ for _ in ()).throw(AssertionError("OCR reran")))
    monkeypatch.setattr("circuitcare.vision.ai_review_service.review_components_with_qwen", run_qwen)
    monkeypatch.setattr("circuitcare.local_llm.OllamaClient", lambda **_kwargs: object())
    st_api = SimpleNamespace(session_state=state, spinner=lambda *_args, **_kwargs: nullcontext(), warning=lambda *_args: None)
    _run_ai_review_workflow(st_api, image_bytes=_png(), scope="scope", rows=rows,
                            records_key="records", ocr_key="ocr", qwen_key="qwen",
                            selected_component_ids=["r1"])
    assert calls == ["qwen"]
    assert state["qwen"]["review_mode"] == "selected_class_only"


def test_shared_component_taxonomy_matches_detection_review_and_ui():
    assert len(COMPONENT_CLASSES) == 18
    assert set(CLASS_OPTIONS) == set(COMPONENT_LABELS)
    assert normalize_component_class("IC") == "ic"
    assert normalize_component_class("not-a-real-class") == "unknown"
    assert normalize_component_class("not_component") == "not_component"


def test_removing_second_pass_migrates_existing_reviewed_and_added_boxes():
    image_bytes = _png()
    old_scope = _legacy_secondary_scope(image_bytes, "project", "photo")
    new_scope = review_scope(image_bytes, "project", "photo")
    _old_result, old_records, _old_error = review_keys(old_scope)
    _new_result, new_records, _new_error = review_keys(new_scope)
    reviewed = {"component_id": "manual-old", "source": "manual", "user_edited": True,
                "상태": "확인", "소자 종류": "capacitor", "x1": 1, "y1": 1, "x2": 8, "y2": 8}
    added = {"component_id": "qwen-added", "source": "qwen_suggestion", "상태": "검수 필요",
             "소자 종류": "unknown", "x1": 10, "y1": 10, "x2": 20, "y2": 20}
    state = {old_records: [reviewed, added], new_records: [{**reviewed, "소자 종류": "resistor"}]}
    _migrate_legacy_review_state(state, image_bytes, "project", "photo", new_scope)
    assert {row["component_id"] for row in state[new_records]} == {"manual-old", "qwen-added"}
    assert next(row for row in state[new_records] if row["component_id"] == "manual-old")["소자 종류"] == "capacitor"
    assert old_records not in state


def test_qwen_uncertain_class_stays_unknown_and_weak_ocr_hint_is_preserved_as_conflict(monkeypatch):
    class TaxonomyClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            allowed = output_schema["properties"]["class_reviews"]["items"]["properties"]["proposed_class"]["enum"]
            assert set(allowed) == set(QWEN_REVIEW_CLASSES)
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [
                    {"detection_id": "r1", "decision": "suggest", "proposed_class": "capacitor",
                     "confidence": .8, "evidence": "Looks like a capacitor."},
                    {"detection_id": "r2", "decision": "uncertain", "proposed_class": "resistor",
                     "confidence": .4, "evidence": "Ambiguous body."},
                ], "missed_component_candidates": [], "notes": [],
            }}

    result = review_components_with_qwen(TaxonomyClient(), "qwen3.5:9b", _png(), [
        {"component_id": "r1", "class_name": "resistor", "bbox_xyxy": [2, 2, 24, 18],
         "ocr_class_hints": ["resistor"], "reference_designators": ["R1"]},
        {"component_id": "r2", "class_name": "unknown", "bbox_xyxy": [30, 2, 50, 18]},
    ], [])
    assert [(row["proposed_class"], row["decision"]) for row in result["class_reviews"]] == [
        ("capacitor", "suggest"), ("unknown", "uncertain")]
    assert result["class_reviews"][0]["review_status"] == "AI_CONFLICTED"
    assert result["class_reviews"][0]["needs_user_review"] is True
    assert result["class_reviews"][0]["detector_class"] == "resistor"


def test_qwen_extension_taxonomy_does_not_change_yolo_18_classes_and_validates_subtypes():
    assert len(COMPONENT_CLASSES) == 18
    assert normalize_qwen_class("FPC-FFC") == "unknown"
    assert normalize_qwen_class("other_component") == "other_component"
    assert normalize_qwen_subtype("other_component", "ic_socket") == "ic_socket"
    assert normalize_qwen_subtype("ic", "fpc_ffc") is None
    assert all(subtype in QWEN_SUBTYPE_DEFINITIONS
               for subtypes in QWEN_SUBTYPES.values() for subtype in subtypes)

    class ExtensionClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [{"detection_id": "j1", "decision": "suggest",
                    "proposed_class": "other_component", "proposed_base_class": "other_component",
                    "proposed_subtype": "ic_socket", "is_physical_component": True,
                    "is_board_feature": False, "confidence": .78,
                    "visual_evidence": "Empty socket body and repeated contacts.",
                    "alternative_classes": ["ic"], "marking_candidates": [],
                    "part_number_candidates": [], "functional_role_candidates": [],
                    "conflicting_evidence": [], "needs_user_review": True}],
                "group_reviews": [],
                "missed_component_candidates": [], "notes": [],
            }}

    result = review_components_with_qwen(ExtensionClient(), "qwen3.5:9b", _png(), [
        {"component_id": "j1", "class_name": "ic", "bbox_xyxy": [2, 2, 30, 30]},
    ], [])
    proposal = result["class_reviews"][0]
    assert proposal["detector_class"] == "ic"
    assert proposal["proposed_base_class"] == "other_component"
    assert proposal["proposed_subtype"] == "ic_socket"
    assert proposal["is_physical_component"] is True
    assert proposal["review_status"] == "AI_SUGGESTED"


@pytest.mark.parametrize(("detector_class", "qwen_class"), [("ic", "connector"), ("connector", "ic")])
def test_qwen_class_review_can_reclassify_ic_connector_both_directions(detector_class, qwen_class):
    class IndependentVisionClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            return {"model": "qwen3.5:9b", "content": {"class_reviews": [{
                "detection_id": "d1", "detector_class": detector_class,
                "decision": "suggest", "proposed_class": qwen_class,
                "proposed_base_class": qwen_class, "proposed_subtype": None,
                "is_physical_component": True, "is_board_feature": False,
                "confidence": .72, "visual_evidence": "Visible housing and contact arrangement.",
                "ocr_evidence": "No type solely inferred from designator.",
                "conflicting_evidence": [], "needs_user_review": True,
                "alternative_classes": [], "marking_candidates": [],
                "part_number_candidates": [], "functional_role_candidates": [],
            }], "group_reviews": [],
                "missed_component_candidates": [], "notes": []}}

    result = review_components_with_qwen(IndependentVisionClient(), "qwen3.5:9b", _png(), [
        {"component_id": "d1", "class_name": detector_class, "bbox_xyxy": [2, 2, 30, 30]},
    ], [])
    assert result["class_reviews"][0]["detector_class"] == detector_class
    assert result["class_reviews"][0]["proposed_base_class"] == qwen_class


def test_large_board_review_batches_cover_every_detection_once():
    detections = [{"detection_id": f"d{i}"} for i in range(53)]
    batches = _review_detection_batches(detections, batch_size=24)
    assert [len(batch) for batch in batches] == [24, 24, 5]
    assert [row["detection_id"] for batch in batches for row in batch] == [
        row["detection_id"] for row in detections]
    compact_batches = _review_detection_batches(detections)
    assert [len(batch) for batch in compact_batches] == [6, 6, 6, 6, 6, 6, 6, 6, 5]


def test_marking_ocr_targets_prioritize_ic_and_passives_and_skip_excluded_rows():
    rows = [{"component_id": f"r{i}", "소자 종류": "resistor", "상태": "검수 필요",
             "신뢰도": .8} for i in range(20)]
    rows += [{"component_id": "ic1", "소자 종류": "ic", "상태": "검수 필요", "신뢰도": .2},
             {"component_id": "excluded", "소자 종류": "ic", "상태": "제외", "신뢰도": .1},
             {"component_id": "mlcc", "소자 종류": "capacitor", "상태": "검수 필요", "신뢰도": .1}]
    targets = _marking_ocr_targets(rows, limit=12)
    assert len(targets) == 12
    assert targets[0]["component_id"] == "ic1"
    assert "excluded" not in {row["component_id"] for row in targets}
    assert "mlcc" not in {row["component_id"] for row in targets}


def test_containment_is_a_review_signal_but_small_true_component_is_not_suppressed():
    assert boxes_overlap_candidate([45, 45, 55, 55], [0, 0, 100, 100])
    assert not same_component_box_pair([45, 45, 55, 55], [0, 0, 100, 100])


def test_location_ensemble_suppresses_same_body_with_different_box_sizes():
    kept, suppressed = deduplicate_component_boxes([
        {"source": "parsa", "score": .78, "box": [10, 10, 110, 110]},
        {"source": "pcb_ai_doctor_v12n", "score": .72, "box": [20, 20, 100, 100]},
    ])
    assert len(kept) == 1
    assert len(suppressed) == 1
    assert kept[0]["source"] == "parsa"


def test_location_ensemble_keeps_partially_overlapping_distinct_boxes():
    kept, suppressed = deduplicate_component_boxes([
        {"source": "parsa", "score": .81, "box": [0, 0, 60, 60]},
        {"source": "pcb_ai_doctor_v12n", "score": .75, "box": [40, 0, 100, 60]},
    ])
    assert len(kept) == 2
    assert suppressed == []


def test_user_edited_yolo_box_and_class_survive_rerun():
    rows = preserve_review_rows([
        {"component_id": "yolo-1", "소자 종류": "capacitor", "상태": "검수 필요",
         "x1": 12, "y1": 12, "x2": 48, "y2": 48, "user_edited": True},
    ], [{"class_name": "resistor", "confidence": .9, "bbox_xyxy": [10, 10, 50, 50]}])
    assert rows[0]["component_id"] == "yolo-1"
    assert rows[0]["소자 종류"] == "capacitor"
    assert rows[0]["x1"] == 12


def test_qwen_still_checks_full_photo_when_yolo_has_no_boxes():
    class EmptyClient:
        def __init__(self):
            self.called = False

        def chat_json(self, _model, messages, *, output_schema, images, output_budget=None):
            self.called = True
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            assert payload["detector_proposals"] == []
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "missed_component_candidates": [{
                    "image_index": 0, "bbox_norm_999": [200, 200, 600, 600],
                    "proposed_class": "IC", "confidence": .7, "evidence": "Visible IC body.",
                }], "notes": [],
            }}

    client = EmptyClient()
    result = review_components_with_qwen(client, "qwen3.5:9b", _png(), [], [])
    assert client.called
    assert len(result["missed_component_candidates"]) == 1


def test_qwen_receives_overlapping_native_resolution_tiles_and_maps_tile_box():
    class TileClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            assert len(images) > 1
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "missed_component_candidates": [{
                    "image_index": 1, "bbox_norm_999": [100, 100, 300, 300],
                    "proposed_class": "capacitor", "confidence": .7, "evidence": "Small part in tile.",
                }], "notes": [],
            }}

    output = io.BytesIO()
    Image.new("RGB", (1800, 700), "white").save(output, format="PNG")
    result = review_components_with_qwen(TileClient(), "qwen3.5:9b", output.getvalue(), [], [])
    assert result["review_tiles"] == 2
    candidate = result["missed_component_candidates"][0]["bbox_xyxy"]
    assert candidate[0] > 0 and candidate[2] > candidate[0]


def test_class_review_avoids_redundant_tiles_on_native_photo():
    class ReviewClient:
        def __init__(self):
            self.images = None
            self.schema = None

        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            self.images = images
            self.schema = output_schema
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [], "group_reviews": [],
                "missed_component_candidates": [], "notes": [],
            }}

    output = io.BytesIO()
    Image.new("RGB", (1310, 1444), "white").save(output, format="PNG")
    detections = [
        {"component_id": "a", "class_name": "resistor", "bbox_xyxy": [100, 100, 180, 150]},
        {"component_id": "b", "class_name": "resistor", "bbox_xyxy": [110, 105, 190, 155]},
    ]
    client = ReviewClient()
    result = review_components_with_qwen(client, "qwen3.5:9b", output.getvalue(), detections, [])

    # The whole image is already at native resolution under the 1600px overview
    # cap, so extra overlapping tiles and duplicate-pair crops are redundant here.
    assert result["review_tiles"] == 0
    assert "duplicate_pair_crop_count" not in result
    assert len(client.images) == 3  # one overview plus one crop for each box
    assert client.schema["properties"]["class_reviews"]["maxItems"] == 6
    assert "duplicate_reviews" not in client.schema["properties"]


def test_ai_review_prompt_bounds_ocr_candidate_count_and_text_size():
    class ReviewClient:
        def __init__(self):
            self.messages = None

        def chat_json(self, _model, messages, *, output_schema, images, output_budget=None):
            self.messages = messages
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [], "group_reviews": [],
                "missed_component_candidates": [], "notes": [],
            }}

    client = ReviewClient()
    records = [{"raw_text": "X" * 1000, "bbox_xyxy": [0, 0, 4, 4], "confidence": .5}
               for _ in range(80)]
    review_components_with_qwen(client, "qwen3.5:9b", _png(), [], records)
    payload = json.loads(client.messages[1]["content"].split("\n", 1)[1])
    assert len(payload["nearby_ocr_text_candidates"]) == 48
    assert all(len(row["text"]) == 120 for row in payload["nearby_ocr_text_candidates"])


def test_cached_qwen_review_is_invalidated_when_source_box_changes():
    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 1, "y1": 1, "x2": 10, "y2": 10}]
    before = _review_fingerprint(rows, None)
    rows[0]["x1"] = 3
    assert _review_fingerprint(rows, None) != before
    _ocr_key, qwen_key = review_aux_keys("scope")
    session_state = {qwen_key: {"class_reviews": ["stale"]}}
    invalidate_qwen_review(session_state, "scope")
    assert qwen_key not in session_state


def test_review_fingerprint_tracks_user_decisions_without_invalidating_inference_cache():
    from circuitcare.vision.component_recognition_panel import (
        _inference_fingerprint, _observation_fingerprint, _ocr_evidence_fingerprint,
    )

    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 1, "y1": 1, "x2": 10, "y2": 10}]
    baseline = _review_fingerprint(rows, None)
    inference = _inference_fingerprint(rows)
    excluded = [{**rows[0], "상태": "제외"}]
    restored = [{**rows[0], "상태": "확인"}]
    assert _review_fingerprint(excluded, None) != baseline
    assert _review_fingerprint(restored, None) != baseline
    assert _inference_fingerprint(excluded) == inference
    assert _inference_fingerprint(restored) == inference
    assert _review_fingerprint([{**rows[0], "user_edited": True}], None) != baseline
    assert _review_fingerprint([{**rows[0], "confirmed_part_number": "LM358"}], None) != baseline
    assert _review_fingerprint([{**rows[0], "소자 종류": "capacitor"}], None) != baseline
    assert _review_fingerprint([{**rows[0], "x1": 2}], None) != baseline


def test_ocr_output_version_does_not_mutate_yolo_job_input_fingerprint():
    from circuitcare.vision.component_recognition_panel import (
        _inference_fingerprint, _observation_fingerprint, _ocr_evidence_fingerprint,
        _review_fingerprint,
    )

    rows = [{"component_id": "r1", "detector_class": "resistor", "소자 종류": "resistor",
             "상태": "검수 필요", "x1": 1, "y1": 2, "x2": 10, "y2": 12}]
    ocr = {"source_image_fingerprint": "image-v1", "engine": "RapidOCR",
           "records": [{"text": "R1", "bbox_xyxy": [1, 1, 4, 4]}]}
    observation_before = _observation_fingerprint(rows)
    review_before = _review_fingerprint(rows, None)
    inference_before_ocr = _inference_fingerprint(rows, None)
    with_ocr_rows = [{**rows[0], "ocr_candidates": ["R1"], "reference_designators": ["R1"]}]
    assert _observation_fingerprint(with_ocr_rows) == observation_before
    assert _review_fingerprint(with_ocr_rows, ocr) == review_before
    assert _ocr_evidence_fingerprint(ocr) != _ocr_evidence_fingerprint(None)
    assert _inference_fingerprint(with_ocr_rows, ocr) != inference_before_ocr


def test_ocr_enrichment_keeps_ai_job_request_identity_stable(monkeypatch):
    import circuitcare.vision.component_recognition_panel as panel
    submitted = []

    def fake_submit(**kwargs):
        submitted.append(kwargs)
        return "job-stable", "PENDING"

    monkeypatch.setattr(panel, "submit_ai_review", fake_submit)
    state = {"circuitcare_v2_current_project_id": "project-1"}
    api = SimpleNamespace(session_state=state)
    initial = [{"component_id": "r1", "detector_class": "resistor", "소자 종류": "resistor",
                "상태": "검수 필요", "x1": 1, "y1": 2, "x2": 10, "y2": 20}]
    enriched = [{**initial[0], "ocr_candidates": ["R1"], "reference_designators": ["R1"]}]
    for rows in (initial, enriched):
        panel.queue_ai_component_review(api, image_bytes=b"photo", scope="scope",
                                        rows=rows, records_key="rows", owner="owner",
                                        selected_component_ids=["r1"])
    assert len(submitted) == 2
    assert submitted[0]["fingerprint"] == submitted[1]["fingerprint"]
    assert submitted[0]["review_fingerprint"] == submitted[1]["review_fingerprint"]


def test_qwen_batch_setting_is_limited_to_benchmarked_sizes(monkeypatch):
    from circuitcare.vision.component_recognition_panel import _qwen_review_batch_size

    for size in (2, 4, 6, 8):
        monkeypatch.setenv("CIRCUITCARE_QWEN_BATCH_SIZE", str(size))
        assert _qwen_review_batch_size() == size
    monkeypatch.setenv("CIRCUITCARE_QWEN_BATCH_SIZE", "3")
    assert _qwen_review_batch_size() == 4
    monkeypatch.setenv("CIRCUITCARE_QWEN_BATCH_SIZE", "invalid")
    assert _qwen_review_batch_size() == 4


def test_qwen_batch_coverage_detects_omitted_or_unvalidated_ids():
    batch = [{"component_id": "c1"}, {"component_id": "c2"}]
    assert _missing_class_review_ids(batch, {"class_reviews": [
        {"detection_id": "c1"}, {"detection_id": "unknown-id"},
    ]}) == ["c2"]


def test_ai_review_worker_merge_preserves_user_rows_and_only_applies_ai_fields():
    from circuitcare.vision.component_recognition_panel import merge_ai_review_rows

    current = [
        {"component_id": "r1", "소자 종류": "capacitor", "상태": "확인", "user_edited": True,
         "confirmed_part_number": "C123", "x1": 5, "measurement_id": "m1"},
        {"component_id": "manual-1", "소자 종류": "resistor", "상태": "확인", "user_edited": True},
    ]
    worker = [
        {"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요", "x1": 1,
         "confirmed_part_number": "wrong", "possible_part_number": "R100", "ocr_candidates": ["R1"]},
        {"component_id": "stale-id", "소자 종류": "IC", "x1": 0, "ocr_candidates": ["U1"]},
    ]
    merged = merge_ai_review_rows(current, worker)
    assert merged[0]["소자 종류"] == "capacitor"
    assert merged[0]["상태"] == "확인" and merged[0]["user_edited"] is True
    assert merged[0]["confirmed_part_number"] == "C123"
    assert merged[0]["measurement_id"] == "m1"
    assert "possible_part_number" not in merged[0]
    assert merged[0]["ocr_candidates"] == ["R1"]
    assert len(merged) == 2 and merged[1]["component_id"] == "manual-1"


def test_user_excluded_duplicate_stays_hidden_after_new_detection_matches():
    old_rows = [
        {"component_id": "active", "소자 종류": "IC", "상태": "검수 필요",
         "x1": 10, "y1": 10, "x2": 110, "y2": 110},
        {"component_id": "duplicate", "소자 종류": "IC", "상태": "제외",
         "x1": 20, "y1": 20, "x2": 100, "y2": 100, "duplicate_of": "active"},
    ]
    rows = preserve_review_rows(old_rows, [{"class_name": "unknown", "confidence": .9,
                                           "bbox_xyxy": [12, 12, 108, 108]}])
    assert {row["component_id"] for row in rows} == {"active", "duplicate"}
    assert next(row for row in rows if row["component_id"] == "duplicate")["상태"] == "제외"
    assert [row["component_id"] for row in board_visible_review_rows(rows)] == ["active"]


def test_group_candidates_require_multiple_independent_contained_boxes():
    group = find_group_box_candidates([
        {"component_id": "outer", "bbox_xyxy": [0, 0, 200, 100]},
        {"component_id": "r1", "bbox_xyxy": [20, 20, 50, 50]},
        {"component_id": "c1", "bbox_xyxy": [120, 20, 150, 50]},
    ])
    assert len(group) == 1
    assert group[0]["parent"]["detection_id"] == "outer"
    assert {row["detection_id"] for row in group[0]["children"]} == {"r1", "c1"}
    assert group[0]["metrics"]["containment_min"] >= .72


def test_duplicate_boxes_and_single_connector_package_are_not_group_candidates():
    duplicate_pair = [
        {"component_id": "outer", "bbox_xyxy": [0, 0, 100, 100]},
        {"component_id": "same", "bbox_xyxy": [10, 10, 90, 90]},
    ]
    connector_package = [
        {"component_id": "connector", "bbox_xyxy": [0, 0, 140, 60]},
        {"component_id": "package_crop", "bbox_xyxy": [12, 10, 128, 50]},
    ]
    assert find_group_box_candidates(duplicate_pair) == []
    assert find_group_box_candidates(connector_package) == []


def test_large_parent_with_zero_or_one_child_is_sent_for_review_with_diagnostics():
    no_children = find_group_box_candidates([
        {"component_id": "large", "class_name": "unknown", "bbox_xyxy": [100, 100, 650, 650]},
    ], image_size=(1000, 1000))
    one_child = find_group_box_candidates([
        {"component_id": "large", "class_name": "unknown", "bbox_xyxy": [100, 100, 650, 650]},
        {"component_id": "c1", "class_name": "capacitor", "bbox_xyxy": [180, 180, 220, 220]},
    ], image_size=(1000, 1000))
    assert len(no_children) == len(one_child) == 1
    assert no_children[0]["children"] == []
    assert len(one_child[0]["children"]) == 1
    rejected = []
    assert find_group_box_candidates([
        {"component_id": "ordinary", "class_name": "capacitor", "confidence": .9,
         "bbox_xyxy": [100, 100, 220, 180]},
    ], image_size=(1000, 1000), diagnostics=rejected) == []
    assert rejected[0]["reason"] == "insufficient_children_or_size_evidence"


def test_known_class_parent_with_one_much_smaller_child_is_reviewed():
    groups = find_group_box_candidates([
        {"component_id": "outer", "class_name": "connector", "confidence": .9,
         "bbox_xyxy": [100, 100, 300, 200]},
        {"component_id": "small", "class_name": "resistor", "bbox_xyxy": [150, 130, 190, 170]},
    ], image_size=(1000, 1000))
    assert len(groups) == 1
    assert groups[0]["parent"]["detection_id"] == "outer"
    assert groups[0]["metrics"]["largest_child_area_ratio"] == pytest.approx(12.5)


def test_qwen_group_review_uses_focused_crop_and_rejects_unlisted_child_ids():
    class GroupClient:
        def chat_json(self, _model, messages, *, output_schema, images, output_budget=None):
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            assert len(payload["group_candidate_crops"]) == 1
            assert payload["group_candidate_crops"][0]["child_ids"] == ["r1", "c1"]
            assert len(images) >= 2
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "group_reviews": [
                    {"parent_id": "outer", "decision": "multiple_components", "child_ids": ["r1", "c1"],
                     "confidence": .91, "evidence": "Two separate component bodies."},
                    {"parent_id": "outer", "decision": "multiple_components", "child_ids": ["ghost", "c1"],
                     "confidence": .99, "evidence": "Invalid child ID."},
                ], "missed_component_candidates": [], "notes": [],
            }}

    result = review_components_with_qwen(GroupClient(), "qwen3.5:9b", _png(), [
        {"component_id": "outer", "class_name": "unknown", "bbox_xyxy": [2, 2, 95, 75]},
        {"component_id": "r1", "class_name": "resistor", "bbox_xyxy": [10, 10, 25, 25]},
        {"component_id": "c1", "class_name": "capacitor", "bbox_xyxy": [65, 10, 80, 25]},
    ], [])
    assert len(result["group_reviews"]) == 1
    assert result["group_reviews"][0]["decision"] == "multiple_components"


def test_group_review_maps_qwen_new_child_box_to_original_pixels():
    class GroupMissingClient:
        def chat_json(self, _model, messages, *, output_schema, images, output_budget=None):
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            assert payload["group_candidate_crops"][0]["parent_id"] == "outer"
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "group_reviews": [{"parent_id": "outer", "decision": "multiple_components",
                    "child_ids": ["c1"], "new_components": [
                        {"bbox_norm_999": [600, 500, 800, 750], "proposed_class": "resistor",
                         "confidence": .82, "evidence": "Separate body without an existing box."},
                    ], "confidence": .91, "evidence": "One known and one unboxed part."}],
                "missed_component_candidates": [], "notes": [],
            }}

    result = review_components_with_qwen(GroupMissingClient(), "qwen3.5:9b", _png(), [
        {"component_id": "outer", "class_name": "unknown", "bbox_xyxy": [0, 0, 98, 78]},
        {"component_id": "c1", "class_name": "capacitor", "bbox_xyxy": [10, 10, 25, 25]},
    ], [])
    review = result["group_reviews"][0]
    proposal = review["new_components"][0]
    assert review["decision"] == "multiple_components"
    assert proposal["bbox_xyxy"] == pytest.approx([60.06, 40.04, 80.08, 60.06], abs=1e-3)
    assert proposal["review_status"] == "검수 필요"
    assert result["review_diagnostics"]["qwen"]["status"] == "complete"


def test_group_review_rejects_new_box_outside_parent_bounds():
    class OutsideParentClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "group_reviews": [{"parent_id": "outer", "decision": "multiple_components",
                    "child_ids": ["c1"], "new_components": [
                        {"bbox_norm_999": [0, 0, 100, 100], "proposed_class": "resistor",
                         "confidence": .9, "evidence": "Outside the proposed parent."},
                    ], "confidence": .9, "evidence": "Bad coordinate proposal."}],
                "missed_component_candidates": [], "notes": [],
            }}

    result = review_components_with_qwen(OutsideParentClient(), "qwen3.5:9b", _png(), [
        {"component_id": "outer", "class_name": "unknown", "bbox_xyxy": [20, 15, 80, 65]},
        {"component_id": "c1", "class_name": "capacitor", "bbox_xyxy": [40, 30, 50, 40]},
    ], [])
    assert result["group_reviews"] == []
    rejected = [item for item in result["review_diagnostics"]["qwen"]["items"]
                if item.get("reason") == "group_candidate_outside_parent_box"]
    assert len(rejected) == 1


def test_qwen_false_split_remains_a_suggestion_until_user_action():
    class IncorrectSplitClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [],
                "group_reviews": [{"parent_id": "connector", "decision": "multiple_components",
                    "child_ids": ["pin_area_a", "pin_area_b"], "confidence": .99,
                    "evidence": "Mistakenly treats sections of one connector as separate."}],
                "missed_component_candidates": [], "notes": [],
            }}

    rows = [
        {"component_id": "connector", "class_name": "connector", "bbox_xyxy": [0, 0, 100, 60],
         "상태": "검수 필요"},
        {"component_id": "pin_area_a", "class_name": "unknown", "bbox_xyxy": [10, 12, 25, 25]},
        {"component_id": "pin_area_b", "class_name": "unknown", "bbox_xyxy": [68, 12, 83, 25]},
    ]
    result = review_components_with_qwen(IncorrectSplitClient(), "qwen3.5:9b", _png(), rows, [])
    assert result["group_reviews"][0]["decision"] == "multiple_components"
    assert rows[0]["상태"] == "검수 필요"
    assert "exclusion_reason" not in rows[0]


def test_not_component_is_exclusion_review_not_new_box_and_can_be_restored():
    class FalsePositiveClient:
        def chat_json(self, _model, _messages, *, output_schema, images, output_budget=None):
            return {"model": "qwen3.5:9b", "content": {
                "class_reviews": [{"detection_id": "noise", "decision": "suggest",
                                   "proposed_class": "not_component", "confidence": .86,
                                   "evidence": "This is a silkscreen mark, not a component."}],
                "group_reviews": [],
                "missed_component_candidates": [{"image_index": 0, "bbox_norm_999": [100, 100, 300, 300],
                    "proposed_class": "not_component", "confidence": .99, "evidence": "Not a component."}],
                "notes": [],
            }}

    result = review_components_with_qwen(FalsePositiveClient(), "qwen3.5:9b", _png(), [
        {"component_id": "noise", "class_name": "unknown", "bbox_xyxy": [10, 10, 40, 30]},
    ], [])
    assert result["class_reviews"][0]["proposed_class"] == "not_component"
    assert result["missed_component_candidates"] == []
    row = {"component_id": "noise", "상태": "검수 필요", "source": "yolo"}
    exclude_review_row(row, reason="qwen_not_component", review=result["class_reviews"][0])
    assert row["상태"] == "제외" and row["user_edited"] is True
    assert board_visible_review_rows([row]) == []
    restored = dict(row, **{"상태": "검수 필요"})
    assert board_visible_review_rows([restored]) == [restored]


def test_group_parent_exclusion_keeps_children_and_review_history():
    rows = [
        {"component_id": "outer", "source": "yolo", "상태": "검수 필요"},
        {"component_id": "r1", "source": "yolo", "상태": "확인"},
        {"component_id": "c1", "source": "qwen_suggestion", "상태": "검수 필요"},
    ]
    exclude_review_row(rows[0], reason="qwen_group_container", review={"child_ids": ["r1", "c1"]})
    assert {row["component_id"] for row in rows} == {"outer", "r1", "c1"}
    assert [row["component_id"] for row in board_visible_review_rows(rows)] == ["r1", "c1"]
    assert rows[0]["ai_review"]["child_ids"] == ["r1", "c1"]
    rerun_rows = preserve_review_rows(rows, [
        {"class_name": "unknown", "confidence": .8, "bbox_xyxy": [0, 0, 100, 100]},
        {"class_name": "resistor", "confidence": .8, "bbox_xyxy": [10, 10, 20, 20]},
        {"class_name": "capacitor", "confidence": .8, "bbox_xyxy": [60, 10, 70, 20]},
    ])
    saved_parent = next(row for row in rerun_rows if row["component_id"] == "outer")
    assert saved_parent["exclusion_reason"] == "qwen_group_container"
    assert saved_parent["ai_review"]["child_ids"] == ["r1", "c1"]


def test_group_child_ui_addition_stays_unconfirmed_and_rejects_duplicate():
    candidate = {"bbox_xyxy": [50, 20, 70, 40], "proposed_class": "resistor", "confidence": .8,
                 "evidence": "Visible separate body."}
    row, error = _make_group_child_row("scope", "outer", 0, candidate, [])
    assert error is None and row["상태"] == "검수 필요"
    assert row["source"] == "qwen_suggestion" and row["group_parent_id"] == "outer"
    duplicate, duplicate_error = _make_group_child_row("scope", "outer", 0, candidate,
        [{"x1": 51, "y1": 21, "x2": 69, "y2": 39}])
    assert duplicate is None and duplicate_error == "duplicates_existing_detection"


def test_qwen_transport_failure_is_reported_as_qwen_stage_error():
    class BrokenClient:
        def chat_json(self, *_args, **_kwargs):
            raise TimeoutError("Qwen timed out")

    with pytest.raises(AIReviewStageError) as error:
        review_components_with_qwen(BrokenClient(), "qwen3.5:9b", _png(), [], [])
    assert error.value.stage == "qwen"
    assert "timed out" in str(error.value)


def test_selected_classification_failure_is_not_misreported_as_candidate_search(monkeypatch):
    from types import SimpleNamespace
    import circuitcare.vision.component_recognition_panel as panel

    state = {}
    class TestClient:
        def model_supports_vision(self, _model): return True
    monkeypatch.setattr(panel, "_ollama_review_client", lambda: TestClient())
    def fail_classification(*_args, **_kwargs):
        raise AIReviewStageError("qwen", "selected-box classification failed")
    monkeypatch.setattr("circuitcare.vision.ai_review_service.review_components_with_qwen", fail_classification)
    st_api = SimpleNamespace(session_state=state)
    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "신뢰도": .8, "x1": 2, "y1": 2, "x2": 20, "y2": 18}]
    with pytest.raises(AIReviewStageError) as error:
        _run_ai_review_workflow(st_api, image_bytes=_png(), scope="s", rows=rows,
                                records_key="rows", ocr_key="ocr", qwen_key="qwen",
                                selected_component_ids=["r1"])
    assert error.value.stage == "qwen"
    assert "qwen" not in state
    assert "cc_component_ocr_s" not in state


def test_unreviewed_legacy_yolo2_rows_removed_but_user_reviewed_rows_remain():
    rows = [
        {"component_id": "old-unreviewed", "source": "yolo_second_pass", "상태": "검수 필요"},
        {"component_id": "old-confirmed", "source": "yolo_second_pass", "상태": "확인"},
        {"component_id": "old-edited", "source": "yolo_second_pass", "상태": "검수 필요", "user_edited": True},
        {"component_id": "manual", "source": "manual", "상태": "검수 필요"},
    ]
    retained = remove_unreviewed_legacy_secondary_rows(rows)
    assert {row["component_id"] for row in retained} == {"old-confirmed", "old-edited", "manual"}


def test_automatic_ai_review_is_cached_by_photo_observations(monkeypatch):
    import circuitcare.vision.component_recognition_panel as panel
    calls = []
    def fake_run(st_api, *, image_bytes, scope, rows, records_key, ocr_key, qwen_key, progress=None,
                 selected_component_ids=None):
        calls.append((scope, rows[0]["x1"]))
        selected = set(selected_component_ids or [])
        detections = panel._selected_classification_detections(rows, selected)
        st_api.session_state[qwen_key] = {
            "verification_status": "AI_SUGGESTIONS_ONLY", "review_mode": "selected_class_only",
            "selected_component_ids": sorted(selected),
            "selected_source_fingerprint": panel._selected_classification_fingerprint(
                b"pcb", detections, selected),
            "class_reviews": [],
        }
    monkeypatch.setattr(panel, "_run_ai_review_workflow", fake_run)
    st_api = SimpleNamespace(session_state={})
    rows = [{"component_id": "r1", "소자 종류": "unknown", "상태": "검수 필요",
             "x1": 2, "y1": 2, "x2": 20, "y2": 18}]
    assert panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="s", rows=rows,
                                            records_key="rows", selected_component_ids=["r1"]) == "complete"
    assert panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="s", rows=rows,
                                            records_key="rows", selected_component_ids=["r1"]) == "complete"
    assert len(calls) == 1
    changed = [{**rows[0], "x1": 3}]
    assert panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="s", rows=changed,
                                            records_key="rows", selected_component_ids=["r1"]) == "complete"
    assert len(calls) == 2


def test_excluding_and_restoring_box_reuses_selected_qwen_cache_without_running_ocr(monkeypatch):
    import circuitcare.vision.component_recognition_panel as panel
    calls = []

    def fake_run(st_api, *, image_bytes, scope, rows, records_key, ocr_key, qwen_key, progress=None,
                 selected_component_ids=None):
        calls.append(scope)
        selected = set(selected_component_ids or [])
        detections = panel._selected_classification_detections(rows, selected)
        st_api.session_state[qwen_key] = {
            "verification_status": "AI_SUGGESTIONS_ONLY", "review_mode": "selected_class_only",
            "selected_component_ids": sorted(selected),
            "selected_source_fingerprint": panel._selected_classification_fingerprint(
                image_bytes, detections, selected),
            "class_reviews": [],
        }

    monkeypatch.setattr(panel, "_run_ai_review_workflow", fake_run)
    state = {}
    st_api = SimpleNamespace(session_state=state)
    active = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
               "x1": 2, "y1": 2, "x2": 20, "y2": 18}]
    panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="photo", rows=active,
                                     records_key="rows", selected_component_ids=["r1"])
    excluded = [{**active[0], "상태": "제외"}]
    panel.invalidate_component_review_state(state, scope="photo", project_id="p", board_id="b")
    panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="photo", rows=excluded,
                                     records_key="rows", selected_component_ids=["r1"])
    panel.ensure_ai_component_review(st_api, image_bytes=b"pcb", scope="photo", rows=active,
                                     records_key="rows", selected_component_ids=["r1"])
    assert calls == ["photo"]
    ocr_key, qwen_key = panel.review_aux_keys("photo")
    assert ocr_key not in state and qwen_key in state


def test_reusing_running_job_preserves_submission_versions_without_copying_caches(monkeypatch):
    import circuitcare.vision.component_recognition_panel as panel
    rows = [{"component_id": "r1", "소자 종류": "resistor", "상태": "검수 필요",
             "x1": 1, "y1": 2, "x2": 10, "y2": 20}]
    state = {"cc_ai_review_job_scope": "job", "cc_ai_review_job_owner_scope": "owner",
             "cc_ai_review_job_fingerprint_scope": panel._observation_fingerprint(rows),
             "cc_ai_review_job_variant_scope": "selected_class_only:r1",
             "cc_ai_review_job_review_fingerprint_scope": panel._review_fingerprint(rows, None),
             "cc_ai_review_job_ocr_fingerprint_scope": panel._ocr_evidence_fingerprint(None)}
    original = state["cc_ai_review_job_review_fingerprint_scope"]
    rows[0]["상태"] = "제외"
    monkeypatch.setattr(panel, "get_ai_review_job", lambda **_kw: {"status": "RUNNING"})
    def forbidden(*_args, **_kwargs):
        raise AssertionError("reused job must not copy caches or submit again")
    monkeypatch.setattr(panel, "deepcopy", forbidden)
    monkeypatch.setattr(panel, "submit_ai_review", forbidden)
    assert panel.queue_ai_component_review(SimpleNamespace(session_state=state),
        image_bytes=b"photo", scope="scope", rows=rows, records_key="rows", owner="owner",
        selected_component_ids=["r1"]) == (
            "job", "owner", "RUNNING")
    assert state["cc_ai_review_job_review_fingerprint_scope"] == original


def test_native_overview_avoids_redundant_tiles_and_followups_keep_component_crops():
    from circuitcare.vision.ai_review_service import _review_images
    output = io.BytesIO()
    Image.new('RGB', (1600, 1400), 'white').save(output, format='PNG')
    detections = [{'detection_id': 'r1', 'bbox_xyxy': [10, 10, 80, 50], 'confidence': .8}]
    first = _review_images(output.getvalue(), detections, include_full_image=True)
    later = _review_images(output.getvalue(), detections, include_full_image=False)
    assert not any(item['kind'] == 'native_resolution_tile' for item in first[2])
    assert any(item['kind'] == 'annotated_full_image' for item in first[2])
    assert {item['kind'] for item in later[2]} == {'detection_crop'}
    assert later[2][0]['detection_id'] == 'r1'
    assert len(later[0]) < len(first[0])
