from circuitcare.vision.detection_evaluation import (
    evaluate_detections, validate_annotation_manifest, validate_board_splits,
)
import pytest


def _manifest_metadata():
    return {"schema_version": "circuitcare.component-annotation.v1", "dataset_id": "test-dataset",
        "dataset_name": "Test Dataset", "dataset_source": "test source", "dataset_source_url": "https://example.org",
        "source_version": "test-version", "dataset_license": "test-license", "annotation_source": "manual",
        "annotation_review_status": "MIXED_REVIEW", "split_name": "validation",
        "split_method": "board-grouped", "board_grouping_policy": "one board revision per split",
        "split_policy": "board-grouped"}


def test_evaluation_reports_map5095_small_recall_confidence_curve_and_duplicate_fp():
    truth = {"a.jpg": [{"class_id": 0, "bbox_xyxy": [0, 0, 20, 20]},
                       {"class_id": 0, "bbox_xyxy": [30, 30, 80, 80]}]}
    predictions = {"a.jpg": [
        {"class_name": "resistor", "confidence": 0.9, "bbox_xyxy": [0, 0, 20, 20]},
        {"class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [0, 0, 20, 20]},
        {"class_name": "resistor", "confidence": 0.7, "bbox_xyxy": [30, 30, 80, 80]},
    ]}
    metrics = evaluate_detections(truth, predictions, class_names=["resistor"],
                                  confidence_threshold=0.25, confidence_thresholds=[0.5])
    assert metrics["true_positive_count"] == 2
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 0
    assert metrics["map50"] == pytest.approx(5 / 6)
    assert metrics["map50_95"] == pytest.approx(5 / 6)
    assert metrics["classes"]["resistor"]["small_object_ground_truth_count"] == 1
    assert metrics["classes"]["resistor"]["small_object_recall_at_confidence_threshold"] == 1.0
    assert metrics["confidence_threshold_curve"][0]["precision"] == 2 / 3


def test_board_split_leakage_detects_related_images_and_duplicate_bytes():
    result = validate_board_splits([
        {"image_id": "board-a-top", "board_id": "board-a", "split": "training", "image_sha256": "same"},
        {"image_id": "board-a-close", "board_id": "board-a", "split": "sealed_test", "image_sha256": "other"},
        {"image_id": "board-b", "board_id": "board-b", "split": "validation", "image_sha256": "same"},
    ])
    assert result["status"] == "LEAKAGE_OR_MANIFEST_ERROR"
    codes = {row["code"] for row in result["errors"]}
    assert "BOARD_SPLIT_LEAKAGE" in codes
    assert "IMAGE_HASH_SPLIT_LEAKAGE" in codes


def test_annotation_manifest_requires_reviewed_truth_to_be_scoreable():
    row = {"component_id": "U1", "class": "integrated_circuit", "bbox_xyxy": [1, 1, 9, 9],
           "occlusion_state": "VISIBLE", "small_component": False,
           "annotation_status": "DRAFT_ANNOTATION", "reviewer": "annotator-a", "source": "manual"}
    image = {"image_id": "image-1", "image_path": "images/image-1.png", "label_path": "labels/image-1.txt",
             "board_id": "board-a", "board_revision": "rev-1",
             "board_side": "TOP", "split": "validation", "image_sha256": "a" * 64,
             "image_size_px": [10, 10], "source": "camera", "label_state": "LABELED", "annotations": [row]}
    manifest = {**_manifest_metadata(), "images": [image]}
    result = validate_annotation_manifest(manifest, class_names=["integrated_circuit"])
    assert result["status"] == "VALID"
    assert result["scoreable_status"] == "BLOCKED_NO_REVIEWED_GROUND_TRUTH"
    row["annotation_status"] = "REVIEWED_GROUND_TRUTH"
    row["reviewer"] = "reviewer-1"
    row["provenance"] = {"reviewed_at": "2026-11-06", "review_method": "manual_visual_review"}
    reviewed = validate_annotation_manifest(manifest, class_names=["integrated_circuit"])
    assert reviewed["scoreable_status"] == "READY_REVIEWED_LABELS"
    draft = {**row, "component_id": "U2", "annotation_status": "DRAFT_ANNOTATION"}
    image["annotations"].append(draft)
    mixed = validate_annotation_manifest(manifest, class_names=["integrated_circuit"])
    assert mixed["scoreable_status"] == "BLOCKED_UNREVIEWED_ANNOTATIONS"


def test_manifest_rejects_out_of_bounds_boxes_unsupported_classes_and_board_leakage():
    base = {"component_id": "R1", "class": "inductor", "bbox_xyxy": [1, 1, 20, 20],
            "occlusion_state": "UNKNOWN", "small_component": True,
            "annotation_status": "REVIEWED_GROUND_TRUTH", "reviewer": "r1", "source": "manual",
            "provenance": {"reviewed_at": "2026-11-06", "review_method": "manual"}}
    images = []
    for image_id, split in (("v1", "training"), ("v2", "sealed_test")):
        images.append({"image_id": image_id, "image_path": f"{image_id}.jpg", "board_id": "same-board", "board_revision": "revA",
            "board_side": "TOP", "split": split, "image_sha256": (image_id * 64)[:64],
            "image_size_px": [10, 10], "source": "camera", "label_state": "LABELED", "annotations": [base]})
    manifest = {**_manifest_metadata(), "images": images}
    result = validate_annotation_manifest(manifest, class_names=["resistor"])
    errors = {row["code"] for row in result["errors"]}
    assert "INVALID_BOUNDING_BOX" in errors
    assert "UNSUPPORTED_CLASS" in errors
    assert "BOARD_SPLIT_LEAKAGE" in errors


def test_empty_ground_truth_images_and_prediction_only_images_count_false_positives():
    result = evaluate_detections({"negative.jpg": []}, {"negative.jpg": [
        {"class_name": "resistor", "confidence": 0.9, "bbox_xyxy": [1, 1, 4, 4]},
    ], "unlisted-negative.jpg": [
        {"class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [2, 2, 5, 5]},
    ]}, class_names=["resistor"])
    assert result["images_evaluated"] == 2
    assert result["true_positive_count"] == 0
    assert result["false_positive_count"] == 2
    assert result["false_negative_count"] == 0


def test_all_missed_and_duplicate_predictions_are_one_to_one():
    ground_truth = {"one.jpg": [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]}
    missed = evaluate_detections(ground_truth, {"one.jpg": []}, class_names=["resistor"])
    assert (missed["true_positive_count"], missed["false_positive_count"], missed["false_negative_count"]) == (0, 0, 1)
    duplicate_predictions = {"one.jpg": [
        {"class_name": "resistor", "confidence": 0.7, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_name": "resistor", "confidence": 0.7, "bbox_xyxy": [0, 0, 10, 10]},
    ]}
    first = evaluate_detections(ground_truth, duplicate_predictions, class_names=["resistor"])
    second = evaluate_detections(ground_truth, duplicate_predictions, class_names=["resistor"])
    assert first == second
    assert first["true_positive_count"] == 1 and first["false_positive_count"] == 1
    assert first["map50"] == pytest.approx(1.0)


def test_wrong_class_box_counts_one_false_positive_and_one_false_negative():
    result = evaluate_detections({"one.jpg": [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]},
        {"one.jpg": [{"class_name": "capacitor", "confidence": 0.95, "bbox_xyxy": [0, 0, 10, 10]}]},
        class_names=["resistor", "capacitor"])
    assert result["true_positive_count"] == 0
    assert result["false_positive_count"] == 1
    assert result["false_negative_count"] == 1


def test_evaluator_rejects_invalid_classes_boxes_and_confidence():
    truth = {"one.jpg": [{"class_id": 1, "bbox_xyxy": [0, 0, 1, 1]}]}
    with pytest.raises(ValueError, match="outside taxonomy"):
        evaluate_detections(truth, {}, class_names=["resistor"])
    with pytest.raises(ValueError, match="Unsupported prediction class"):
        evaluate_detections({}, {"one.jpg": [{"class_name": "unknown", "confidence": 0.9,
            "bbox_xyxy": [0, 0, 1, 1]}]}, class_names=["resistor"])
    with pytest.raises(ValueError, match="confidence"):
        evaluate_detections({}, {"one.jpg": [{"class_name": "resistor", "confidence": float("nan"),
            "bbox_xyxy": [0, 0, 1, 1]}]}, class_names=["resistor"])


def test_manifest_keeps_unresolved_boards_and_empty_images_unscoreable_until_reviewed():
    manifest = _manifest_metadata()
    manifest["annotation_review_status"] = "HUMAN_REVIEWED"
    manifest["images"] = [{"image_id": "negative", "image_path": "negative.jpg", "board_id": "board-1",
        "board_revision": "rev-1", "board_side": "UNKNOWN", "split": "validation",
        "image_sha256": "b" * 64, "image_size_px": [100, 100], "source": "camera",
        "label_state": "REVIEWED_EMPTY", "reviewer": "reviewer-1",
        "review_provenance": {"reviewed_at": "2026-11-06", "review_method": "full_image_visual_review"},
        "annotations": []}]
    result = validate_annotation_manifest(manifest, class_names=["resistor"])
    assert result["scoreable_status"] == "READY_REVIEWED_LABELS"
    manifest["images"][0]["board_id"] = "BOARD_ID_UNRESOLVED"
    unresolved = validate_annotation_manifest(manifest, class_names=["resistor"])
    assert unresolved["board_split"]["board_count"] == 0
    assert unresolved["scoreable_status"] == "BLOCKED_BOARD_ID_UNRESOLVED"


def test_malformed_manifest_rows_return_structured_errors_without_crashing():
    split_result = validate_board_splits([
        {"image_id": ["not", "a", "string"], "board_id": "b", "split": [], "image_sha256": None},
        None,
    ])
    assert {error["code"] for error in split_result["errors"]} == {
        "MISSING_ID_OR_INVALID_SPLIT", "INVALID_SPLIT_RECORD"
    }

    manifest = {**_manifest_metadata(), "images": [{
        "image_id": "bad-image", "image_path": "bad.jpg", "board_id": "b", "board_revision": "r",
        "board_side": [], "split": "validation", "image_sha256": "c" * 64,
        "image_size_px": [10, 10], "source": "camera", "label_state": [],
        "annotations": [{"component_id": [], "class": "resistor", "bbox_xyxy": "1234",
            "occlusion_state": "UNKNOWN", "small_component": None,
            "annotation_status": [], "reviewer": None, "source": "manual"}],
    }]}
    result = validate_annotation_manifest(manifest, class_names=["resistor"])
    codes = {error["code"] for error in result["errors"]}
    assert "INVALID_SIDE_OR_IMAGE_SIZE" in codes
    assert "INVALID_LABEL_STATE" in codes
    assert "INVALID_ANNOTATION_STATUS" in codes
    assert "INVALID_BOUNDING_BOX" in codes
