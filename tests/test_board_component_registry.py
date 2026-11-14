import json
from types import SimpleNamespace

import pytest

from circuitcare.frontend.board_component_registry import (
    REGISTRY_KEY, build_board_component_registry, confirm_component_match,
    current_source_observation_fingerprint, diagnosis_components, measurement_component_aliases, remove_component_match,
    resolve_measurement_board_component, source_component_key,
    measurement_view_model_for_board_component,
)


def _component(component_id, *, status="UNCONFIRMED", component_type="resistor", bbox=(1, 2, 8, 9)):
    return SimpleNamespace(
        component_id=component_id, component_type=component_type, bbox_xyxy=bbox,
        coordinate_frame="SOURCE_PHOTO_PIXEL", identification_status=status,
        presence_status="CONFIRMED" if status == "USER_CONFIRMED" else "UNCONFIRMED",
        component_type_status=status, ocr_marking="103" if status == "USER_CONFIRMED" else None,
        ocr_candidates=("R1",), possible_part_number=None, confirmed_functional_roles=(),
        evidence=("test evidence",), overlay_source="yolo",
    )


def _snapshots():
    return {
        "front": {"components": (_component("local-1", status="USER_CONFIRMED"),),
                  "image_fingerprint": "front-hash", "review_version": "front-review"},
        "back": {"components": (_component("local-1", bbox=(1, 2, 8, 9)),),
                 "image_fingerprint": "back-hash", "review_version": "back-review"},
        "close": {"components": (_component("close-1", bbox=(1, 2, 8, 9)),),
                  "image_fingerprint": "close-hash", "review_version": "close-review"},
    }


def test_registry_keeps_close_boxes_independent_without_explicit_confirmation():
    state = {}
    registry = build_board_component_registry(
        state, project_id="p1", board_id="b1", photo_snapshots=_snapshots(),
    )
    assert len(registry["components"]) == 3
    assert len(set(registry["source_to_board_id"].values())) == 3
    # No pixel coordinates are projected into board-level diagnosis inputs.
    assert all(item.bbox_xyxy is None for item in diagnosis_components(registry))


def test_user_confirmed_link_merges_sources_preserves_canonical_id_and_can_split():
    state = {}
    first = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    front_id = first["source_to_board_id"][front]
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    merged = build_board_component_registry(state, project_id="p1", board_id="b1",
                                            photo_snapshots=_snapshots())
    assert len(merged["components"]) == 2
    assert merged["source_to_board_id"][front] == merged["source_to_board_id"][back] == front_id
    merged_row = next(row for row in merged["components"] if row["board_component_id"] == front_id)
    assert merged_row["identity_status"] == "USER_CONFIRMED"
    assert set(merged_row["source_photo_ids"]) == {"front", "back"}

    assert remove_component_match(state, project_id="p1", board_id="b1",
                                  left_source_key=front, right_source_key=back)
    split = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    assert len(split["components"]) == 3
    assert split["source_to_board_id"][front] == front_id
    assert split["source_to_board_id"][front] != split["source_to_board_id"][back]


def test_replacing_a_photo_disables_old_identity_link_until_reconfirmed():
    state = {}
    before = _snapshots()
    build_board_component_registry(state, project_id="p1", board_id="b1", photo_snapshots=before)
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    replaced = _snapshots()
    replaced["front"] = {**replaced["front"], "image_fingerprint": "replacement-image"}
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=replaced)
    assert registry["source_to_board_id"][front] != registry["source_to_board_id"][back]
    assert any(item["reason"] == "PHOTO_REPLACED_RECONFIRMATION_REQUIRED"
               for item in registry["inactive_links"])


def test_redetection_review_version_disables_link_until_user_reconfirms():
    state = {}
    snapshots = _snapshots()
    build_board_component_registry(state, project_id="p1", board_id="b1",
                                   photo_snapshots=snapshots)
    front = source_component_key("front", "local-1")
    back = source_component_key("back", "local-1")
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=front, right_source_key=back)
    changed = _snapshots()
    changed["front"] = {**changed["front"], "review_version": "redetected"}
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=changed)
    assert registry["source_to_board_id"][front] != registry["source_to_board_id"][back]
    assert any(item["reason"] == "PHOTO_REDETECTED_RECONFIRMATION_REQUIRED"
               for item in registry["inactive_links"])


def test_merge_rejects_direct_and_transitive_same_photo_collisions_without_mutating_state():
    state = {}
    snapshots = {
        "a": {"components": (_component("a1"), _component("a2", bbox=(20, 20, 30, 30))),
              "image_fingerprint": "a-hash"},
        "b": {"components": (_component("b1"),), "image_fingerprint": "b-hash"},
    }
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    a1, a2, b1 = (source_component_key(photo, local)
                  for photo, local in (("a", "a1"), ("a", "a2"), ("b", "b1")))
    confirm_component_match(state, project_id="p1", board_id="b1",
                            left_source_key=a1, right_source_key=b1)
    merged = build_board_component_registry(state, project_id="p1", board_id="b1",
                                            photo_snapshots=snapshots)
    owner = next(iter(state[REGISTRY_KEY]))
    before_links = state[REGISTRY_KEY][owner]["confirmed_links"]
    with pytest.raises(ValueError, match="같은 사진"):
        confirm_component_match(state, project_id="p1", board_id="b1",
                                left_source_key=a2, right_source_key=b1)
    after_links = state[REGISTRY_KEY][owner]["confirmed_links"]
    assert after_links == before_links
    assert merged["source_to_board_id"][a1] == merged["source_to_board_id"][b1]
    assert merged["source_to_board_id"][a2] != merged["source_to_board_id"][b1]


def test_corrupt_transitive_same_photo_link_is_retained_but_never_merged():
    state = {}
    snapshots = {
        "a": {"components": (_component("a1"), _component("a2", bbox=(20, 20, 30, 30))),
              "image_fingerprint": "a-hash"},
        "b": {"components": (_component("b1"),), "image_fingerprint": "b-hash"},
    }
    registry = build_board_component_registry(state, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    a1 = source_component_key("a", "a1")
    a2 = source_component_key("a", "a2")
    b1 = source_component_key("b", "b1")
    links = [tuple(sorted((a1, b1))), tuple(sorted((a2, b1)))]
    owner = next(iter(state[REGISTRY_KEY]))
    entry = state[REGISTRY_KEY][owner]
    source_map = {source["source_key"]: source for row in registry["components"]
                  for source in row["source_components"]}
    entry["confirmed_links"] = [list(link) for link in links]
    entry["link_versions"] = {
        json.dumps(list(link), separators=(",", ":")):
            [{"image_fingerprint": ("a-hash" if json.loads(key)[0] == "a" else "b-hash"),
              "review_version": None, "component_map_version": None,
              "source_observation_fingerprint": source_map[key]["source_observation_fingerprint"]}
             for key in link]
        for link in links
    }
    rebuilt = build_board_component_registry(state, project_id="p1", board_id="b1",
                                             photo_snapshots=snapshots)
    assert rebuilt["source_to_board_id"][a1] == rebuilt["source_to_board_id"][b1]
    assert rebuilt["source_to_board_id"][a2] != rebuilt["source_to_board_id"][b1]
    assert any(item["reason"] == "SAME_PHOTO_COMPONENT_COLLISION_RECONFIRMATION_REQUIRED"
               for item in rebuilt["inactive_links"])


def test_measurement_aliases_are_only_emitted_when_local_id_is_unambiguous():
    registry = build_board_component_registry(
        {}, project_id="p1", board_id="b1", photo_snapshots=_snapshots(),
    )
    aliases = measurement_component_aliases(registry)
    assert aliases["local-1"] is None
    assert aliases["close-1"].startswith("bc-")
    observation_fp = current_source_observation_fingerprint(
        registry, photo_id="front", component_id="local-1")
    board_id, status = resolve_measurement_board_component(
        registry, photo_id="front", component_id="local-1", observation_fingerprint=observation_fp)
    assert board_id is not None and status == "VERIFIED_SOURCE_IDENTITY"
    assert resolve_measurement_board_component(
        registry, photo_id="front", component_id="local-1",
        photo_fingerprint="replacement-hash", observation_fingerprint=observation_fp,
    ) == (None, "PHOTO_REPLACED_RECONFIRMATION_REQUIRED")
    assert resolve_measurement_board_component(
        registry, photo_id=None, component_id="local-1") == (None, "UNRESOLVED")
    unique_id, unique_status = resolve_measurement_board_component(
        registry, photo_id=None, component_id="close-1")
    assert unique_id is None
    assert unique_status == "RECONFIRMATION_REQUIRED"


def test_project_and_board_registry_data_are_isolated_and_photo_manifest_versions_changes():
    state = {}
    first = build_board_component_registry(state, project_id="p1", board_id="b1",
                                           photo_snapshots=_snapshots())
    second = build_board_component_registry(state, project_id="p2", board_id="b1",
                                            photo_snapshots=_snapshots())
    assert first["components"][0]["board_component_id"] != second["components"][0]["board_component_id"]
    assert len(state[REGISTRY_KEY]) == 2
    altered = _snapshots()
    altered["top_extra"] = {"components": (), "image_fingerprint": "added", "review_version": "v1"}
    updated = build_board_component_registry(state, project_id="p1", board_id="b1",
                                             photo_snapshots=altered)
    assert updated["version"] != first["version"]


def test_excluded_photo_box_is_retained_in_registry_but_not_sent_to_diagnosis():
    snapshots = {"front": {"registry_components": ({
        "component_id": "excluded-1", "component_type": "ic", "bbox_xyxy": [0, 0, 10, 10],
        "identification_status": "EXCLUDED", "presence_status": "EXCLUDED",
        "component_type_status": "UNCONFIRMED",
    },)}}
    registry = build_board_component_registry({}, project_id="p1", board_id="b1",
                                              photo_snapshots=snapshots)
    assert registry["components"][0]["source_components"][0]["review_status"] == "EXCLUDED"
    assert diagnosis_components(registry) == ()


def test_measurement_adapter_resolves_real_source_geometry_and_rejects_missing_or_excluded():
    registry = build_board_component_registry({}, project_id="p1", board_id="b1",
                                             photo_snapshots=_snapshots())
    front_source = source_component_key("front", "local-1")
    board_id = registry["source_to_board_id"][front_source]
    model = measurement_view_model_for_board_component(
        registry, board_component_id=board_id, preferred_photo_id="front")
    assert model is not None
    assert model.canonical_frame_id == "front"
    assert model.component_id == "local-1"
    assert model.bbox_xyxy == (1.0, 2.0, 8.0, 9.0)
    assert model.board_component_id == board_id

    excluded_registry = build_board_component_registry({}, project_id="p1", board_id="b1",
        photo_snapshots={"front": {"components": (_component("x", status="EXCLUDED"),),
                                    "image_fingerprint": "x", "review_version": "x"}})
    excluded_id = excluded_registry["components"][0]["board_component_id"]
    assert measurement_view_model_for_board_component(
        excluded_registry, board_component_id=excluded_id) is None


def test_qwen_type_suggestion_stays_unverified_in_diagnosis_dto():
    state = {}
    snapshots = _snapshots()
    snapshots["front"]["components"] = (_component("R1", status="UNCONFIRMED"),)
    source = snapshots["front"]["components"][0]
    source.qwen_component_type_candidate = "capacitor"
    source.qwen_component_type_evidence = "visual suggestion"
    source.qwen_component_type_status = "AI_PROPOSED_UNVERIFIED"
    registry = build_board_component_registry(
        state, project_id="p-qwen", board_id="b-qwen", photo_snapshots={"front": snapshots["front"]})
    diagnosis_row = diagnosis_components(registry)[0]
    assert diagnosis_row.component_type == "unknown"
    assert diagnosis_row.candidate_component_types == ("capacitor",)
    assert diagnosis_row.ai_review_evidence == ("visual suggestion",)


def test_qwen_extension_suggestion_and_subtype_survive_registry_without_becoming_diagnosis_facts():
    snapshots = _snapshots()
    snapshots["front"]["components"] = (_component("J1", status="UNCONFIRMED"),)
    source = snapshots["front"]["components"][0]
    source.qwen_component_type_candidate = "other_component"
    source.qwen_component_subtype_candidate = "ic_socket"
    source.qwen_component_type_evidence = "Open socket cavity and contacts."
    source.qwen_component_type_status = "AI_PROPOSED_UNVERIFIED"
    source.qwen_review_status = "AI_CONFLICTED"
    registry = build_board_component_registry({}, project_id="p-ext", board_id="b-ext",
        photo_snapshots={"front": snapshots["front"]})
    row = diagnosis_components(registry)[0]
    assert row.component_type == "unknown"
    assert row.candidate_component_types == ("other_component",)
    assert row.candidate_component_subtypes == ("ic_socket",)
    assert row.ai_review_statuses == ("AI_CONFLICTED",)


def test_board_feature_qwen_proposal_is_not_exposed_as_a_component_type():
    snapshots = _snapshots()
    snapshots["front"]["components"] = (_component("TP1", status="UNCONFIRMED"),)
    source = snapshots["front"]["components"][0]
    source.qwen_component_type_candidate = "board_feature"
    source.qwen_component_type_status = "AI_PROPOSED_UNVERIFIED"
    registry = build_board_component_registry({}, project_id="p-feature", board_id="b-feature",
        photo_snapshots={"front": snapshots["front"]})
    row = diagnosis_components(registry)[0]
    assert row.component_type == "unknown"
    assert row.candidate_component_types == ()
    assert row.candidate_board_features == ("TP1",)


def test_user_confirmed_board_feature_is_kept_in_registry_but_not_diagnosis_components():
    snapshot = {"image_fingerprint": "hash", "registry_components": [{
        "component_id": "TP1", "component_type": "board_feature", "effective_class": "board_feature",
        "bbox_xyxy": [1, 1, 10, 10], "identification_status": "USER_CONFIRMED",
        "presence_status": "CONFIRMED", "component_type_status": "USER_CONFIRMED_EXTENSION",
        "confirmed_base_class": "board_feature", "confirmed_subtype": "test_point",
        "qwen_component_type_status": "USER_CONFIRMED_EXTENSION",
    }]}
    registry = build_board_component_registry({}, project_id="p-feature-confirmed", board_id="b-feature-confirmed",
                                              photo_snapshots={"front": snapshot})
    assert registry["components"][0]["component_type"] == "board_feature"
    assert diagnosis_components(registry) == ()


def test_user_confirmed_extension_keeps_detector_qwen_and_effective_taxonomies_separate():
    from circuitcare.component_taxonomy import review_taxonomy

    connector = review_taxonomy({
        "소자 종류": "ic", "qwen_component_type_candidate": "connector",
        "qwen_component_subtype_candidate": "fpc_ffc", "qwen_component_type_status": "USER_CONFIRMED",
    })
    assert connector["detector_class"] == "ic"
    assert connector["confirmed_base_class"] == "connector"
    assert connector["confirmed_subtype"] == "fpc_ffc"
    assert connector["diagnosis_class"] == "connector"

    socket = review_taxonomy({
        "소자 종류": "ic", "confirmed_base_class": "other_component",
        "confirmed_subtype": "ic_socket", "qwen_component_type_status": "USER_CONFIRMED_EXTENSION",
    })
    assert socket["detector_class"] == "ic"
    assert socket["effective_class"] == "other_component"
    assert socket["diagnosis_class"] == "unknown"

    feature = review_taxonomy({
        "소자 종류": "ic", "confirmed_base_class": "board_feature",
        "confirmed_subtype": "test_point", "qwen_component_type_status": "USER_CONFIRMED_EXTENSION",
    })
    assert feature["effective_class"] == "board_feature"
    assert feature["diagnosis_class"] == "unknown"


@pytest.mark.parametrize(("qwen_class", "subtype", "expected_registry_type", "expected_diagnosis"), [
    ("connector", "fpc_ffc", "connector", "connector"),
    ("other_component", "ic_socket", "other_component", "unknown"),
    ("board_feature", "test_point", "board_feature", None),
])
def test_confirmed_qwen_class_flows_through_registry_to_safe_diagnosis(
    qwen_class, subtype, expected_registry_type, expected_diagnosis,
):
    from circuitcare.dashboard.views.inspection_panel import confirm_qwen_component_type
    from circuitcare.component_taxonomy import review_taxonomy

    review_row = {"component_id": "U1", "소자 종류": "ic", "x1": 1, "y1": 1,
                  "x2": 20, "y2": 20}
    confirm_qwen_component_type(review_row, base_class=qwen_class, subtype=subtype)
    resolved = review_taxonomy(review_row)
    snapshot = {"image_fingerprint": "image", "registry_components": [{
        **review_row, **resolved, "component_type": resolved["effective_class"],
        "bbox_xyxy": [1, 1, 20, 20], "coordinate_frame": "SOURCE_PHOTO_PIXEL",
        "identification_status": "USER_CONFIRMED", "presence_status": "CONFIRMED",
    }]}
    registry = build_board_component_registry({}, project_id="p-confirmed-class", board_id="b-confirmed-class",
                                              photo_snapshots={"front": snapshot})
    assert registry["components"][0]["component_type"] == expected_registry_type
    diagnosis = diagnosis_components(registry)
    if expected_diagnosis is None:
        assert diagnosis == ()
    else:
        assert diagnosis[0].component_type == expected_diagnosis
    source = registry["components"][0]["source_components"][0]
    assert source["detector_class"] == "ic"
    assert source["confirmed_base_class"] == qwen_class
    assert source["confirmed_subtype"] == subtype


def test_registry_preserves_part_number_and_qwen_ocr_conflict_candidates():
    snapshot = {"image_fingerprint": "image-hash", "review_version": "v1",
                "registry_components": [{
                    "component_id": "U1", "component_type": "ic", "bbox_xyxy": [1, 2, 10, 12],
                    "identification_status": "UNCONFIRMED", "presence_status": "UNCONFIRMED",
                    "component_type_status": "UNCONFIRMED", "ocr_marking": "AMS1117",
                    "ocr_candidates": ["AMS1117"], "ocr_source": "RapidOCR", "ocr_score": .83,
                    "possible_part_number": "AMS1117", "part_number_status": "UNVERIFIED_PART_ID",
                    "qwen_marking_candidates": ["AMS1117-3.3"],
                    "qwen_conflicting_evidence": ["OCR candidates differ"],
                }]}
    registry = build_board_component_registry({}, project_id="p-ocr", board_id="b-ocr",
                                              photo_snapshots={"front": snapshot})
    source = registry["components"][0]["source_components"][0]
    assert source["possible_part_number"] == "AMS1117"
    assert source["ocr_marking"] == "AMS1117"
    assert source["qwen_marking_candidates"] == ["AMS1117-3.3"]
    assert source["qwen_conflicting_evidence"] == ["OCR candidates differ"]
