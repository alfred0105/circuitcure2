import pytest

from circuitcare.frontend.viewer_geometry import requested_view_transform


def test_requested_geometry_uses_original_source_pixels_and_preserves_aspect_ratio():
    transform = requested_view_transform(
        source_size=(4000, 2000), display_size=(800, 400), active_photo_id="photo-a",
    )
    assert transform["active_photo_id"] == "photo-a"
    assert transform["source_image_width"] == 4000
    assert transform["source_image_height"] == 2000
    assert transform["display_width"] == 800
    assert transform["display_height"] == 400
    assert transform["scale_x"] == pytest.approx(0.2)
    assert transform["scale_y"] == pytest.approx(0.2)
    assert transform["coordinate_frame"] == "source_image_px"
    assert transform["coordinate_status"] == "UNVERIFIED_RESPONSIVE_CSS"
    assert transform["display_size_kind"] == "requested"


@pytest.mark.parametrize(
    "source_size,display_size", [((0, 100), (500, 200)), ((100, 0), (500, 200)),
                                  ((100, 100), (500, 0))],
)
def test_invalid_geometry_is_rejected(source_size, display_size):
    with pytest.raises(ValueError):
        requested_view_transform(
            source_size=source_size, display_size=display_size, active_photo_id="photo-a",
        )
