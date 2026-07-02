import numpy as np
import pytest

from gohan.track.geometry import curvature_from_points, heading_between, project_point_to_segment, signed_lateral_error
from gohan.track.track_map import TrackMap
from gohan.track.waypoint_loader import WaypointData


def make_track():
    return TrackMap(
        WaypointData(
            name="test",
            points=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=float),
            yaws=None,
            track_width_m=4.0,
            closed_loop=True,
        )
    )


def test_project_point_to_segment():
    projection = project_point_to_segment(np.array([5.0, 2.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert projection.t == pytest.approx(0.5)
    assert projection.point.tolist() == pytest.approx([5.0, 0.0])
    assert projection.distance == pytest.approx(2.0)


def test_heading_and_lateral_error():
    heading = heading_between(np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert heading == pytest.approx(0.0)
    assert signed_lateral_error(np.array([1.0, 2.0]), np.array([1.0, 0.0]), heading) == pytest.approx(2.0)


def test_curvature_from_points_nonzero():
    curvature = curvature_from_points(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert curvature > 0.0


def test_track_map_nearest_progress_and_boundaries():
    track = make_track()
    query = track.nearest(5.0, 1.0)
    assert query.segment_index == 0
    assert query.progress_m == pytest.approx(5.0)
    assert query.lateral_error_m == pytest.approx(1.0)
    left, right = track.boundary_distances(query.lateral_error_m)
    assert left == pytest.approx(1.0)
    assert right == pytest.approx(3.0)


def test_track_map_lookahead_points():
    track = make_track()
    query = track.nearest(0.0, 0.0)
    points = track.lookahead_points_local(query, 0.0, 0.0, 0.0, distances_m=(1.0, 2.0, 3.0))
    assert len(points) == 3
    assert points[0][0] > 0.0
