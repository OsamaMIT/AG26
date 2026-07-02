import numpy as np

from gohan.telemetry.features import OBSERVATION_SIZE, build_observation
from gohan.telemetry.telemetry_state import VehicleCommand, VehicleTelemetry
from gohan.track.track_map import TrackMap
from gohan.track.waypoint_loader import WaypointData


def make_track():
    return TrackMap(
        WaypointData(
            name="line",
            points=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=float),
            yaws=None,
            track_width_m=10.0,
            closed_loop=True,
        )
    )


def test_observation_shape_dtype_and_range():
    packet = build_observation(
        VehicleTelemetry(x=10.0, y=1.0, yaw=0.0, speed_mps=30.0, longitudinal_velocity_mps=30.0),
        make_track(),
        VehicleCommand(steering=0.2, throttle=0.4),
        {},
    )
    assert packet.observation.shape == (OBSERVATION_SIZE,)
    assert packet.observation.dtype == np.float32
    assert np.all(packet.observation <= 1.0)
    assert np.all(packet.observation >= -1.0)
    assert np.all(np.isfinite(packet.observation))


def test_non_finite_telemetry_is_sanitized():
    packet = build_observation(
        VehicleTelemetry(x=np.nan, y=0.0, yaw=np.inf, speed_mps=np.nan),
        make_track(),
        VehicleCommand(),
        {},
    )
    assert np.all(np.isfinite(packet.observation))
