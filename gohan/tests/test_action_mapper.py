import numpy as np
import pytest

from gohan.control.action_mapper import ActionMapper, VehicleCommand


def test_map_positive_action_to_throttle():
    command = ActionMapper().map_action(np.array([0.5, 0.75], dtype=np.float32))
    assert isinstance(command, VehicleCommand)
    assert command.steering == pytest.approx(0.5)
    assert command.throttle == pytest.approx(0.75)
    assert command.brake == pytest.approx(0.0)


def test_map_negative_action_to_brake():
    command = ActionMapper().map_action(np.array([-0.25, -0.4], dtype=np.float32))
    assert command.steering == pytest.approx(-0.25)
    assert command.throttle == pytest.approx(0.0)
    assert command.brake == pytest.approx(0.4)


def test_action_is_clipped():
    command = ActionMapper().map_action(np.array([5.0, -2.0], dtype=np.float32))
    assert command.steering == pytest.approx(1.0)
    assert command.brake == pytest.approx(1.0)


def test_invalid_action_shape_raises():
    with pytest.raises(ValueError):
        ActionMapper().map_action(np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_non_finite_action_raises():
    with pytest.raises(ValueError):
        ActionMapper().map_action(np.array([0.0, np.nan], dtype=np.float32))
