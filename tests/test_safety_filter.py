import pytest

from gohan.control.safety_filter import SafetyConfig, SafetyFilter
from gohan.telemetry.telemetry_state import VehicleCommand, VehicleTelemetry


def test_rate_limits_commands():
    safety = SafetyFilter(
        SafetyConfig(
            max_steering_rate_per_s=1.0,
            max_throttle_rate_per_s=1.0,
            max_brake_rate_per_s=1.0,
            telemetry_timeout_s=10.0,
        )
    )
    telemetry = VehicleTelemetry(timestamp=0.0)
    safety.reset(timestamp=0.0)
    command = safety.filter(VehicleCommand(steering=1.0, throttle=1.0), telemetry, timestamp=0.1)
    assert command.steering == pytest.approx(0.1)
    assert command.throttle == pytest.approx(0.1)
    assert command.brake == pytest.approx(0.0)


def test_stale_telemetry_emergency_brake():
    safety = SafetyFilter(SafetyConfig(telemetry_timeout_s=0.5, emergency_brake_on_no_telemetry=True))
    command = safety.filter(VehicleCommand(throttle=1.0), VehicleTelemetry(timestamp=0.0), timestamp=2.0)
    assert command.throttle == 0.0
    assert command.brake == 1.0


def test_off_track_emergency_brake():
    safety = SafetyFilter(SafetyConfig(emergency_brake_on_offtrack=True, telemetry_timeout_s=10.0))
    command = safety.filter(VehicleCommand(throttle=1.0), VehicleTelemetry(timestamp=1.0, off_track=True), timestamp=1.1)
    assert command.throttle == 0.0
    assert command.brake == 1.0


def test_early_training_speed_cap_removes_throttle():
    safety = SafetyFilter(SafetyConfig(early_training_speed_cap_mps=5.0, telemetry_timeout_s=10.0))
    telemetry = VehicleTelemetry(timestamp=1.0, speed_mps=10.0)
    safety.reset(timestamp=1.0)
    command = safety.filter(VehicleCommand(throttle=1.0), telemetry, timestamp=1.1)
    assert command.throttle == 0.0
