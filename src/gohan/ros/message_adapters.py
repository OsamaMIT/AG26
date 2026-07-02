"""Defensive adapters between AWSIM ROS2 messages and GOHAN dataclasses."""

from __future__ import annotations

import importlib
import math
import warnings
from typing import Any

from gohan.telemetry.telemetry_state import VehicleCommand, VehicleTelemetry
from gohan.utils.math_utils import quaternion_to_euler
from gohan.utils.timing import now_s

_missing_field_warnings: set[str] = set()


def unsupported_command_type_message(ros_type: str | None) -> str:
    """Return the standard user-facing adapter error."""
    display_type = ros_type or "<unknown>"
    return (
        f"Unsupported AWSIM command message type: {display_type}. Run `ros2 topic info <topic>` and update "
        "src/gohan/ros/message_adapters.py."
    )


def import_message_class(ros_type: str) -> type:
    """Import a ROS2 Python message class from a ROS type string."""
    package, name = _split_ros_type(ros_type)
    module = importlib.import_module(f"{package}.msg")
    return getattr(module, name)


def command_message_class(ros_type: str) -> type:
    """Return a supported command message class or raise an explicit adapter error."""
    normalized = ros_type.replace("/", "/msg/", 1) if "/msg/" not in ros_type and "/" in ros_type else ros_type
    if normalized in {
        "autonoma_msgs/msg/VehicleInputs",
        "autonoma_msgs/msg/ToRaptor",
    }:
        return import_message_class(normalized)
    raise RuntimeError(unsupported_command_type_message(ros_type))


def build_command_message(
    command: VehicleCommand,
    msg_cls: type,
    ros_type: str,
    config: dict[str, Any] | None = None,
    counters: dict[str, int] | None = None,
) -> Any:
    """Build a ROS2 command message for a supported AWSIM high-level command topic."""
    config = config or {}
    counters = counters if counters is not None else {"steering": 0, "throttle": 0, "brake": 0, "rolling": 0}
    command_cfg = dict(config.get("command", {}))
    normalized = ros_type.replace("/", "/msg/", 1) if "/msg/" not in ros_type and "/" in ros_type else ros_type
    msg = msg_cls()
    if normalized == "autonoma_msgs/msg/VehicleInputs":
        max_steering_cmd = float(command_cfg.get("max_steering_cmd", 200.0))
        max_brake_cmd = float(command_cfg.get("max_brake_cmd", 1000.0))
        _set_any(msg, ["steering_cmd"], float(command.steering) * max_steering_cmd)
        _set_any(msg, ["throttle_cmd"], float(command.throttle) * 100.0)
        _set_any(msg, ["brake_cmd"], float(command.brake) * max_brake_cmd)
        _set_any(msg, ["gear_cmd"], int(command_cfg.get("gear_cmd", 1)))
        _set_any(msg, ["steering_cmd_count"], _next_counter(counters, "steering"))
        _set_any(msg, ["throttle_cmd_count"], _next_counter(counters, "throttle"))
        _set_any(msg, ["brake_cmd_count"], _next_counter(counters, "brake"))
        return msg
    if normalized == "autonoma_msgs/msg/ToRaptor":
        # TODO(AWSIM): ToRaptor controls Raptor state, not steering/throttle/brake.
        # Use /vehicle_inputs for direct vehicle actuation. This adapter is included
        # so users can deliberately publish a control-state command when needed.
        _set_any(msg, ["ct_state"], int(command_cfg.get("ct_state", 5)))
        _set_any(msg, ["rolling_counter"], _next_counter(counters, "rolling"))
        return msg
    raise RuntimeError(unsupported_command_type_message(ros_type))


def adapt_odometry(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Merge nav_msgs/msg/Odometry into VehicleTelemetry."""
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = _stamp_to_seconds(getattr(msg, "header", None)) or now_s()
    pose = msg.pose.pose
    twist = msg.twist.twist
    telemetry.x = float(pose.position.x)
    telemetry.y = float(pose.position.y)
    telemetry.z = float(pose.position.z)
    telemetry.roll, telemetry.pitch, telemetry.yaw = quaternion_to_euler(
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    telemetry.longitudinal_velocity_mps = float(twist.linear.x)
    telemetry.lateral_velocity_mps = float(twist.linear.y)
    telemetry.speed_mps = math.hypot(float(twist.linear.x), float(twist.linear.y))
    telemetry.yaw_rate_radps = float(twist.angular.z)
    telemetry.raw["odometry"] = _message_summary(msg)
    return telemetry


def adapt_inspva(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Merge novatel_oem7_msgs/msg/INSPVA into VehicleTelemetry.

    AWSIM racing publishes NovAtel INSPVA rather than nav_msgs/Odometry.
    We convert latitude/longitude to a local ENU frame using the first
    INSPVA message as the origin.
    """
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = _stamp_to_seconds(getattr(msg, "header", None)) or now_s()

    lat = float(getattr(msg, "latitude", 0.0))
    lon = float(getattr(msg, "longitude", 0.0))
    height = float(getattr(msg, "height", telemetry.z))
    origin = telemetry.raw.get("_gnss_origin")
    if not isinstance(origin, dict):
        origin = {"latitude": lat, "longitude": lon, "height": height}
        telemetry.raw["_gnss_origin"] = origin

    telemetry.x, telemetry.y = _latlon_to_local_enu(
        lat,
        lon,
        float(origin["latitude"]),
        float(origin["longitude"]),
    )
    telemetry.z = height - float(origin.get("height", height))

    telemetry.roll = math.radians(float(getattr(msg, "roll", 0.0)))
    telemetry.pitch = math.radians(float(getattr(msg, "pitch", 0.0)))
    azimuth_rad = math.radians(float(getattr(msg, "azimuth", 0.0)))
    telemetry.yaw = _wrap_to_pi(math.pi / 2.0 - azimuth_rad)

    north_velocity = float(getattr(msg, "north_velocity", 0.0))
    east_velocity = float(getattr(msg, "east_velocity", 0.0))
    telemetry.speed_mps = math.hypot(east_velocity, north_velocity)
    telemetry.longitudinal_velocity_mps = math.cos(telemetry.yaw) * east_velocity + math.sin(telemetry.yaw) * north_velocity
    telemetry.lateral_velocity_mps = -math.sin(telemetry.yaw) * east_velocity + math.cos(telemetry.yaw) * north_velocity
    telemetry.raw["inspva"] = _message_summary(msg)
    return telemetry


def adapt_imu(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Merge sensor_msgs/msg/Imu into VehicleTelemetry."""
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = max(telemetry.timestamp, _stamp_to_seconds(getattr(msg, "header", None)) or now_s())
    telemetry.acceleration_x = float(getattr(msg.linear_acceleration, "x", 0.0))
    telemetry.acceleration_y = float(getattr(msg.linear_acceleration, "y", 0.0))
    telemetry.yaw_rate_radps = float(getattr(msg.angular_velocity, "z", telemetry.yaw_rate_radps))
    telemetry.raw["imu"] = _message_summary(msg)
    return telemetry


def adapt_vehicle_data(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Best-effort merge of autonoma_msgs/VehicleData into VehicleTelemetry."""
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = now_s()
    telemetry.steering_angle = _get_any_warn(
        msg,
        ["steering_wheel_angle", "steering_angle"],
        telemetry.steering_angle,
        "VehicleData steering angle",
    )
    telemetry.throttle = _normalize_percent(
        _get_any_warn(msg, ["accel_pedal_input", "accel_pedal_output", "throttle"], telemetry.throttle, "VehicleData throttle")
    )
    brake_pressure = max(
        float(_get_any(msg, ["front_brake_pressure"], 0.0)),
        float(_get_any(msg, ["rear_brake_pressure"], 0.0)),
    )
    telemetry.brake = _normalize_brake(
        _get_any_warn(msg, ["brake_pressure", "brake"], brake_pressure, "VehicleData brake")
    )
    telemetry.off_track = bool(_get_any(msg, ["off_track"], telemetry.off_track))
    telemetry.collision_detected = bool(
        _get_any(msg, ["collision_detected"], telemetry.collision_detected)
    )
    telemetry.raw["vehicle_data"] = _message_summary(msg)
    return telemetry


def adapt_powertrain_data(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Best-effort merge of autonoma_msgs/PowertrainData into VehicleTelemetry."""
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = now_s()
    speed_kmph = _get_any_warn(msg, ["vehicle_speed_kmph"], None, "PowertrainData vehicle speed")
    if speed_kmph is not None:
        telemetry.speed_mps = float(speed_kmph) / 3.6
    telemetry.rpm = float(_get_any_warn(msg, ["engine_rpm"], telemetry.rpm, "PowertrainData engine rpm"))
    telemetry.gear = int(_get_any_warn(msg, ["current_gear"], telemetry.gear, "PowertrainData gear"))
    telemetry.throttle = _normalize_percent(
        _get_any(msg, ["throttle_position"], telemetry.throttle)
    )
    telemetry.raw["powertrain_data"] = _message_summary(msg)
    return telemetry


def adapt_race_control(msg: Any, base: VehicleTelemetry | None = None) -> VehicleTelemetry:
    """Best-effort merge of autonoma_msgs/RaceControl into VehicleTelemetry."""
    telemetry = _copy_telemetry(base)
    telemetry.timestamp = now_s()
    telemetry.lap_count = int(_get_any(msg, ["lap_count"], telemetry.lap_count))
    telemetry.lap_time = float(_get_any(msg, ["lap_time"], telemetry.lap_time))
    telemetry.raw["race_control"] = _message_summary(msg)
    return telemetry


def _split_ros_type(ros_type: str) -> tuple[str, str]:
    parts = ros_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise RuntimeError(f"Unsupported ROS message type string: {ros_type}")


def _copy_telemetry(base: VehicleTelemetry | None) -> VehicleTelemetry:
    if base is None:
        return VehicleTelemetry(timestamp=now_s())
    return VehicleTelemetry(**{field: getattr(base, field) for field in base.__dataclass_fields__})


def _stamp_to_seconds(header: Any) -> float | None:
    if header is None or not hasattr(header, "stamp"):
        return None
    stamp = header.stamp
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def _get_any(msg: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(msg, name):
            return getattr(msg, name)
    return default


def _get_any_warn(msg: Any, names: list[str], default: Any, label: str) -> Any:
    for name in names:
        if hasattr(msg, name):
            return getattr(msg, name)
    if label not in _missing_field_warnings:
        warnings.warn(
            f"AWSIM message {type(msg).__name__} is missing expected field(s) {names}; using safe default for {label}.",
            RuntimeWarning,
        )
        _missing_field_warnings.add(label)
    return default


def _set_any(msg: Any, names: list[str], value: Any) -> None:
    for name in names:
        if hasattr(msg, name):
            setattr(msg, name, value)
            return
    setattr(msg, names[0], value)


def _next_counter(counters: dict[str, int], key: str) -> int:
    value = int(counters.get(key, 0)) % 256
    counters[key] = (value + 1) % 256
    return value


def _normalize_percent(value: Any) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _normalize_brake(value: Any) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if value > 1.0:
        return min(value / 1000.0, 1.0)
    return max(value, 0.0)


def _message_summary(msg: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    slots = getattr(msg, "__slots__", [])
    for slot in slots:
        if slot.startswith("_"):
            continue
        value = getattr(msg, slot, None)
        if isinstance(value, (str, int, float, bool)):
            summary[slot] = value
    return summary


def _latlon_to_local_enu(lat_deg: float, lon_deg: float, origin_lat_deg: float, origin_lon_deg: float) -> tuple[float, float]:
    earth_radius_m = 6378137.0
    lat = math.radians(lat_deg)
    origin_lat = math.radians(origin_lat_deg)
    d_lat = lat - origin_lat
    d_lon = math.radians(lon_deg - origin_lon_deg)
    x_east = earth_radius_m * d_lon * math.cos(origin_lat)
    y_north = earth_radius_m * d_lat
    return float(x_east), float(y_north)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
