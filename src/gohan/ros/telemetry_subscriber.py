"""ROS2 telemetry subscription manager for AWSIM."""

from __future__ import annotations

import threading
from typing import Any, Callable

from gohan.ros.message_adapters import (
    adapt_inspva,
    adapt_imu,
    adapt_odometry,
    adapt_powertrain_data,
    adapt_race_control,
    adapt_vehicle_data,
    import_message_class,
)
from gohan.ros.topic_discovery import TopicDiscoveryReport, TopicSelection
from gohan.telemetry.telemetry_state import VehicleTelemetry
from gohan.utils.timing import now_s


class TelemetrySubscriber:
    """Subscribe to selected AWSIM telemetry topics and expose latest state."""

    def __init__(self, node: Any, report: TopicDiscoveryReport) -> None:
        self.node = node
        self.report = report
        self._lock = threading.Lock()
        self._latest: VehicleTelemetry | None = None
        self._last_update_s = 0.0
        self._subscriptions: list[Any] = []
        self._subscribe("odometry", report.odometry, self._primary_state_adapter(report.odometry), required=True)
        self._subscribe("imu", report.imu, adapt_imu, required=False)
        self._subscribe("vehicle_data", report.vehicle_data, adapt_vehicle_data, required=False)
        self._subscribe("powertrain", report.powertrain, adapt_powertrain_data, required=False)
        self._subscribe("race_control", report.race_control, adapt_race_control, required=False)

    @property
    def latest(self) -> VehicleTelemetry | None:
        with self._lock:
            return self._latest

    @property
    def last_update_s(self) -> float:
        return self._last_update_s

    def wait_for_telemetry(self, timeout_s: float, spin_once: Callable[[], None]) -> VehicleTelemetry | None:
        """Spin until telemetry is received or timeout expires."""
        deadline = now_s() + float(timeout_s)
        while now_s() < deadline:
            if self.latest is not None:
                return self.latest
            spin_once()
        return self.latest

    def _subscribe(self, label: str, selection: TopicSelection, adapter: Callable, required: bool) -> None:
        if not selection.name or not selection.type:
            if required:
                raise RuntimeError(f"Missing required AWSIM telemetry topic for {label}: {selection.missing_candidates}")
            print(f"Optional AWSIM telemetry topic not found for {label}: {selection.missing_candidates}")
            return
        try:
            msg_cls = import_message_class(selection.type)
        except Exception as exc:
            message = f"Could not import ROS message type {selection.type} for topic {selection.name}: {exc}"
            if required:
                raise RuntimeError(message) from exc
            print(message)
            return

        def callback(msg: Any) -> None:
            with self._lock:
                receipt_time = now_s()
                self._latest = adapter(msg, self._latest)
                self._latest.timestamp = receipt_time
                self._last_update_s = receipt_time

        self._subscriptions.append(self.node.create_subscription(msg_cls, selection.name, callback, 10))

    @staticmethod
    def _primary_state_adapter(selection: TopicSelection) -> Callable:
        ros_type = selection.type or ""
        normalized = ros_type.replace("/", "/msg/", 1) if "/msg/" not in ros_type and "/" in ros_type else ros_type
        if normalized == "novatel_oem7_msgs/msg/INSPVA":
            return adapt_inspva
        return adapt_odometry
