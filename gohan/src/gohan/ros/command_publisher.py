"""ROS2 vehicle command publisher for AWSIM."""

from __future__ import annotations

from typing import Any

from gohan.ros.message_adapters import (
    build_command_message,
    command_message_class,
    unsupported_command_type_message,
)
from gohan.ros.topic_discovery import TopicSelection
from gohan.telemetry.telemetry_state import VehicleCommand


class CommandPublisher:
    """Publish normalized VehicleCommand objects to AWSIM."""

    def __init__(self, node: Any, selection: TopicSelection, config: dict[str, Any] | None = None) -> None:
        if not selection.name or not selection.type:
            raise RuntimeError(unsupported_command_type_message(selection.type))
        self.node = node
        self.topic = selection.name
        self.ros_type = selection.type
        self.config = config or {}
        self._counters = {"steering": 0, "throttle": 0, "brake": 0, "rolling": 0}
        try:
            self.msg_cls = command_message_class(self.ros_type)
        except Exception as exc:
            raise RuntimeError(unsupported_command_type_message(self.ros_type)) from exc
        self.publisher = self.node.create_publisher(self.msg_cls, self.topic, 10)
        print(f"GOHAN command publisher selected {self.topic} [{self.ros_type}]")
        if "ToRaptor" in self.ros_type:
            print("Warning: /to_raptor does not carry steering/throttle/brake. Prefer /vehicle_inputs for GOHAN.")

    def publish(self, command: VehicleCommand) -> None:
        msg = build_command_message(command, self.msg_cls, self.ros_type, self.config, self._counters)
        self.publisher.publish(msg)
