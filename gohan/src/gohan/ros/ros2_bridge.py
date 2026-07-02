"""High-level ROS2 bridge for AWSIM Racing Simulator."""

from __future__ import annotations

from typing import Any

from gohan.ros.command_publisher import CommandPublisher
from gohan.ros.telemetry_subscriber import TelemetrySubscriber
from gohan.ros.topic_discovery import TopicDiscoveryReport, discover_from_topic_list, format_startup_report
from gohan.telemetry.telemetry_state import VehicleCommand, VehicleTelemetry


def require_ros2() -> Any:
    """Import rclpy or raise a user-actionable error."""
    try:
        import rclpy
    except Exception as exc:
        raise RuntimeError(
            "rclpy is unavailable. Source your ROS2 installation before running GOHAN, e.g. "
            "source /opt/ros/<distro>/setup.bash"
        ) from exc
    return rclpy


class Ros2Bridge:
    """ROS2 bridge that discovers AWSIM topics, receives telemetry, and publishes commands."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rclpy = require_ros2()
        if not self.rclpy.ok():
            self.rclpy.init()
        ros_cfg = dict(config.get("ros", {}))
        self.node = self.rclpy.create_node(str(ros_cfg.get("node_name", "gohan_awsim_bridge")))
        self.spin_timeout_s = 1.0 / max(float(ros_cfg.get("spin_hz", 50.0)), 1.0)
        self.report = self.discover_topics()
        print(format_startup_report(self.report))
        if not self.report.odometry.name:
            raise RuntimeError(f"Missing required odometry topics: {self.report.odometry.missing_candidates}")
        if not self.report.command.name:
            raise RuntimeError(f"Missing required command topics: {self.report.command.missing_candidates}")
        self.telemetry_subscriber = TelemetrySubscriber(self.node, self.report)
        self.command_publisher = CommandPublisher(self.node, self.report.command, config)

    def discover_topics(self) -> TopicDiscoveryReport:
        topic_names_and_types = self.node.get_topic_names_and_types()
        return discover_from_topic_list(topic_names_and_types, self.config)

    @property
    def latest_telemetry(self) -> VehicleTelemetry | None:
        return self.telemetry_subscriber.latest

    def spin_once(self, timeout_s: float | None = None) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=self.spin_timeout_s if timeout_s is None else timeout_s)

    def wait_for_telemetry(self, timeout_s: float) -> VehicleTelemetry | None:
        return self.telemetry_subscriber.wait_for_telemetry(timeout_s, lambda: self.spin_once(self.spin_timeout_s))

    def publish_command(self, command: VehicleCommand) -> None:
        self.command_publisher.publish(command)

    def close(self) -> None:
        try:
            self.publish_command(VehicleCommand(steering=0.0, throttle=0.0, brake=1.0))
        except Exception:
            pass
        self.node.destroy_node()
        try:
            self.rclpy.shutdown()
        except Exception:
            pass
