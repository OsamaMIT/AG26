"""ROS2 topic discovery and candidate matching for AWSIM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TopicSelection:
    """Selected topic name/type pair."""

    name: str | None = None
    type: str | None = None
    missing_candidates: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicDiscoveryReport:
    """Complete startup topic discovery report."""

    found_topics: dict[str, list[str]]
    odometry: TopicSelection
    imu: TopicSelection
    vehicle_data: TopicSelection
    powertrain: TopicSelection
    race_control: TopicSelection
    command: TopicSelection

    def selected_telemetry(self) -> dict[str, TopicSelection]:
        return {
            "odometry": self.odometry,
            "imu": self.imu,
            "vehicle_data": self.vehicle_data,
            "powertrain": self.powertrain,
            "race_control": self.race_control,
        }


def topics_to_dict(topic_names_and_types: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    """Convert rclpy topic tuples to a dictionary."""
    return {name: list(types) for name, types in topic_names_and_types}


def select_topic(found_topics: dict[str, list[str]], candidates: list[str], required: bool = False) -> TopicSelection:
    """Select the first configured candidate found in the ROS graph."""
    for candidate in candidates:
        if candidate in found_topics:
            types = found_topics[candidate]
            return TopicSelection(name=candidate, type=types[0] if types else None, missing_candidates=[])
    if required:
        return TopicSelection(name=None, type=None, missing_candidates=list(candidates))
    return TopicSelection(name=None, type=None, missing_candidates=list(candidates))


def discover_from_topic_list(
    topic_names_and_types: list[tuple[str, list[str]]],
    config: dict[str, Any],
) -> TopicDiscoveryReport:
    """Build a discovery report from an rclpy topic list."""
    found = topics_to_dict(topic_names_and_types)
    topics = dict(config.get("ros", {}).get("topics", config.get("topics", {})))
    return TopicDiscoveryReport(
        found_topics=found,
        odometry=select_topic(found, list(topics.get("odometry_candidates", [])), required=True),
        imu=select_topic(found, list(topics.get("imu_candidates", [])), required=False),
        vehicle_data=select_topic(found, list(topics.get("vehicle_data_candidates", [])), required=False),
        powertrain=select_topic(found, list(topics.get("powertrain_candidates", [])), required=False),
        race_control=select_topic(found, list(topics.get("race_control_candidates", [])), required=False),
        command=select_topic(found, list(topics.get("command_candidates", [])), required=True),
    )


def format_startup_report(report: TopicDiscoveryReport) -> str:
    """Format a clear startup report for users."""
    lines = ["GOHAN AWSIM ROS2 startup report", "Detected topics:"]
    for name, types in sorted(report.found_topics.items()):
        lines.append(f"  {name}: {', '.join(types) if types else '<unknown type>'}")
    if _only_ros_system_topics(report.found_topics):
        lines.extend(
            [
                "",
                "Diagnosis:",
                "  ROS2 is running, but AWSIM is not visible on this ROS graph.",
                "  Start AWSIM in ROS/autonomy mode, confirm it is publishing, and check ROS_DOMAIN_ID.",
            ]
        )
    lines.append("Selected AWSIM telemetry topics:")
    for key, selection in report.selected_telemetry().items():
        if selection.name:
            lines.append(f"  {key}: {selection.name} [{selection.type}]")
        else:
            requirement = "required" if key == "odometry" else "optional"
            lines.append(f"  {key}: missing {requirement} candidates {selection.missing_candidates}")
    if report.command.name:
        lines.append(f"Selected command topic: {report.command.name} [{report.command.type}]")
    else:
        lines.append(f"Missing required command topic candidates: {report.command.missing_candidates}")
    if not report.odometry.name:
        lines.append(f"Missing required odometry topic candidates: {report.odometry.missing_candidates}")
    unsupported = _unsupported_topic_types(report)
    if unsupported:
        lines.append("Unsupported topic types:")
        lines.extend(f"  {item}" for item in unsupported)
    return "\n".join(lines)


def _only_ros_system_topics(found_topics: dict[str, list[str]]) -> bool:
    system_topics = {"/parameter_events", "/rosout"}
    return bool(found_topics) and set(found_topics).issubset(system_topics)


def _unsupported_topic_types(report: TopicDiscoveryReport) -> list[str]:
    unsupported: list[str] = []
    expected = {
        "odometry": {"nav_msgs/msg/Odometry", "novatel_oem7_msgs/msg/INSPVA"},
        "imu": {"sensor_msgs/msg/Imu"},
        "vehicle_data": {"autonoma_msgs/msg/VehicleData"},
        "powertrain": {"autonoma_msgs/msg/PowertrainData"},
        "race_control": {"autonoma_msgs/msg/RaceControl"},
        "command": {"autonoma_msgs/msg/VehicleInputs", "autonoma_msgs/msg/ToRaptor"},
    }
    selections = {**report.selected_telemetry(), "command": report.command}
    for label, selection in selections.items():
        if not selection.name or not selection.type:
            continue
        normalized = _normalize_ros_type(selection.type)
        if normalized not in expected[label]:
            unsupported.append(f"{label}: {selection.name} [{selection.type}]")
    return unsupported


def _normalize_ros_type(ros_type: str) -> str:
    return ros_type.replace("/", "/msg/", 1) if "/msg/" not in ros_type and "/" in ros_type else ros_type
