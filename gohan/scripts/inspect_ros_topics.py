#!/usr/bin/env python3
"""Inspect ROS2 topics and highlight likely AWSIM interfaces."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from gohan.ros.ros2_bridge import require_ros2
from gohan.ros.topic_discovery import discover_from_topic_list, format_startup_report
from gohan.utils.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awsim.yaml")
    args = parser.parse_args()
    try:
        config = load_yaml(args.config)
        rclpy = require_ros2()
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("gohan_topic_inspector")
        report = discover_from_topic_list(node.get_topic_names_and_types(), config)
        print(format_startup_report(report))
        print("\nSuggested config updates:")
        for name in sorted(report.found_topics):
            lower = name.lower()
            if any(
                token in lower
                for token in ["odom", "imu", "rawimux", "inspva", "vehicle", "powertrain", "race_control", "raptor"]
            ):
                print(f"  consider: {name} -> {report.found_topics[name]}")
        node.destroy_node()
        rclpy.shutdown()
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
