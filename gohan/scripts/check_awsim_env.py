#!/usr/bin/env python3
"""Check ROS2, AWSIM topics, telemetry adapters, observation, and command publishing."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from gohan.ros.ros2_bridge import Ros2Bridge
from gohan.telemetry.features import build_observation
from gohan.telemetry.telemetry_state import VehicleCommand
from gohan.track.track_map import TrackMap
from gohan.utils.config import load_yaml
from gohan.utils.paths import resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awsim.yaml")
    args = parser.parse_args()
    bridge = None
    try:
        config = load_yaml(args.config)
        track = TrackMap.from_file(resolve_project_path(config["env"]["track_config"], ROOT))
        bridge = Ros2Bridge(config)
        telemetry = bridge.wait_for_telemetry(float(config["ros"].get("telemetry_timeout_s", 2.0)))
        if telemetry is None:
            print("No telemetry received before timeout.")
            return 3
        packet = build_observation(telemetry, track, VehicleCommand(), config)
        print(f"Observation shape: {packet.observation.shape}")
        print(f"Observation values: {packet.observation}")
        bridge.publish_command(VehicleCommand())
        print("Published one neutral command.")
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
