#!/usr/bin/env python3
"""Record AWSIM telemetry to CSV without training."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from gohan.telemetry.features import build_observation
from gohan.telemetry.logger import TelemetryLogger
from gohan.telemetry.telemetry_state import EpisodeStepRecord, VehicleCommand
from gohan.track.track_map import TrackMap
from gohan.ros.ros2_bridge import Ros2Bridge
from gohan.utils.config import load_yaml
from gohan.utils.paths import make_run_dir, resolve_project_path
from gohan.utils.timing import now_s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awsim.yaml")
    parser.add_argument("--run-name", default="telemetry_test")
    parser.add_argument("--duration-s", type=float, default=30.0)
    args = parser.parse_args()
    bridge = None
    logger = None
    try:
        config = load_yaml(args.config)
        track = TrackMap.from_file(resolve_project_path(config["env"]["track_config"], ROOT))
        run_dir = make_run_dir(args.run_name, ROOT)
        logger = TelemetryLogger(run_dir, [resolve_project_path(args.config, ROOT)])
        bridge = Ros2Bridge(config)
        first = bridge.wait_for_telemetry(float(config["ros"].get("telemetry_timeout_s", 2.0)))
        if first is None:
            print("No telemetry received. Is AWSIM running and publishing ROS2 topics?")
            return 3
        deadline = now_s() + args.duration_s
        step = 0
        while now_s() < deadline:
            bridge.spin_once()
            telemetry = bridge.latest_telemetry
            if telemetry is None:
                continue
            packet = build_observation(telemetry, track, VehicleCommand(), config)
            record = EpisodeStepRecord(
                timestamp=now_s(),
                episode=1,
                step=step,
                x=telemetry.x,
                y=telemetry.y,
                yaw=telemetry.yaw,
                speed_mps=telemetry.speed_mps,
                progress_m=packet.track_features.progress_m,
                lap_progress=packet.track_features.lap_progress,
                lateral_error_m=packet.track_features.lateral_error_m,
                heading_error_rad=packet.track_features.heading_error_rad,
                steering=telemetry.steering_angle,
                throttle=telemetry.throttle,
                brake=telemetry.brake,
                reward=0.0,
                terminated=False,
                truncated=False,
                collision_detected=telemetry.collision_detected,
                off_track=telemetry.off_track or packet.track_features.off_track,
            )
            logger.log_step(record)
            step += 1
        print(f"Recorded telemetry to {run_dir / 'telemetry.csv'}")
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if bridge is not None:
            bridge.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
