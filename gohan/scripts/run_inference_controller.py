#!/usr/bin/env python3
"""Run a trained GOHAN model as a live ROS2 controller."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from gohan.control.action_mapper import ActionMapper
from gohan.control.safety_filter import SafetyFilter
from gohan.ros.ros2_bridge import Ros2Bridge
from gohan.telemetry.features import build_observation
from gohan.telemetry.logger import TelemetryLogger
from gohan.telemetry.telemetry_state import EpisodeStepRecord, VehicleCommand
from gohan.track.track_map import TrackMap
from gohan.utils.config import load_yaml
from gohan.utils.paths import make_run_dir, resolve_project_path
from gohan.utils.timing import RateTimer, now_s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/awsim.yaml")
    parser.add_argument("--run-name", default="inference")
    args = parser.parse_args()
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        print(f"stable-baselines3 is required for inference: {exc}")
        return 2

    config = load_yaml(args.config, ROOT)
    track = TrackMap.from_file(resolve_project_path(config["env"]["track_config"], ROOT))
    bridge = None
    logger = None
    mapper = ActionMapper()
    safety = SafetyFilter(config)
    previous_command = VehicleCommand()
    model = PPO.load(args.model, device="cpu")
    try:
        bridge = Ros2Bridge(config)
        run_dir = make_run_dir(args.run_name, ROOT)
        logger = TelemetryLogger(run_dir, [resolve_project_path(args.config, ROOT)])
        telemetry = bridge.wait_for_telemetry(float(config["ros"].get("telemetry_timeout_s", 2.0)))
        if telemetry is None:
            print("No telemetry received before timeout.")
            return 3
        rate = RateTimer(float(config["ros"].get("command_rate_hz", 20.0)))
        step = 0
        while True:
            bridge.spin_once(0.0)
            telemetry = bridge.latest_telemetry
            if telemetry is None:
                command = safety.filter(VehicleCommand(brake=1.0), None, now_s())
            else:
                packet = build_observation(telemetry, track, previous_command, config)
                action, _ = model.predict(packet.observation, deterministic=True)
                command = safety.filter(mapper.map_action(action), telemetry, now_s())
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
                    steering=command.steering,
                    throttle=command.throttle,
                    brake=command.brake,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    collision_detected=telemetry.collision_detected,
                    off_track=telemetry.off_track or packet.track_features.off_track,
                )
                logger.log_step(record)
                previous_command = command
                step += 1
            bridge.publish_command(command)
            rate.sleep()
    except KeyboardInterrupt:
        print("Stopping controller and sending brake command.")
        return 130
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if bridge is not None:
            try:
                bridge.publish_command(VehicleCommand(brake=1.0))
            except Exception:
                pass
            bridge.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
