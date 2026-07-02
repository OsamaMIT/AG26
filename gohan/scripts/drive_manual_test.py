#!/usr/bin/env python3
"""Publish safe manual commands to verify AWSIM command topics."""

from __future__ import annotations

import argparse
import math

from _bootstrap import bootstrap

bootstrap()

from gohan.ros.ros2_bridge import Ros2Bridge
from gohan.telemetry.telemetry_state import VehicleCommand
from gohan.utils.config import load_yaml
from gohan.utils.timing import RateTimer, now_s


def command_for_mode(mode: str, elapsed: float) -> VehicleCommand:
    if mode == "neutral":
        return VehicleCommand()
    if mode == "brake":
        return VehicleCommand(brake=1.0)
    if mode == "slow_forward":
        return VehicleCommand(throttle=0.12)
    if mode == "sine_steer_low_speed":
        return VehicleCommand(steering=0.25 * math.sin(2.0 * math.pi * 0.25 * elapsed), throttle=0.10)
    raise ValueError(f"Unsupported manual test mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awsim.yaml")
    parser.add_argument("--mode", choices=["neutral", "brake", "slow_forward", "sine_steer_low_speed"], default="neutral")
    parser.add_argument("--duration-s", type=float, default=5.0)
    args = parser.parse_args()
    bridge = None
    try:
        config = load_yaml(args.config)
        bridge = Ros2Bridge(config)
        rate = RateTimer(float(config["ros"].get("command_rate_hz", 20.0)))
        start = now_s()
        while now_s() - start < args.duration_s:
            elapsed = now_s() - start
            bridge.publish_command(command_for_mode(args.mode, elapsed))
            bridge.spin_once(0.0)
            rate.sleep()
        bridge.publish_command(VehicleCommand(brake=1.0))
        print(f"Manual command test complete: {args.mode}")
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
