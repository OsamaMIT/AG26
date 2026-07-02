#!/usr/bin/env python3
"""Evaluate a trained GOHAN policy against live AWSIM."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from gohan.envs.awsim_racing_env import AWSIMRacingEnv
from gohan.utils.config import load_config_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/awsim.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--run-name", default="evaluation")
    args = parser.parse_args()
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        print(f"stable-baselines3 is required for evaluation: {exc}")
        return 2

    config = load_config_bundle(args.config, reward_config=args.reward_config)
    env = None
    model = PPO.load(args.model, device="cpu")
    total_reward = 0.0
    speeds: list[float] = []
    off_track_events = 0
    collision_events = 0
    best_lap_time = None
    try:
        env = AWSIMRacingEnv(config, run_name=args.run_name, enable_logging=True)
        for episode in range(args.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                telemetry = info["telemetry"]
                total_reward += reward
                speeds.append(telemetry.speed_mps)
                off_track_events += int(telemetry.off_track)
                collision_events += int(telemetry.collision_detected)
                if telemetry.lap_time > 0.0:
                    best_lap_time = telemetry.lap_time if best_lap_time is None else min(best_lap_time, telemetry.lap_time)
                done = terminated or truncated
            print(f"Episode {episode + 1} complete.")
        print(f"episodes: {args.episodes}")
        print(f"total reward: {total_reward:.3f}")
        print(f"lap progress: {env.last_track_features.lap_progress if env.last_track_features else 0.0:.3f}")
        print(f"off-track events: {off_track_events}")
        print(f"collision events: {collision_events}")
        print(f"mean speed: {sum(speeds) / max(len(speeds), 1):.3f} m/s")
        print(f"best lap time: {best_lap_time if best_lap_time is not None else 'unavailable'}")
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
