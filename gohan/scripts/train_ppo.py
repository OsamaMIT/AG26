#!/usr/bin/env python3
"""Train PPO against live AWSIM telemetry/control."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from gohan.envs.awsim_racing_env import AWSIMRacingEnv
from gohan.training.callbacks import make_checkpoint_callback
from gohan.training.run_manager import RunManager
from gohan.training.sb3_factory import make_ppo
from gohan.utils.config import load_config_bundle, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/awsim.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--reward-config", default="configs/reward.yaml")
    parser.add_argument("--run-name", default="awsim_ppo_v1")
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        print(f"stable-baselines3 is required for training: {exc}")
        return 2

    run = RunManager(args.run_name, ROOT)
    config = load_config_bundle(args.config, args.training_config, args.reward_config, ROOT)
    training_cfg = load_yaml(args.training_config, ROOT)
    if args.total_timesteps is not None:
        training_cfg.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    env = None
    model = None
    try:
        env = Monitor(AWSIMRacingEnv(config, run_name=args.run_name, enable_logging=True), filename=str(run.run_dir / "monitor.csv"))
        model = make_ppo(env, training_cfg)
        checkpoint = make_checkpoint_callback(training_cfg["training"].get("checkpoint_freq", 10000), run.models_dir)
        total_timesteps = int(training_cfg["training"].get("total_timesteps", 100000))
        print(f"Training PPO for {total_timesteps} timesteps. Outputs: {run.run_dir}")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint)
        model.save(run.models_dir / "final_model")
        print(f"Saved final model to {run.models_dir / 'final_model.zip'}")
        return 0
    except KeyboardInterrupt:
        if model is not None:
            model.save(run.models_dir / "interrupted_model")
            print(f"Saved interrupted model to {run.models_dir / 'interrupted_model.zip'}")
        return 130
    except RuntimeError as exc:
        print(exc)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
