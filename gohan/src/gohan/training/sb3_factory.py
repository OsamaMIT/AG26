"""Stable-Baselines3 model construction helpers."""

from __future__ import annotations

from typing import Any


def make_ppo(env: Any, training_config: dict[str, Any]):
    """Create a CPU-first PPO MlpPolicy model from config."""
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for training. Install requirements.txt first.") from exc

    cfg = dict(training_config.get("training", training_config))
    if cfg.get("algorithm", "PPO") != "PPO":
        raise ValueError("GOHAN v1 supports PPO only.")
    hidden_sizes = list(cfg.get("policy_hidden_sizes", [128, 128]))
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 1024)),
        batch_size=int(cfg.get("batch_size", 64)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        seed=int(cfg.get("seed", 42)),
        device=str(cfg.get("device", "cpu")),
        policy_kwargs={"net_arch": hidden_sizes},
        verbose=1,
    )
