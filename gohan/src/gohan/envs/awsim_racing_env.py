"""Gymnasium environment backed by live AWSIM ROS2 telemetry and commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gohan.control.action_mapper import ActionMapper
from gohan.control.emergency_logic import EmergencyState, update_emergency_state
from gohan.control.safety_filter import SafetyFilter
from gohan.rewards.reward_function import RacingReward
from gohan.ros.ros2_bridge import Ros2Bridge
from gohan.telemetry.features import OBSERVATION_SIZE, build_observation
from gohan.telemetry.logger import TelemetryLogger
from gohan.telemetry.metrics import summarize_episode
from gohan.telemetry.telemetry_state import EpisodeStepRecord, VehicleCommand, VehicleTelemetry
from gohan.track.track_map import TrackMap
from gohan.utils.config import load_config_bundle
from gohan.utils.paths import make_run_dir, project_root, resolve_project_path
from gohan.utils.timing import now_s


class AWSIMRacingEnv(gym.Env):
    """Telemetry-only GOHAN environment using live AWSIM Racing Simulator over ROS2."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any] | str | Path = "configs/awsim.yaml",
        reward_config: dict[str, Any] | str | Path | None = None,
        run_name: str | None = None,
        enable_logging: bool = True,
    ) -> None:
        super().__init__()
        self.project_root = project_root()
        if isinstance(config, (str, Path)):
            self.config = load_config_bundle(config, reward_config=reward_config if isinstance(reward_config, (str, Path)) else None)
            if isinstance(reward_config, dict):
                self.config.update(reward_config)
        else:
            self.config = dict(config)
            if isinstance(reward_config, dict):
                self.config.update(reward_config)

        self.env_cfg = dict(self.config.get("env", {}))
        self.ros_cfg = dict(self.config.get("ros", {}))
        self.vehicle_cfg = dict(self.config.get("vehicle", {}))
        self.dt = float(self.env_cfg.get("dt", 0.05))
        self.max_episode_steps = int(self.env_cfg.get("max_episode_steps", 3000))
        self.telemetry_timeout_s = float(self.ros_cfg.get("telemetry_timeout_s", 2.0))
        self.reset_mode = str(self.env_cfg.get("reset_mode", "manual"))

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        track_config = resolve_project_path(self.env_cfg.get("track_config", "configs/tracks/default_track.yaml"), self.project_root)
        self.track_map = TrackMap.from_file(track_config)
        self.action_mapper = ActionMapper()
        self.safety_filter = SafetyFilter(self.config)
        self.reward_fn = RacingReward(self.config)
        self.bridge: Ros2Bridge | None = None
        if bool(self.env_cfg.get("require_awsim_connection", True)):
            self.bridge = Ros2Bridge(self.config)

        self.episode = 0
        self.step_count = 0
        self.previous_command = VehicleCommand()
        self.last_telemetry: VehicleTelemetry | None = None
        self.last_track_features = None
        self.emergency_state = EmergencyState()
        self._episode_records: list[EpisodeStepRecord] = []
        self.logger: TelemetryLogger | None = None
        if enable_logging:
            run_dir = make_run_dir(run_name or f"gohan_run_{int(now_s())}", self.project_root)
            self.logger = TelemetryLogger(run_dir)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_count = 0
        self.reward_fn.reset()
        self.safety_filter.reset()
        self.emergency_state.reset()
        self._episode_records.clear()

        if self.bridge is None:
            raise RuntimeError("AWSIMRacingEnv requires a live AWSIM ROS2 bridge; mock simulation is intentionally not provided.")

        if self.reset_mode == "manual":
            self._publish_command_for_reset()
        telemetry = self.bridge.wait_for_telemetry(self.telemetry_timeout_s)
        if telemetry is None:
            raise TimeoutError(
                "Timed out waiting for AWSIM telemetry. Start AWSIM, source ROS2, and inspect topics before training."
            )
        self.last_telemetry = telemetry
        packet = build_observation(telemetry, self.track_map, self.previous_command, self.config)
        self.last_track_features = packet.track_features
        return packet.observation, {"telemetry": telemetry, "track_features": packet.track_features}

    def step(self, action: np.ndarray):
        if self.bridge is None:
            raise RuntimeError("AWSIMRacingEnv requires a live AWSIM ROS2 bridge; mock simulation is intentionally not provided.")
        self.step_count += 1
        raw_command = self.action_mapper.map_action(action)
        telemetry_for_safety = self.last_telemetry
        command = self.safety_filter.filter(raw_command, telemetry_for_safety, now_s(), early_training=True)
        self._publish_action(command)

        telemetry = self._wait_for_next_telemetry()
        truncated = False
        if telemetry is None:
            telemetry = self.last_telemetry
            truncated = True
        if telemetry is None:
            raise TimeoutError("No AWSIM telemetry received before or during step.")

        packet = build_observation(telemetry, self.track_map, command, self.config)
        reward, reward_components = self._compute_reward(telemetry, packet.track_features, command)
        self.emergency_state = update_emergency_state(
            self.emergency_state,
            telemetry,
            packet.track_features,
            float(self.vehicle_cfg.get("max_yaw_rate_radps", 3.0)),
        )
        terminated = self._check_terminated(telemetry)
        truncated = truncated or self._check_truncated()

        record = EpisodeStepRecord(
            timestamp=now_s(),
            episode=self.episode,
            step=self.step_count,
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
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            collision_detected=telemetry.collision_detected,
            off_track=telemetry.off_track or packet.track_features.off_track,
            info={"reward_components": reward_components},
        )
        self._episode_records.append(record)
        if self.logger is not None:
            self.logger.log_step(record, reward_components)
            if terminated or truncated:
                self.logger.log_episode_summary(self.episode, summarize_episode(self._episode_records))

        self.last_telemetry = telemetry
        self.last_track_features = packet.track_features
        self.previous_command = command
        info = {
            "telemetry": telemetry,
            "track_features": packet.track_features,
            "reward_components": reward_components,
            "command": command,
        }
        return packet.observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        if self.logger is not None:
            self.logger.close()
            self.logger = None

    def _get_obs(self) -> np.ndarray:
        if self.last_telemetry is None:
            return np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
        return build_observation(self.last_telemetry, self.track_map, self.previous_command, self.config).observation

    def _compute_reward(self, telemetry, track_features, command):
        return self.reward_fn.compute(telemetry, track_features, command, self.dt)

    def _check_done(self) -> tuple[bool, bool]:
        if self.last_telemetry is None:
            return False, True
        return self._check_terminated(self.last_telemetry), self._check_truncated()

    def _check_terminated(self, telemetry: VehicleTelemetry) -> bool:
        if telemetry.collision_detected:
            return True
        if self.emergency_state.off_track_steps >= int(self.env_cfg.get("max_off_track_steps", 40)):
            return True
        if self.emergency_state.spin_steps >= int(self.env_cfg.get("max_spin_steps", 20)):
            return True
        if telemetry.lap_count > 0 and self.step_count > 10:
            return True
        return False

    def _check_truncated(self) -> bool:
        if self.step_count >= self.max_episode_steps:
            return True
        if self.emergency_state.stuck_steps >= int(self.env_cfg.get("max_stuck_steps", 100)):
            return True
        if self.last_telemetry is not None and now_s() - self.last_telemetry.timestamp > self.telemetry_timeout_s:
            return True
        return False

    def _publish_action(self, command: VehicleCommand) -> None:
        if self.bridge is None:
            raise RuntimeError("Cannot publish action without AWSIM ROS2 bridge.")
        self.bridge.publish_command(command)

    def _publish_command_for_reset(self) -> None:
        assert self.bridge is not None
        self.bridge.publish_command(VehicleCommand(steering=0.0, throttle=0.0, brake=0.5))
        for _ in range(3):
            self.bridge.spin_once(self.dt)

    def _wait_for_next_telemetry(self) -> VehicleTelemetry | None:
        assert self.bridge is not None
        start_update = self.bridge.telemetry_subscriber.last_update_s
        deadline = now_s() + self.dt
        latest = self.bridge.latest_telemetry
        while now_s() < deadline:
            self.bridge.spin_once(min(self.dt, 0.01))
            latest = self.bridge.latest_telemetry
            if self.bridge.telemetry_subscriber.last_update_s > start_update:
                return latest
        return latest
