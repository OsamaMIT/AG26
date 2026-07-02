"""Safety filtering for normalized vehicle commands."""

from __future__ import annotations

from dataclasses import dataclass

from gohan.telemetry.telemetry_state import VehicleCommand, VehicleTelemetry
from gohan.utils.math_utils import clamp
from gohan.utils.timing import now_s


@dataclass(slots=True)
class SafetyConfig:
    """Safety-filter configuration."""

    enable: bool = True
    max_steering_rate_per_s: float = 3.0
    max_throttle_rate_per_s: float = 4.0
    max_brake_rate_per_s: float = 6.0
    early_training_speed_cap_mps: float | None = 25.0
    emergency_brake_on_no_telemetry: bool = True
    emergency_brake_on_offtrack: bool = True
    telemetry_timeout_s: float = 2.0

    @classmethod
    def from_config(cls, config: dict) -> "SafetyConfig":
        safety = dict(config.get("safety", {}))
        ros = dict(config.get("ros", {}))
        if "telemetry_timeout_s" not in safety:
            safety["telemetry_timeout_s"] = ros.get("telemetry_timeout_s", cls.telemetry_timeout_s)
        return cls(**{key: value for key, value in safety.items() if key in cls.__dataclass_fields__})


class SafetyFilter:
    """Apply emergency logic and rate limits to vehicle commands."""

    def __init__(self, config: SafetyConfig | dict | None = None) -> None:
        if isinstance(config, SafetyConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SafetyConfig.from_config(config)
        else:
            self.config = SafetyConfig()
        self._last_command = VehicleCommand()
        self._last_time: float | None = None

    @property
    def last_command(self) -> VehicleCommand:
        return self._last_command

    def reset(self, command: VehicleCommand | None = None, timestamp: float | None = None) -> None:
        self._last_command = command or VehicleCommand()
        self._last_time = timestamp

    def filter(
        self,
        command: VehicleCommand,
        telemetry: VehicleTelemetry | None,
        timestamp: float | None = None,
        early_training: bool = True,
    ) -> VehicleCommand:
        """Return a filtered command for the current telemetry state."""
        if not self.config.enable:
            self._last_command = self._clip_command(command)
            self._last_time = timestamp if timestamp is not None else now_s()
            return self._last_command

        timestamp = timestamp if timestamp is not None else now_s()
        if self._last_time is None:
            dt = 1.0 / 20.0
        else:
            dt = max(timestamp - self._last_time, 1e-6)

        stale = telemetry is None
        if telemetry is not None:
            stale = timestamp - float(telemetry.timestamp) > self.config.telemetry_timeout_s

        if stale and self.config.emergency_brake_on_no_telemetry:
            return self._emergency(command.steering, timestamp)
        if telemetry is not None and telemetry.off_track and self.config.emergency_brake_on_offtrack:
            return self._emergency(command.steering, timestamp)

        clipped = self._clip_command(command)
        if (
            early_training
            and telemetry is not None
            and self.config.early_training_speed_cap_mps is not None
            and telemetry.speed_mps > self.config.early_training_speed_cap_mps
        ):
            clipped = VehicleCommand(steering=clipped.steering, throttle=0.0, brake=clipped.brake)

        filtered = VehicleCommand(
            steering=self._rate_limit(
                self._last_command.steering,
                clipped.steering,
                self.config.max_steering_rate_per_s,
                dt,
            ),
            throttle=self._rate_limit(
                self._last_command.throttle,
                clipped.throttle,
                self.config.max_throttle_rate_per_s,
                dt,
            ),
            brake=self._rate_limit(
                self._last_command.brake,
                clipped.brake,
                self.config.max_brake_rate_per_s,
                dt,
            ),
        )
        self._last_command = filtered
        self._last_time = timestamp
        return filtered

    def _emergency(self, steering: float, timestamp: float) -> VehicleCommand:
        emergency = VehicleCommand(steering=clamp(steering, -1.0, 1.0), throttle=0.0, brake=1.0)
        self._last_command = emergency
        self._last_time = timestamp
        return emergency

    @staticmethod
    def _clip_command(command: VehicleCommand) -> VehicleCommand:
        return VehicleCommand(
            steering=clamp(command.steering, -1.0, 1.0),
            throttle=clamp(command.throttle, 0.0, 1.0),
            brake=clamp(command.brake, 0.0, 1.0),
        )

    @staticmethod
    def _rate_limit(previous: float, target: float, rate_per_s: float, dt: float) -> float:
        delta = clamp(target - previous, -rate_per_s * dt, rate_per_s * dt)
        return clamp(previous + delta, -1.0, 1.0)
