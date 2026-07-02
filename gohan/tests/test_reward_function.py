from gohan.rewards.reward_function import RacingReward
from gohan.telemetry.telemetry_state import TrackFeatures, VehicleCommand, VehicleTelemetry


def test_progress_reward_positive_after_forward_progress():
    reward_fn = RacingReward()
    telemetry = VehicleTelemetry(speed_mps=20.0)
    reward_fn.previous_progress_m = 1.0
    reward, components = reward_fn.compute(
        telemetry,
        TrackFeatures(progress_m=3.0, lap_length_m=100.0),
        VehicleCommand(throttle=0.2),
        dt=0.05,
    )
    assert components["progress"] > 0.0
    assert reward > 0.0


def test_collision_penalty_is_negative():
    reward_fn = RacingReward()
    reward, components = reward_fn.compute(
        VehicleTelemetry(collision_detected=True),
        TrackFeatures(progress_m=0.0, lap_length_m=100.0),
        VehicleCommand(),
        dt=0.05,
    )
    assert components["collision"] < 0.0
    assert reward < 0.0


def test_reverse_progress_penalty():
    reward_fn = RacingReward()
    reward_fn.previous_progress_m = 10.0
    reward, components = reward_fn.compute(
        VehicleTelemetry(speed_mps=5.0),
        TrackFeatures(progress_m=9.0, lap_length_m=100.0),
        VehicleCommand(),
        dt=0.05,
    )
    assert components["reverse"] < 0.0
    assert reward < 0.0
