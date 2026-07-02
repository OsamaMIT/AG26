# GOHAN

GOHAN means Guided Optimization for High-speed Autonomous Navigation.

GOHAN is a telemetry-only reinforcement learning autonomy stack for Autonoma Labs AWSIM Racing Simulator and the Dallara AV-21R. AWSIM publishes ROS2 telemetry, GOHAN converts that telemetry into a normalized vector observation, a Stable-Baselines3 policy predicts steering and throttle/brake, and GOHAN publishes high-level vehicle commands back to AWSIM.

GOHAN is a separate Python/ROS2 autonomy project. It does not include a simulator, modify simulator source, use camera perception, or assume CUDA.

## Architecture

```text
AWSIM Racing Simulator
  -> ROS2 telemetry
GOHAN Ros2Bridge
  -> telemetry feature extractor
Stable-Baselines3 PPO/SAC-ready policy
  -> action mapper and safety filter
GOHAN Ros2Bridge
  -> ROS2 vehicle command
AWSIM Dallara AV-21R
  -> logging, reward, and training loop
```

## Layout

```text
gohan/
  configs/
  scripts/
  src/gohan/
  tests/
  runs/
```

## Prerequisites

- Linux with ROS2 installed externally.
- Python 3.10+.
- AWSIM Racing Simulator running separately, from any install location.
- ROS2 messages for the Autonoma high-level racing interface, such as `autonoma_msgs`.
- CPU is the default training device.

`rclpy` is intentionally not listed in `requirements.txt`; it comes from the sourced ROS2 installation.

## One-Time GOHAN Setup

Use system Python for ROS2 compatibility:

```bash
cd /home/osama/Documents/RL/AG26/gohan
conda deactivate
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Build/source the external ROS2 message workspace separately. On this machine the expected workspace is:

```bash
source /opt/ros/jazzy/setup.bash
source /home/osama/Documents/RL/AG26/ros2_ws/install/setup.bash
```

## Every New GOHAN Terminal

Run these commands before any GOHAN ROS script:

```bash
cd /home/osama/Documents/RL/AG26/gohan
conda deactivate
source /opt/ros/jazzy/setup.bash
source /home/osama/Documents/RL/AG26/ros2_ws/install/setup.bash
source .venv/bin/activate
export ROS_DOMAIN_ID=0
```

## AWSIM Must Run Separately

GOHAN does not need an AWSIM clone or build directory. Start AWSIM however you installed it, keep it open, and put the car into the Drive scene with ROS2 control enabled. GOHAN then connects automatically through ROS2 topic discovery.

Example AWSIM terminal:

```bash
export ROS_DOMAIN_ID=0
# launch AWSIM using your installed executable, launcher, or desktop shortcut
```

In the AWSIM GUI, use Scenario Setup, choose a ROS2-controlled vehicle, and enter Drive. Topics will not appear while AWSIM is closed or still sitting outside the running scenario.

Manual reset is the default. Automatic reset requires an AWSIM-side reset service or simulator integration. Add that later in `AWSIMRacingEnv.reset()` once a reset API is available.

## Inspect ROS2 Topics

```bash
ros2 topic list -t
python scripts/inspect_ros_topics.py --config configs/awsim.yaml
```

Expected high-level topics include:

```text
/novatel_top/inspva or /novatel_bottom/inspva
/novatel_top/rawimux or /novatel_bottom/rawimux
/vehicle_data
/powertrain_data
/race_control
/vehicle_inputs
/to_raptor
```

If you only see `/parameter_events` and `/rosout`, ROS2 is alive but AWSIM is not visible on the graph. Keep AWSIM open in Drive mode and confirm both terminals use the same `ROS_DOMAIN_ID`.

If a command topic type is unsupported, inspect it:

```bash
ros2 topic info /vehicle_inputs
```

Then update:

```text
src/gohan/ros/message_adapters.py
```

GOHAN will fail clearly rather than publishing an unknown command shape.

## Record Telemetry

```bash
python scripts/record_telemetry.py --config configs/awsim.yaml --run-name telemetry_test
```

This checks that AWSIM telemetry is visible and that the adapters can build GOHAN telemetry rows.

## Test Command Publishing

```bash
python scripts/drive_manual_test.py --config configs/awsim.yaml --mode slow_forward
```

Modes:

- `neutral`
- `brake`
- `slow_forward`
- `sine_steer_low_speed`

Run this before training.

## Check Environment

```bash
python scripts/check_awsim_env.py --config configs/awsim.yaml
```

This waits for telemetry, creates one observation vector, publishes one neutral command, and exits.

## Train PPO

Run training only after topic inspection, telemetry recording, environment check, and manual command publishing work.

```bash
python scripts/train_ppo.py \
  --config configs/awsim.yaml \
  --training-config configs/training.yaml \
  --reward-config configs/reward.yaml \
  --run-name awsim_ppo_v1 \
  --total-timesteps 100000
```

Models and logs are written under:

```text
runs/awsim_ppo_v1/
```

## Evaluate

```bash
python scripts/evaluate_policy.py \
  --config configs/awsim.yaml \
  --model runs/awsim_ppo_v1/models/final_model.zip \
  --episodes 3
```

## Live Controller

```bash
python scripts/run_inference_controller.py \
  --config configs/awsim.yaml \
  --model runs/awsim_ppo_v1/models/final_model.zip
```

Ctrl+C sends a brake command before shutdown.

## Plot Telemetry

```bash
python scripts/plot_telemetry.py --input runs/awsim_ppo_v1/telemetry.csv --output runs/awsim_ppo_v1/plots
```

## Observation Vector

GOHAN uses a 20D `np.float32` vector clipped to `[-1, 1]`:

```text
[
  lateral_error_norm,
  heading_error_sin,
  heading_error_cos,
  speed_norm,
  longitudinal_velocity_norm,
  lateral_velocity_norm,
  yaw_rate_norm,
  track_curvature_norm,
  distance_to_left_boundary_norm,
  distance_to_right_boundary_norm,
  lookahead_1_dx_norm,
  lookahead_1_dy_norm,
  lookahead_2_dx_norm,
  lookahead_2_dy_norm,
  lookahead_3_dx_norm,
  lookahead_3_dy_norm,
  previous_steering,
  previous_throttle_brake,
  lap_progress_sin,
  lap_progress_cos
]
```

The observation is built from ROS2 telemetry and track-relative features, not images.

## Action Mapping

The action space is:

```python
gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

`action[0]` is steering. `action[1]` is combined throttle/brake:

```text
if action[1] >= 0:
  throttle = action[1]
  brake = 0
else:
  throttle = 0
  brake = abs(action[1])
```

## Reward

`RacingReward` combines progress, speed, line tracking, heading alignment, yaw stability, action smoothness, off-track, collision, reverse-progress, and stuck penalties.

## Limitations

- Automatic reset requires AWSIM-side reset support or manual reset.
- Command message types may need adapter updates for a specific AWSIM build.
- The first goal is stable lap completion, not optimal racing.

## Roadmap

- v1: direct telemetry-to-control PPO.
- v1.5: reward and curriculum refinement.
- v2: telemetry-guided racing target correction.
- v3: hybrid controller plus RL target optimizer.

## Tests

```bash
pytest
```

Unit tests do not require ROS2 or AWSIM.
