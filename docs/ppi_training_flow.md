# PPI Policy Training Flow

This note summarizes the algorithm implemented in
`map4d/backbone/policy/ppi.py`.

## Overview

`PPI` is a diffusion policy for predicting bimanual action trajectories. It
encodes observations with a point-cloud based observation encoder, then uses a
diffusion head to denoise left-arm and right-arm trajectory tokens. When
`predict_point_flow=True`, it also predicts point flow as an auxiliary structured
target.

The default config in `map4d/backbone/config/ppi.yaml` uses:

- `horizon_keyframe = 4`
- `horizon_continuous = 50`
- `horizon = horizon_keyframe + horizon_continuous`
- `n_obs_steps = 1`
- `n_action_steps = 54`
- `action_dim = 16`
- `what_condition = "ppi"`
- `predict_point_flow = true`

The action layout is treated as:

- left end-effector position: `0:3`
- left end-effector rotation quaternion: `3:7`
- right end-effector position: `7:10`
- right end-effector rotation quaternion: `10:14`
- left gripper openess: `14:15`
- right gripper openess: `15:16`

## Initialization

`PPI.__init__` builds the policy components.

1. Observation encoder

   `ObservationEncoder` converts normalized observations into fixed conditioning
   features:

   - point cloud context coordinates/features
   - language features when enabled
   - robot state features
   - sampled point features
   - point-flow features and coordinates when `predict_point_flow=True`

2. Diffusion schedulers

   The policy uses separate DDPM schedulers for translation and rotation:

   - position scheduler: `scaled_linear`
   - rotation scheduler: `squaredcos_cap_v2`

   Both use `prediction_type` from `noise_scheduler_cfg`.

3. Diffusion head

   `what_condition` selects the denoising model:

   - `continuous` or `keyframe`: `DiffusionHeadPure`
   - `keypose_continuous`: `DiffusionHeadKeyposeContinuous`
   - `pointflow_continuous`: `DiffusionHeadPointflowContinuous`
   - `ppi`: `DiffusionHeadPPI`

   The `ppi` mode is the full model: it conditions continuous action prediction
   on keypose and point-flow information.

4. Masking and normalization

   `LowdimMaskGenerator` creates action masks for conditional diffusion, and
   `LinearNormalizer` stores observation/action/point-flow normalization stats.

## Training Forward Pass

Training is implemented in `PPI.forward(batch)`.

### 1. Normalize Inputs

The batch contains observations and action trajectories. If point-flow
prediction is enabled, it also contains point-flow targets.

```python
nobs = self.normalizer.normalize(batch["obs"])
nactions = self.normalizer["action"].normalize(batch["action"])
npoint_flow = self.normalizer["point_flow"].normalize(batch["point_flow"])
```

Only XYZ point-cloud coordinates are kept:

```python
nobs["point_cloud"] = nobs["point_cloud"][..., :3]
```

### 2. Encode Observations

The policy uses the first `n_obs_steps` observations, flattens the batch/time
dimensions, and passes them through `ObservationEncoder`.

The encoder output is packed into `fixed_inputs`, which conditions the diffusion
head during denoising.

When `predict_point_flow=True`, `fixed_inputs` contains:

```text
context_coord, context_feat, lang_feat, state_feat,
pn_coord, pn_feat, pointflow_feat, pointflow_coords
```

Otherwise it contains:

```text
context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat
```

### 3. Add Diffusion Noise

The normalized action trajectory is used as the clean denoising target:

```python
trajectory = nactions
```

The code samples Gaussian noise and one diffusion timestep per batch item:

```python
noise = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(
    0,
    self.noise_scheduler_cfg.num_train_timesteps,
    (batch_size,),
    device=trajectory.device,
).long()
```

Position and rotation are noised separately:

- left position: `trajectory[..., :3]`
- left rotation: `trajectory[..., 3:7]`
- right position: `trajectory[..., 7:10]`
- right rotation: `trajectory[..., 10:14]`

Gripper openess is not diffused; the ground-truth values are appended directly:

```python
noisy_trajectory = torch.cat(
    (pos_left, rot_left, pos_right, rot_right, gt_openess_left, gt_openess_right),
    -1,
)
```

### 4. Predict Noise Residuals

The noisy trajectory is split into left and right 7D pose streams:

```python
noisy_trajectory_left = noisy_trajectory[..., :7]
noisy_trajectory_right = noisy_trajectory[..., 7:14]
```

The diffusion head predicts denoising outputs:

```python
pred_left, pred_right, pred_point_flow = self.model(
    noisy_trajectory_left,
    noisy_trajectory_right,
    timesteps,
    fixed_inputs,
)
```

Each prediction is a list of layer outputs, and the training loss supervises all
layers.

### 5. Compute Loss

For each left/right prediction layer, the policy applies weighted L1 losses:

- translation noise loss: weight `30`
- rotation noise loss: weight `10`
- gripper openess loss: weight `30`

For point-flow prediction, it adds:

- point-flow L1 loss: weight `600`

The method returns:

```python
return total_loss, loss_dict
```

`loss_dict` includes:

- `action_loss`
- `point_flow_loss` when enabled
- `bc_loss`

## Inference

Inference is implemented in `PPI.predict_action(obs_dict)` and
`PPI.conditional_sample_diffuser_actor(...)`.

### 1. Encode Observations

The observation dictionary is normalized and encoded the same way as training.
The encoded features are packed into `fixed_inputs`.

### 2. Initialize Diffusion State

No ground-truth action is available at inference time. The code initializes:

```python
cond_data = torch.zeros((B, T, action_dim))
cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

Then `conditional_sample_diffuser_actor` starts from random Gaussian noise.

### 3. Reverse Diffusion

For each inference timestep:

1. Run the diffusion head on the current left/right trajectory samples.
2. Keep the last layer output.
3. Use the position scheduler to update position samples.
4. Use the rotation scheduler to update rotation samples.
5. Concatenate left pose, right pose, and predicted gripper openess into the
   full 16D action trajectory.

If point-flow prediction is enabled, the last point-flow prediction is returned
alongside the action trajectory.

### 4. Unnormalize and Slice Executed Actions

The predicted normalized action trajectory is unnormalized:

```python
action_pred = self.normalizer["action"].unnormalize(naction_pred)
```

Only the execution window is returned:

```python
start = To - 1
end = start + self.n_action_steps
action = action_pred[:, start:end]
```

The returned dictionary contains:

- `action`: actions to execute
- `action_pred`: full predicted trajectory
- `point_flow_pred` when `predict_point_flow=True`

## Key Takeaways

- The policy is a conditional diffusion model over bimanual action trajectories.
- Translation and rotation use separate noise schedules.
- Gripper openess is supervised directly instead of being diffused.
- `what_condition="ppi"` is the full PPI mode, conditioning on both keypose and
  point-flow information.
- Point flow is trained as an auxiliary structured prediction with a large loss
  weight.
