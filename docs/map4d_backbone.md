# map4d backbone

single robot arm first.

## DiT

### Input

transformer tokens

noisy target tokens

- Noised trajectory [7] = delta_pos(3) + delta_quat(4)
- Noised Keyframe object pose [n*9] = n object \* local_delta_pos(3)+delta_rot6d(6)
- Noised Keyframe tcp pose [7] = local_delta_pos(3) + delta_quat(4)

context tokens

- RGB Image Features (RGB -> DINOv3 -> feature)
- Depth Image Features (reserved interface; not inserted into DiT context tokens in v1)
- map4d object history tokens (per-frame per-object pose -> MLP -> tokens)
- size parameters and relation tokens (parameters -> MLP -> tokens)

Context token sources:

- RGB Image Features: optional visual context extracted from observation RGB frames.
- Depth Image Features: interface reserved. In v1, depth is used upstream for map4d construction and is not directly inserted as DiT transformer context tokens.
- map4d object history tokens: v1 uses the same observation horizon as RGB frames, default `T_obs=2`. Each per-frame per-object pose is independently projected by an MLP into one object token.
- size parameters: static object geometry from map4d.
- relation parameters: pairwise object geometry/relation features derived from map4d.
- robot state: encoded by MLP and used as FiLM/AdaLN global conditioning.
- denoising step: timestep embedding used as FiLM/AdaLN global conditioning.

global conditioning / modulation

- robot state (MLP+FiLM)
- Denoising Step (MLP+FiLM)

### Output

- DeNoised trajectory [7] = delta_pos(3) + delta_quat(4)
- DeNoised Keyframe object pose [n*9] = n object \* local_delta_pos(3)+delta_rot6d(6)
- DeNoised Keyframe tcp pose [7] = local_delta_pos(3) + delta_quat(4)

Pose target convention:

- Trajectory target uses delta pose in formal settings: delta_pos(3) + delta_quat(4). StackCube may use delta_pos only for debugging/compatibility.
- Keyframe object pose target uses local_delta_pos(3) + delta_rot6d(6).
- Keyframe tcp pose target uses local_delta_pos(3) + delta_quat(4).
- local_delta_pos is expressed in the current pose frame: `R_current^T * (p_future - p_current)`.
- Rotation targets are relative rotations from current pose to future keyframe pose.

Gripper openness:

- map4d follows PPI's gripper openness design.
- Gripper openness is not part of the noisy diffusion target.
- It is predicted by a direct regression head from trajectory token features.
- It is supervised with L1 loss against GT openness.
- During reverse diffusion, only delta_pos and delta_quat are scheduler-stepped; openness is read from the latest model output.

Inference:

- Keyframe object/tcp tokens are latent plan tokens.
- Trajectory, keyframe object, and keyframe tcp tokens are jointly denoised during training and inference.
- Keyframe object/tcp tokens participate in transformer attention and guide trajectory generation.
- Only trajectory tokens are sent to the controller.
  
### Loss

diffusion losses are L1 losses on predicted noise

weight

- trajectory delta pose noise: 1
- keyframe tcp delta pose noise: 1
- keyframe object delta pose noise: 0.3

direct regression losses

- gripper openness: L1 loss on GT openness

### Scheduler

Rotation representation:

- TCP/action rotation uses normalized quaternion rot(4). Quaternions are canonicalized to `w >= 0` before training and renormalized after each denoising step.
- Object rotation uses rot6d(6). rot6d is denoised in 6D Euclidean space and orthogonalized with `rotation_6d_to_matrix` before geometric use.
- Position and rotation use separate schedulers. Quaternion/rot6d fields are not mean-std normalized as ordinary vectors.
- Gaussian diffusion noise moves quaternions off the unit sphere, so every noisy/denoised quaternion must be projected back to a canonical unit quaternion. This is a required map4d addition over the original PPI implementation.

### Normalizer

- map4d follows PPI's `LinearNormalizer` design for non-rotation numeric fields.
- delta_pos/local_delta_pos fields use the PPI limits normalizer and are scaled to `[-1, 1]`.
- robot state, RGB/depth features, and other learned context features follow their existing encoder/input normalization.
- Quaternion and rot6d fields are not mean-std normalized as ordinary vectors.
- Quaternion fields are canonicalized to `w >= 0` and renormalized after each denoising step.
- Noisy quaternion inputs are also renormalized before entering the denoising transformer.
- rot6d fields are denoised in raw 6D representation space and orthogonalized with `rotation_6d_to_matrix` before geometric use.
- Gripper openness follows PPI's direct regression path and uses the action/task normalizer if the dataset provides one.
