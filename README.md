# 4D Map

## Roadmap

TODO: validate 4d map representation and losses in **ManiSkill** (on task **StackCube-v1** first).

- [x] 4d map representation, including losses (```map4d/representation/maps4d```)
- [x] 4d map construction (```map4d/construction```)
- [ ] 4d map encoder (```map4d/encoder```)
- [ ] insert into baselines, like dp and act

StackCube 的 4D map representation 已放在 ```map4d/representation/maps4d/maniskill_stackcube.py```，基于 ```Map_4d``` 保存 scene-level Objects，并用 red cube / green cube / desk 的 Cuboid 节点描述结构参数。Construction 入口在 ```map4d/construction/map_constructor.py```，当前流程是 RGB-D -> Grounded-SAM2/manual masks -> structural parameter estimator -> FoundationPose poses -> instantiated Map4d，后续 encoder 开发可直接通过 ```map4d/map4d_encoder.py``` 的 ```Map4d_Encoder.construction(...)``` 调用。

## benckmark

### ManiSkill

Data Collection:

**Remember to change paths.**

```bash
python -m mani_skill.examples.motionplanning.panda.run -h

# generating trajectory by motion planning
python -m mani_skill.examples.motionplanning.panda.run \
-e StackCube-v1 \
-n 1000 \
--only-count-success \
-b cpu \
--traj-name StackCube \
--record-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill \
--num-procs 10

# replay to control mode we need
python -m mani_skill.trajectory.replay_trajectory \
--traj-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.h5 \
-o rgb \
-c pd_ee_delta_pose \
--no-verbose \
--max-retry 3 \
--no-allow-failure \
--save-traj \
-n 1

# convert to lerobot format
python -m mani_skill.trajectory.convert_to_lerobot \
--traj-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5 \
--output-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/dataset/maniskill \
--fps 30

# --count 100
```

Train dp
```bash
seed=1
demos=100
python train_rgbd.py --env-id StackCube-v1 \
  --demo-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/StackCube-v1/motionplanning/StackCube.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 1000 \
  --total_iters 400000 --obs-mode "rgb" \
  --exp-name diffusion_policy-StackCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
  --track
```

Train dp with 4D map

The current StackCube 4D map training path uses ManiSkill GT to build a 4D map tensor for each observation sequence. The training config is kept in ```baselines/diffusion_policy/configs/stackcube_map4d_train.conf``` and loads 100 trajectories by default to match the baseline setting.

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
conda activate 4dmap

source baselines/diffusion_policy/configs/stackcube_map4d_train.conf
mkdir -p "$(dirname "${LOG_FILE}")"
python baselines/diffusion_policy/train_rgbd.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
```

For a background run:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
conda activate 4dmap

source baselines/diffusion_policy/configs/stackcube_map4d_train.conf
mkdir -p "$(dirname "${LOG_FILE}")"
nohup python baselines/diffusion_policy/train_rgbd.py "${TRAIN_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
tail -f "${LOG_FILE}"
```

The config uses ```--use-map4d```, ```--map4d-source maniskill_gt```, ```--obs-mode rgb+depth```, and ```--num-demos 100```. Wandb upload is disabled by default through ```--no-track```; set ```WANDB_MODE=offline``` if you later enable tracking.

Smoke test:

```bash
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
conda activate 4dmap
python baselines/diffusion_policy/smoke_map4d_pipeline.py
```

The smoke test writes review artifacts under ```outputs/map4d_pipeline_smoke```, including tensors, arrays, RGB images, and 4D map visualizations.


## requirements

for dp, you only need to install requirements regarding **maniskill** and **diffusion policy** like diffusers in ```requirements.txt```.
