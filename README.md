# 4D Map

## benckmark

### ManiSkill

Data Collection:

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

# --count 100
```

Train dp
```bash
seed=1
demos=100
python train_rgbd.py --env-id StackCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 30000 --obs-mode "rgb" \
  --exp-name diffusion_policy-PickCube-v1-rgb-${demos}_motionplanning_demos-${seed} \
  --track
```


## requirements

for dp, you only need to install requirements regarding **maniskill** and **diffusion policy** like diffusers.

```bash
######################### cuda 12.8

# pytorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# pytorch-geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
# pytorch3d
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation

# image processing
pip install opencv-python imageio

# 3d utils
pip install open3d trimesh

# clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# diffusers
pip install diffusers

### benchmark
# maniskill
pip install mani_skill
pip install numpy==1.26.4

```