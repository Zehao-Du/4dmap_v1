#!/bin/bash
# DP baseline on PlugCharger-v1, 800 demos
set -o pipefail
ROOT_DIR=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy
DP_DIR="$ROOT_DIR/baselines/diffusion_policy"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$DP_DIR"
mkdir -p outputs/train_logs
RUN_NAME="plugcharger_dp_baseline_603demos_seed1"
LOG_FILE="outputs/train_logs/${RUN_NAME}.log"
echo "[$(date +%H:%M:%S)] running $RUN_NAME -> $LOG_FILE"
python train_rgbd.py \
  --exp-name "$RUN_NAME" --seed 1 --env-id PlugCharger-v1 \
  --demo-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/dataset/ManiSkill/PlugCharger-v1/motionplanning/PlugCharger.rgb.pd_ee_delta_pose.physx_cpu.filtered.h5 \
  --num-demos 1000 --control-mode pd_ee_delta_pose --sim-backend physx_cpu \
  --obs-mode rgb --max-episode-steps 400 --total-iters 100000 --batch-size 64 \
  --obs-horizon 2 --act-horizon 8 --pred-horizon 16 --lr 0.0001 \
  --visual-encoder plain_conv \
  --log-freq 100 --eval-freq 10000 --num-eval-episodes 100 --num-eval-envs 10 \
  --num-dataload-workers 0 --no-track --no-capture-video 2>&1 | tee "$LOG_FILE"
