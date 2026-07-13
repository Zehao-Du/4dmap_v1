"""Hydra/DDP training entry point for the standalone Map4D DiT policy."""

from __future__ import annotations

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

import copy
import json
import os
import pathlib
import random
import time
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import tqdm
import yaml

from map4d.backbone.common.checkpoint_util import TopKCheckpointManager
from map4d.backbone.common.pytorch_util import dict_apply, optimizer_to
from map4d.backbone.dataset.base_dataset import BaseDataset
from map4d.backbone.eval_maniskill import build_rollout_evaluator
from map4d.backbone.model.common.lr_scheduler import get_scheduler
from map4d.backbone.model.diffusion.ema_model import EMAModel
from map4d.backbone.policy.map4d_dit_policy import Map4DDiTPolicy
from map4d.representation.maps4d.metadata import get_task_metadata_value

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("map4d_meta", get_task_metadata_value, replace=True)


def _world_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def _init_dist():
    local_rank, rank, world_size = _world_info()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return local_rank, rank, world_size


class TrainMap4DDiTWorkspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: DictConfig, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.local_rank, self.global_rank, self.world_size = _init_dist()
        self.distributed = self.world_size > 1

        seed = int(cfg.training.seed) + self.global_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model: Map4DDiTPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: Optional[Map4DDiTPolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.epoch = 0

    @property
    def output_dir(self):
        if self._output_dir is not None:
            return self._output_dir
        return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    def _rank0(self) -> bool:
        return self.global_rank == 0

    def _log(self, message: str) -> None:
        if self._rank0():
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _policy_for_eval(self):
        if self.ema_model is not None:
            return self.ema_model
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if torch.cuda.is_available() and cfg.training.device.startswith("cuda"):
            device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        self._log(f"Using device={device}, world_size={self.world_size}")

        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 2 if cfg.training.max_train_steps is None else cfg.training.max_train_steps
            cfg.training.max_val_steps = 1 if cfg.training.max_val_steps is None else cfg.training.max_val_steps
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.checkpoint.save_ckpt = False

        self._log("Instantiating dataset")
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"dataset must be BaseDataset, got {type(dataset)}")
        self._log(f"Dataset ready: type={type(dataset).__name__}, train_len={len(dataset)}")
        self._log("Normalizing dataset")
        normalizer = dataset.get_normalizer()
        self._log("Normalizer ready")
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            shuffle = False
        else:
            train_sampler = None
            shuffle = bool(cfg.dataloader.shuffle)
        self._log(
            "Building train dataloader "
            f"batch_size={int(cfg.dataloader.batch_size)}, num_workers={int(cfg.dataloader.num_workers)}"
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=shuffle,
            num_workers=int(cfg.dataloader.num_workers),
            pin_memory=bool(cfg.dataloader.pin_memory),
            persistent_workers=bool(cfg.dataloader.persistent_workers),
            sampler=train_sampler,
        )

        val_dataset = dataset.get_validation_dataset()
        self._log(f"Validation dataset ready: val_len={len(val_dataset)}")
        val_dataloader = None
        val_sampler = None
        if len(val_dataset) > 0:
            if self.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            self._log(
                "Building validation dataloader "
                f"batch_size={int(cfg.val_dataloader.batch_size)}, "
                f"num_workers={int(cfg.val_dataloader.num_workers)}"
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=int(cfg.val_dataloader.batch_size),
                shuffle=False,
                num_workers=int(cfg.val_dataloader.num_workers),
                pin_memory=bool(cfg.val_dataloader.pin_memory),
                persistent_workers=bool(cfg.val_dataloader.persistent_workers),
                sampler=val_sampler,
            )

        self._log("Moving model to device")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        self._log("Building optimizer and scheduler")
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        optimizer_to(self.optimizer, device)
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=max(1, len(train_dataloader) * int(cfg.training.num_epochs)),
            last_epoch=-1,
        )
        ema: Optional[EMAModel] = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        if self.distributed:
            ddp_find_unused_parameters = bool(
                OmegaConf.select(cfg, "training.ddp_find_unused_parameters", default=True)
            )
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if device.type == "cuda" else None,
                find_unused_parameters=ddp_find_unused_parameters,
            )

        if self._rank0():
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
            self._log(f"Output dir: {self.output_dir}")

        rollout_evaluator = None
        if self._rank0():
            rollout_evaluator = build_rollout_evaluator(cfg, device=device, output_dir=self.output_dir)

        wandb_run = None
        if self._rank0() and cfg.logging.mode != "disabled":
            try:
                import wandb

                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging,
                )
            except Exception as exc:
                print(f"wandb disabled: {exc}")

        local_metrics_path = None
        local_metrics_every = int(OmegaConf.select(cfg, "metrics.local_every", default=100))
        local_metrics_enabled = bool(OmegaConf.select(cfg, "metrics.local", default=True))
        if self._rank0() and local_metrics_enabled:
            local_metrics_name = OmegaConf.select(
                cfg, "metrics.local_name", default="train_metrics.jsonl"
            )
            local_metrics_path = os.path.join(self.output_dir, str(local_metrics_name))

        def write_local_metrics(record):
            if local_metrics_path is None:
                return
            serializable = {}
            for key, value in record.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    serializable[key] = value
                elif hasattr(value, "item"):
                    serializable[key] = value.item()
            with open(local_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(serializable, sort_keys=True) + "\n")

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        train_sampling_batch = None

        for local_epoch_idx in range(int(cfg.training.num_epochs)):
            self._log(f"Starting training epoch {self.epoch}")
            if train_sampler is not None:
                train_sampler.set_epoch(local_epoch_idx)
            train_losses = []
            train_metric_sums = {}
            train_metric_count = 0
            step_log = {}
            self.model.train()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=float(cfg.training.tqdm_interval_sec),
                disable=not self._rank0(),
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    raw_loss, loss_dict = self.model(batch)
                    loss = raw_loss / int(cfg.training.gradient_accumulate_every)
                    loss.backward()
                    if (batch_idx + 1) % int(cfg.training.gradient_accumulate_every) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.lr_scheduler.step()
                        if ema is not None:
                            ema.step(self.model.module if isinstance(self.model, DDP) else self.model)

                    raw_loss_value = float(raw_loss.detach().cpu())
                    train_losses.append(raw_loss_value)
                    for key, value in loss_dict.items():
                        train_metric_sums[key] = train_metric_sums.get(key, 0.0) + float(value)
                    train_metric_count += 1
                    if self._rank0():
                        tepoch.set_postfix(loss=raw_loss_value, refresh=False)
                        step_log = {
                            "train_loss": raw_loss_value,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            **loss_dict,
                        }
                        if wandb_run is not None:
                            wandb_run.log(step_log, step=self.global_step)
                        if local_metrics_every > 0 and self.global_step % local_metrics_every == 0:
                            write_local_metrics({**step_log, "record_type": "train_step"})
                    self.global_step += 1
                    if cfg.training.max_train_steps is not None and batch_idx >= int(cfg.training.max_train_steps) - 1:
                        break

            if self._rank0() and train_losses:
                step_log["train_loss"] = float(np.mean(train_losses))
                if train_metric_count > 0:
                    for key, value in train_metric_sums.items():
                        step_log[key] = value / train_metric_count

            if val_dataloader is not None and self.epoch % int(cfg.training.val_every) == 0:
                if val_sampler is not None:
                    val_sampler.set_epoch(local_epoch_idx)
                eval_policy = self._policy_for_eval()
                eval_policy.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss, _ = eval_policy(batch)
                        val_losses.append(float(loss.detach().cpu()))
                        if cfg.training.max_val_steps is not None and batch_idx >= int(cfg.training.max_val_steps) - 1:
                            break
                if self._rank0() and val_losses:
                    step_log["val_loss"] = float(np.mean(val_losses))

            sample_every = int(cfg.training.sample_every)
            if train_sampling_batch is not None and sample_every > 0 and self.epoch % sample_every == 0:
                eval_policy = self._policy_for_eval()
                eval_policy.eval()
                with torch.no_grad():
                    result = eval_policy.predict_action(train_sampling_batch["obs"])
                    quat_norm = result["trajectory_pred"][..., 3:7].norm(dim=-1).mean()
                if self._rank0():
                    step_log["sample_quat_norm"] = float(quat_norm.detach().cpu())

            if (
                rollout_evaluator is not None
                and self.epoch > 0
                and self.epoch % int(cfg.rollout.every) == 0
            ):
                eval_policy = self._policy_for_eval()
                eval_policy.eval()
                rollout_metrics = rollout_evaluator.evaluate(
                    eval_policy,
                    epoch=self.epoch,
                    iteration=self.global_step,
                )
                if self._rank0():
                    for key, value in rollout_metrics.items():
                        step_log[f"rollout/{key}"] = float(value)
                    print(
                        "rollout "
                        + ", ".join(f"{key}={value:.4f}" for key, value in rollout_metrics.items())
                    )

            if (
                self._rank0()
                and cfg.checkpoint.save_ckpt
                and self.epoch % int(cfg.training.checkpoint_every) == 0
            ):
                os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag="latest")
                metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                if cfg.checkpoint.topk.monitor_key in metric_dict:
                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)

            if self._rank0() and wandb_run is not None and step_log:
                wandb_run.log(step_log, step=self.global_step)

            # Environment evaluation
            eval_env_every = int(cfg.training.get("eval_env_every", 0))
            if (
                self._rank0()
                and eval_env_every > 0
                and self.epoch > 0
                and self.epoch % eval_env_every == 0
            ):
                from map4d.backbone.evaluate_maniskill import evaluate_maniskill

                eval_policy = self._policy_for_eval()
                eval_policy.eval()
                env_metrics = evaluate_maniskill(
                    eval_policy,
                    cfg.task_name,
                    num_eval_episodes=int(cfg.training.get("num_eval_episodes", 100)),
                    num_eval_envs=int(cfg.training.get("num_eval_envs", 10)),
                    n_obs_steps=int(cfg.n_obs_steps),
                    robot_state_dim=int(cfg.robot_state_dim),
                    size_parameter_dim=int(cfg.size_parameter_dim),
                    relation_parameter_dim=int(cfg.relation_parameter_dim),
                    device=device,
                    use_rgb=False,
                    rgb_feature_dim=int(cfg.policy.model_cfg.get("rgb_feature_dim", 384)),
                )
                print(f"[Epoch {self.epoch}] Env eval: {env_metrics}")
                step_log.update({f"eval/{k}": v for k, v in env_metrics.items()})
                if wandb_run is not None:
                    wandb_run.log({f"eval/{k}": v for k, v in env_metrics.items()}, step=self.global_step)
                success_once = env_metrics.get("success_once", 0.0)
                if success_once > getattr(self, "_best_success_once", 0.0):
                    self._best_success_once = success_once
                    print(f"  New best success_once: {success_once:.4f}. Saving checkpoint.")
                    self.save_checkpoint(tag="best_success")

            if self._rank0() and step_log:
                write_local_metrics({**step_log, "record_type": "epoch"})
            self.epoch += 1
            if self.distributed:
                dist.barrier()

        if rollout_evaluator is not None:
            rollout_evaluator.close()

    def save_checkpoint(self, path=None, tag="latest"):
        if path is None:
            path = os.path.join(self.output_dir, "checkpoints", f"{tag}.pth.tar")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict() if self.ema_model is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "_output_dir": self._output_dir,
        }
        torch.save(checkpoint, path)
        return path


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")),
    config_name="map4d_dit",
)
def main(cfg: DictConfig):
    workspace = TrainMap4DDiTWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    t0 = time.time()
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"total time: {time.time() - t0:.3f}s")
