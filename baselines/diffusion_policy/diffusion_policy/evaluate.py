from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common


def _append_episode_metrics(eval_metrics, episode_info):
    for k, v in episode_info.items():
        if k.startswith("_"):
            continue
        if torch.is_tensor(v):
            v = v.float().cpu().numpy()
        eval_metrics[k].append(v)


def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if "final_info" in info and isinstance(info["final_info"], dict):
                    _append_episode_metrics(eval_metrics, info["final_info"]["episode"])
                elif "final_info" in info:
                    for final_info in info["final_info"]:
                        _append_episode_metrics(eval_metrics, final_info["episode"])
                elif "episode" in info:
                    _append_episode_metrics(eval_metrics, info["episode"])
                else:
                    raise KeyError(
                        f"Expected episode metrics in info, got keys: {list(info.keys())}"
                    )
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
