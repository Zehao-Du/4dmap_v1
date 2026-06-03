"""Evaluate ACT + map4d with GT map4d injected by observation wrapper."""
from collections import defaultdict
import numpy as np
import torch
from mani_skill.utils import common


def evaluate_map4d(n: int, agent, eval_envs, eval_kwargs):
    stats, num_queries, temporal_agg, max_timesteps, device, sim_backend, map4d_pre_horizon = eval_kwargs.values()

    use_visual_obs = isinstance(eval_envs.single_observation_space.sample(), dict)
    delta_control = not stats
    if not delta_control:
        if sim_backend == "physx_cpu":
            pre_process = lambda s_obs: (s_obs - stats['state_mean'].cpu().numpy()) / stats['state_std'].cpu().numpy()
        else:
            pre_process = lambda s_obs: (s_obs - stats['state_mean']) / stats['state_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    action_dim = eval_envs.action_space.shape[-1]
    num_envs = eval_envs.num_envs
    if temporal_agg:
        query_frequency = 1
        all_time_actions = torch.zeros([num_envs, max_timesteps+1, max_timesteps+1+num_queries, action_dim], device=device)
    else:
        query_frequency = num_queries
        actions_to_take = torch.zeros([num_envs, num_queries, action_dim], device=device)

    agent.eval()
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        ts, eps_count = 0, 0

        while eps_count < n:
            # pre-process obs
            if use_visual_obs:
                obs['state'] = pre_process(obs['state']) if not delta_control else obs['state']
                obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            else:
                obs = pre_process(obs) if not delta_control else obs
                obs = common.to_tensor(obs, device)

            # query policy
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)

            if temporal_agg:
                assert query_frequency == 1
                all_time_actions[:, ts, ts:ts+num_queries] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                actions_populated = torch.zeros(max_timesteps+1, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - num_queries):ts+1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                k = 0.01
                if ts < num_queries:
                    exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step[0]), device=device))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.tile(exp_weights, (num_envs, 1))
                    exp_weights = torch.unsqueeze(exp_weights, -1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            action = post_process(raw_action) if not delta_control else raw_action
            if sim_backend == "physx_cpu":
                action = action.cpu().numpy()

            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            if truncated.any():
                assert truncated.all() == truncated.any()
                if "final_info" in info and isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                elif "final_info" in info:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                else:
                    for k, v in info["episode"].items():
                        if isinstance(v, np.ndarray):
                            eval_metrics[k].append(v)
                        else:
                            eval_metrics[k].append(v.float().cpu().numpy())
                eps_count += num_envs
                ts = 0
                if temporal_agg:
                    all_time_actions = torch.zeros([num_envs, max_timesteps+1, max_timesteps+1+num_queries, action_dim], device=device)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
