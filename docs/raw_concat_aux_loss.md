# Raw Concat + Auxiliary Future Prediction

## 动机

实验表明 raw concat（直接将map4d flatten拼接到state）效果最好（91%/82%），但缺少temporal建模。
本方案在raw concat基础上，增加一个辅助头从policy内部feature预测未来物体状态，
迫使policy学习物理动力学表示，同时不引入encoder瓶颈。

## ACT 方案

```
训练时数据流:
obs = {state(25), rgb, map4d(pre_horizon, 3, 12), future_map4d(future_horizon, 3, 12)}

1. raw_concat: state' = cat(state, map4d[-1].flatten()) → (61,)
2. input_proj_robot_state: state'(61) → proprio_token(256)
3. Transformer Encoder: [latent_z, proprio_token, img_patches...] → memory
4. 取 memory[1] (proprio slot after attention) → (256,)
5. Auxiliary head: memory[1] → MLP(256→256→future_horizon*3*9) → predict future
6. Transformer Decoder: queries cross-attend memory → actions

Loss = L1(action) + KL + aux_future_prediction_loss

需要改动：
- Transformer.forward() 额外返回 memory
- DETRVAE.forward() 把 proprio memory slot 传出来
- Agent.compute_loss() 加 auxiliary head + loss
```

## DP 方案

```
训练时数据流:
obs_cond = encode_obs(visual_feat, state+map4d_raw) → (obs_horizon × total_dim,)

1. obs_cond 只依赖 rgb + state + map4d，不含 diffusion noise
2. Auxiliary head: obs_cond → MLP(obs_cond_dim→256→future_horizon*3*9) → predict future
3. UNet forward: noisy_action + obs_cond → noise_pred (正常diffusion流程)

Loss = MSE(noise) + aux_future_prediction_loss

需要改动：
- Agent.compute_loss() 加 auxiliary head + loss
- 不需要改 UNet 代码
```

## 共同点

- 辅助头结构相同：MLP预测未来物体 delta position + delta rotation
- Loss相同：L1(pred_future, gt_future)
- GT来源相同：dataset中 future_map4d 切片
- 推理时不调用辅助头，无额外开销

## GT 构造

```
future_map4d: (B, future_horizon, N, 12)
current_map4d = map4d[:, -1]  # (B, N, 12)

gt_future_pos = future_map4d[..., 3:6]    # (B, H, N, 3)
gt_future_rot = future_map4d[..., 6:12]   # (B, H, N, 6)
current_pos = current_map4d[:, :, 3:6]    # (B, N, 3)
current_rot = current_map4d[:, :, 6:12]   # (B, N, 6)

gt_delta_pos = gt_future_pos - current_pos.unsqueeze(1)  # (B, H, N, 3)
gt_delta_rot = gt_future_rot - current_rot.unsqueeze(1)  # (B, H, N, 6)
gt_target = cat(gt_delta_pos, gt_delta_rot, dim=-1).flatten(1)  # (B, H*N*9)
```

## 配置参数

- `--map4d-raw-concat`: 启用raw concat
- `--map4d-aux-loss`: 启用辅助future prediction loss（仅raw_concat模式下生效）
- `--map4d-aux-weight`: 辅助loss权重（默认1.0）
- `--map4d-future-horizon`: 预测未来帧数（默认30）
- `--map4d-pre-horizon`: 输入历史帧数（默认30）
