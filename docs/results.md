# Results

## **StackCube-v1 实验结果**

Task: StackCube-v1, control_mode=pd_ee_delta_pos, max_episode_steps=1000, 100k iterations

Eval: 100 episodes (10 envs × 10 rounds), eval时提供GT map4d

### **Diffusion Policy**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| DP baseline | 100 | 9% | 8% |
| DP + map4d GRU encoder | 100 | 11% | 6% |
| DP + map4d Transformer encoder | 100 | 15% | 10% |
| DP + map4d raw concat | 100 | 61% | 46% |
| DP + map4d raw concat + aux loss | 100 | 72% | 61% |
| DP + map4d raw concat + keyframe aux (TCP pose) | 100 | 14% | 1% |
| DP + map4d raw concat + keyframe aux (TCP pos) | 100 | 60% | 49% |
| DP baseline | 990 | 39% | 25% |
| DP + map4d GRU encoder | 990 | 46% | 33% |
| DP + map4d Transformer encoder | 990 | 33% | 7% |
| DP + map4d raw concat | 990 | 88% | 85% |
| DP + map4d raw concat + aux loss | 990 | 81% | 73% |
| DP + map4d raw concat + keyframe aux (TCP pose) | 990 | OOM | OOM |
| DP + map4d raw concat + keyframe aux (TCP pos) | 990 | **97%** | **97%** |

### **ACT**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| ACT baseline | 100 | 72% | 45% |
| ACT + map4d GRU encoder (context token) | 100 | 73% | 50% |
| ACT + map4d Transformer encoder (context token) | 100 | 76% | 54% |
| ACT + map4d MLP token | 100 | 67% | 44% |
| ACT + map4d raw concat | 100 | 73% | 51% |
| ACT + map4d raw concat + aux loss | 100 | 64% | 42% |
| ACT + map4d raw concat + keyframe aux (TCP pose) | 100 | 36% | 24% |
| ACT + map4d raw concat + keyframe aux (TCP pos) | 100 | 69% | 52% |
| ACT baseline | 990 | 85% | 71% |
| ACT + map4d GRU encoder (context token) | 990 | 80% | 69% |
| ACT + map4d Transformer encoder (context token) | 990 | 81% | 72% |
| ACT + map4d MLP token | 990 | 86% | 80% |
| ACT + map4d raw concat | 990 | 91% | 82% |
| ACT + map4d raw concat + aux loss | 990 | 84% | 65% |
| ACT + map4d raw concat + keyframe aux (TCP pose) | 990 | 89% | 73% |
| ACT + map4d raw concat + keyframe aux (TCP pos) | 990 | **91%** | **83%** |

### **StackCube Keyframe 结果备注**

- TCP pose 版本使用 `future_keyframe_tcp_pose` 的 7D pose target。
- TCP pos 版本只监督 TCP position，`future_keyframe_tcp_pose` 为 3D target，并使用 `--map4d-tcp-dim 3`。
- DP + TCP pos 990 demos 是目前 StackCube 最好结果，90k eval 达到 `success_once=97%`、`success_at_end=97%`。
- DP + TCP pose 990 demos 之前在加载 990 demos 到 GPU 时 OOM，未形成有效训练结果。

## **PlugCharger-v1**

Task: PlugCharger-v1, control_mode=pd_ee_delta_pose, max_episode_steps=400, 100k iterations

Dataset: 1000 demos (filtered ≤400 steps, mean 202 steps)

Eval: 100 episodes (10 envs × 10 rounds), eval时提供GT map4d

### **ACT**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| ACT baseline | 100 | 1% | 1% |
| ACT baseline | 1000 | 7% | 7% |
| ACT + map4d raw concat | 100 | 2% | 2% |
| ACT + map4d raw concat | 1000 | 9% | 9% |
| ACT + map4d raw concat + keyframe aux (TCP pose) | 100 | 1% | 1% |

### **Diffusion Policy**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| DP baseline | 100 | 0% | 0% |
| DP baseline | 1000 | 0% | 0% |
| DP + map4d raw concat | 100 | 3% | 3% |
| DP + map4d raw concat | 1000 | 7% | 4% |
| DP + map4d raw concat + keyframe aux (TCP pose) | 100 | 1% | 1% |

### **PlugCharger Keyframe 结果备注**

- PlugCharger 100-demo keyframe aux 当前没有明显收益；ACT / DP best 都只有约 1%。
- 训练 loss 能下降到较低值，但 eval success 几乎为 0，后续应优先排查 control/eval 设置、keyframe target 是否和任务阶段对齐，以及 1000-demo 设置。

## **实验说明**

- **map4d GRU encoder (context token)**: map4d序列经过GRU encoder编码为128维feature，作为独立context token融入policy
- **map4d Transformer encoder (context token)**: 同上，将GRU替换为Transformer (pre_horizon=30, future_horizon=30)
- **map4d MLP token**: 30帧map4d flatten为1080维，经MLP(1080→256→128)压缩为1个context token融入transformer memory
- **map4d raw concat**: 跳过encoder，直接取当前帧map4d flatten后拼接到robot state
- **map4d raw concat + aux loss**: raw concat基础上，从policy内部feature预测未来物体状态作为辅助loss
- **map4d raw concat + keyframe aux (TCP pose)**: raw concat基础上，预测未来关键帧的物体状态和7D TCP pose
- **map4d raw concat + keyframe aux (TCP pos)**: 同上，但TCP target只使用3D position；物体target仍为object delta position + rotation 6D
