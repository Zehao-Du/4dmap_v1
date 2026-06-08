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
| DP baseline | 990 | 39% | 25% |
| DP + map4d GRU encoder | 990 | 46% | 33% |
| DP + map4d Transformer encoder | 990 | 33% | 7% |
| DP + map4d raw concat | 990 | 88% | 85% |
| DP + map4d raw concat + aux loss | 990 | 81% | 73% |

### **ACT**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| ACT baseline | 100 | 72% | 45% |
| ACT + map4d GRU encoder (context token) | 100 | 73% | 50% |
| ACT + map4d Transformer encoder (context token) | 100 | 76% | 54% |
| ACT + map4d MLP token | 100 | 67% | 44% |
| ACT + map4d raw concat | 100 | 73% | 51% |
| ACT + map4d raw concat + aux loss | 100 | 64% | 42% |
| ACT baseline | 990 | 85% | 71% |
| ACT + map4d GRU encoder (context token) | 990 | 80% | 69% |
| ACT + map4d Transformer encoder (context token) | 990 | 81% | 72% |
| ACT + map4d MLP token | 990 | 86% | 80% |
| ACT + map4d raw concat | 990 | 91% | 82% |
| ACT + map4d raw concat + aux loss | 990 | 84% | 65% |

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

### **Diffusion Policy**

| 方法 | Demos | best success_once | best success_at_end |
| --- | --- | --- | --- |
| DP baseline | 100 | 0% | 0% |
| DP baseline | 1000 | 0% | 0% |
| DP + map4d raw concat | 100 | 3% | 3% |
| DP + map4d raw concat | 1000 | 7% | 4% |

## **实验说明**

- **map4d GRU encoder (context token)**: map4d序列经过GRU encoder编码为128维feature，作为独立context token融入policy
- **map4d Transformer encoder (context token)**: 同上，将GRU替换为Transformer (pre_horizon=30, future_horizon=30)
- **map4d MLP token**: 30帧map4d flatten为1080维，经MLP(1080→256→128)压缩为1个context token融入transformer memory
- **map4d raw concat**: 跳过encoder，直接取当前帧map4d flatten后拼接到robot state
- **map4d raw concat + aux loss**: raw concat基础上，从policy内部feature预测未来物体状态作为辅助loss
