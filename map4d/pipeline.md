# 4D Map Pipeline

本文档整理 DP baseline、DP + 4D map encoder，以及 4D map encoder 内部的训练和推理流程。

## 记号

- `B`: batch size。
- `ho`: observation horizon，也就是 4D map encoder 的 `pre_horizon = h1`。
- `ha`: action horizon。
- `h2`: 4D map auxiliary prediction 的 `future_horizon`。
- `dim_action`: robot action space 维度，当前动作空间为 `{x, y, z, rx, ry, rz, wg}`。
- `dim_fi`: image feature 维度。
- `dim_fm`: 4D map feature 维度。
- `dim_ff`: fusion 后的 policy feature 维度。

输入张量约定：

- RGB image: `[B, ho, 3, W, H]`。
- Depth: `[B, ho, 1, W, H]`。
- Action: `[B, ha, dim_action]`。
- 4D map sequence: `[B, ho, ...]`，其中 `...` 表示每帧场景内所有 objects 的结构参数和 6D pose。

## DP Baseline

训练时，真实 action 先加噪得到 noised action：

```text
action [B, ha, dim_action]
  -> add noise
noised action [B, ha, dim_action]
```

policy denoiser 使用 `noised action + timestep + feature` 预测 noise 或 denoised action：

```text
RGB image [B, ho, 3, W, H]
  -> image encoder
image feature [B, dim_fi]

noised action + timestep + image feature
  -> denoise network
action prediction [B, ha, dim_action]
```

推理时，从 sampled noise 开始反复 denoise：

```text
sampled noise [B, ha, dim_action]
  -> denoise network(noised action, timestep, image feature)
action [B, ha, dim_action]
```

## DP + 4D Map

DP + 4D map 在 baseline 的 image feature 之外，额外从 RGB-D observation 构建并编码 4D map：

```text
RGB image [B, ho, 3, W, H]
  -> image encoder
image feature [B, dim_fi]

RGB image [B, ho, 3, W, H] + depth [B, ho, 1, W, H]
  -> 4D map construction
4D map sequence [B, ho, ...]
  -> 4D map encoder
4D map feature [B, dim_fm]

image feature [B, dim_fi] + 4D map feature [B, dim_fm]
  -> fusion
final feature [B, dim_ff]

noised action + timestep + final feature
  -> denoise network
action prediction [B, ha, dim_action]
```

训练和推理的 denoise 主链路一致；区别在于训练阶段有 action 加噪和辅助监督，推理阶段从 sampled noise 开始生成 action。

## 4D Map Encoder

在时刻 `T`，4D map encoder 使用最近 `h1 = ho` 帧 observation：

```text
RGB:   T - h1 + 1, ..., T - 1, T
Depth: T - h1 + 1, ..., T - 1, T
```

### Step 1: 4D Map Construction

对每一帧构建一个实例化后的 4D map，共得到 `h1` 个 map：

```text
RGB [B, ho, 3, W, H] + depth [B, ho, 1, W, H]
  -> construction
4D map [B, ho, ...]
```

在 ManiSkill/StackCube 当前阶段，construction 可以直接从 simulator GT 读取实例化参数：

- 物体 3D position: `{x, y, z}`。
- 物体 rotation: `{rx, ry, rz}` 或等价旋转表示。
- 物体结构参数: `{w, h, d}` 等 object-specific parameters。

这些参数写入每帧场景内所有 `Object`，形成 `Map4d_StackCube` 序列。

### Step 2: Map Encoding

map encoder 对 `h1` 帧 4D map 做时序编码：

```text
4D map [B, ho, ...]
  -> map encoder
map feature [B, dim_fm]
```

如果具体实现保留 per-frame feature，也可以先得到 `[B, ho, dim_fm]`，再通过 last-frame pooling、temporal pooling 或 flatten/fusion 得到供 policy 使用的 `[B, dim_fm]`。

### Step 3: Auxiliary Future Prediction

辅助 loss 只在训练时使用。map encoder 以 observation horizon 内的 map feature 为输入，预测未来 `h2` 帧 4D map 参数：

```text
map feature [B, dim_fm]
  -> future prediction head
pred future 4D map parameters [B, h2, ...]
```

GT future map 来自 simulator：

```text
future GT 4D map parameters:
T + 1, ..., T + h2
```

辅助监督包括但不限于：

- 6D pose/position 预测 loss。
- rotation 预测 loss。
- temporal consistency / denoise-style prediction loss。
- physics or structure consistency loss。

辅助 loss 不直接改变 diffusion action space；它用于约束 4D map feature 更好地表达场景几何、结构和未来动态。

## 当前实现目标

- DP baseline 保持原有 action diffusion 逻辑。
- DP + 4D map 的 policy input feature 应为 `fusion(image feature, map feature)`。
- `pre_horizon` 与 DP `obs_horizon` 对齐。
- `future_horizon` 独立设置，只服务于 4D map auxiliary loss。
- StackCube 阶段优先使用 ManiSkill GT 构建 4D map，后续再替换或扩展为 RGB-D perception-based construction。
