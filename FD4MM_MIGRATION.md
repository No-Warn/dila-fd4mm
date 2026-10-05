# FD4MM → DiLA 数据迁移说明

本文档记录将 **FD4MM**（Frequency Decoupling for Motion Magnification, CVPR 2024）
的帧数据接入 **DiLA**（Disentangled Latent Action World Models）训练流程的实现，
以及一个**必须在训练前知晓的重要发现**。

---

## 0. 结论速览（TL;DR）

| 项目 | 迁移文档的方案 | 实测结果 | 本实现 |
|---|---|---|---|
| 时序假设 | `i` 与 `i+1` 时间连续 | ❌ **不成立** | 使用样本内真实时序轴 `A → C → B` |
| 序列构造 | `[A₁,B₁,A₂,B₂,…]`（T=2K） | ❌ 拼接了无关场景 | 默认 `sequence_mode: acb` |
| `frameC` | 文档未提及 | ✅ 存在，且是 A、B 之间的真实中间帧 | 作为真实时序帧使用 |
| 归一化 | `[-1, 1]` | ❌ repo 用 `[0,1]`（RAE 内部再归一化） | 跟随 repo：`Resize+ToTensor` |
| 代码结构 | 独立 `dila_fd4mm/` 包 | — | 集成进 DiLA 现有 loader |

**迁移文档中「方案一」的前提经实测为假。** 按文档自己的验证清单（第一节第一步/第二步）
要求，此时必须回退；文档提到的「方案二」并未定义，因此本实现定义了正确的替代方案。

---

## 1. 关键发现：索引 `i` 与 `i+1` 之间没有时序连续性

FD4MM 的 `train` split 是 **Oh et al., ECCV 2018《Learning-based Video Motion
Magnification》的合成数据集**。`frameA / frameB / frameC / meta / train_mf.txt`
确实来自该数据集，而非真实视频帧。

### 1.1 证据一：`meta/` 表明每个索引都是独立合成的

```json
// data/train/meta/000000.json
{"background_file": ".../coco_train2017/000000225345.jpg",
 "third_frame_motion": 1.0,
 "sample_properties": [{"motion_amount": 0.386, "motion_dir": 5.899, "center": [190.4, 208.1], ...}, ...]}
```

- 每个索引自带一张随机 **COCO 背景图** 和随机前景物体的运动参数；
- 前 200 个索引的背景图 **200/200 全部不同**（不是同一段视频的连续帧）；
- `train_mf.txt` 的 100000 个放大因子 **全部互不相同**（不存在「同一视频的连续 run」）。

### 1.2 证据二：像素距离统计

在 300 个样本上实测（0–255 灰度尺度的像素 L2）：

| 距离 | 均值 | 含义 |
|---|---|---|
| `‖Aᵢ − Cᵢ‖` | 9.2 | 样本内真实运动 |
| `‖Cᵢ − Bᵢ‖` | 8.3 | 样本内真实运动 |
| `‖Aᵢ − Bᵢ‖` | 13.5 | 样本内总位移 |
| **`‖Bᵢ − A₍ᵢ₊₁₎‖`** | **94.6** | ← 文档方案一依赖的「跨索引衔接」 |
| `‖Aᵢ − A₍ᵢ₊₁₎‖` | 94.7 | 与随机配对同量级 |

- `‖Bᵢ − A₍ᵢ₊₁₎‖` ≈ 随机两帧的距离，是样本内运动量的 **约 7 倍**；
- 300/300 个样本中 `‖Bᵢ − A₍ᵢ₊₁₎‖` 都**不小于**样本内 `‖Aᵢ − Bᵢ‖`；
- 用 DiLA 实际训练变换（`Resize(256)+ToTensor`，值域 `[0,1]`）复测：比值 **7.0×**；
- 用本实现的文档方案一模式复测：相邻帧 L2 均值 **0.196**，而正确构造仅 **0.030**（**6.5 倍**差距，最大 0.562 vs 0.155）。

### 1.3 证据三：可视化

对样本 `i` 取 `[Aᵢ, Cᵢ, Bᵢ, A₍ᵢ₊₁₎]`：前三帧是**同一场景**（同一只狗/同一辆车）
的细微运动；第 4 帧 `A₍ᵢ₊₁₎` 是**完全不同的场景**（另一个人、另一条街）。

### 1.4 为什么这很重要

DiLA 的潜动作来自结构嵌入的时序差分。若把无关场景相接，
模型学到的是「潜动作 = 瞬移到随机场景」，
**content-structure 解耦会直接退化** —— 正是迁移文档开头警告的失败模式。

### 1.5 文档的验证门槛确实触发了

迁移文档第六节第一步写明：

> 如果差异波动剧烈（std > 2×mean），说明 A/B 帧可能不是时间连续的，
> 方案一的前提不成立，需要回退到方案二或重新审视数据语义。

实测比值 7.0×，**该门槛明确指出方案一不可用**。

---

## 2. 正确的迁移方案：使用样本内真实时序轴 `A → C → B`

FD4MM 数据唯一真实的时序轴在**样本内部**：

```
frameA/%06d.png   t = 0.0   （起始）
frameC/%06d.png   t = 0.5   （中间帧，真实渲染）
frameB/%06d.png   t = 1.0   （结束）
```

`frameC` 在 300/300 个样本中都落在 A 与 B 之间（`‖A−B‖` 恒为三者最大），
即真实的中间时刻。因此：

- **最忠实的配置是 `seq_len: 3`**，输出 `[A, C, B]`，全部为真实帧、真实时序；
- 当 `seq_len > 3` 时，本实现提供 `frame_fill` 策略扩展序列（见 §3）。

> 单个样本只有 2 次状态转移，但数据集有 **10 万条独立场景**，
> 合计 20 万次转移，样本量充足；DiLA 的长时能力本身由 phase-2 的
> **自回归 latent rollout** 微调承担（见官方 README），不要求单条序列很长。

---

## 3. `seq_len > 3` 时的扩展策略

| `frame_fill` | 行为 | 实测相邻帧 L2 | 评价 |
|---|---|---|---|
| `reflect`（**默认**） | 回文扩展 `A,C,B,C,A,C,B,C…` | 0.030（max 0.155） | ✅ 所有帧都是**真实观测**；折返处速度反向 |
| `interpolate` | 像素线性插值到 `seq_len` | 0.004 | ⚠️ 时间单调，但**运动物体会重影** |
| `repeat_last` | 末帧填充 | 0.004 | ❌ 静止尾巴会教出「空动作」，文档亦明确不推荐 |

关于 `interpolate` 的重要实测：`‖C − (A+B)/2‖ / ‖A−B‖ ≈ 0.50`。
若物体是像素线性运动，`C` 应等于 `(A+B)/2`；实测偏差达到帧间位移的一半，
说明合成器是**重新渲染**而非像素混合，像素插值只会产生半透明重影，
**不是**忠实的中间帧。因此默认选择 `reflect` 而非 `interpolate`。

---

## 4. 实现内容（集成进现有 DiLA 代码，而非另起一个包）

迁移文档提议新建独立 `dila_fd4mm/` 包并自行实现 `train.py`/模型配置
（`dinov2_vitb14`、`latent_action_dim` 等）。但本仓库已有完整的
DiLA 训练栈（Hydra + Accelerate + `MixedDataset` 工厂注册表 +
RAE/DINOv2 编码器）。另起一套会与真实模型定义冲突，故本实现**接入现有管线**。

### 新增

| 文件 | 作用 |
|---|---|
| `utils/fd4mm_dataset.py` | `FD4MMDataset`：`A→C→B` 真实时序、`frame_fill` 策略、连续性度量与可视化、CLI |
| `configs/data/fd4mm.yaml` | 仅用 FD4MM 的数据配置（`data=fd4mm`） |
| `verify_fd4mm_dila.py` | 迁移文档第六节的验证清单，一键运行 |
| `FD4MM_MIGRATION.md` | 本文档 |

### 修改

| 文件 | 改动 |
|---|---|
| `utils/mixed_dataset.py` | 新增 `FD4MMFactory` 并注册为 `fd4mm` |
| `utils/load_mixed_datasets.py` | `create_dataset_configs_from_config` 支持 `fd4mm` |
| `configs/data/multidataset.yaml` | 新增 `fd4mm` 配置块（默认未启用，可混训） |
| `train.py` | `_setup_multidataset_loaders` 传入并检查 `fd4mm` 配置 |
| `.gitignore` | 忽略根目录 `/data/`（数据软链）与验证输出图 |

数据集遵循现有约定：`__getitem__` 返回 `{'images': (T, C, H, W)}`，
由 `FD4MMFactory.normalize_batch` 取出 `images`，与 `LoopNavDataset` 一致。

### 数据接入

```bash
# 仓库内软链到真实数据（已建立）
data/train -> /home/apricity/data/train
```

---

## 5. 使用方法

```bash
# 训练前验证（迁移文档第六节清单，务必先跑）
python verify_fd4mm_dila.py --root ./data/train --seq-len 3

# 仅用 FD4MM 训练（最忠实：T=3 全部为真实帧）
accelerate launch main.py data=fd4mm seq_len=3 \
  batch_size=8 num_train_steps=30000 \
  work_dir=./checkpoints exp_name=DiLA-fd4mm

# 需要 T=16 时（reflect 扩展，所有帧仍为真实观测）
accelerate launch main.py data=fd4mm seq_len=16 \
  data.multidataset.fd4mm.frame_fill=reflect \
  batch_size=8 num_train_steps=30000 \
  work_dir=./checkpoints exp_name=DiLA-fd4mm-t16

# 与官方数据集混训：把 fd4mm 加入 dataset_names 并追加 ratio
#   dataset_names: [ssv2, openx, nwm, loopnav, fd4mm]
#   ratios:        [1,     1,     1,   1,      1]
```

也可直接作为库使用：

```python
import torchvision.transforms as T
from utils.fd4mm_dataset import FD4MMDataset

ds = FD4MMDataset(
    root="./data/train",
    seq_len=3,                 # 真实时序轴长度
    sequence_mode="acb",       # A -> C -> B
    frame_fill="reflect",      # seq_len > 3 时如何扩展
    split="train",
    transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]),
)
seq = ds[0]["images"]          # (3, 3, 256, 256), 值域 [0, 1]
```

---

## 6. `sequence_mode` 选项

| 模式 | 输出 | 状态 |
|---|---|---|
| `acb`（默认） | `[A, C, B]`（`frameC` 缺失时退化为 `[A, B]`） | ✅ 推荐 |
| `ab` | `[A, B]` | ✅ 正确但信息更少（仅 1 次转移） |
| `concat_pairs` | `[Aₛ, Bₛ, Aₛ₊₁, Bₛ₊₁, …]` | ❌ **本数据集上无效**，仅用于复现/诊断，会打印警告 |

`concat_pairs` 即迁移文档的「方案一」。保留它是为了让文档作者能复现该失败：
其相邻帧 L2 为 0.196，是正确构造（0.030）的 **6.5 倍**。

---

## 7. 对迁移文档若干说法的更正

1. **时序假设**：`i` 与 `i+1` 不连续（§1）。方案一不可用。
2. **字段表遗漏 `frameC`**：文档表格只列了 `amplified/frameA/frameB`，但数据中存在
   `frameC/`，且它正是 A、B 之间的真实中间帧 —— 这是本数据**唯一可用的真实时序轴**。
3. **`amplified` 与 `mag`**：确实不应使用（无监督设定），这一点文档正确。
4. **归一化 `[-1,1]`**：本仓库训练循环使用 `Resize+ToTensor`（`[0,1]`），
   RAE/DINOv2 编码器内部再做归一化。若在此处再转 `[-1,1]` 会造成**双重归一化**。
   本实现因此不做 `[-1,1]` 变换。
5. **噪声增强**：已实现为可选（`add_noise`/`noise_std`），但**默认关闭**，
   且作用在 `[0,1]` 值域上；训练配置需显式开启。
6. **「边界丢弃策略」**：在 `acb`/`ab` 模式下不存在边界问题（每个样本自成一序）。
   仅在 `concat_pairs` 模式下按文档实现丢弃策略。
7. **独立工程结构**：未采用。接入现有 `MixedDataset` 工厂注册表，
   避免与真实 DiLA 模型/配置冲突。
8. **图像尺寸**：文档称「不要降低 `image_size` 以免影响 patch 网格」。
   本仓库全局固定 `Resize((256,256))`，且 patch 网格随输入自适应，
   该顾虑在此实现中不适用。

---

## 8. 验证结果

`python verify_fd4mm_dila.py --root ./data/train --seq-len 3` 全部通过：

```
STEP 1/2  temporal continuity of the source indices
           within_AB =    0.0530
           within_AC =    0.0358
           within_CB =    0.0325
    boundary_BA_next =    0.3721
  boundary(B_i, A_(i+1)) / within(A_i, B_i) = 7.0x
  VERDICT: indices are NOT temporally continuous. Plan 1 is INVALID.

STEP 3  emitted tensor shape / value range
  shape=(3, 3, 256, 256) dtype=torch.float32 min=0.0000 max=1.0000

STEP 2  visual check -> fd4mm_sequence_check.png   （同一场景，无跳变）

STEP 4  real DiLA MixedDataset / MixedBatchSampler batch
  observations: (4, 3, 3, 256, 256) torch.float32 range=[0.0000, 1.0000]
  adjacent-frame L2: mean=0.0216 max=0.0400

ALL CHECKS PASSED
```

已验证：`seq_len` 3/8/16、`sequence_mode` 全部三种、
`frame_fill` 全部三种、`num_workers=0/2`、
train/val 切分（90000/10000）、值域 `[0,1]`、与 `MixedBatchSampler` 混合采样打通。

### 尚未验证

- **端到端训练 loss 下降**（文档第六节第四步）：本机 **无 CUDA**，
  且需要下载 DINOv2 权重（`RAE.encoder_cache_dir: ./pretrained`），
  未在 CPU 上跑完整训练。建议在 GPU 机器上先跑
  `num_train_steps=100` 的小规模过拟合测试确认 loss 下降。
- `seq_len=3` 是否足够支撑 DiLA 的解耦质量，属于训练层面的超参问题，
  需实验确认；若发现潜动作过弱，可先用 `seq_len=16 + reflect` 对比。

---

## 9. 推荐配置

| 场景 | 配置 |
|---|---|
| 最忠实（推荐先跑） | `seq_len: 3`, `sequence_mode: acb` |
| 需要长序列 / 混训 | `seq_len: 16`, `sequence_mode: acb`, `frame_fill: reflect` |
| 复现文档方案一（诊断） | `sequence_mode: concat_pairs`, `pairs_per_sample: 4` |
| 不建议 | `frame_fill: repeat_last`（静止尾巴），`frame_fill: interpolate`（重影） |

---

## 10. 训练环境搭建（2026-10 实测）

### 10.1 预训练权重（已完成，位于 `pretrained/`）

| 文件 | 大小 | 来源 |
|---|---|---|
| `models--facebook--dinov2-with-registers-base/` | 330 MiB | `facebook/dinov2-with-registers-base` |
| `stats/dinov2/wReg_base/imagenet1k/stat.pt` | 0.8 MiB | `nyu-visionx/RAE-collections` |
| `decoders/dinov2/wReg_base/ViTXL/dinov2_decoder.pt` | 1.6 GiB | `nyu-visionx/RAE-collections` |
| `rae/dinov2_stats.pt`、`rae/dinov2_decoder.pt` | 软链 | → 上面两个（对齐 `train_model.yaml` 的路径） |

下载时有两个坑：

```bash
# 1) 环境里的 ALL_PROXY=socks://... 非法，且未装 socksio；NO_PROXY 里的 [::1] 会让 httpx 解析失败
unset ALL_PROXY all_proxy no_proxy NO_PROXY
export HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897

# 2) HF_HOME 与 --cache-dir 混用会让离线缓存找不到，必须统一用 HF_HUB_CACHE
export HF_HUB_CACHE="$PWD/pretrained"       # RAE 离线加载依赖它
```

> 若 `torch.load` 报 Xet 缓存只读（沙箱/受限 HOME 场景），加 `HF_HUB_DISABLE_XET=1`
> 并把 `TMPDIR`、`HF_XET_CACHE` 指到可写目录。

### 10.2 已修复的兼容性问题（否则训练无法启动）

1. **`configs/decoder/ViTXL/config.json`**：`"patch_size": "SHOULD BE RELOADED"`
   是占位字符串。`transformers` 5.x 用 pydantic 严格校验字段类型，构造
   `AutoConfig` 时直接抛 `StrictDataclassFieldValidationError`。
   `RAE.__init__` 本来就会覆盖 `patch_size`/`image_size`，故改为合法整数 `16`。
2. **`train.py::_init_tracker`**：`settings={"start_method": "thread"}` 在
   wandb 0.30 已被移除，严格校验下即使 `wandb.enabled=false` 也会抛
   `extra_forbidden`。已删除该键（`WANDB_START_METHOD` 环境变量仍保留）。
3. **`train.py::_init_tracker`**：禁用 wandb 时 tracker 是 `NoopRun`，
   没有 `get_url()`，原代码必然 `AttributeError`。已加 `hasattr` 保护。

### 10.3 仍需你处理：PyTorch 版本

本机 GPU 为 **RTX 5060 Max-Q（GB206M，Blackwell，sm_120）**，驱动 595.91.07 已加载。
但当前 torch 2.5.1+cu121 的编译架构为 `sm_50…sm_90`，**不含 sm_120**，
会报 `no kernel image is available`。需要：

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 10.4 运行

```bash
cd /home/apricity/disentangled-latent-action-world-models
conda activate dila
export HF_HUB_CACHE="$PWD/pretrained"

python verify_fd4mm_dila.py --root ./data/train --seq-len 3

accelerate launch main.py data=fd4mm seq_len=3 batch_size=2 \
  num_train_steps=200 val_every_n_steps=100 wandb.enabled=false \
  work_dir=./checkpoints exp_name=fd4mm-smoke
```

### 10.5 已验证：CPU 上完整训练打通

用 `accelerate launch --cpu ... num_train_steps=2 seq_len=3 batch_size=1`
跑通了 **数据 → RAE 编码 → 模型前向 → 反向 → 优化器 → 验证 → 存 ckpt → 存重建图**
全链路，loss 从 4.50 降到 3.96，val 从 4.12 降到 3.74：

```
Step 0/2: {'sensory_gen': 2.23, 'grid': 1.08, 'action': 0.20, 'l2': 1.74,
           'symmetry': 2.31, 'total_loss_accumulated_step': 4.50, 'grad_norm': 14.88}
Step 1/2: {'sensory_gen': 1.96, 'grid': 1.22, 'action': 0.08, 'l2': 3.97,
           'symmetry': 3.57, 'total_loss_accumulated_step': 3.96, 'grad_norm': 6.18}
Validation step 1: {'val/total_loss_avg': 3.74}
```

> 注意：这只是「管线可运行 + loss 在动」的证据，**不是收敛性结论**（2 步）。
> 真实训练仍需在 GPU 上完成。
