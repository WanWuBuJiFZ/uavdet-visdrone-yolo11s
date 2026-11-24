# 云端完整运行流程（获得用户授权后执行）

> 本文**不授权**现在租用算力、上传数据或运行正式训练。它描述的是：代码通过验收后，
> 用户准备云资源 → 云端 smoke → 正式 300 轮 → 取回结果 的完整流程与检查点。

## 1. 环境与版本

本机（macOS arm64, CPU）已验证：Python 3.11.3 + torch 2.0.1 + torchvision 0.15.2 +
ultralytics 8.4.215 + numpy 1.26.4 + opencv 4.10.0.84。**没有任何 GPU/CUDA 组合被本仓库验证过**，
下面是选择流程而不是"照抄就能用"的固定组合：

```bash
# 1) 先确认驱动与 CUDA
nvidia-smi                      # 记录驱动版本与支持的 CUDA 上限

# 2) 按 https://pytorch.org/get-started/locally/ 选择匹配的 torch/torchvision
#    例如 CUDA 12.1（示例，不代表已验证）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 其余依赖 + 可编辑安装
pip install -r requirements-core.txt
pip install -e . --no-deps

# 4) 真实可用性检查（不是只看配置里的 device 字符串）
python - <<'PY'
import torch, ultralytics, cv2, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpus", [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])
print("ultralytics", ultralytics.__version__, "numpy", numpy.__version__, "cv2", cv2.__version__)
PY
```

要求：`torch.cuda.is_available() == True`，GPU 名称与显存符合预期。ultralytics 若提示版本/
依赖问题，先解决再继续；不要用 `--device cpu` 掩盖云端 GPU 不可用。

磁盘：数据副本约 1.6 GB；工作副本（`workspace/data_views/`）再占约 1.9 GB（复制模式下 ×1.15
余量检查）；每个正式 run 的权重 + 图表通常 < 1 GB。建议预留 ≥ 20 GB。

## 2. 代码与数据上传

只上传本目录需要的内容，**不要**上传旧项目目录、旧 `best.pt`、旧用户数据或本地缓存：

```bash
rsync -av --exclude '.venv' --exclude 'outputs' --exclude 'workspace' \
      --exclude 'app_data' --exclude '__pycache__' \
      ./uav_detection_rebuild/ user@gpu-host:/workspace/uav_detection_rebuild/

# 数据单独传（或直接挂载云盘），保持目录结构：
#   /workspace/uav_detection_rebuild/dataset_visdrone/
#     data.yaml  VisDrone2019-DET-train/{images,labels}  VisDrone2019-DET-val/{images,labels}
```

> 提示：`check`/`train` 现在会核对"审计是不是针对当前数据根做的"。挂载路径不对时不会再复用旧审计，
> 而是记录 `audit_reuse_rejected` 并重新审计（预计 1–3 分钟）；这是有意为之，用来挡住错误的云挂载。

设置数据路径（任选其一，源码里没有硬编码路径）：

```bash
export UAV_DATA_ROOT="/workspace/uav_detection_rebuild/dataset_visdrone"
# 或者每次命令加 --data-root "$UAV_DATA_ROOT"
# 或者改配置 data.root
```

预训练权重（显式下载或手动上传到 `weights/yolo11s.pt`）：

```bash
python -m uavdet --config configs/project.yaml fetch-weights --name yolo11s.pt
sha256sum weights/yolo11s.pt      # 记下来，训练日志会再记录一次
```

## 3. 先跑 smoke（必须）

```bash
python -m uavdet --config configs/smoke.yaml pipeline \
  --run-id smoke_gpu_001 --data-root "$UAV_DATA_ROOT" --device 0
```

退出条件（验收人依据真实结果确认，不以"没报错"代替检查）：

- `outputs/smoke_gpu_001/environment.json` 记录到真实 GPU 名称/显存、torch/CUDA 版本；
- 审计从**完整数据**通过（train 6471 / val 548），smoke 视图只复制 20/10 张；
- 训练 2 轮完成，`train/results.csv` 有 2 行、loss 有限；
- `evaluation/metrics.json`、`per_class.csv`、`curves.npz`、`confusion_matrix.npz`、
  `predictions.jsonl`、`ground_truth.jsonl` 存在且能解析；
- `figures/<tag>/figure_manifest.json`：**适用的图必须全部产出、0 个渲染错误**；smoke 允许
  `status=partial`，但 `missing` 必须有明确原因（例如 2 轮模型在 conf=0.25 下没有合格的成功案例，
  F11 如实缺项）。`status=ok` 不是 smoke 的通过条件，formal 才是；
- 样本太小导致的 N/A / 低 mAP 是正常的，**smoke 指标不能当作模型性能结论**；
- `evaluation/metrics.json` 里 `complete=true`（评价覆盖了 smoke 视图的全部图片），并记下
  `max_det_hits` / `zero_area_detections` / `weights_path` 三个字段；
- 同时做一次 batch=16 的显存试跑（1–2 迭代即可）；batch=2 能跑不代表 batch=16 能跑。

```bash
# 显存/批大小检查建议：复制 smoke 配置并临时覆盖，跑完只看是否 OOM，不进正式报告
python -m uavdet --config configs/smoke.yaml check --run-id smoke_batch16 \
  --data-root "$UAV_DATA_ROOT" --device 0
python -m uavdet --config configs/smoke.yaml train --run-id smoke_batch16 \
  --data-root "$UAV_DATA_ROOT" --device 0 --batch 16 --epochs 1
```

## 4. 正式训练与完整验证

仅在 smoke 与显存检查通过、用户已授权后：

```bash
# 建议放在可续会话里（SSH 断开不杀进程）
tmux new -s uavdet            # 或 nohup ... > formal_001.out 2>&1 &
python -m uavdet --config configs/project.yaml pipeline \
  --run-id formal_001 --data-root "$UAV_DATA_ROOT" --device 0
```

分阶段执行（便于查错，两种方式只选一种）：

```bash
python -m uavdet --config configs/project.yaml check --run-id formal_001 --data-root "$UAV_DATA_ROOT" --device 0
python -m uavdet --config configs/project.yaml train --run-id formal_001 --data-root "$UAV_DATA_ROOT" --device 0
python -m uavdet --config configs/project.yaml val --run-id formal_001 \
  --weights outputs/formal_001/train/weights/best.pt --device 0
python -m uavdet --config configs/project.yaml benchmark --run-id formal_001 \
  --weights outputs/formal_001/train/weights/best.pt --device 0
python -m uavdet --config configs/project.yaml plot --run-id formal_001 --figure-tag v1
```

后续子命令没再传 `--data-root` 时读取该 run 已记录的数据视图；若根路径迁移，需显式更新
（`--data-root`），并在 `manifest.json` 留 `config_rebase` 事件。

## 5. 中断恢复

```bash
# 训练中断（进程被杀/OOM/机器重启）后
python -m uavdet --config configs/project.yaml train --run-id formal_001 \
  --resume outputs/formal_001/train/weights/last.pt --data-root "$UAV_DATA_ROOT" --device 0

# 已有阶段全部完成、只想补齐后面的阶段
python -m uavdet --config configs/project.yaml pipeline --run-id formal_001 --continue
```

规则：恢复只接续**同一个 run** 的 checkpoint；数据内容与训练协议指纹必须一致（改 batch/lr/数据
必须新 run）；恢复后 `results.csv` 的 epoch 必须单调不重复，否则显式失败；训练未达目标轮数时
stage 状态不会是 `complete`。`--continue` 只跳过"已完成且产物齐全并校验通过"的阶段，
不会自动改参数，也不能代替训练恢复的显式 checkpoint 选择。

## 6. 观察与日志

```bash
ls outputs/formal_001/
cat outputs/formal_001/manifest.json          # 各 stage 状态、事件、错误
cat outputs/formal_001/logs/events.jsonl      # 逐事件时间线
tail -f outputs/formal_001/train/train.log
python -m uavdet --config configs/project.yaml info | head -60   # 解析配置 + 已有 run 概览
```

失败或中断时：`manifest.json` 中对应 stage 为 `failed`/`interrupted`，错误在
`stages.<name>.error`；日志保留，不会出现"已完成"。修好后从失败阶段继续，已完成的阶段不会被
静默覆盖。

## 7. 取回结果

```bash
# 注意：max_det=3000 时 548 图的 predictions.jsonl 可能有数百 MB；只带图表/指标时可用下面的 min 包
tar -czf formal_001_results.tar.gz -C outputs formal_001
# 必带：manifest.json、config_resolved.yaml、environment.json、
#       train/results.csv、train/args.yaml、train/weights/{best,last}.pt、
#       evaluation/*、benchmark/*、figures/*、logs/*
```

本地复检（CPU、无 GPU、无 Qt 也可）：把 tar 解开到某个 `outputs/formal_001`，然后

```bash
python -m uavdet --config configs/project.yaml info
python -m uavdet --config configs/project.yaml plot --run-id formal_001 --figure-tag v2
```

`plot` 只读已保存结果（不加载模型、不需要 GPU），可用于核对图表与源数据。

## 8. 正式完成的判定

| 检查 | 通过标准 |
|---|---|
| 初始化来源 | `train/args.yaml`/manifest 记录 `weights/yolo11s.pt` 的 SHA256；无旧 best.pt 起点 |
| 数据规模 | train 6471、val 548；`evaluation` 的图片 ID 摘要与清单一致 |
| 训练轮数 | `results.csv` 到 300 轮；未完成则状态不是 complete |
| 权重 | `best.pt`/`last.pt` 可加载，10 类顺序正确；重新验证的确实是实际 best.pt |
| 指标 | 主表数字 = `evaluation/metrics.json`；逐类 = `per_class.csv`；`complete=true` |
| 评价上限 | `max_det_hits` 为 0，或已在报告中说明为何该上限可以接受（本机 smoke 是 10/10 图触顶） |
| 零面积框 | `zero_area_detections` 已复核（本机 smoke 为 634，来自裁剪，不参与匹配） |
| 图表 | `figure_manifest.json` 的 `status=ok`，逐图有输入 SHA256 与 `source_data/` |
| 一致性 | 新 run 的 CSV/权重/日志/图表同源；旧数据与旧权重 SHA256 未变 |

**未通过以上任一项，就按真实状态交付，不得宣布"正式实验完成"。**
