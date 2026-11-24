# 实现与验证报告（IMPLEMENTATION_REPORT）

> 交付日期：2026-11-17。
> 仓库说明：本报告中的 `outputs/...` 是本机运行产物，**未随仓库提交**；仓库内可直接查看的结果图在
> `正式结果_看图/` 与 `docs/证据截图/`，结果数字见 §11。
> 本报告只写**实际发生**的检查与结果；没有运行条件的内容明确标为「未测」，不以「预计可用」替代证据。
> 正式 300 轮训练已在云端完成（2026-11-18，RTX 4090 D）；本报告 §11 只引用这次运行的结果。

## 1. 状态总览（按计划 03 §8 模板）

| 项目 | 状态 | 证据路径 / 命令 | 原因 / 后续动作 |
|---|---|---|---|
| 数据审计与配置 | **完成（真实全量数据已跑）** | `outputs/preflight_local_002/`（`audit/data_manifest.json`、`dataset_stats.json`、`issues.csv`、`label_repairs.csv`） | 见 §4.1。train 6471 / val 548，381,963 个实例；1 条 0 面积标注行按记录策略丢弃 |
| 训练与恢复 | **完成（正式 300 轮已在云端完成；本机另有 2 轮 smoke）** | `outputs/smoke_local_001/train/`（`results.csv` 2 轮、`weights/best.pt`、`last.pt`、`args.yaml`、`train.log`） | 见 §4.2。恢复训练的单元测试见 `tests/test_training_evaluation.py`；真实 resume 未跑（会再花 1 分钟，但不改变结论，见 §7 未测项） |
| 评价与原始数组导出 | **完成（云端正式 548 图；本机 smoke 10 图）** | 主结果 `outputs/formal_001/evaluation/`（548 图，见 §11）；本机 smoke 证据 `outputs/smoke_local_001/evaluation/` | 同一 validator pass 导出预测/GT；主表数字只取 §11 |
| SCI 图表 | **完成（正式 run 12/12；smoke 11/12）** | 主图集 `outputs/formal_001/figures/v1/`（12 张×3 格式 + `source_data/` + 清单，见 §11）；本机 smoke 图集 `outputs/smoke_local_001/figures/final_v9/` | 正式 run 的 F11 有 2 个合格案例；smoke 的 F11 如实缺项（不造图） |
| 检测引擎与 GUI | **完成（真实权重 + 真实图片已跑）** | 无 GUI：predict 输出目录；GUI 截图 `docs/证据截图/` | 见 §4.5/4.6。摄像头实机未测 |
| 核心单元测试 | **完成（230 tests, 0 fail, 0 skip；1 个需真实权重的集成测试按环境变量跳过）** | `.venv/bin/python -m unittest discover -s tests -v` | 见 §5 |
| 云端 GPU smoke | **未单独取回产物**（云端正式 run 已证明环境可用） | `outputs/formal_001/environment.json`（RTX 4090 D / torch 2.5.1+cu121） | 若要严格按 03 §5，可另跑一次 smoke 并回传 |
| 正式 300 轮 | **已执行（云端 HPC，300 轮），产物已取回并核对** | `outputs/formal_001/`（manifest 五阶段 complete；548 图评价；12/12 图表） | 见 §11；独立复核仍待进行 |

## 2. 环境与版本（实际测试）

| 项 | 值 |
|---|---|
| 平台 | macOS 27.0.1 (arm64, Apple M4)，CPU 运行 |
| Python | 3.11.3（venv，`--system-site-packages`） |
| torch / torchvision | 2.0.1 / 0.15.2（CPU；MPS 可用但未用于训练） |
| ultralytics | **8.4.215**（所有 adapter 字段按此版本源码核对） |
| numpy / pandas / matplotlib | 1.26.4 / 1.5.3 / 3.7.1 |
| opencv-python | 4.10.0.84 |
| Pillow / PyYAML / scipy | **13.0.0** / 6.0.3 / 1.10.1 |
| PyQt6 | 6.12.1（Qt 6.12.0，离屏 `QT_QPA_PLATFORM=offscreen`） |
| 版本锁定 | `requirements-verified.txt`（精确版本；venv 是 `--system-site-packages` 建的，README §1 已写明） |

约束：numpy 必须 <2.0（scipy 1.10 / torch 2.0.1 均为 numpy-1.x 构建）。**没有任何 GPU/CUDA 组合经过验证**。

## 3. 实际完成范围

- **模块A 配置/数据/实验记录**：`config.py`（严格 schema、`extends`、CLI 覆盖、路径相对配置文件解析、`run_id` 校验、协议指纹）、`artifacts.py`（原子 JSON、阶段状态机、环境记录、run 摘要）、`data.py`（全量审计、内容哈希、重复检测、只读原数据、可写数据视图、**受记录的 0 面积标注修复**）。
- **模块B 训练**：`training.py`（权重来源检查+SHA256、全部参数显式传递、真实 `save_dir` 回流、恢复约束、epoch 单调性、未达轮数不得 complete）。
- **模块C 验证/分析/测速**：`evaluation.py` + `ultralytics_adapter.py`（同一 validator pass 记录预测/GT、曲线/矩阵/N-A 导出、版本适配、固定阈值案例匹配）、`benchmark.py`（计时窗口只含预处理→模型→后处理，GPU/MPS 同步）。
- **模块D 图表**：`plotting/{style,dataset,training,evaluation,cases}.py` + 内置 `assets/sci-report.mplstyle`，F01–F12、`source_data/`、`figure_manifest.json`、`figure_index.md`；缺项诚实报告。
- **模块E 检测引擎**：`inference.py`（BGR 约定、显式阈值、10 类校验、CPU 回退告知、标注、批量保存结果）。
- **模块F 桌面应用**：`app/`（四页、深色 QSS、工作线程持有模型、可注入工厂、线程安全停止、资源释放、结果页直接读 manifest/metrics/figures）。
- 入口：`python -m uavdet …`（check/train/val/benchmark/plot/predict/pipeline/fetch-weights/info）与 `python -m uavdet.app`。

## 4. 真实运行证据

### 4.1 全量数据审计（formal scope）

```bash
python -m uavdet --config configs/project.yaml check --run-id preflight_local_002
```

结果（`outputs/preflight_local_002/audit/`）：

- train 6471 图 / 343,204 实例；val 548 图 / 38,759 实例；合计 7,019 图 / 381,963 实例。
- 每图目标数：train 中位 42、P95 132、**最大 902**；val 中位 65、P95 158、**最大 317**。
- 相对面积 `w_norm*h_norm`：train P50=4.62e-4、P95=6.32e-3；val P50=5.52e-4、P95=6.14e-3。
- 类别分布（train）：pedestrian 79,337 / people 27,059 / bicycle 10,480 / car 144,866 / van 24,956 / truck 12,875 / tricycle 4,812 / awning-tricycle 3,246 / bus 5,926 / motor 29,647。
- 问题清单：**1 条 0 面积标注**（`VisDrone2019-DET-train/labels/9999985_00000_d_0000020.txt` 第 11 行 `3 0.437857 0.150476 0.002857 0.000000`，高度 0 像素）、4 条重复标注、3 组内容重复图片（均在 split 内，**无跨 split 重复**）。
- 数据内容摘要 `manifest_digest=3a26e40b…be9d01`；原数据目录 SHA256 全程未变（单元测试 A03 断言）。
- `dataset_visdrone/data.yaml` 是任务资料包自带的（mtime 18:09，早于本次实现开始时间 18:18），实现过程没有写入该目录；`tests/test_data.py` 的 A03 断言可复现。
- 另保留 `outputs/preflight_local_001`：同一份数据在 `label_repair=strict` 下审计**按设计失败**
  （`audit` stage = failed，错误指向同一条 0 面积标注），证明"非法标签阻止正式训练"这条规则真的生效，
  而不是只写在文档里。
- 该 1 条坏标注由 `data.label_repair=drop_degenerate_boxes` 记录后丢弃：写入 `audit/label_repairs.csv`，只在 `workspace/data_views/<digest>/labels/` 的副本生效。源文件 `9999985_00000_d_0000020.txt` 共有 **29 个非空行**，第 11 行就是那条 0 面积标注；派生视图为 28 行，源文件 SHA256 与审计记录一致（未改动）。

### 4.2 真实小样本训练（smoke，CPU）

```bash
python -m uavdet --config configs/smoke.yaml train --run-id smoke_local_001 --device cpu
```

- 数据视图：`workspace/data_views/21832c9621ee6dbf_smoke/`（20 train / 10 val，复制模式；源数据只读）。
- 起点：`weights/yolo11s.pt`（官方 COCO 权重，19,313,732 B，SHA256 `85a76fe8…d502d5`；`fetch-weights` 显式下载）。
- 实际训练：2 epochs，batch=2，imgsz=640，workers=0，close_mosaic=0；每轮 ~15 s，CPU（Apple M4）。
- 模型规模（可复算）：`ultralytics.utils.torch_utils.model_info` 对 `best.pt.model` 输出 **181 层 / 9,431,662 参数 / 21.7 GFLOPs**；ultralytics 在验证时打印的 fused 摘要为 **100 层 / 9,416,670 参数 / 21.4 GFLOPs**（BN 融合后少 14,992 个参数）。两个数都真实，区别是融合前后，报告不再混用。
- 产物：`train/args.yaml`、`train/results.csv`（2 行）、`train/weights/{best,last}.pt`（各 ~19.2 MB）、`train/train.log`；best.pt SHA256 `60e74df6…fd3b6c`。
- 该 smoke 训练自身验证 mAP50=0.127（2 轮、20 张，**不能当作模型性能结论**）。

### 4.3 真实验证与原始数组导出（同一 validator pass）

```bash
python -m uavdet --config configs/smoke.yaml val --run-id smoke_local_001 \
  --weights outputs/smoke_local_001/train/weights/best.pt --device cpu
```

- 评价对象：同一 run 的实际 `best.pt`（`metrics.json.weights_path` 记录完整路径），完整 smoke val（10 图 / 544 实例），imgsz=640、conf=0.001、NMS IoU=0.70、**max_det=3000（§7 变更 2 的实测决策）**、FP32。
- 指标（`evaluation/metrics.json`，0–1 原值，最终一次运行）：mAP50=**0.129959**、mAP50-95=**0.101680**、P=**0.49383**、R=**0.11719**；`evaluator_version=8.4.215`、`max_det=3000`、`complete=true`。
- max_det 实测决策（同一 smoke val，仅改这一项）：mAP50 = 0.126582(300) / 0.127566(500) / 0.128731(1000) / 0.129476(2000) / 0.129959(3000) / 0.130398(5000)，推理耗时基本不变（~56 ms），P/R（best-F1 点）不变。取 3000：3000→5000 只再涨 0.00044。
- 披露：该 smoke 模型在 conf=0.001 下**每张图都触到 max_det**（`max_det_hits=10`），`metrics.json` 因此写入告警；另有 634 个检测框裁剪后面积为 0（`zero_area_detections`，同样写入告警）。正式模型上要复核这两个数。
- 固定阈值案例：`cases.json`（同时写 `case_selection.json`，见 §7 变更 15）记录 `successful_criterion=strict_no_fn_no_fp`、successful=0、failure=6（TP=101/FP=142/FN=443），每条案例带 `criterion`、`classes`、`gt_median_rel_area` 等特征；两级判据都无候选时 F11 不造图（见 §7 变更 10）。
- 导出：`per_class.csv`（10 类逐类 P/R/AP50/AP50-95）、`curves.npz`（`pr_precision`/`f1`/`p_conf`/`r_conf` 各 11×1000，末行为均值；`valid_classes` 掩码）、`curves_meta.json`、`confusion_matrix.npz`（11×11 原始计数）+ `confusion_meta.json`（true=row / pred=column / background=10 / conf=0.001 / IoU=0.5）、`predictions.jsonl`（10 行，原图像素 `xyxy`）、`ground_truth.jsonl`（10 行）、`cases.json`（固定 conf=0.25、IoU=0.5：TP=101, FP=142, FN=443）。
- 评价器记录：`ultralytics.DetectionValidator 8.4.215`；`pr_definition` 明写这是原生 P/R 汇总定义，不是固定 conf 的 micro 值。
- 副作用已修并实测：早先版本让 ultralytics 把自身 val 图写到 `workspace/val_runs/`；现在固定写到 `outputs/smoke_local_001/evaluation/ultralytics/`（本次运行的 `Results saved to …/evaluation/ultralytics` 为证），旧的 `workspace/val_runs` 残留已核对内容后删除并在本报告说明。

### 4.4 真实测速（CPU，固定 10 图清单）

```bash
python -m uavdet --config configs/smoke.yaml benchmark --run-id smoke_local_001 \
  --weights outputs/smoke_local_001/train/weights/best.pt --device cpu
```

- batch=1、FP32、warmup=5、repeats=3、num_images=10（smoke val 只有 10 张，summary 记录 `num_images=10` 并给出告警）、30 次计时。
- mean=**128.65 ms**、P50=**116.41 ms**、P95=**182.10 ms**、FPS=**7.77**（CPU Apple M4；同一机器前后两次分别为 185 ms / 129 ms，CPU 计时存在负载波动，比较设备时必须固定设置）。
- 短清单已显式告警：`requested_images=100`、`available_images=10`、`warnings=["benchmark.num_images=100 but the 'val' split … has only 10 image(s) …"]`，并在 `manifest.json` 记 `benchmark_shortfall` 事件。
- `timing_scope` 写明：已解码原图→预处理→模型→后处理→结构化结果；不含权重加载、磁盘读图、绘图、保存。

### 4.4b 一条命令的真实 pipeline（smoke，CPU）

```bash
python -m uavdet --config configs/smoke.yaml pipeline --run-id smoke_e2e_001 --device cpu
```

- 阶段结果：`audit=complete`、`train=complete`（2 轮，真实 YOLO11s）、`evaluation=complete`、
  `benchmark=complete`、`figures=failed`（11/12 张已生成，F11 缺失）；命令退出码 **7**（图表 partial）。
- 验证器自身输出落在 `outputs/smoke_e2e_001/evaluation/ultralytics/`；benchmark 打印短清单告警；
  图表清单 `figure_manifest.json` 给出 `missing=[F11]` 的原因。
- 这次运行是"一条命令可跑通"的直接证据，也是**唯一一次**完整 pipeline（不存在第二个同名 run）。
- 该 run 的图表 tag 为 `figures/final_v2/`（与 `smoke_local_001/figures/final_v9/` 同一版代码生成）。
- `figures` 阶段记为 `failed` 而不是 `complete`：计划只允许
  pending/running/complete/failed/interrupted/skipped，缺一张已适用图就不能宣称完整。

### 4.5 真实图片推理（headless）

```bash
python -m uavdet --config configs/smoke.yaml predict \
  --weights outputs/smoke_local_001/train/weights/best.pt \
  --source dataset_visdrone/VisDrone2019-DET-val/images/0000271_01401_d_0000380.jpg \
  --out /tmp/uavdet_predict_test --device cpu
```

- 21 个目标（people 0.971、car 0.845 …），`detections.csv` + `predictions.json` + `run_meta.json`（记录 10 类、SHA256、device、耗时）。
- 标注图人工打开核对：颜色与类别一致、框落在真实车辆上、无 BGR/RGB 颠倒。

### 4.6 桌面应用（真实 10 类权重，离屏）

以 smoke `best.pt` 在 `QT_QPA_PLATFORM=offscreen` 下真实运行：加载模型 → 真实图片检测 → 21 目标 → 导出 PNG/CSV → 结果页载入 `smoke_local_001`。

- 主线程/工作线程 id 断言不同（推理不在 GUI 线程）；截图显示深色界面、分类计数、明细表（原图像素坐标）、耗时 136.8 ms。
- 结果页显示：scope=`smoke 小样本`、阶段表（图表 failed，因 F11 缺项）、指标 ×100 只换算一次（12.658% / 10.040% / 49.383% / 11.719%）、11 张 PNG 缩略图。
- 截图：`docs/证据截图/界面_运行中.png`、`docs/证据截图/界面_中文.png`、`docs/证据截图/界面_英文_结果页.png`（1400×880）。
- 修复的环境级问题：macOS + PyQt6 下**首次在 QThread 内 import torch 会 SIGBUS**（已实测复现：`EXIT:138`）；现在 `MainWindow.load_model` 先在主线程 `warm_up_runtime()` 导入 torch/ultralytics，再启动工作线程（`app/workers.py`），修复后同一脚本 `EXIT:0`，21 个目标、线程 id 断言不同。

### 4.7 真实图表

- smoke 运行：`figures/final_v9/` 11 张图（F01–F10、F12）×PNG/PDF/SVG，`status=partial`，`missing=[F11]`，原因写明「没有图片满足成功的固定阈值判据」；`F09` 输入 SHA256 与实际 `evaluation/curves.npz` 一致（哈希逐项核对）。逐张打开检查：F01 类别分布、F02 相对面积（log 轴、共享分箱、非 COCO AP_small）、F03 每图目标数+ECDF、F04/F05/F06 训练曲线、F07 逐类 AP、F08 混淆矩阵（原始计数 + 按 GT 行归一化，background 单独标注；真实数据核对时发现初版按列归一化的 broadcast 错误，已改为 cell/row-sum 并加 D03 逐值测试）、F09 PR、F10 F1–conf、F12 失败案例（context + 同一裁剪框的 original/GT/prediction，取消逐框文字避免密集场景不可读；初版在真实数据上文字重叠不可读，按实际渲染结果返工后复核）。
- 全量数据：`outputs/preflight_local_002/figures/dataset_v1/` 的 F01–F03（F04–F12 记 `not_applicable`：该 run 没有训练/评价阶段）。
- 图面文字、图注模板、索引全部英文（`tests/test_figures.py::EnglishOnlyFigureOutputTests` 守护）；
  产物内路径为 run-relative 或 `external/<文件名>`，不含本机绝对路径。
- 每张图 `figure_manifest.json` 记录输入路径+SHA256、变量/单位/类别序、筛选/变换/归一化、样本数、软件与样式版本、输出尺寸；`source_data/` 保存实际绘图数值。照片面板明确为栅格内容。

## 5. 测试证据

```bash
cd Project/uav_detection_rebuild
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/uavdet-mpl \
  .venv/bin/python -m unittest discover -s tests -v
# Ran 230 tests in 61.6s — OK (skipped=1: real-model test, needs UAVDET_REAL_WEIGHTS)
```

| 文件 | 数量 | 覆盖的验收编号 |
|---|---|---|
| `tests/test_config_runs.py` | 23 | A01/A02、run 生命周期、阶段状态机、协议指纹、原子写入、`extends` 循环 |
| `tests/test_data.py` | 24 | A03–A08、坐标换算 A06（100×80 → `[40,30,60,50]`）、重复检测、视图复制/复用、smoke 子集、修复策略 |
| `tests/test_training_evaluation.py` | 61 | B01–B08（参数传递、初始化来源、resume 约束、曲线/矩阵导出、N/A、预测证据、失败状态） |
| `tests/test_inference_benchmark.py` | 32 | C01–C06（参数、BGR、max_det/空结果、模型切换、资源释放、计时边界） |
| `tests/test_figures.py` | 31 | D01–D04（百分比只换算一次、缺项不造图、3 类/3 轮逐值核对、scope 混用拒绝） |
| `tests/test_gui.py` | 42 | E01–E04 + 导出、设置持久化、失败恢复、停止/关窗释放 |
| `tests/test_integration.py` | 17 | CLI 全链路 check→train→val→benchmark→plot、`pipeline --continue` 不重训、figure tag 不可覆盖、scope 混用拒绝、reval run 的 `not_applicable` |

其它已执行检查：

```bash
python -m compileall -q src tests     # 通过
python -m uavdet --help               # 退出 0，不导入 torch/Qt
python -m uavdet --config configs/project.yaml check --help   # 退出 0
python -m uavdet --config configs/project.yaml pipeline --help# 退出 0
python -m uavdet.app --help           # 退出 0（不创建 QApplication）
```

安装形态检查（`E04`）：`pip install --target /tmp/uavdet_install_target .` 后
`importlib.resources` 能定位 `plotting/assets/sci-report.mplstyle` 与 `app/styles.qss`，
且 `import uavdet.cli` 不加载 torch/PyQt6（已实测 `style exists: True qss exists: True`）。

验收编号覆盖：计划 03 §3 的 30 条行为（A01–A08、B01–B08、C01–C06、D01–D04、E01–E04）逐条映射到测试（下表是代表性测试，不是穷举；完整清单可用测试名 grep）。

| 编号 | 行为 | 代表性测试（不是穷举；完整清单可 grep 测试名） |
|---|---|---|
| A01 | 不同 cwd 解析同一配置、CLI 覆盖 | `test_config_runs.py::test_paths_resolve_against_config_file_not_cwd`、`::test_command_line_override_wins` |
| A02 | 拒绝 `../`/分隔符 run_id | `::test_valid_and_invalid_ids`、`::test_run_id_cannot_escape_output_root` |
| A03 | 原数据/旧权重不变 | `test_data.py::test_audit_never_writes_into_the_data_root`、`::test_view_is_a_copy_and_source_is_untouched`、`test_integration.py::AuditReuseTests` |
| A04 | 缺图/坏图/非法类别/负宽 | `test_data.py::test_corrupt_image_fails_the_audit`、`::test_negative_box_is_rejected`、`::test_out_of_range_class_is_rejected`、`::test_orphan_label_is_rejected` |
| A05 | 空标签、无预测 | `test_data.py::test_empty_label_is_legal_and_counted`、`test_inference_benchmark.py::test_c03_empty_result_is_a_valid_result`、`::test_zero_detection_images_are_still_saved` |
| A06 | 坐标换算 100×80 | `test_data.py::test_reference_case_from_acceptance_list` |
| A07 | 固定类序、总数一致 | `test_data.py::test_clean_dataset_passes_and_counts_add_up`、`test_figures.py::test_dataset_figures_match_audit` |
| A08 | 重复图/跨 split | `test_data.py::test_cross_split_duplicate_fails_and_same_split_duplicate_warns`、`::test_duplicate_annotation_lines_are_reported` |
| B01 | train 参数实际传递 | `test_training_evaluation.py::test_b01_every_configured_parameter_is_passed`、`::test_b01_artefacts_are_copied_into_the_run` |
| B02 | val 参数实际传递 | `::test_b02_val_arguments_are_passed_explicitly`、`::test_non_default_max_det_conf_and_iou_reach_the_validator`、`::test_split_and_output_location_reach_the_validator` |
| B03 | 错误初始化权重被拒 | `::test_missing_file_is_a_weight_error`、`::test_best_pt_without_flag_is_rejected`、`::test_checkpoint_with_visdrone_class_names_is_rejected` |
| B04 | resume 约束、epoch 不叠加 | `::test_b04_resume_uses_the_same_run_checkpoint_without_adding_epochs`、`::test_b04_non_monotonic_results_csv_is_rejected`、`::test_b04_unfinished_training_raises_but_keeps_artefacts` |
| B05 | 数组/矩阵/曲线导出 | `::test_b05_per_class_maps_by_class_id`、`::test_b05_curves_keep_class_order_and_mean_row`、`::test_b05_confusion_is_exported_true_by_predicted` |
| B06 | 无 GT 为 N/A | `::test_b06_no_gt_class_is_none_not_zero`、`::test_b06_per_class_uses_na_for_classes_without_gt` |
| B07 | 同一次验证的预测证据 | `::test_b07_jsonl_ids_are_stable_and_coordinates_are_original_pixels`、`::test_b07_capture_batch_records_ids_and_original_pixel_coords` |
| B08 | 失败不写完成 | `::test_b08_training_failure_raises_and_never_claims_complete`、`test_config_runs.py::test_failed_stage_is_not_complete` |
| C01 | 引擎参数（图片/视频/摄像头同一套） | `test_inference_benchmark.py::test_c01_predict_passes_every_configured_parameter`、`::test_cpu_fallback_is_recorded_when_the_gpu_is_missing` |
| C02 | BGR 不重复转换 | `::test_c02_bgr_frame_reaches_the_model_unchanged`、`::test_c02_annotate_keeps_colours_and_the_original_frame`、`test_gui.py::test_bgr_to_qimage_converts_exactly_once` |
| C03 | max_det 与空结果 | `::test_c03_max_det_caps_the_output`、`::test_c03_empty_result_is_a_valid_result` |
| C04 | 模型切换失败保留旧模型 | `::test_c04_load_failure_raises_and_leaves_nothing_loaded`、`::test_c04_wrong_class_count_is_rejected`、`::test_failed_switch_keeps_the_previous_model_usable` |
| C05 | 停止/异常释放资源 | `::test_c05_close_releases_the_model_and_is_idempotent`、`test_gui.py::test_stop_releases_capture_and_recovers`、`::test_error_releases_capture_and_recovers` |
| C06 | 测速边界 | `::test_c06_timing_covers_only_predict`、`::test_c06_image_selection_is_deterministic_and_from_the_view`、`::test_short_of_the_requested_image_count_is_reported` |
| D01 | 百分比只换算一次 | `test_figures.py::test_percent_conversion_happens_once`、`::test_map50_percent_is_multiplied_exactly_once` |
| D02 | 缺输入不造图 | `::test_missing_curves_npz`、`::test_missing_results_csv`、`::test_missing_confusion_matrix`、`::test_no_successful_case_is_reported_not_invented` |
| D03 | 数据到图逐值一致 | `::test_dataset_figures_match_audit`、`::test_training_figures_match_results_csv`、`::test_per_class_ap_matches_csv`、`::test_confusion_matrix_orientation_and_normalisation` |
| D04 | smoke 不混入正式 | `test_integration.py::test_scope_mismatch_refuses_a_report`、`test_gui.py::test_smoke_run_is_badged_as_smoke` |
| E01 | GUI 后台任务、状态恢复 | `test_gui.py::test_e01_predict_runs_in_a_worker_thread`、`::test_session_refuses_inference_on_main_thread`、`::test_failed_detection_recovers_the_ui` |
| E02 | 关窗/停止释放 | `::test_e02_close_during_image_task_stops_it`、`::test_e02_close_during_stream_releases_capture` |
| E03 | 无 Qt 核心可用 | `::test_core_import_does_not_pull_in_qt`、`::test_app_modules_import_without_qt`、`test_figures.py::test_importing_renderers_is_light` |
| E04 | 安装后资源定位 | `test_gui.py::test_importlib_resources_locates_stylesheet`、`::test_stylesheet_is_applied` + 普通安装实测（§5 末） |


单元测试全部在临时目录的合成数据上运行：不下载权重、不读真实数据集、不写 `dataset_visdrone/`。

测试迭代中由真实数据暴露并修复的问题：F12 密集场景不可读（改裁剪+去逐框文字）、F08 归一化方向错误（列→行）、验证器输出目录外泄、benchmark 短清单静默、GUI 首次线程内 import torch 崩溃、标注文字贴边被裁。这些都按「先复现→再修→再在真实产物上复核」处理，不是删测试或改断言。

## 6. 参考旧代码与改进

| 旧文件 | 复用 | 新实现改掉的缺陷 |
|---|---|---|
| `train_yolo11.py` | 训练参数默认值、YOLO 调用方式 | 配置集中化；`validate()` 传 `project/name`；真实 `save_dir` 回流；不写原数据；不接受旧 best.pt 起点 |
| `detection_app/utils/detector.py` | 10 类中英映射、结果解析思路 | 全局无锁单例 → 工作线程持有；所有推理显式传 conf/iou/max_det/imgsz；结果不提前四舍五入 |
| `main_window.py` | 侧栏+页面组织、深色风格 | 去掉登录；模型切换真实生效；关闭有完整生命周期 |
| `pages/image_page.py` 等 | 预览/结果表/计数/导出 | 统一走公共 `Detector`，不复制加载逻辑；流式页合并视频与摄像头 |
| `pages/model_page.py` | 权重与参数控件 | 检查 10 类；加载成功才切换；失败保留旧模型 |
| `pages/metrics_page.py` | 看曲线与指标 | 明确区分训练曲线/最终验证/smoke，不把 CSV 末轮当 best.pt 复测值 |
| `generate_metrics_charts.py` | 损失/mAP/P-R 图思路 | 新增数据特点/逐类/PR-F1/案例；出图只读结果、可溯源、可无 GPU 重画 |
| `utils/chart_style.py`、`sci-skills.mplstyle` | SCI 配色与线型 | 复制为包内资源，云端不依赖本机 skills 目录 |
| `utils/storage.py` | CSV 导出思路 | 不继承账户数据/硬编码模型目录 |

## 7. 偏离与需要说明的决定

1. **`data.label_repair`（新增配置键 + 记录的 0 面积标注修复）**：计划要求非法标签阻止正式训练。全量审计发现 train 有 1 条 0 像素高度标注（共 381,963 实例中 1 条），任何检测器都无法使用该框。默认 `drop_degenerate_boxes`：只丢弃这一类，逐行写入 `audit/label_repairs.csv`，只作用于派生视图副本；`strict` 保留可复现"原始数据确实不合法"。若验收要求严格拒绝，把两个 config 改回 `strict` 即可（随后正式训练会被阻止，直到人工修数据）。
2. **`eval.max_det` 由 300 改为 3000（实测决定）**：审计显示单图 GT 最多 902（train）/317（val），计划要求"报告真实最大目标数并统一决定"。在同一 smoke val 上只改这一项测得 mAP50 = 0.126582(300) / 0.127566(500) / 0.128731(1000) / 0.129476(2000) / 0.129959(3000) / 0.130398(5000)，耗时不变；取 3000 作为拐点。**评价**用 3000，GUI/predict/benchmark 的应用速度仍为 300。`metrics.json` 写入 `max_det_hits`（本 smoke 为 10/10 图触顶）与 `zero_area_detections`（634）两条告警，正式 run 必须复核这两项。代价：`predictions.jsonl` 体积随 max_det 增长（548 图全量约数百 MB，取回时注意）。
3. **`report.build_figures` 引入 `not_applicable` 与 `missing` 的区分**：证据来自本 run 且缺失 → `partial`（CLI 退出 7）；本 run 根本没有该阶段 → `not_applicable`，不阻塞（reval run 的 plot 仍返回 0）。这落实了"缺输入要非零退出、缺失要清单"而不把"未做该阶段"误报为错误。
4. **配置新增 `meta` 只读块**：写入解析后的配置文件路径、extends 链、加载时间，便于复现；用户配置里出现 `meta` 会被拒绝。
5. **新增小文件 `devices.py`**：统一设备解析与显式 CPU 回退告知（训练/推理/测速共用），避免三处各写一份。
6. **`plotting/dataset.py` 内含共享绘图工具箱**（`figure_settings`/`prepare_render`/`finish_figure` 等），供同属一个交付物的 training/evaluation/cases 复用，避免四份重复；`tests/test_figures.py` 断言各模块 `FIGURE_META` 与 `report.FIGURE_META` 一致。
7. **GUI 主线程 `warm_up_runtime()`**：macOS + PyQt6 下首次在 QThread 内 `import torch` 触发 SIGBUS（已实测复现）；改为加载模型前在主线程导入 torch/ultralytics。这是环境适配，不改变"推理在后台线程"的约束。
8. **CLI 增加 `--override section.key=value`**：计划要求"命令行覆盖优先"，但未规定参数形态；该形式可覆盖任意已校验字段（如 `data.label_repair`），并在 run 内记录最终配置。
9. **额外产物 `audit/label_repairs.csv`**：让第 1 条偏离可审计；空表也会写出。
10. **F11 成功案例的两级判据**：严格判据是「conf≥0.25、IoU≥0.5 下无漏检无误检」。VisDrone
    单图 40–900 个目标，训练好的模型也极少整图零错误，因此增加一个**有记录的第二级**：来自本次
    验证、`tp>=3`、`precision>=0.8`、`tp/num_gt>=0.4`，每条记录写 `criterion`，图注与
    `cases.json` 都标明用的是哪一级；两级都没有候选时仍然如实输出「无合格案例」。2 轮 smoke
    模型两级都不满足，所以 F11 仍缺失——这是数据事实，不是渲染错误。
11. **验证器自身输出重定向**：`model.val()` 的原生图/缓存现在写进 `outputs/<run>/evaluation/ultralytics/`，避免在 `workspace/` 留下无归属目录。

12. **已完成阶段默认拒绝重跑**（本轮补齐）：`val`/`benchmark` 在同一 run 上第二次执行会返回
    退出码 4 并提示用新 `--run-id`（可用 `--parent-run-id` 记录来源 run）；操作者确实要重跑时
    加 `--force`，stage 的 `attempts` 会递增并留事件。这对应计划 01 §5「重验证使用新 run_id」。此前
    实现允许直接覆盖，属于与规格不一致，已修正并加测试。
13. **正式 pipeline 的两道完整性闸门**（本轮补齐）：`scope=formal` 时，评价必须覆盖
    `data.expected_val_images`（548）张，测速必须拿到配置的固定清单（100 张），否则对应 stage 记
    `failed`、pipeline 退出非零。`metrics.json` 同时新增 `expected_images`/`complete` 字段与
    不完整告警，单独 `val` 也能看出是否覆盖了全部图片。期望值按 scope 取：formal=548/6471，
    smoke=10/20（否则正确的 smoke 会被误判为不完整，已修复并加测试）。
14. **CLI 声明但此前未生效的参数**（本轮修复）：`val --parent-run-id`、`benchmark --images`
    现在真正传递（此前被解析后忽略）。已加 `tests/test_integration.py::CliContractTests` 覆盖。
15. **`eval.case_images` + `case_selection.json` + 案例特征（本轮补齐）**：计划 02 §2 要求
    `case_selection.json`、按案例的类别/尺寸特征、以及"用户显式指定 image_id"。现在 `cases.json`
    与 `case_selection.json` 同时写出（同一 payload），每条案例带 `criterion`/`classes`/
    `gt_median_rel_area`，`eval.case_images` 里的图片固定排在最前并标 `criterion=user_selected`。
16. **`check` 必须按当前数据重新审计（本轮修复，阻断级）**：独立审查发现 `check` 在
    `--data-root` 变化时直接复用旧审计（错误的云端挂载会通过关卡）。现在复用前会核对数据根、
    期望数量、类别序，并对每 split 首/中/末样本重算 SHA256；任一不符即记 `audit_reuse_rejected`
    并真正重审。回归测试：`tests/test_integration.py::AuditReuseTests`。
17. **拒绝的执行不得改动阶段记录（本轮修复）**：`plot` 遇到已存在的 figure tag 时，之前会先
    `begin_stage` 再把阶段标成 `failed`。现在 tag 检查提前到阶段转换之前，回归测试断言
    `attempts` 与状态不变。
18. **权重来源透明（本轮修复）**：`best.pt` 缺失时回退 `last.pt` 会打印并记录事件，且
    `metrics.json` 新增 `weights_path`；`val --split test` 在数据集没有 test split 时会在审计前
    直接失败（退出码 3），不再浪费一次全量审计；`pipeline --continue` 遇到"阶段 complete 但产物
    丢失"会明确说明不能自动重写。
20. **图表输出强制英文 + 路径可移植（本轮）**：`src/uavdet/plotting/*` 与 `report.py` 没有任何中文；
    渲染后的图像文字、`figure_index.md`、`figure_manifest.json` 的说明字段全部英文，测试会扫描这些
    字符串并在**路径含中文的工程目录**下渲染验证仍然全英文；图表产物里的路径改为
    run-relative 或 `external/<文件名>`（加 SHA256 保证可追溯），不再写入本机绝对路径。
21. **桌面端中英文切换（本轮新增需求）**：`app/i18n.py` 提供 231 条翻译，中文原文即 key，
    缺失翻译回落到中文（不会出现空标签）；`MainWindow.toggle_language()` 在**空闲时**重建界面并
    保留已加载模型、当前图片/检测结果、已选 run；语言写进 `settings.json`，下次启动保持。
    测试 `tests/test_gui.py::LanguageToggleTest` 覆盖：默认中文、切换后可见控件无中文、切回、
    持久化、任务运行时拒绝、状态保留。图表面板与 CLI/报告仍是英文，不受此开关影响。
19. **额外文件（此前未声明）**：
    `src/uavdet/app/main.py`（`uavdet-app` 入口）、`docs/证据截图/界面_运行中.png`（实机截图）、
    `check --allow-fail`（只用于人工查看失败审计）、`requirements-verified.txt`（精确版本）。
    这些不改变计划范围，职责如上。

## 8. 未验证 / 待测项（不得当作已通过）

1. **云端 GPU smoke 单独产物**：正式 run 已完成，但未单独回传 smoke 的 `outputs/`（环境信息见 §11）。
2. ~~真实 548 图完整验证、100 图固定清单测速~~：**已完成**（见 §11，548 图 / 100 图测速）。
3. **真实训练恢复（`--resume` 实跑）**：仍只有单元测试覆盖（本次正式 run 未中断）。
4. **真实模型集成测试**：已实现为 `UAVDET_REAL_WEIGHTS=<10 类权重>` 的显式开关（`test_integration.py::RealModelIntegrationTests`），默认 skip；本机未用该开关跑过。
5. **摄像头实机 / 真实视频解码**：无设备，仅假 `VideoCapture`。
6. **GPU 上的 AMP 具体回退行为**：正式 run 开 AMP 完成（results.csv 正常），但 AMP 检查细节未逐行核对；batch=16 已在 4090 D 上跑通。
7. **F11 成功案例的成图路径**：2 轮 smoke 模型在两级判据下都无候选，真实数据只验证了"缺失"路径；成图路径由合成夹具覆盖。
8. **`max_det_hits` 与 `zero_area_detections` 在正式模型上的取值**：smoke 模型 10/10 图触顶（3000）、634 个零面积框；正式模型必须复核（`metrics.json` 已写入）。
9. **独立人工视觉复核**：实现者已逐张打开 PNG 检查；`figure_manifest.json` 的 `scientific_review`/`visual_review` 仍为 `pending`，表示**独立复核人尚未签字**。
10. **F08 在 conf=0.001 下的可读性**：评价阈值下 background 行会被大量低置信预测占据（已在图注如实标注）。若报告需要更干净的矩阵，可另起 run 用 `--override eval.conf=0.25` 专出该图（与主指标 run 分开、并注明用途）。
11. **Linux/云端中文字体、HiDPI、非 ASCII 路径**：未验证。

## 9. 旧资产指纹核对（计划 03 §9）

2026-11-17 只读核对 `Project/无人机小目标检测系统/`，五个指纹与计划完全一致，说明旧项目未被本次实现改动：

| 文件 | 实测 SHA256 | 计划值 |
|---|---|---|
| `train_yolo11.py` | `b19fc785…6cf68c4` | 一致 |
| `generate_metrics_charts.py` | `264a064a…2ff23b7` | 一致 |
| `runs/.../weights/best.pt` | `10ffb3e1…55fbc339` | 一致 |
| `runs/.../results.csv` | `c383fe77…dc902462` | 一致 |
| `runs/.../args.yaml` | `98de905e…0155463` | 一致 |

本次实现没有读取旧权重作为训练起点，没有复制旧账户/历史/缓存；`dataset_visdrone/` 在全部检查与训练后内容哈希未变（A03 测试 + 人工核对）。

## 9b. 交付目录中的运行时产物

| 路径 | 内容 | 说明 |
|---|---|---|
| `outputs/preflight_local_001/` | strict 策略下审计失败的 run | 证明非法标签确实阻止正式训练 |
| `outputs/preflight_local_002/` | 全量审计 + 数据集图表（`figures/dataset_v1/`） | formal scope，真实 7,019 图 |
| `outputs/smoke_local_001/` | 分阶段：审计→训练→验证→测速→图表（`figures/final_v9/`） | 2 轮真实训练，最终证据集 |
| `outputs/smoke_e2e_001/` | 一条命令 `pipeline` 的完整 run（图表 `figures/final_v2/`） | figures 阶段 failed（F11 缺项），退出码 7 |
| `outputs/formal_001/` | **云端正式 run（300 轮，RTX 4090 D）**，含权重、548 图评价、测速、12 张图表 | 五阶段 complete；结果摘要见 §11 |
| `workspace/data_views/` | smoke（30 图）与 formal（7,019 图，1.6 GB）两个数据视图副本 | 源数据只读；视图可复用 |
| `weights/yolo11s.pt` | 官方 COCO 预训练权重 | SHA256 `85a76fe8…d502d5`，由 `fetch-weights` 显式下载 |

中间迭代产生的 figure tag（`smoke`、`smoke2`–`smoke7`、`final_v1`–`final_v6`）已清理，
只保留每个 run 的最终 tag（`smoke_local_001/figures/final_v9`、`smoke_e2e_001/figures/final_v2`）；
清理属于本实现自己的运行产物，未触碰任何输入数据。

## 10. 复现下一步（给验证者 / 用户）

```bash
# 1) 本地复跑验收（无需 GPU/Qt 即可跑核心）
cd Project/uav_detection_rebuild
pip install -r requirements-core.txt && pip install -e . --no-deps
python -m compileall -q src tests
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v

# 2) 检查已有真实产物（不需要重新训练）
python -m uavdet --config configs/project.yaml info
python -m uavdet --config configs/smoke.yaml plot --run-id smoke_local_001 --figure-tag v2

# 3) 云端（需授权）：先 smoke，再正式
python -m uavdet --config configs/smoke.yaml pipeline  --run-id smoke_gpu_001 --device 0
python -m uavdet --config configs/project.yaml pipeline --run-id formal_001     --device 0
```

第 3 步的完整门槛见 `docs/CLOUD_RUN.md`。正式 300 轮已在云端完成（结果见 §11），独立复核仍待进行。

## 11. 云端正式运行结果（formal_001，2026-11-18）

本次交付的唯一结果来源：`outputs/formal_001/`（Slurm 日志在 `logs/slurm-formal-263462.out`）。

| 项 | 值 |
|---|---|
| 硬件 / 环境 | NVIDIA GeForce RTX 4090 D（23.52 GB）· torch 2.5.1+cu121 · ultralytics 8.4.215 · numpy 1.26.4 |
| 协议 | scope=formal · 300 epochs · batch 16 · imgsz 640 · SGD lr0=0.01 · seed 0 · patience 0 · close_mosaic 10 · AMP 开 |
| 数据 | train 6471 / val 548；评价覆盖 548 图 / 38,759 实例（`complete=true`） |
| 评价设置 | conf=0.001 · NMS IoU=0.70 · max_det=3000 · FP32 |
| **mAP@0.5** | **40.64%** |
| **mAP@0.5:0.95** | **23.46%** |
| Precision / Recall | 54.58% / 40.30% |
| 训练曲线 | mAP50 峰值在**第 115 轮**（40.59%）；150 轮后缓降，第 300 轮 39.43% |
| 训练耗时 | 300 轮 **2.61 h**，约 31.3 s/轮 |
| 测速（batch1、FP32、100 图×3） | mean **9.10 ms** · P50 8.99 ms · P95 10.02 ms · **FPS 109.9** |
| 图表 | `figures/v1/` **12/12 全部产出**，`status=ok`，`missing=[]`，输入 SHA256 与 run 文件逐项一致 |
| 案例 | 严格判据（conf≥0.25、IoU≥0.5、无漏检无误检）**2 个成功案例**，6 个失败案例 |
| 告警（已记录） | 9/548 图触到 max_det=3000（截断风险）；144 个裁剪后零面积框 |

逐类 AP@0.5：pedestrian 45.08% · people 33.88% · bicycle 14.51% · car 79.65% · van 45.28% · truck 39.09% · tricycle 29.18% · awning-tricycle 15.50% · bus 57.29% · motor 46.96%（逐类 AP@0.5:0.95 见 `evaluation/per_class.csv`）。

两点如实说明：

1. **best 出现在第 115 轮**，150 轮后缓慢下降（300 轮时 39.43%）。`best.pt` 选到的是峰值权重；
   如果重训一次，150–200 轮更省时（协议仍是 300 轮，未改动）。
2. `max_det` 触顶与零面积框都写进了 `evaluation/metrics.json` 的 warnings，属于如实披露；
   主表数字只来自本次运行的同一评价器、同一阈值。

