# FineLocation 推理工具

这个文件夹用于运行焊缝起点精定位两阶段网络。当前内置模型是“参考点 3 mm 数据增强”版本。

## 1. 创建环境

推荐先用 CPU 环境，兼容性最好：

```bash
conda create -n fineloc python=3.11 -y
conda activate fineloc

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy open3d
```

如果机器有 NVIDIA 显卡，也可以安装 GPU 版：

```bash
conda create -n fineloc python=3.11 -y
conda activate fineloc

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy open3d
```

检查环境：

```bash
python - <<'PY'
import torch, open3d, numpy
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("open3d:", open3d.__version__)
print("numpy:", numpy.__version__)
PY
```

## 2. 输入文件

推理需要：

- `cloud_*.pcd`：原始点云。
- `param*.txt`：参数 JSON，必须包含 `startPos` 和 `endPos1`。
- `result_*.txt`：可选，用于显示传统工具输出点和计算误差；没有也能推理。

## 3. 直接运行一组自带样本

进入本文件夹：

```bash
cd ToolFineLocation_infer_ref3
```

运行 normal 样本：

```bash
python fineloc_infer.py \
  --cloud examples/raw_samples/normal/cloud_2025-12-12_09_15_19_M0_P0L0P1识别成功.pcd \
  --param examples/raw_samples/normal/param2025-12-12_09_15_19_M0_P0L0P1识别成功.txt \
  --result examples/raw_samples/normal/result_2025-12-12_09_15_19_M0_P0L0P1识别成功.txt \
  --device cpu
```

也可以用脚本运行同一组样本：

```bash
bash examples/run_one.sh
```

如果要打开三维可视化窗口，加上：

```bash
--visualize
```

## 4. 批量推理

推理一个文件夹：

```bash
python fineloc_infer.py \
  --raw-dir /path/FineLocationData2026 \
  --output-jsonl predictions.jsonl \
  --device cpu
```

递归推理多级目录：

```bash
python fineloc_infer.py \
  --raw-dir /path/PointCloudData \
  --recursive \
  --output-jsonl predictions.jsonl \
  --device cpu
```

自带 4 组示例数据也可以批量推理：

```bash
python fineloc_infer.py \
  --raw-dir examples/raw_samples \
  --recursive \
  --output-jsonl examples/predictions.jsonl \
  --device cpu
```

## 5. 输出含义

主要输出字段：

- `final_start_point`：网络预测的焊缝起点，原始坐标系。
- `coarse_point`：一阶段粗定位点，原始坐标系。
- `stage2_knn_radius_mm`：二阶段从原始点云 KNN 裁剪时覆盖的半径。
- `final_l2_mm`：如果提供了有效 `result.weldStart`，这里是网络输出和工具/标签起点的 L2 误差。
- `final_parallel_abs_mm`：沿参考焊缝方向的误差。
- `final_perp_mm`：垂直参考焊缝方向的误差。

## 6. 自带样本说明

`examples/raw_samples/` 下包含：

- `normal/`：正常样本。
- `tool_failed/`：传统工具识别失败样本。
- `high_error_1/`：大误差样本。
- `high_error_2/`：端点附近点云质量较差的大误差样本。

## 7. 模型流程

默认流程：

```text
原始点云 + param
-> FPS 下采样 8192 点
-> 一阶段网络预测粗定位点
-> 以粗定位点为中心，从原始点云 KNN 取 16384 点
-> 二阶段网络预测修正量
-> 输出焊缝起点
```
