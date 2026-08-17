# C++ 部署版

该目录把当前“随机采样 + 平面法向非正交基 + 两阶段精定位”模型封装为 C++17 接口。运行时不需要 Python、PyTorch、Open3D 或 PCL，只依赖 ONNX Runtime。

C++ 库和 CLI 只负责推理，不包含三维可视化窗口；如需可视化，请使用仓库根目录的 Python 推理工具。

## 接口

```cpp
#include <fineloc/fineloc.hpp>

fineloc::FineLocationEngine engine("stage1.onnx", "stage2.onnx");
fineloc::Prediction prediction = engine.infer(raw_points, prior);
```

`raw_points` 是原始坐标系下的 `std::vector<fineloc::Vec3>`。`prior` 需要参考点、第一终点和三个候选平面法向。输出包括最终起点、焊缝方向、两个平面法向和二阶段 KNN 覆盖半径。

## Ubuntu 编译

下载并解压 ONNX Runtime C/C++ CPU 包，然后执行：

```bash
cmake -S cpp -B cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.23.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

运行自带样本：

```bash
./cpp/build/fineloc_cli \
  --cloud "examples/raw_samples/normal/cloud_2025-12-12_09_15_19_M0_P0L0P1识别成功.pcd" \
  --param "examples/raw_samples/normal/param2025-12-12_09_15_19_M0_P0L0P1识别成功.txt" \
  --stage1 models/stage1.onnx \
  --stage2 models/stage2.onnx
```

如运行时找不到 `libonnxruntime.so`：

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-1.23.2/lib:$LD_LIBRARY_PATH
```

## Windows 编译

下载 Windows x64 ONNX Runtime C/C++ 包，在 “x64 Native Tools Command Prompt for VS” 中执行：

```powershell
cmake -S cpp -B cpp\build -A x64 `
  -DONNXRUNTIME_ROOT=C:\libs\onnxruntime-win-x64-1.23.2
cmake --build cpp\build --config Release
```

生成的程序位于 `cpp\build\Release\fineloc_cli.exe`。CMake 会将 `onnxruntime.dll` 复制到程序目录。

## 重新导出模型

仓库已包含 ONNX，无需在部署电脑重新导出。如更换权重，在 Python 训练环境执行：

```bash
pip install onnx
python tools/export_onnx.py
```

CLI 的 PCD 读取器与当前 Python 工具一致，只支持 `DATA binary`、前三个字段为 float32 `x y z` 的 PCD。实际项目可跳过 CLI，直接调用 `FineLocationEngine::infer()`。
