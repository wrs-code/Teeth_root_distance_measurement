# Quick Start Guide - 快速开始指南

本指南帮助您快速开始使用牙齿实例分割批量处理系统。

## 30秒快速开始

```bash
# 1. 设置环境 (首次运行)
chmod +x setup_venv.sh
./setup_venv.sh

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载预训练模型 (如果没有)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# 4. 运行批量处理
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt
```

## 前置条件检查清单

在开始之前，确保您有：

- [ ] Python 3.8 - 3.10
- [ ] NVIDIA GPU (推荐，非必需)
- [ ] CUDA 11.8 (如果使用GPU) - [查看CUDA_SETUP.md](CUDA_SETUP.md)
- [ ] 至少8GB RAM (推荐16GB+)
- [ ] 牙科X光图像文件

## 详细步骤

### 步骤1: 环境设置

#### 选项A: 自动设置 (推荐)

```bash
# 给脚本执行权限
chmod +x setup_venv.sh

# 运行设置脚本
./setup_venv.sh
```

这个脚本会：
- 检查Python和CUDA版本
- 创建虚拟环境
- 安装所有依赖
- 验证GPU支持

#### 选项B: 手动设置

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 准备模型权重

#### ⚠️ 重要：开源仓库未提供预训练的牙齿模型！

**Instance_seg_teeth仓库没有提供训练好的模型文件**，您需要：
- 自己训练模型（推荐，2-4小时）
- 或使用通用YOLOv8模型快速测试（效果很差，仅用于测试pipeline）

---

#### YOLOv8模型

**选项1: 使用通用YOLOv8模型（⚠️ 仅测试pipeline，检测效果差）**

```bash
# 下载通用YOLOv8模型（在COCO数据集上训练，不认识牙齿）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# ⚠️ 警告：此模型不包含牙齿类别，会将牙齿识别为其他物体
# 只能用于测试代码是否运行，不能用于实际牙齿检测
```

**为什么通用模型不行？**
- COCO数据集：人、车、动物等80类（❌ 无牙齿）
- 会误检牙齿为其他物体
- 仅用于验证pipeline是否工作

**选项2: 训练牙齿专用模型（✅ 推荐）**

这是正确的使用方式：

```bash
# 1. 下载UFBA-425数据集
# 访问: https://figshare.com/articles/dataset/UFBA-425/29827475

# 2. 使用训练notebook
jupyter notebook Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb

# 3. 或命令行训练（数据准备好后）
yolo task=detect mode=train \
    model=yolov8x.pt \
    data=./data.yaml \
    epochs=30 \
    batch=10 \
    imgsz=640

# 4. 训练完成后使用
# 模型保存在: runs/detect/train/weights/best.pt
```

**训练时间**：RTX 3090约2-4小时（425张图像）

**选项3: 使用已有的牙齿专用模型**

如果您已经训练好或获得了模型：
```bash
ls models/
# yolo_teeth_best.pt
```

#### UNet模型 (可选)

如果您有训练好的UNet模型：
```bash
ls models/
# unet_teeth_model.h5
```

参考 `Instance_seg_teeth/notebooks/Unet/` 或 `Instance_seg_teeth/notebooks/yolov8+unet/` 训练。

### 步骤3: 准备输入图像

创建一个包含牙科X光图像的文件夹：

```bash
# 创建测试图像目录
mkdir -p test_images

# 将您的图像复制到这个目录
cp /path/to/your/xray/*.jpg test_images/
# 或
cp /path/to/your/xray/*.png test_images/

# 检查图像
ls test_images/
```

支持的图像格式：
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

### 步骤4: 运行批量处理

#### 基本使用 (仅YOLOv8检测)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt
```

#### 高级使用 (YOLOv8 + UNet分割)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./models/yolo_teeth_best.pt \
    --unet_weights ./models/unet_teeth_model.h5
```

#### 自定义输出目录

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt \
    --output_dir ./my_results
```

#### CPU模式 (无GPU)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt \
    --device cpu
```

### 步骤5: 查看结果

处理完成后，结果会保存在时间戳命名的目录中：

```bash
# 查看输出目录
ls output/

# 进入最新的批次目录
cd output/batch_20251121_123456/

# 目录结构
ls
# original_images/              - 原始图像副本
# yolo_visualizations/          - 检测结果可视化
# bounding_boxes_json/          - 边界框JSON数据
# binary_masks/                 - 二值掩码
# binary_masks_visualizations/  - 掩码可视化
# unet_predictions/             - UNet预测 (如果启用)
# unet_visualizations/          - UNet可视化 (如果启用)
# summary/                      - 处理汇总
# processing_log.txt            - 详细日志
```

#### 查看可视化结果

```bash
# 查看检测可视化
eog yolo_visualizations/*.jpg  # Linux
# 或
open yolo_visualizations/*.jpg  # Mac

# 查看掩码可视化
eog binary_masks_visualizations/*.png
```

#### 查看处理汇总

```bash
# 查看汇总JSON
cat summary/processing_summary.json

# 查看处理日志
cat processing_log.txt
```

## 完整示例工作流

```bash
# ==========================================
# 完整的批量处理工作流
# ==========================================

# 1. 首次设置 (只需一次)
chmod +x setup_venv.sh
./setup_venv.sh

# 2. 每次使用前激活环境
source venv/bin/activate

# 3. 准备数据
mkdir -p test_images
cp ~/Downloads/dental_xrays/*.jpg test_images/

# 4. 下载或准备模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# 5. 运行批量处理
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./yolov8x.pt \
    --output_dir ./results

# 6. 查看结果
cd results/batch_*/
ls -lh
cat summary/processing_summary.json | python -m json.tool

# 7. 处理完成后退出环境
deactivate
```

## 性能优化建议

### 如果处理速度慢

1. **使用GPU**: 确保CUDA正确安装并使用GPU模式
   ```bash
   # 检查GPU
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **使用更小的模型**: YOLOv8n比YOLOv8x快得多
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./yolov8n.pt
   ```

3. **批量处理**: 一次处理多个图像比多次单独处理更高效

### 如果内存不足

1. **减少图像大小**: 在输入前调整图像大小
2. **一次处理较少图像**: 分批处理
3. **使用更小的模型**: 使用yolov8n或yolov8s

## 故障排除快速参考

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'ultralytics'` | 运行 `pip install ultralytics==8.0.28` |
| `No GPU detected` | 检查CUDA安装，或使用 `--device cpu` |
| `CUDA out of memory` | 一次处理较少图像或使用CPU |
| `Model file not found` | 检查模型路径是否正确 |
| `No images found` | 检查输入目录路径和图像格式 |

详细故障排除请参考 [README.md](README.md#troubleshooting)

## 输出文件说明

### JSON格式数据 (可编程访问)

```python
import json
import numpy as np

# 读取边界框数据
with open('bounding_boxes_json/image1_boxes.json', 'r') as f:
    boxes = json.load(f)
    print(f"检测到 {boxes['num_detections']} 颗牙齿")

# 读取二值掩码
mask = np.load('binary_masks/image1_mask.npy')
print(f"掩码形状: {mask.shape}")  # (32, height, width)

# 读取UNet预测
if os.path.exists('unet_predictions/image1_unet_pred.npy'):
    pred = np.load('unet_predictions/image1_unet_pred.npy')
    print(f"预测形状: {pred.shape}")
```

### 可视化图像 (直接查看)

- **yolo_visualizations/**: 带边界框和标签的检测结果
- **binary_masks_visualizations/**: 彩色编码的牙齿类型掩码
- **unet_visualizations/**: UNet分割结果

## 下一步

1. **训练自定义模型**: 使用您自己的数据训练模型
   - 参考 `Instance_seg_teeth/notebooks/`

2. **集成到您的流程**: 将批量处理脚本集成到您的工作流
   ```python
   from batch_process_teeth import TeethBatchProcessor
   processor = TeethBatchProcessor(yolo_weights='...', ...)
   processor.process_batch('input_dir')
   ```

3. **自动化**: 创建定时任务自动处理新图像
   ```bash
   # 添加到crontab
   0 2 * * * cd /path/to/project && source venv/bin/activate && python batch_process_teeth.py --input_dir ./new_images --yolo_weights ./yolov8x.pt
   ```

## 获取帮助

- 查看完整文档: [README.md](README.md)
- CUDA设置帮助: [CUDA_SETUP.md](CUDA_SETUP.md)
- 报告问题: 在GitHub上创建issue

## 快速命令参考

```bash
# 激活环境
source venv/bin/activate

# 运行处理 (基本)
python batch_process_teeth.py --input_dir ./images --yolo_weights ./model.pt

# 运行处理 (完整)
python batch_process_teeth.py --input_dir ./images --yolo_weights ./yolo.pt --unet_weights ./unet.h5

# 查看帮助
python batch_process_teeth.py --help

# 退出环境
deactivate
```

祝您使用愉快！
