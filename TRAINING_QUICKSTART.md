# 训练快速开始 - 严格按照原始仓库流程

**本训练方案严格按照Instance_seg_teeth原始仓库的训练流程**

参考: `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb`

## 🚀 2步开始训练

### 1️⃣ 激活环境

```bash
# 激活虚拟环境
source venv/bin/activate

# 确认Python和依赖
python --version
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 2️⃣ 运行训练脚本（自动化全流程）

```bash
# 方法1: 使用Python脚本（推荐）
python train_teeth_yolo.py

# 方法2: 使用Shell脚本
chmod +x train_teeth_yolo.sh
./train_teeth_yolo.sh
```

**训练脚本会自动完成以下步骤（与原始notebook完全相同）：**

1. ✅ 检查CUDA环境
2. ✅ 安装ultralytics==8.0.28
3. ✅ 使用Roboflow API下载UFBA-425数据集（与原始仓库相同的配置）
   - Workspace: teeth-segmentation
   - Project: teeth-segmentation-evs6x
   - Version: 15
4. ✅ 使用原始notebook的完全相同的训练参数进行训练
5. ✅ 验证训练好的模型

### 3️⃣ 使用训练好的模型

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./runs/detect/train/weights/best.pt
```

## ✅ 完成！

训练好的模型在：`runs/detect/train/weights/best.pt`

---

## 📊 预期结果

- **训练时间**: 2-4小时 (RTX 3090)
- **训练参数**: 与原始notebook完全相同
  - Model: yolov8x.pt
  - Epochs: 30
  - Batch: 10
  - Dropout: 0.6
  - Close Mosaic: 0
  - Cosine LR: True
  - Warmup Epochs: 10
  - LRF: 0.005

## 🔧 训练配置（与原始仓库一致）

训练脚本使用与原始notebook完全相同的命令：

```bash
yolo task=detect mode=train \
    model=yolov8x.pt \
    data={dataset.location}/data.yaml \
    epochs=30 \
    batch=10 \
    imgsz=640 \
    cache=True \
    single_cls=False \
    val=True \
    dropout=0.6 \
    close_mosaic=0 \
    cos_lr=True \
    exist_ok=True \
    warmup_epochs=10 \
    lrf=0.005
```

## 📁 数据集下载（使用Roboflow）

训练脚本会自动使用Roboflow API下载数据集，配置与原始仓库完全相同：

```python
from roboflow import Roboflow
rf = Roboflow(api_key="XMzlZ50lfNikO4rN8iV2")
project = rf.workspace("teeth-segmentation").project("teeth-segmentation-evs6x")
dataset = project.version(15).download("yolov8")
```

## ⚠️ 常见问题

| 问题 | 解决方案 |
|------|---------|
| CUDA不可用 | 检查CUDA 11.8安装，运行 `nvidia-smi` |
| Roboflow下载失败 | 训练脚本使用内置API key，无需手动配置 |
| 训练中断 | 使用 `runs/detect/train/weights/last.pt` 继续训练 |

## 📖 详细文档

完整训练指南: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## 🎯 与原始仓库的对比

| 组件 | 原始notebook | 本训练脚本 | 状态 |
|------|------------|-----------|------|
| ultralytics版本 | 8.0.28 | 8.0.28 | ✅ 完全相同 |
| 数据下载方式 | Roboflow API | Roboflow API | ✅ 完全相同 |
| 训练参数 | dropout=0.6等 | dropout=0.6等 | ✅ 完全相同 |
| 模型 | yolov8x.pt | yolov8x.pt | ✅ 完全相同 |
| 数据集版本 | version 15 | version 15 | ✅ 完全相同 |

---

**严格按照原始仓库流程，保证训练结果一致！** 🎉

问题？查看原始notebook: `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb`
