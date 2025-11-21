# YOLOv8 牙齿检测模型训练完整指南

本指南严格遵循 [Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth) 开源仓库的官方训练流程，确保复现论文中报告的最佳效果。

## 📊 预期训练效果

根据论文 [OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors](https://arxiv.org/abs/2406.03747)：

- **mAP**: 74.9%
- **AP50**: 94.6%
- **训练时间**: 2-4小时 (RTX 3090)

## 🎯 训练方案概述

Instance_seg_teeth 仓库提供了三种主要方案：

| 方案 | 模型 | mAP | AP50 | 推荐场景 |
|-----|------|-----|------|---------|
| **YOLOv8检测** | YOLOv8x | 74.9% | 94.6% | ✅ 牙齿编号、定位（最佳） |
| UNet分割 | UNet | ~68% | - | 像素级分割 |
| OralBBNet | YOLOv8+UNet | 88.4% | - | 实例分割（最复杂） |

**本指南专注于YOLOv8检测方案**，因为：
1. ✅ 效果最好（mAP 74.9%，在5种模型中排名第一）
2. ✅ 训练最简单（单阶段检测器）
3. ✅ 推理速度快（适合实时应用）
4. ✅ 已有完整数据集和配置

## 📁 项目结构

```
Teeth_root_distance_measurement/
├── Instance_seg_teeth/                # 官方仓库（已克隆）
│   ├── notebooks/
│   │   └── yolov8/
│   │       ├── yolov8_train.ipynb    # 官方训练notebook
│   │       └── yolo_test.ipynb       # 官方测试notebook
│   └── Dataset/
│       └── yolo_train_dataset/        # UFBA-425数据集（Roboflow版本）
│           ├── data.yaml              # 原始配置
│           ├── train/                 # 894张训练图像
│           │   ├── images/
│           │   └── labels/
│           ├── valid/                 # 64张验证图像
│           │   ├── images/
│           │   └── labels/
│           └── test/                  # 64张测试图像
│               ├── images/
│               └── labels/
│
├── yolo_training_config.yaml          # 训练配置（绝对路径）
├── train_yolov8_teeth.py             # Python训练脚本
├── train_yolov8.sh                   # Bash训练脚本
└── batch_process_teeth.py            # 批量推理脚本
```

## 🔧 环境准备

### 1. 系统要求

- **CUDA**: 11.8 (推荐)
- **Python**: 3.8 - 3.10
- **GPU**: NVIDIA GPU with ≥8GB VRAM (推荐RTX 3060或更好)
- **RAM**: 16GB+ 推荐

### 2. 安装依赖

```bash
# 激活虚拟环境（如果已创建）
source venv/bin/activate

# 安装YOLOv8（使用官方推荐版本）
pip install ultralytics==8.0.28

# 验证安装
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 3. 验证CUDA

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU
nvidia-smi

# 验证PyTorch CUDA支持
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 📦 数据集说明

### UFBA-425 数据集（Roboflow版本）

- **来源**: Roboflow (teeth-segmentation-evs6x v15)
- **总图像数**: 1022张（包含数据增强）
  - 训练集: 894张
  - 验证集: 64张
  - 测试集: 64张

- **类别数**: 32类（FDI牙齿编号系统）
  - 11-18: 右上象限（8颗牙）
  - 21-28: 左上象限（8颗牙）
  - 31-38: 左下象限（8颗牙）
  - 41-48: 右下象限（8颗牙）

- **预处理**:
  - Resize到640x640（拉伸）
  - 直方图均衡化（对比度增强）

- **数据增强**（每张原始图像生成3个版本）:
  - 随机裁剪：0-20%
  - 随机亮度调整：0-10%

### 标签格式

YOLOv8标准格式（每行一个检测框）：
```
class_id x_center y_center width height
```

所有坐标都是归一化的（0-1之间）。

示例：
```
7 0.2140625 0.2578125 0.06015625 0.10859375
6 0.23515625 0.34375 0.0625 0.20390625
```

## 🚀 开始训练

### 方法1：使用Python脚本（推荐）

**基础训练（使用官方参数）:**
```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml
```

**查看所有选项:**
```bash
python train_yolov8_teeth.py --help
```

**自定义训练:**
```bash
# 训练50个epochs
python train_yolov8_teeth.py --data yolo_training_config.yaml --epochs 50

# 使用更大的batch size（需要更多显存）
python train_yolov8_teeth.py --data yolo_training_config.yaml --batch 16

# 使用较小的模型（更快但精度稍低）
python train_yolov8_teeth.py --data yolo_training_config.yaml --model yolov8l.pt

# 恢复中断的训练
python train_yolov8_teeth.py --data yolo_training_config.yaml --resume

# CPU训练（非常慢，不推荐）
python train_yolov8_teeth.py --data yolo_training_config.yaml --device cpu
```

### 方法2：使用Bash脚本（快速启动）

```bash
# 一键启动训练（使用官方配置）
./train_yolov8.sh
```

### 方法3：使用YOLO CLI（命令行）

```bash
# 直接使用YOLO命令行（与官方notebook完全一致）
yolo task=detect mode=train \
    model=yolov8x.pt \
    data=yolo_training_config.yaml \
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

## ⚙️ 官方训练参数详解

根据Instance_seg_teeth官方notebook，以下是训练参数及其作用：

| 参数 | 值 | 说明 |
|-----|---|------|
| `model` | `yolov8x.pt` | 使用最大的YOLOv8模型（81M参数），精度最高 |
| `epochs` | `30` | 训练30轮（在论文中取得最佳效果） |
| `batch` | `10` | 批量大小10（适配大多数GPU显存） |
| `imgsz` | `640` | 图像大小640x640（与数据集预处理一致） |
| `dropout` | `0.6` | Dropout率60%（防止过拟合） |
| `warmup_epochs` | `10` | 前10个epoch进行学习率预热 |
| `lrf` | `0.005` | 最终学习率因子（学习率衰减到初始的0.5%） |
| `cos_lr` | `True` | 使用余弦学习率调度器 |
| `cache` | `True` | 缓存图像到内存（加速训练） |
| `close_mosaic` | `0` | 在epoch 0关闭mosaic增强 |
| `single_cls` | `False` | 多类别检测（32类牙齿） |
| `val` | `True` | 训练过程中验证 |

### 为什么选择这些参数？

1. **yolov8x**: 最大模型，在牙齿检测任务中取得74.9% mAP，优于所有其他模型
2. **epochs=30**: 经过实验验证的最佳训练轮数，平衡了效果和训练时间
3. **batch=10**: 在8-16GB显存的GPU上稳定运行
4. **dropout=0.6**: 较高的dropout率，因为牙齿数据集相对较小（894张训练图）
5. **warmup_epochs=10**: 长预热期帮助模型稳定收敛

## 📈 训练过程监控

### 训练输出

训练过程中会实时显示：
```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss  Instances       Size
  1/30      5.12G      1.234      2.345      1.123        280        640
```

- **box_loss**: 边界框定位损失
- **cls_loss**: 分类损失
- **dfl_loss**: 分布焦点损失
- **Instances**: 检测到的实例数
- **mAP50**: 在IoU=0.5时的平均精度
- **mAP50-95**: IoU从0.5到0.95的平均精度

### 查看训练结果

训练完成后，结果保存在 `runs/detect/teeth_detection_TIMESTAMP/`：

```bash
runs/detect/teeth_detection_20251121_123456/
├── weights/
│   ├── best.pt              # 最佳模型（验证集mAP最高）
│   └── last.pt              # 最后一个epoch的模型
├── results.png              # 训练曲线（loss、mAP等）
├── confusion_matrix.png     # 混淆矩阵
├── F1_curve.png            # F1曲线
├── P_curve.png             # 精确率曲线
├── R_curve.png             # 召回率曲线
├── PR_curve.png            # PR曲线
└── args.yaml               # 训练参数记录
```

### 使用TensorBoard（可选）

```bash
# 启动TensorBoard
tensorboard --logdir runs/detect

# 在浏览器中打开 http://localhost:6006
```

## 🧪 训练后测试

### 方法1：YOLO CLI快速测试

```bash
# 使用最佳模型进行预测
yolo predict \
    model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    source=./test_images \
    conf=0.5 \
    save=True
```

### 方法2：使用批量处理脚本

```bash
python batch_process_teeth.py \
    --yolo_weights runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    --input_dir ./test_images \
    --output_dir ./output
```

### 方法3：在测试集上评估

```bash
# 在测试集上评估模型性能
yolo val \
    model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    data=yolo_training_config.yaml \
    split=test
```

## 🎓 不同YOLOv8模型对比

如果GPU显存有限或需要更快的推理速度，可以考虑使用较小的模型：

| 模型 | 参数量 | 显存占用 | 推理速度 | 预期mAP | 训练时间 |
|------|--------|---------|---------|---------|---------|
| yolov8n | 3.2M | ~2GB | 最快 | ~60% | 0.5-1h |
| yolov8s | 11.2M | ~3GB | 很快 | ~65% | 1-1.5h |
| yolov8m | 25.9M | ~5GB | 快 | ~70% | 1.5-2h |
| yolov8l | 43.7M | ~7GB | 中等 | ~73% | 2-3h |
| **yolov8x** | **81.0M** | **~10GB** | **较慢** | **74.9%** | **2-4h** |

**推荐**:
- 📊 **研究/复现论文**: 使用 `yolov8x.pt`（官方配置）
- ⚡ **快速实验**: 使用 `yolov8l.pt`（性能接近，速度更快）
- 💻 **显存受限**: 使用 `yolov8m.pt` 或 `yolov8s.pt`
- 📱 **嵌入式设备**: 使用 `yolov8n.pt`

修改模型很简单：
```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml --model yolov8l.pt
```

## 🔧 常见问题排查

### 1. CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小batch size
python train_yolov8_teeth.py --data yolo_training_config.yaml --batch 4

# 或使用较小的模型
python train_yolov8_teeth.py --data yolo_training_config.yaml --model yolov8l.pt --batch 8
```

### 2. 训练很慢

**可能原因**:
- ❌ 没有使用GPU
- ❌ cache=False（没有缓存图像）
- ❌ 使用了CPU

**检查**:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. mAP很低

**可能原因**:
- 训练epochs不足（< 30）
- 学习率不合适
- 数据集问题

**建议**:
```bash
# 延长训练
python train_yolov8_teeth.py --data yolo_training_config.yaml --epochs 50

# 查看训练曲线，确认收敛
```

### 4. 训练中断

**恢复训练**:
```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml --resume
```

### 5. 数据集路径错误

**症状**: `FileNotFoundError` 或 `Dataset not found`

**检查**:
```bash
# 验证数据集存在
ls Instance_seg_teeth/Dataset/yolo_train_dataset/train/images/ | head -5

# 检查配置文件
cat yolo_training_config.yaml
```

## 📊 预期训练曲线

### 正常的训练曲线应该显示:

1. **Loss曲线**:
   - box_loss: 从 ~1.5 降到 ~0.8
   - cls_loss: 从 ~2.0 降到 ~0.5
   - dfl_loss: 从 ~1.2 降到 ~0.9

2. **mAP曲线**:
   - mAP50: 从 ~0.5 升到 ~0.94
   - mAP50-95: 从 ~0.3 升到 ~0.75

3. **学习率**:
   - 前10个epoch预热（从0逐渐增加）
   - 然后余弦衰减（平滑下降到接近0）

### 如果出现异常:

- **Loss不下降**: 学习率可能太小，或数据有问题
- **Loss震荡**: batch size可能太小，或学习率太大
- **mAP不提升**: 可能需要更多epochs，或模型欠拟合

## 🎯 训练完成后的下一步

### 1. 模型评估

```bash
# 在测试集上评估
yolo val model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    data=yolo_training_config.yaml \
    split=test

# 查看详细指标
cat runs/detect/teeth_detection_TIMESTAMP/results.txt
```

### 2. 批量推理

```bash
# 处理整个文件夹的图像
python batch_process_teeth.py \
    --yolo_weights runs/detect/teeth_detection_TIMESTAMP/weights/best.pt \
    --input_dir ./patient_xrays \
    --output_dir ./results
```

### 3. 导出模型

```bash
# 导出为ONNX格式（更快的推理）
yolo export model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt format=onnx

# 导出为TensorRT格式（GPU加速）
yolo export model=runs/detect/teeth_detection_TIMESTAMP/weights/best.pt format=engine
```

### 4. 模型优化

如果需要进一步提升性能：

```bash
# 使用更多epochs
python train_yolov8_teeth.py --data yolo_training_config.yaml --epochs 50

# 调整学习率
yolo train model=yolov8x.pt data=yolo_training_config.yaml epochs=30 lr0=0.01

# 使用测试时增强（TTA）
yolo val model=best.pt data=yolo_training_config.yaml augment=True
```

## 📚 参考资料

### 论文和数据集

- **论文**: [OralBBNet: Spatially Guided Dental Segmentation](https://arxiv.org/abs/2406.03747)
- **数据集**: [UFBA-425 on FigShare](https://figshare.com/articles/dataset/UFBA-425/29827475)
- **开源代码**: [Instance_seg_teeth GitHub](https://github.com/devichand579/Instance_seg_teeth)

### YOLOv8文档

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [YOLOv8 Detection Tasks](https://docs.ultralytics.com/tasks/detect/)

### FDI牙齿编号系统

```
        右上象限              左上象限
    18 17 16 15 14 13 12 11 | 21 22 23 24 25 26 27 28
    ─────────────────────────┼─────────────────────────
    48 47 46 45 44 43 42 41 | 31 32 33 34 35 36 37 38
        右下象限              左下象限
```

## 🆘 获取帮助

### 本项目问题

如遇到问题，请：
1. 检查本文档的"常见问题排查"部分
2. 查看训练日志和错误信息
3. 在GitHub Issues中搜索类似问题

### Instance_seg_teeth原始问题

访问: https://github.com/devichand579/Instance_seg_teeth/issues

### Ultralytics YOLOv8问题

访问: https://github.com/ultralytics/ultralytics/issues

## 📝 训练检查清单

开始训练前，确认：

- [ ] CUDA 11.8已安装并可用
- [ ] ultralytics==8.0.28已安装
- [ ] Instance_seg_teeth仓库已克隆
- [ ] 数据集存在：Instance_seg_teeth/Dataset/yolo_train_dataset/
- [ ] yolo_training_config.yaml文件存在
- [ ] GPU显存 ≥ 8GB (对于yolov8x + batch=10)
- [ ] 硬盘空间足够（至少10GB用于模型和结果）
- [ ] 虚拟环境已激活（如果使用）

训练期间监控：

- [ ] GPU利用率 > 90% (使用 `nvidia-smi` 检查)
- [ ] Loss持续下降
- [ ] mAP持续上升
- [ ] 没有CUDA OOM错误
- [ ] 训练时间合理（2-4小时左右）

训练完成后验证：

- [ ] best.pt文件存在
- [ ] mAP接近74.9%（±2%是正常的）
- [ ] AP50接近94.6%
- [ ] 训练曲线正常（无异常震荡）
- [ ] 在测试图像上效果良好

## ✅ 总结

本指南提供了基于Instance_seg_teeth官方仓库的完整YOLOv8训练方案：

1. ✅ **数据集**: UFBA-425 (1022张图像，32类牙齿)
2. ✅ **模型**: YOLOv8x (81M参数)
3. ✅ **训练参数**: 严格遵循官方配置
4. ✅ **预期效果**: mAP=74.9%, AP50=94.6%
5. ✅ **训练时间**: 2-4小时 (现代GPU)

**开始训练**:
```bash
python train_yolov8_teeth.py --data yolo_training_config.yaml
```

**祝训练顺利！** 🚀
