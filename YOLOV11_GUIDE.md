# YOLOv11 使用指南

## YOLOv8 vs YOLOv11

### YOLOv8 (当前配置)
- ✅ 开源仓库Instance_seg_teeth使用的版本
- ✅ 稳定，经过测试
- ✅ `ultralytics==8.0.28` (2023年版本)
- ✅ 与原始训练代码完全兼容

### YOLOv11 (可选升级)
- ✅ 最新版本，性能更好
- ✅ 架构改进，精度提升
- ⚠️ 需要升级ultralytics到最新版
- ⚠️ 可能与原始notebook代码不完全兼容

## 如何使用YOLOv11

### 选项1: 仅在批量处理中使用YOLOv11（推荐）

**步骤1: 创建新的虚拟环境**

```bash
# 创建YOLOv11专用环境
python3 -m venv venv_yolov11
source venv_yolov11/bin/activate

# 安装YOLOv11依赖
pip install -r requirements_yolov11.txt
```

**步骤2: 下载YOLOv11预训练模型**

```bash
# 使用Python下载
python << EOF
from ultralytics import YOLO
# 自动下载YOLOv11x模型
model = YOLO('yolo11x.pt')
print("YOLOv11x模型下载完成！")
EOF
```

或手动下载：
```bash
# YOLOv11n (nano - 最快)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# YOLOv11s (small)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

# YOLOv11m (medium)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# YOLOv11l (large)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

# YOLOv11x (extra large - 最准确)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

**步骤3: 训练YOLOv11在牙齿数据集上**

```bash
# 使用命令行训练
yolo task=detect mode=train \
    model=yolo11x.pt \
    data=./path/to/data.yaml \
    epochs=30 \
    batch=10 \
    imgsz=640

# 训练完成后，模型在: runs/detect/train/weights/best.pt
```

**步骤4: 使用批量处理**

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./runs/detect/train/weights/best.pt
```

### 选项2: 在原始notebook中使用YOLOv11

⚠️ **不推荐**，因为可能有兼容性问题。

如果坚持要用：

```bash
# 在原有环境中升级ultralytics
pip install --upgrade ultralytics

# 检查版本
python -c "import ultralytics; print(ultralytics.__version__)"
```

修改notebook中的代码：
```python
# 原代码（YOLOv8）
model = YOLO('yolov8x.pt')

# 改为（YOLOv11）
model = YOLO('yolo11x.pt')
```

## YOLOv11性能对比

### 模型大小和速度

| 模型 | 参数量 | 速度 (ms) | mAP50 | 推荐用途 |
|-----|-------|----------|-------|---------|
| YOLOv11n | 2.6M | 1.5 | ~39% | 实时检测 |
| YOLOv11s | 9.4M | 2.5 | ~47% | 平衡 |
| YOLOv11m | 20.1M | 4.5 | ~51% | 高精度 |
| YOLOv11l | 25.3M | 6.0 | ~53% | 高精度 |
| YOLOv11x | 56.9M | 12.0 | ~54% | 最高精度 |

### YOLOv8 vs YOLOv11 (COCO数据集)

| 指标 | YOLOv8x | YOLOv11x | 提升 |
|-----|---------|----------|------|
| mAP50-95 | 53.9% | 54.7% | +0.8% |
| 速度 | 更快 | 稍慢 | - |
| 精度 | 高 | 更高 | ✅ |

## 推荐配置

### 对于生产环境

**推荐使用YOLOv11**：
```bash
# 1. 使用YOLOv11环境
source venv_yolov11/bin/activate

# 2. 训练YOLOv11模型
yolo task=detect mode=train \
    model=yolo11x.pt \
    data=./data.yaml \
    epochs=30 \
    batch=10 \
    imgsz=640

# 3. 批量处理
python batch_process_teeth.py \
    --input_dir ./images \
    --yolo_weights ./runs/detect/train/weights/best.pt
```

### 对于学习和复现论文

**推荐使用YOLOv8**（与原始代码一致）：
```bash
# 使用原始环境
source venv/bin/activate

# 按照原始notebook训练
jupyter notebook Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb
```

## 快速测试YOLOv11

### 1. 创建测试脚本

```bash
cat > test_yolov11.py << 'EOF'
from ultralytics import YOLO
import torch

print("=" * 50)
print("YOLOv11测试")
print("=" * 50)

# 检查CUDA
print(f"\nCUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 加载YOLOv11模型（自动下载）
print("\n加载YOLOv11x模型...")
model = YOLO('yolo11x.pt')

print("\n✅ YOLOv11配置成功!")
print(f"模型类型: {type(model)}")
print("\n您现在可以使用YOLOv11进行训练和推理。")
EOF

python test_yolov11.py
```

### 2. 快速推理测试

```bash
# 使用YOLOv11检测（通用模型，不认识牙齿）
yolo predict model=yolo11x.pt source=./test_images conf=0.5
```

## 兼容性说明

### ✅ 兼容的组件
- batch_process_teeth.py - 完全兼容YOLOv11
- PyTorch CUDA 11.8 - 兼容
- TensorFlow 2.13 - 兼容

### ⚠️ 需要注意的
- 原始训练notebooks - 可能需要小幅修改
- UNet部分 - 不受影响
- 依赖版本 - 使用requirements_yolov11.txt

## 常见问题

### Q1: YOLOv11比YOLOv8好多少？
**A**: 在COCO数据集上提升约0.8% mAP，对于牙齿检测具体提升需要实验验证。

### Q2: 可以同时安装YOLOv8和YOLOv11吗？
**A**: 不建议在同一环境，建议创建两个虚拟环境：
- `venv` - YOLOv8（原始配置）
- `venv_yolov11` - YOLOv11（升级版）

### Q3: 训练时间会变化吗？
**A**: YOLOv11通常稍慢一些，但精度更高。

### Q4: 我应该用哪个版本？
**A**:
- **复现论文/学习** → YOLOv8（与原始代码一致）
- **生产部署/追求性能** → YOLOv11（最新最好）

## 总结

### YOLOv8 (推荐用于学习)
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### YOLOv11 (推荐用于生产)
```bash
python3 -m venv venv_yolov11
source venv_yolov11/bin/activate
pip install -r requirements_yolov11.txt
```

两个版本都可以完美运行batch_process_teeth.py！
