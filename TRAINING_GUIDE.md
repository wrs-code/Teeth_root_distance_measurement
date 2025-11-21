# YOLO牙齿检测模型训练完整指南

**严格按照Instance_seg_teeth原始仓库的训练流程**

本指南详细说明如何训练牙齿检测模型，完全基于原始仓库的训练notebook。

参考: `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb`

---

## 📋 目录

1. [快速开始](#快速开始)
2. [原始训练流程详解](#原始训练流程详解)
3. [使用自动化训练脚本](#使用自动化训练脚本)
4. [训练参数说明](#训练参数说明)
5. [数据集详情](#数据集详情)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

如果您只想快速开始训练，请查看 [TRAINING_QUICKSTART.md](TRAINING_QUICKSTART.md)

**最简单的方法：**
```bash
source venv/bin/activate
python train_teeth_yolo.py
```

---

## 📖 原始训练流程详解

本节详细说明原始notebook的每个步骤。

### 步骤1: 检查CUDA环境

**原始notebook代码：**
```python
import torch
torch.cuda.is_available()
```

**说明：** 验证PyTorch能够访问GPU

**我们的实现：**
```python
# train_teeth_yolo.py 中的 check_cuda() 函数
cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 步骤2: 设置工作目录

**原始notebook代码：**
```python
import os
HOME = os.getcwd()
print(HOME)
```

**说明：** 记录当前工作目录

### 步骤3: 安装ultralytics==8.0.28

**原始notebook代码：**
```python
!pip install ultralytics
!pip install ultralytics==8.0.28

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
```

**为什么使用8.0.28？**
- 这是原始仓库测试和验证过的版本
- 确保API兼容性
- 训练参数经过该版本验证

**我们的实现：**
```python
# train_teeth_yolo.py 中的 install_ultralytics() 函数
subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.0.28"], check=True)
```

### 步骤4: 导入YOLO模块

**原始notebook代码：**
```python
from ultralytics import YOLO
from IPython.display import display, Image
```

### 步骤5: 下载数据集（使用Roboflow）

**原始notebook代码：**
```python
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="XMzlZ50lfNikO4rN8iV2")
project = rf.workspace("teeth-segmentation").project("teeth-segmentation-evs6x")
dataset = project.version(15).download("yolov8")
```

**关键信息：**
- **Workspace**: teeth-segmentation
- **Project**: teeth-segmentation-evs6x
- **Version**: 15
- **Format**: yolov8
- **API Key**: XMzlZ50lfNikO4rN8iV2（内置于原始代码）

**数据集结构：**
```
datasets/Teeth-Segmentation-15/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

**我们的实现：**
```python
# train_teeth_yolo.py 中的 download_dataset() 函数
from roboflow import Roboflow
rf = Roboflow(api_key="XMzlZ50lfNikO4rN8iV2")
project = rf.workspace("teeth-segmentation").project("teeth-segmentation-evs6x")
dataset = project.version(15).download("yolov8")
```

### 步骤6: 训练模型

**原始notebook代码（完整的训练命令）：**
```bash
!yolo task=detect mode=train \
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

**我们的实现：**
```python
# train_teeth_yolo.py 中的 train_model() 函数
train_cmd = [
    "yolo",
    "task=detect",
    "mode=train",
    "model=yolov8x.pt",
    f"data={data_yaml}",
    "epochs=30",
    "batch=10",
    "imgsz=640",
    "cache=True",
    "single_cls=False",
    "val=True",
    "dropout=0.6",
    "close_mosaic=0",
    "cos_lr=True",
    "exist_ok=True",
    "warmup_epochs=10",
    "lrf=0.005"
]
subprocess.run(train_cmd, check=True)
```

---

## 🤖 使用自动化训练脚本

我们提供了自动化脚本，严格按照原始notebook的流程执行所有步骤。

### 方法1: Python脚本（推荐）

```bash
source venv/bin/activate
python train_teeth_yolo.py
```

**脚本会自动：**
1. ✅ 检查CUDA环境
2. ✅ 安装/验证ultralytics==8.0.28
3. ✅ 使用Roboflow下载数据集
4. ✅ 使用原始参数训练模型
5. ✅ 验证训练好的模型

### 方法2: Shell脚本

```bash
source venv/bin/activate
chmod +x train_teeth_yolo.sh
./train_teeth_yolo.sh
```

Shell脚本会调用Python脚本。

---

## ⚙️ 训练参数说明

以下参数与原始notebook完全相同：

### 基础参数

| 参数 | 值 | 说明 | 来源 |
|------|-----|------|------|
| `model` | yolov8x.pt | YOLOv8 Extra-Large模型 | 原始notebook |
| `task` | detect | 目标检测任务 | 原始notebook |
| `mode` | train | 训练模式 | 原始notebook |
| `data` | data.yaml | 数据集配置文件 | 原始notebook |
| `epochs` | 30 | 训练轮数 | 原始notebook |
| `batch` | 10 | 批次大小 | 原始notebook |
| `imgsz` | 640 | 图像尺寸 | 原始notebook |

### 训练策略参数

| 参数 | 值 | 说明 | 来源 |
|------|-----|------|------|
| `dropout` | 0.6 | Dropout正则化率 | ⭐ 原始notebook关键参数 |
| `close_mosaic` | 0 | 关闭Mosaic增强的轮数 | ⭐ 原始notebook关键参数 |
| `cos_lr` | True | 余弦学习率调度 | 原始notebook |
| `warmup_epochs` | 10 | 学习率预热轮数 | ⭐ 原始notebook关键参数 |
| `lrf` | 0.005 | 最终学习率因子 | ⭐ 原始notebook关键参数 |

### 其他参数

| 参数 | 值 | 说明 | 来源 |
|------|-----|------|------|
| `cache` | True | 缓存图像以加速训练 | 原始notebook |
| `single_cls` | False | 多类别检测（32类牙齿） | 原始notebook |
| `val` | True | 启用验证 | 原始notebook |
| `exist_ok` | True | 允许覆盖现有项目 | 原始notebook |

**为什么这些参数重要？**

这些参数是原始作者经过实验优化的结果，特别是：
- **dropout=0.6**: 针对医学图像的较高正则化
- **close_mosaic=0**: 整个训练过程都使用Mosaic数据增强
- **warmup_epochs=10**: 较长的预热期以稳定训练
- **lrf=0.005**: 较低的最终学习率以获得更好的收敛

---

## 📊 数据集详情

### UFBA-425数据集

**来源：** Roboflow（teeth-segmentation项目，版本15）

**原始数据集：** FigShare - UFBA-425

**内容：**
- 425张牙科全景X光片
- 32类牙齿（FDI编号系统）
- YOLOv8格式标注

**类别（32类）：**
```
0: Tooth 11 (上右中切牙)
1: Tooth 12 (上右侧切牙)
...
31: Tooth 48 (下左第三磨牙)
```

**训练/验证划分：**
- 训练集: ~340张图像
- 验证集: ~85张图像

### data.yaml配置

Roboflow下载的数据集包含自动生成的`data.yaml`：

```yaml
train: ../train/images
val: ../valid/images

nc: 32
names: ['11', '12', '13', '14', '15', '16', '17', '18',
        '21', '22', '23', '24', '25', '26', '27', '28',
        '31', '32', '33', '34', '35', '36', '37', '38',
        '41', '42', '43', '44', '45', '46', '47', '48']
```

---

## 🎯 训练结果

### 预期指标

根据原始仓库和类似研究：

- **训练时间**: 2-4小时 (RTX 3090)
- **mAP50**: >90% (可能达到94%)
- **mAP50-95**: >70% (可能达到74%)

### 输出文件

训练完成后，模型和结果保存在：

```
runs/detect/train/
├── weights/
│   ├── best.pt          # 最佳模型（验证集上最高mAP）
│   └── last.pt          # 最后一轮的模型
├── results.png          # 训练曲线
├── confusion_matrix.png # 混淆矩阵
├── F1_curve.png         # F1曲线
├── P_curve.png          # Precision曲线
├── R_curve.png          # Recall曲线
└── val_batch*.jpg       # 验证集预测可视化
```

### 使用训练好的模型

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./runs/detect/train/weights/best.pt
```

---

## ❓ 常见问题

### Q1: 为什么必须使用ultralytics==8.0.28？

**A**: 原始仓库使用该版本进行开发和测试，训练参数针对该版本优化。使用不同版本可能导致：
- API不兼容
- 训练行为差异
- 结果不一致

### Q2: 可以使用FigShare手动下载数据集吗？

**A**: 可以，但需要自己转换为YOLOv8格式。我们的脚本使用Roboflow以保持与原始仓库完全一致。Roboflow版本已经是正确的YOLOv8格式。

### Q3: 为什么使用dropout=0.6这么高？

**A**: 这是原始作者针对医学图像数据集的优化选择。医学图像通常样本较少，需要更强的正则化防止过拟合。

### Q4: 可以调整batch size吗？

**A**: 原始notebook使用batch=10。如果GPU内存不足，可以降低，但可能影响训练效果。如果GPU内存充足，可以增加。

### Q5: 训练中断了怎么办？

**A**: 使用last.pt继续训练：
```bash
yolo task=detect mode=train \
    model=runs/detect/train/weights/last.pt \
    data=datasets/Teeth-Segmentation-15/data.yaml \
    epochs=30 \
    # ... 其他参数相同
```

### Q6: 可以使用YOLOv11吗？

**A**: 可以，但需要修改配置。详见 [YOLOV11_GUIDE.md](YOLOV11_GUIDE.md)。
建议：
- **学习/复现原始工作** → 使用YOLOv8（本指南）
- **生产部署/追求最新性能** → 使用YOLOv11

### Q7: Roboflow API key会过期吗？

**A**: 该API key是原始仓库公开的key。如果失效，可以：
1. 注册Roboflow账号
2. Fork项目或创建新项目
3. 使用自己的API key

---

## 📚 参考资源

### 原始仓库
- **GitHub**: https://github.com/devichand579/Instance_seg_teeth.git
- **训练Notebook**: `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb`

### 数据集
- **FigShare**: https://figshare.com/articles/dataset/UFBA-425/29827475
- **Roboflow**: teeth-segmentation/teeth-segmentation-evs6x (version 15)

### YOLO文档
- **Ultralytics**: https://docs.ultralytics.com/
- **YOLOv8**: https://github.com/ultralytics/ultralytics

---

## 🔄 完整训练流程总结

1. **环境准备**
   ```bash
   source venv/bin/activate
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **运行训练脚本**
   ```bash
   python train_teeth_yolo.py
   ```

   脚本会自动执行：
   - ✅ CUDA检查
   - ✅ 安装ultralytics==8.0.28
   - ✅ 下载数据集（Roboflow）
   - ✅ 训练模型（原始参数）
   - ✅ 验证模型

3. **使用模型**
   ```bash
   python batch_process_teeth.py \
       --input_dir ./test_images \
       --yolo_weights ./runs/detect/train/weights/best.pt
   ```

---

## ✅ 验证训练是否成功

### 检查1: 模型文件存在
```bash
ls -lh runs/detect/train/weights/best.pt
# 应该看到约136MB的文件（YOLOv8x）
```

### 检查2: 查看训练曲线
```bash
eog runs/detect/train/results.png
```

应该看到：
- Loss曲线下降
- mAP曲线上升
- 无明显过拟合迹象

### 检查3: 验证指标
训练结束时应显示：
```
mAP50-95: >0.70
mAP50: >0.90
```

### 检查4: 快速推理测试
```bash
yolo predict \
    model=runs/detect/train/weights/best.pt \
    source=datasets/Teeth-Segmentation-15/valid/images \
    save=True \
    conf=0.5
```

检查 `runs/detect/predict/` 中的结果图像。

---

**本指南严格基于原始仓库，确保训练流程和结果的一致性！** 🎉

有问题？查看原始notebook或提issue。
