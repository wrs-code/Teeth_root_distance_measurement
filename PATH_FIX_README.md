# 路径问题修复说明

## 问题
之前的配置文件使用了硬编码的绝对路径 `/home/user/...`，导致在其他用户目录下无法使用。

## 解决方案

我们提供了两种方法：

### 方法1：自动生成配置（推荐）✅

训练脚本 `train_yolov8.sh` 已更新，会自动为你的环境生成正确的配置文件。

**直接运行即可**：
```bash
./train_yolov8.sh
```

脚本会自动：
1. 检测你的当前目录
2. 生成带有正确绝对路径的配置文件
3. 开始训练

### 方法2：手动生成配置

如果需要手动生成配置文件：

```bash
python generate_config.py
```

这会生成 `yolo_training_config.yaml`，包含你系统的正确路径。

## 使用说明

### 前提条件

确保你在项目根目录：
```bash
cd /home/wrs-kc/Teeth_root_distance_measurement
```

### 开始训练

```bash
# 方法1: 使用训练脚本（会自动生成配置）
./train_yolov8.sh

# 方法2: 使用Python脚本（需要先生成配置）
python generate_config.py
python train_yolov8_teeth.py --data yolo_training_config.yaml

# 方法3: 直接使用YOLO CLI（需要先生成配置）
python generate_config.py
yolo task=detect mode=train model=yolov8x.pt data=yolo_training_config.yaml epochs=30 batch=10
```

## 验证配置

检查配置是否正确：

```bash
python generate_config.py
cat yolo_training_config.yaml
```

应该看到类似这样的路径：
```yaml
path: /home/wrs-kc/Teeth_root_distance_measurement/Instance_seg_teeth/Dataset/yolo_train_dataset
```

## 故障排除

### 如果仍然报错"Dataset not found"

1. **检查当前目录**：
   ```bash
   pwd
   # 应该输出: /home/wrs-kc/Teeth_root_distance_measurement
   ```

2. **检查数据集是否存在**：
   ```bash
   ls Instance_seg_teeth/Dataset/yolo_train_dataset/
   # 应该看到: train/ valid/ test/ data.yaml
   ```

3. **手动生成配置并验证**：
   ```bash
   python generate_config.py
   cat yolo_training_config.yaml
   ```

4. **检查路径是否正确**：
   配置文件中的 `path:` 应该指向你的实际数据集位置

### 如果 generate_config.py 不存在

从仓库重新拉取：
```bash
git pull origin claude/fix-yolo-training-01WxZfccetxZHTmmAVSKTxVG
```

## 技术细节

- `generate_config.py` 使用 Python 的 `os.path.abspath(__file__)` 来获取当前脚本的绝对路径
- 然后根据这个路径构建数据集的绝对路径
- 这确保了无论在哪个用户目录下，配置都是正确的

## 需要帮助？

如果问题仍未解决，请提供以下信息：
```bash
pwd
ls -la Instance_seg_teeth/Dataset/yolo_train_dataset/
cat yolo_training_config.yaml
```
