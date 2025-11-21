#!/usr/bin/env python3
"""
YOLO牙齿检测模型训练脚本
严格按照Instance_seg_teeth原始仓库的训练流程
参考: Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb
"""

import os
import sys
import subprocess
import torch

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

def check_cuda():
    """步骤1: 检查CUDA可用性"""
    print_header("步骤 1/5: 检查CUDA环境")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")

    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: CUDA不可用，训练将非常慢！")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("✓ CUDA检查完成")

def install_ultralytics():
    """步骤2: 安装ultralytics==8.0.28（与原始仓库一致）"""
    print_header("步骤 2/5: 安装ultralytics==8.0.28")

    try:
        import ultralytics
        current_version = ultralytics.__version__
        print(f"检测到ultralytics版本: {current_version}")

        if current_version != "8.0.28":
            print("⚠️ 版本不匹配，需要安装8.0.28以保持与原始仓库一致")
            response = input("是否重新安装? (y/n): ")
            if response.lower() == 'y':
                print("正在安装ultralytics==8.0.28...")
                subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.0.28"], check=True)
        else:
            print("✓ ultralytics版本正确")
    except ImportError:
        print("ultralytics未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.0.28"], check=True)

    # 验证安装
    from ultralytics import YOLO
    print("✓ ultralytics安装完成")

def download_dataset():
    """步骤3: 使用Roboflow API下载数据集（与原始仓库完全相同）"""
    print_header("步骤 3/5: 下载UFBA-425数据集（使用Roboflow）")

    HOME = os.getcwd()
    datasets_dir = os.path.join(HOME, "datasets")

    # 检查数据集是否已存在
    dataset_path = os.path.join(datasets_dir, "Teeth-Segmentation-15")
    if os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, "data.yaml")):
        print(f"✓ 数据集已存在: {dataset_path}")
        return dataset_path

    # 创建datasets目录
    os.makedirs(datasets_dir, exist_ok=True)
    os.chdir(datasets_dir)

    print("正在安装roboflow...")
    subprocess.run([sys.executable, "-m", "pip", "install", "roboflow"], check=True)

    print("\n正在使用Roboflow API下载数据集...")
    print("使用原始仓库的API key和项目配置:")
    print("  - Workspace: teeth-segmentation")
    print("  - Project: teeth-segmentation-evs6x")
    print("  - Version: 15")
    print("\n这是原始notebook使用的完全相同的数据集配置。\n")

    # 使用原始notebook的Roboflow配置
    from roboflow import Roboflow
    rf = Roboflow(api_key="XMzlZ50lfNikO4rN8iV2")
    project = rf.workspace("teeth-segmentation").project("teeth-segmentation-evs6x")
    dataset = project.version(15).download("yolov8")

    os.chdir(HOME)

    print(f"✓ 数据集下载完成: {dataset.location}")
    return dataset.location

def train_model(dataset_location):
    """步骤4: 训练模型（使用原始notebook的完全相同的参数）"""
    print_header("步骤 4/5: 训练YOLOv8x模型")

    print("训练配置（与原始notebook完全相同）:")
    print("  - 模型: yolov8x.pt")
    print("  - 数据集: " + dataset_location + "/data.yaml")
    print("  - Epochs: 30")
    print("  - Batch: 10")
    print("  - Image Size: 640")
    print("  - Cache: True")
    print("  - Single Class: False")
    print("  - Dropout: 0.6")
    print("  - Close Mosaic: 0")
    print("  - Cosine LR: True")
    print("  - Warmup Epochs: 10")
    print("  - LRF: 0.005")
    print("\n这些参数与原始仓库的notebook完全一致。\n")

    response = input("确认开始训练? 这将需要2-4小时。(y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        sys.exit(0)

    # 构建训练命令（与原始notebook完全相同）
    data_yaml = os.path.join(dataset_location, "data.yaml")

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

    print("\n执行训练命令:")
    print(" ".join(train_cmd))
    print("\n")

    # 执行训练
    try:
        subprocess.run(train_cmd, check=True)
        print("\n✓ 训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return False

def validate_model():
    """步骤5: 验证训练好的模型"""
    print_header("步骤 5/5: 验证训练好的模型")

    best_model_path = "runs/detect/train/weights/best.pt"
    last_model_path = "runs/detect/train/weights/last.pt"

    if not os.path.exists(best_model_path):
        print("✗ 未找到训练好的模型")
        return False

    print(f"✓ 找到训练好的模型:")
    print(f"  - 最佳模型: {best_model_path}")
    print(f"  - 最终模型: {last_model_path}")

    print("\n正在验证最佳模型...")

    from ultralytics import YOLO

    try:
        model = YOLO(best_model_path)
        metrics = model.val()

        print("\n验证指标:")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP75: {metrics.box.map75:.4f}")

        print("\n✓ 模型验证完成！")
        return True
    except Exception as e:
        print(f"\n✗ 模型验证失败: {e}")
        return False

def main():
    """主函数 - 严格按照原始notebook的流程"""
    print_header("YOLO牙齿检测模型训练")
    print("严格按照Instance_seg_teeth原始仓库的训练流程")
    print("参考: Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb")

    try:
        # 步骤1: 检查CUDA
        check_cuda()

        # 步骤2: 安装ultralytics
        install_ultralytics()

        # 步骤3: 下载数据集
        dataset_location = download_dataset()

        # 步骤4: 训练模型
        success = train_model(dataset_location)

        if not success:
            print("\n训练未完成")
            sys.exit(1)

        # 步骤5: 验证模型
        validate_model()

        # 完成
        print_header("训练流程完成！")
        print("训练结果:")
        print("  - 模型权重: runs/detect/train/weights/")
        print("  - 训练日志: runs/detect/train/")
        print("  - 训练曲线: runs/detect/train/results.png")
        print("\n下一步:")
        print("  1. 查看训练曲线:")
        print("     eog runs/detect/train/results.png")
        print("\n  2. 使用训练好的模型进行批量处理:")
        print("     python batch_process_teeth.py \\")
        print("         --input_dir ./test_images \\")
        print("         --yolo_weights ./runs/detect/train/weights/best.pt")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
