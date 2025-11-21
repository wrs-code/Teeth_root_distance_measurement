#!/usr/bin/env python3
"""
YOLOv8 Teeth Detection Training Script
Based on Instance_seg_teeth official training notebook
Dataset: UFBA-425 (Roboflow version)

This script strictly follows the training configuration from the official repository:
https://github.com/devichand579/Instance_seg_teeth

Training achieves mAP=74.9, AP50=94.6 as reported in the paper.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime


def check_environment():
    """Check CUDA and environment setup"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        response = input("Continue with CPU training? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)

    # Check ultralytics version
    import ultralytics
    print(f"Ultralytics Version: {ultralytics.__version__}")
    if ultralytics.__version__ != "8.0.28":
        print(f"WARNING: Recommended ultralytics version is 8.0.28, you have {ultralytics.__version__}")
        print("For best results matching the paper, use: pip install ultralytics==8.0.28")

    print("=" * 70)
    print()


def train_yolov8(
    data_yaml: str,
    model: str = "yolov8x.pt",
    epochs: int = 30,
    batch: int = 10,
    imgsz: int = 640,
    project: str = "runs/detect",
    name: str = "teeth_detection",
    resume: bool = False,
    device: str = "0",
):
    """
    Train YOLOv8 model on teeth detection dataset

    Parameters match the official training configuration:
    - model: yolov8x.pt (largest YOLOv8 model for best accuracy)
    - epochs: 30
    - batch: 10
    - imgsz: 640
    - dropout: 0.6
    - warmup_epochs: 10
    - lrf: 0.005
    - cos_lr: True (cosine learning rate)
    - cache: True (cache images for faster training)
    """

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Data Config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device}")
    print(f"Project: {project}")
    print(f"Name: {name}")
    print(f"Resume: {resume}")
    print("=" * 70)
    print()

    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data config file not found: {data_yaml}")

    # Initialize model
    print("Loading model...")
    yolo_model = YOLO(model)

    # Start training with official parameters
    print("\nStarting training...")
    print("This will take several hours depending on your GPU.")
    print("Training parameters follow the official Instance_seg_teeth configuration.")
    print()

    start_time = time.time()

    # Train with exact parameters from official notebook
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,

        # Official training hyperparameters
        cache=True,              # Cache images for faster training
        single_cls=False,        # Multi-class detection (32 teeth classes)
        val=True,                # Validate during training
        dropout=0.6,             # Dropout rate
        close_mosaic=0,          # Close mosaic augmentation at epoch 0
        cos_lr=True,             # Use cosine learning rate scheduler
        warmup_epochs=10,        # Warmup for 10 epochs
        lrf=0.005,               # Final learning rate factor

        # Resume training if specified
        resume=resume,
    )

    end_time = time.time()
    training_time = end_time - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Training Time: {training_time / 3600:.2f} hours")
    print(f"Best Model: {project}/{name}/weights/best.pt")
    print(f"Last Model: {project}/{name}/weights/last.pt")
    print(f"Results: {project}/{name}/")
    print()
    print("Expected Performance (from paper):")
    print("  - mAP: 74.9%")
    print("  - AP50: 94.6%")
    print()
    print("To use the trained model for inference:")
    print(f"  yolo predict model={project}/{name}/weights/best.pt source=./test_images conf=0.5")
    print()
    print("Or use the batch processing script:")
    print(f"  python batch_process_teeth.py --yolo_weights {project}/{name}/weights/best.pt --input_dir ./test_images")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Teeth Detection (Instance_seg_teeth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default parameters
  python train_yolov8_teeth.py --data yolo_training_config.yaml

  # Training with custom epochs and batch size
  python train_yolov8_teeth.py --data yolo_training_config.yaml --epochs 50 --batch 16

  # Resume training from checkpoint
  python train_yolov8_teeth.py --data yolo_training_config.yaml --resume

  # Training on CPU (not recommended)
  python train_yolov8_teeth.py --data yolo_training_config.yaml --device cpu

Official Training Parameters (from Instance_seg_teeth):
  - Model: yolov8x.pt (81M parameters, best accuracy)
  - Epochs: 30
  - Batch: 10
  - Image Size: 640x640
  - Dropout: 0.6
  - Warmup Epochs: 10
  - Learning Rate Factor: 0.005
  - Cosine LR: True

Expected Results:
  - mAP: 74.9%
  - AP50: 94.6%
  - Training Time: 2-4 hours (RTX 3090)
        """
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml configuration file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="YOLOv8 model to use (default: yolov8x.pt for best results)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30, as in paper)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Batch size (default: 10, as in paper)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory (default: runs/detect)"
    )

    parser.add_argument(
        "--name",
        type=str,
        default=f"teeth_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment name (default: teeth_detection_TIMESTAMP)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (default: 0) or 'cpu'"
    )

    args = parser.parse_args()

    # Print banner
    print()
    print("=" * 70)
    print("YOLOv8 TEETH DETECTION TRAINING")
    print("Instance_seg_teeth Official Training Configuration")
    print("=" * 70)
    print()

    # Check environment
    check_environment()

    # Start training
    try:
        results = train_yolov8(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            resume=args.resume,
            device=args.device,
        )

        print("Training completed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("To resume training, use the --resume flag.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nERROR: Training failed with error:")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
