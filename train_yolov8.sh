#!/bin/bash
# YOLOv8 Teeth Detection Training Script
# Based on Instance_seg_teeth official training configuration
# Quick start script for training

set -e  # Exit on error

echo "=========================================="
echo "YOLOv8 Teeth Detection Training"
echo "Instance_seg_teeth Official Configuration"
echo "=========================================="
echo ""

# Configuration
MODEL="yolov8x.pt"
EPOCHS=30
BATCH=10
IMGSZ=640
DEVICE="0"

# Generate config file with absolute paths for current user
echo "Generating configuration file with absolute paths..."
if [ -f "generate_config.py" ]; then
    python generate_config.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate configuration file"
        exit 1
    fi
    DATA_YAML="yolo_training_config_auto.yaml"
else
    echo "WARNING: generate_config.py not found, using template config"
    DATA_YAML="yolo_training_config.yaml"
fi
echo ""

# Check if data.yaml exists
if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: Data configuration file not found: $DATA_YAML"
    exit 1
fi

# Check if dataset exists
if [ ! -d "Instance_seg_teeth/Dataset/yolo_train_dataset" ]; then
    echo "ERROR: Dataset not found. Please ensure Instance_seg_teeth is cloned."
    echo "Run: git clone https://github.com/devichand579/Instance_seg_teeth.git"
    exit 1
fi

echo "Training Configuration:"
echo "  - Data Config: $DATA_YAML"
echo "  - Model: $MODEL"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH"
echo "  - Image Size: $IMGSZ"
echo "  - Device: $DEVICE"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if ultralytics is installed
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "ERROR: ultralytics not installed"
    echo "Please run: pip install ultralytics==8.0.28"
    exit 1
fi

# Start training using YOLO CLI
echo "Starting training..."
echo "This will take 2-4 hours on a modern GPU (e.g., RTX 3090)"
echo ""

# Official training command from Instance_seg_teeth notebook
yolo task=detect mode=train \
    model=$MODEL \
    data=$DATA_YAML \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMGSZ \
    device=$DEVICE \
    cache=True \
    single_cls=False \
    val=True \
    dropout=0.6 \
    close_mosaic=0 \
    cos_lr=True \
    exist_ok=True \
    warmup_epochs=10 \
    lrf=0.005

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo ""
echo "Model weights saved in: runs/detect/train/weights/"
echo "  - Best model: runs/detect/train/weights/best.pt"
echo "  - Last model: runs/detect/train/weights/last.pt"
echo ""
echo "Expected Performance (from paper):"
echo "  - mAP: 74.9%"
echo "  - AP50: 94.6%"
echo ""
echo "To use the trained model:"
echo "  yolo predict model=runs/detect/train/weights/best.pt source=./test_images conf=0.5"
echo ""
echo "Or use batch processing:"
echo "  python batch_process_teeth.py --yolo_weights runs/detect/train/weights/best.pt --input_dir ./test_images"
echo ""
