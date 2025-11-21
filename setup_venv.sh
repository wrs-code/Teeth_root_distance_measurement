#!/bin/bash

# Setup script for Teeth Instance Segmentation Batch Processing
# This script creates a virtual environment and installs all dependencies
# Compatible with CUDA 11.8

set -e  # Exit on error

echo "=========================================="
echo "Teeth Segmentation Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Check CUDA version
echo ""
echo "Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo ""
    echo "Note: This setup is optimized for CUDA 11.8"
    echo "TensorFlow 2.13 is compatible with CUDA 11.8 and cuDNN 8.6"
else
    echo "Warning: nvcc not found. Make sure CUDA 11.8 is installed."
    echo "You may need to install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
fi

# Check if CUDA 11.8 is installed
echo ""
echo "Checking CUDA installation..."
if [ -d "/usr/local/cuda-11.8" ]; then
    echo "CUDA 11.8 found at /usr/local/cuda-11.8"
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
elif [ -d "/usr/local/cuda" ]; then
    echo "CUDA found at /usr/local/cuda"
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "Unknown")
    echo "CUDA Version: $CUDA_VERSION"
else
    echo "Warning: CUDA installation not found in standard locations"
    echo "Please ensure CUDA 11.8 is properly installed"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify TensorFlow GPU support
echo ""
echo "Verifying TensorFlow GPU support..."
python3 << EOF
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is properly configured!")
else:
    print("Warning: No GPU detected. Training will be slow on CPU.")
EOF

# Verify YOLOv8 installation
echo ""
echo "Verifying YOLOv8 installation..."
python3 << EOF
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")
ultralytics.checks()
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the batch processing demo, use:"
echo "  python batch_process_teeth.py --input_dir <input_folder> --output_dir <output_folder> --yolo_weights <path_to_weights>"
echo ""
echo "CUDA 11.8 Notes:"
echo "- TensorFlow 2.13 is installed (compatible with CUDA 11.8)"
echo "- You need cuDNN 8.6 for optimal performance"
echo "- If you don't have CUDA 11.8, you may need to install it"
echo "- Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
echo ""
