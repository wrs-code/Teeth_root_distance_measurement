# Teeth Instance Segmentation - Batch Processing Demo

This project provides a batch processing pipeline for dental X-ray images using the [Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth.git) repository. It combines YOLOv8 for teeth detection and optionally UNet for detailed segmentation, with all intermediate outputs saved for debugging.

## Features

- **Batch Processing**: Process multiple dental X-ray images automatically
- **YOLOv8 Detection**: Detect and localize individual teeth (32 classes)
- **Optional UNet Segmentation**: Detailed pixel-level segmentation
- **Complete Intermediate Outputs**: All processing steps saved with timestamps
- **Organized Output Structure**: Easy to navigate results
- **Debug-Friendly**: Comprehensive logging and error handling
- **CUDA 11.8 Compatible**: Optimized for CUDA 11.8 with TensorFlow 2.13

## Project Structure

```
Teeth_root_distance_measurement/
├── Instance_seg_teeth/          # Cloned repository
│   ├── notebooks/               # Original Jupyter notebooks
│   ├── Dataset/                 # Dataset information
│   └── ...
├── batch_process_teeth.py       # Main batch processing script
├── requirements.txt             # Python dependencies
├── setup_venv.sh               # Virtual environment setup script
├── README.md                   # This file
└── output/                     # Output directory (created during processing)
    └── batch_YYYYMMDD_HHMMSS/  # Timestamped batch results
        ├── original_images/
        ├── yolo_detections/
        ├── yolo_visualizations/
        ├── bounding_boxes_json/
        ├── binary_masks/
        ├── binary_masks_visualizations/
        ├── unet_predictions/
        ├── unet_visualizations/
        ├── summary/
        └── processing_log.txt
```

## Requirements

### System Requirements

- **CUDA**: 11.8 (recommended)
- **cuDNN**: 8.6 (for TensorFlow GPU support)
- **Python**: 3.8 - 3.10
- **GPU**: NVIDIA GPU with CUDA support (recommended for performance)
- **RAM**: 8GB minimum, 16GB+ recommended

### CUDA 11.8 Installation

If you don't have CUDA 11.8 installed:

1. **Check current CUDA version**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Install CUDA 11.8** (if needed):
   - Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Follow the installation instructions for your OS
   - Install cuDNN 8.6 from: https://developer.nvidia.com/cudnn

3. **Set environment variables**:
   ```bash
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

**Note**: If you already have CUDA 11.8, you can skip the reinstallation. The setup script will work with your existing installation.

## Installation

### Step 1: Clone this repository

The Instance_seg_teeth repository is already cloned in this directory.

### Step 2: Run the setup script

```bash
# Make the setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

This script will:
1. Check your Python and CUDA versions
2. Create a virtual environment
3. Install all required dependencies
4. Verify TensorFlow GPU support
5. Verify YOLOv8 installation

### Step 3: Activate the virtual environment

```bash
source venv/bin/activate
```

### Manual Installation (Alternative)

If you prefer to install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The project uses the following main dependencies (see `requirements.txt` for complete list):

- **TensorFlow 2.13**: Compatible with CUDA 11.8
- **Ultralytics 8.0.28**: YOLOv8 framework
- **OpenCV**: Image processing
- **scikit-image**: Image preprocessing (CLAHE)
- **NumPy, Matplotlib, Pillow**: Scientific computing and visualization
- **psutil, scikit-learn**: Utilities

## Usage

### Basic Usage (YOLOv8 Detection Only)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./path/to/yolov8_weights.pt
```

### Full Pipeline (YOLOv8 + UNet)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./path/to/yolov8_weights.pt \
    --unet_weights ./path/to/unet_weights.h5
```

### Custom Output Directory

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./path/to/yolov8_weights.pt \
    --output_dir ./my_results
```

### Using CPU (if no GPU available)

```bash
python batch_process_teeth.py \
    --input_dir ./test_images \
    --yolo_weights ./path/to/yolov8_weights.pt \
    --device cpu
```

## Command Line Arguments

- `--input_dir`: Directory containing input dental X-ray images (required)
- `--yolo_weights`: Path to YOLOv8 model weights file (.pt) (required)
- `--unet_weights`: Path to UNet model weights file (.h5) (optional)
- `--output_dir`: Base directory for output files (default: `./output`)
- `--device`: Device to use - `0` for first GPU, `cpu` for CPU (default: `0`)

## Output Structure

Each batch processing run creates a timestamped directory with the following structure:

```
output/batch_20251121_123456/
├── original_images/              # Copies of input images
│   ├── image1.jpg
│   └── image2.jpg
│
├── yolo_detections/              # Raw YOLO detection results
│
├── yolo_visualizations/          # Annotated images with bounding boxes
│   ├── image1_detection.jpg
│   └── image2_detection.jpg
│
├── bounding_boxes_json/          # Bounding box coordinates in JSON format
│   ├── image1_boxes.json
│   └── image2_boxes.json
│
├── binary_masks/                 # Binary masks from YOLO detections
│   ├── image1_mask.npy
│   └── image2_mask.npy
│
├── binary_masks_visualizations/  # Colored visualizations of binary masks
│   ├── image1_mask_vis.png
│   └── image2_mask_vis.png
│
├── unet_predictions/             # UNet segmentation predictions (if enabled)
│   ├── image1_unet_pred.npy
│   └── image2_unet_pred.npy
│
├── unet_visualizations/          # Colored visualizations of UNet predictions
│   ├── image1_unet_vis.png
│   └── image2_unet_vis.png
│
├── summary/                      # Processing summary
│   └── processing_summary.json
│
└── processing_log.txt           # Detailed processing log
```

### File Formats

- **Images**: JPG, PNG (visualizations)
- **Masks**: NumPy arrays (.npy) for programmatic access
- **Bounding Boxes**: JSON format for easy parsing
- **Logs**: Plain text with timestamps

## Output File Examples

### Bounding Box JSON Format

```json
{
  "image_name": "sample_xray",
  "timestamp": "2025-11-21 12:34:56",
  "num_detections": 28,
  "detections": [
    {
      "class_id": 0,
      "bbox_xywh": [320.5, 150.2, 45.3, 60.8]
    },
    ...
  ]
}
```

### Processing Summary JSON

```json
{
  "timestamp": "2025-11-21 12:45:00",
  "total_images": 10,
  "successful": 10,
  "failed": 0,
  "total_detections": 285,
  "results": [...]
}
```

## Color Coding for Tooth Types

The visualizations use color coding to distinguish different tooth types:

- **Blue** (#2828a2): Incisors (teeth 0,1,8,9,16,17,24,25)
- **Cyan** (#09c5c5): Canines (teeth 2,10,18,26)
- **Green** (#1caf1c): Premolars (teeth 3,4,11,12,19,20,27,28)
- **Yellow** (#eeee25): Molars (teeth 5,6,7,13,14,15,21,22,23,29,30,31)

## Troubleshooting

### CUDA/GPU Issues

1. **Check GPU availability**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Verify CUDA version**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **If GPU not detected**:
   - Verify CUDA 11.8 and cuDNN 8.6 are installed
   - Check TensorFlow installation: `pip list | grep tensorflow`
   - Use `--device cpu` flag to run on CPU

### Memory Issues

If you encounter out-of-memory errors:

1. **Reduce batch size** in the code (currently processes one image at a time)
2. **Process fewer images** at once
3. **Use a machine with more RAM/VRAM**

### Model Loading Errors

If you get errors loading the models:

1. **Check model file paths** - ensure they exist
2. **Verify model format** - YOLOv8 uses .pt, UNet uses .h5
3. **Check model compatibility** - models must be trained with compatible versions

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or install specific package
pip install ultralytics==8.0.28
```

## Getting Model Weights

### YOLOv8 Weights

You can use pre-trained weights or train your own:

1. **Download pre-trained weights**:
   ```bash
   # YOLOv8x (extra large)
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   ```

2. **Train your own**: Follow the notebooks in `Instance_seg_teeth/notebooks/yolov8/`

### UNet Weights

To use UNet segmentation:

1. **Train your own**: Follow the notebooks in `Instance_seg_teeth/notebooks/Unet/` or `Instance_seg_teeth/notebooks/yolov8+unet/`
2. Save the trained model as `.h5` file

## Performance Tips

1. **GPU Acceleration**: Always use GPU if available (50-100x faster than CPU)
2. **Batch Processing**: Process multiple images in one run to amortize startup costs
3. **Image Size**: Larger images take longer to process
4. **Model Choice**: YOLOv8x is more accurate but slower than smaller variants (yolov8n, yolov8s, yolov8m)

## Citation

If you use this code or the Instance_seg_teeth repository, please cite:

```bibtex
@article{Budagam2025,
  author = "Devichand Budagam and Azamat Zhanatuly Imanbayev and Iskander Rafailovich Akhmetov and Aleksandr Sinitca and Sergey Antonov and Dmitrii Kaplun",
  title = "{UFBA-425}",
  year = "2025",
  month = "8",
  url = "https://figshare.com/articles/dataset/UFBA-425/29827475",
  doi = "10.6084/m9.figshare.29827475.v1"
}

@misc{budagam2025oralbbnetspatiallyguideddental,
  title={OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors},
  author={Devichand Budagam and Azamat Zhanatuly Imanbayev and Iskander Rafailovich Akhmetov and Aleksandr Sinitca and Sergey Antonov and Dmitrii Kaplun},
  year={2025},
  eprint={2406.03747},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2406.03747}
}
```

## License

This project uses code from the Instance_seg_teeth repository. Please refer to the LICENSE file in that repository for licensing information.

## Support

For issues related to:
- **This batch processing script**: Open an issue in this repository
- **Original Instance_seg_teeth code**: Visit https://github.com/devichand579/Instance_seg_teeth

## Author

Auto-generated batch processing demo for Instance_seg_teeth
Date: 2025-11-21

## Changelog

### Version 1.0 (2025-11-21)
- Initial release
- YOLOv8 batch detection
- Optional UNet segmentation
- Comprehensive output structure with timestamps
- CUDA 11.8 compatibility
