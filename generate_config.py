#!/usr/bin/env python3
"""
Generate yolo_training_config.yaml with absolute paths
This ensures the config works from any location
"""
import os
import yaml

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "Instance_seg_teeth/Dataset/yolo_train_dataset")

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Please make sure Instance_seg_teeth is cloned in the current directory.")
    exit(1)

# Configuration
config = {
    'path': dataset_path,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 32,
    'names': ['11', '12', '13', '14', '15', '16', '17', '18',
              '21', '22', '23', '24', '25', '26', '27', '28',
              '31', '32', '33', '34', '35', '36', '37', '38',
              '41', '42', '43', '44', '45', '46', '47', '48'],
    'roboflow': {
        'workspace': 'teeth-segmentation',
        'project': 'teeth-segmentation-evs6x',
        'version': 15,
        'license': 'Private',
        'url': 'https://app.roboflow.com/teeth-segmentation/teeth-segmentation-evs6x/15'
    }
}

# Save to file
output_file = 'yolo_training_config.yaml'
with open(output_file, 'w') as f:
    f.write("# YOLOv8 Training Configuration for Teeth Detection\n")
    f.write("# Dataset: UFBA-425 (Roboflow version with augmentation)\n")
    f.write("# Total images: 1022 (894 train, 64 valid, 64 test)\n")
    f.write("# Auto-generated with absolute paths\n\n")
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Configuration file generated: {output_file}")
print(f"   Dataset path: {dataset_path}")
print(f"\nVerifying paths:")
print(f"   Train: {os.path.exists(os.path.join(dataset_path, 'train/images'))}")
print(f"   Valid: {os.path.exists(os.path.join(dataset_path, 'valid/images'))}")
print(f"   Test: {os.path.exists(os.path.join(dataset_path, 'test/images'))}")
