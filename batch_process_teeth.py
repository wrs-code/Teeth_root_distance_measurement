"""
Batch Processing Demo for Teeth Instance Segmentation
This script processes multiple dental X-ray images using the Instance_seg_teeth pipeline.

Features:
- Batch processing of multiple images
- YOLOv8 detection for teeth localization
- Optional UNet segmentation for detailed masks
- All intermediate outputs saved with timestamps
- Organized output directory structure
- Debug-friendly with verbose logging

Author: Auto-generated
Date: 2025-11-21
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from skimage import exposure
import tensorflow as tf

# Add the Instance_seg_teeth to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Instance_seg_teeth'))

from ultralytics import YOLO


class TeethBatchProcessor:
    """Batch processor for teeth segmentation using YOLOv8 and optionally UNet"""

    def __init__(self, yolo_weights, unet_weights=None, output_base_dir='output', device=0):
        """
        Initialize the batch processor

        Args:
            yolo_weights: Path to YOLOv8 model weights
            unet_weights: Path to UNet model weights (optional)
            output_base_dir: Base directory for outputs
            device: GPU device ID (0 for first GPU, 'cpu' for CPU)
        """
        self.yolo_weights = yolo_weights
        self.unet_weights = unet_weights
        self.output_base_dir = output_base_dir
        self.device = device

        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_base_dir, f'batch_{timestamp}')
        self.create_output_structure()

        # Load models
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_weights)
        print(f"YOLO model loaded from {yolo_weights}")

        self.unet_model = None
        if unet_weights:
            print("Loading UNet model...")
            self.unet_model = tf.keras.models.load_model(
                unet_weights,
                custom_objects={
                    'dice_loss_with_l2_regularization': self.dice_loss_with_l2_regularization,
                    'dice_coef': self.dice_coef
                }
            )
            print(f"UNet model loaded from {unet_weights}")

        # Setup logging
        self.log_file = os.path.join(self.output_dir, 'processing_log.txt')
        self.log(f"Batch processing started at {timestamp}")
        self.log(f"Output directory: {self.output_dir}")

    def create_output_structure(self):
        """Create organized output directory structure"""
        subdirs = [
            'original_images',
            'yolo_detections',
            'yolo_visualizations',
            'bounding_boxes_json',
            'binary_masks',
            'binary_masks_visualizations',
            'unet_predictions',
            'unet_visualizations',
            'summary'
        ]

        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    @staticmethod
    def dice_loss_with_l2_regularization(target, predicted, epsilon=1e-7, l2_weight=0.1):
        """Dice loss with L2 regularization for UNet training"""
        intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
        predicted_square = tf.square(predicted)
        target_square = tf.square(target)
        union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
        dice = (2 * intersection + epsilon) / (union + epsilon)
        mean_dice_loss = tf.reduce_mean(dice)

        l2_norm = tf.reduce_sum(tf.square(predicted - target), axis=[1, 2])
        l2_regularization = l2_weight * tf.reduce_mean(l2_norm)

        total_loss = mean_dice_loss + l2_regularization
        return total_loss

    @staticmethod
    def dice_coef(target, predicted, epsilon=1e-7):
        """Dice coefficient metric"""
        predicted = tf.where(predicted < 0.51, 0.00, 1.00)
        intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
        predicted_square = tf.square(predicted)
        target_square = tf.square(target)
        union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
        dice = (2 * intersection + epsilon) / (union + epsilon)
        mean_dice_loss = -tf.reduce_mean(dice)
        return -mean_dice_loss

    def apply_clahe(self, image, clip_limit=0.02):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        image_float = image.astype(float) / 255.0
        image_clahe = exposure.equalize_adapthist(image_float, clip_limit=clip_limit)
        image_clahe = (image_clahe * 255).astype(image.dtype)
        return image_clahe

    def process_yolo_detection(self, image_path, image_name):
        """
        Process image with YOLO detection

        Returns:
            results: YOLO detection results
            boxes: List of bounding boxes
            classes: List of class IDs
            binary_mask: Binary mask from bounding boxes
        """
        self.log(f"Processing YOLO detection for {image_name}...")

        # Run YOLO prediction
        results = self.yolo_model.predict(
            source=image_path,
            iou=0.7,
            conf=0.5,
            save=False,
            device=self.device
        )

        # Extract boxes and classes
        boxes_list = []
        classes_list = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_xywh = boxes.xywh.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                boxes_list.append(boxes_xywh)
                classes_list.append(classes)

        # Save detection visualization
        if len(results) > 0:
            result_img = results[0].plot()
            output_path = os.path.join(self.output_dir, 'yolo_visualizations', f'{image_name}_detection.jpg')
            cv2.imwrite(output_path, result_img)
            self.log(f"  Saved detection visualization to {output_path}")

        # Create binary masks from bounding boxes
        binary_mask = self.create_binary_masks(boxes_list, classes_list, image_path)

        # Save bounding box information as JSON
        boxes_info = {
            'image_name': image_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_detections': sum(len(b) for b in boxes_list),
            'detections': []
        }

        for boxes, classes in zip(boxes_list, classes_list):
            for box, cls in zip(boxes, classes):
                boxes_info['detections'].append({
                    'class_id': int(cls),
                    'bbox_xywh': box.tolist()
                })

        json_path = os.path.join(self.output_dir, 'bounding_boxes_json', f'{image_name}_boxes.json')
        with open(json_path, 'w') as f:
            json.dump(boxes_info, f, indent=2)
        self.log(f"  Saved bounding box JSON to {json_path}")

        return results, boxes_list, classes_list, binary_mask

    def create_binary_masks(self, boxes_list, classes_list, image_path):
        """Create binary masks from YOLO bounding boxes"""
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            self.log(f"  Warning: Could not read image {image_path}")
            return None

        h, w = img.shape[:2]

        # Create binary mask (32 classes, one for each tooth)
        binary_mask = np.zeros((32, h, w), dtype=np.uint8)

        for boxes, classes in zip(boxes_list, classes_list):
            for box, cls in zip(boxes, classes):
                x, y, box_w, box_h = box
                x1, y1 = int(x - box_w/2), int(y - box_h/2)
                x2, y2 = int(x + box_w/2), int(y + box_h/2)

                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                class_idx = int(cls)
                if 0 <= class_idx < 32:
                    binary_mask[class_idx, y1:y2, x1:x2] = 1

        return binary_mask

    def save_binary_mask(self, binary_mask, image_name):
        """Save binary mask as numpy array and visualization"""
        if binary_mask is None:
            return

        # Save as numpy array
        mask_path = os.path.join(self.output_dir, 'binary_masks', f'{image_name}_mask.npy')
        np.save(mask_path, binary_mask)
        self.log(f"  Saved binary mask to {mask_path}")

        # Create and save visualization
        vis_mask = self.visualize_mask(binary_mask)
        vis_path = os.path.join(self.output_dir, 'binary_masks_visualizations', f'{image_name}_mask_vis.png')
        plt.imsave(vis_path, vis_mask)
        self.log(f"  Saved mask visualization to {vis_path}")

    def visualize_mask(self, mask, img_size=(640, 640)):
        """Visualize mask with different colors for different tooth types"""
        if mask.ndim == 3:
            # mask is (32, H, W), transpose to (H, W, 32)
            mask = np.transpose(mask, (1, 2, 0))

        h, w = mask.shape[:2]
        combined_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Color mapping for different tooth types
        colors = {
            'incisors': [40, 40, 162],      # Blue: 0,1,8,9,16,17,24,25
            'canines': [9, 197, 197],       # Cyan: 2,10,18,26
            'premolars': [28, 175, 28],     # Green: 3,4,11,12,19,20,27,28
            'molars': [238, 238, 37]        # Yellow: 5,6,7,13,14,15,21,22,23,29,30,31
        }

        incisor_indices = [0, 1, 8, 9, 16, 17, 24, 25]
        canine_indices = [2, 10, 18, 26]
        premolar_indices = [3, 4, 11, 12, 19, 20, 27, 28]
        molar_indices = [5, 6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31]

        for i in range(min(32, mask.shape[2])):
            if i in incisor_indices:
                color = colors['incisors']
            elif i in canine_indices:
                color = colors['canines']
            elif i in premolar_indices:
                color = colors['premolars']
            elif i in molar_indices:
                color = colors['molars']
            else:
                continue

            mask_channel = mask[:, :, i]
            mask_binary = (mask_channel > 0).astype(np.uint8)

            for c in range(3):
                combined_image[:, :, c] += mask_binary * color[c]

        return combined_image

    def process_unet_segmentation(self, image_path, binary_mask, image_name):
        """Process image with UNet segmentation"""
        if self.unet_model is None:
            return None

        self.log(f"Processing UNet segmentation for {image_name}...")

        # Load and preprocess image
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)

        # Apply CLAHE
        img_array = self.apply_clahe(img_array)

        # Resize to 512x512 (UNet input size)
        img_resized = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Normalize
        img_normalized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))

        # Resize binary mask to match
        if binary_mask is not None:
            mask_resized = np.transpose(binary_mask, (1, 2, 0))
            mask_resized = cv2.resize(mask_resized, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = np.zeros((512, 512, 32), dtype=np.float32)

        # Concatenate image and binary mask (35 channels: 32 for mask + 3 for image if RGB, or 33 if grayscale)
        if len(img_normalized.shape) == 2:
            img_normalized = np.expand_dims(img_normalized, axis=-1)

        # For the model expecting 35 channels (32 binary masks + 3 image channels)
        # If grayscale, replicate to 3 channels
        if img_normalized.shape[-1] == 1:
            img_normalized = np.repeat(img_normalized, 3, axis=-1)

        input_data = np.concatenate([mask_resized, img_normalized], axis=-1)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Predict
        prediction = self.unet_model.predict(input_data, verbose=0)

        # Post-process prediction
        pred_mask = prediction[0]  # Remove batch dimension
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

        # Save prediction
        pred_path = os.path.join(self.output_dir, 'unet_predictions', f'{image_name}_unet_pred.npy')
        np.save(pred_path, pred_mask)
        self.log(f"  Saved UNet prediction to {pred_path}")

        # Save visualization
        vis_pred = self.visualize_mask(pred_mask_binary)
        vis_path = os.path.join(self.output_dir, 'unet_visualizations', f'{image_name}_unet_vis.png')
        plt.imsave(vis_path, vis_pred)
        self.log(f"  Saved UNet visualization to {vis_path}")

        return pred_mask

    def process_single_image(self, image_path):
        """Process a single image through the pipeline"""
        image_name = Path(image_path).stem
        self.log(f"\n{'='*60}")
        self.log(f"Processing: {image_name}")
        self.log(f"{'='*60}")

        # Copy original image to output
        output_img_path = os.path.join(self.output_dir, 'original_images', Path(image_path).name)
        shutil.copy(image_path, output_img_path)
        self.log(f"Copied original image to {output_img_path}")

        try:
            # Step 1: YOLO Detection
            results, boxes_list, classes_list, binary_mask = self.process_yolo_detection(image_path, image_name)

            # Step 2: Save binary masks
            if binary_mask is not None:
                self.save_binary_mask(binary_mask, image_name)

            # Step 3: UNet Segmentation (if model is available)
            unet_pred = None
            if self.unet_model is not None:
                unet_pred = self.process_unet_segmentation(image_path, binary_mask, image_name)

            self.log(f"Successfully processed {image_name}")

            return {
                'image_name': image_name,
                'status': 'success',
                'num_detections': sum(len(b) for b in boxes_list),
                'has_unet': unet_pred is not None
            }

        except Exception as e:
            self.log(f"ERROR processing {image_name}: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return {
                'image_name': image_name,
                'status': 'error',
                'error': str(e)
            }

    def process_batch(self, input_dir, image_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
        """Process all images in the input directory"""
        self.log(f"\nStarting batch processing from: {input_dir}")

        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))

        image_files = sorted(set(image_files))

        self.log(f"Found {len(image_files)} images to process")

        if len(image_files) == 0:
            self.log("No images found! Please check the input directory.")
            return

        # Process each image
        results_summary = []
        for i, image_path in enumerate(image_files, 1):
            self.log(f"\n[{i}/{len(image_files)}] Processing {image_path.name}")
            result = self.process_single_image(str(image_path))
            results_summary.append(result)

        # Generate summary
        self.generate_summary(results_summary)

        self.log(f"\n{'='*60}")
        self.log(f"Batch processing complete!")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"{'='*60}\n")

    def generate_summary(self, results_summary):
        """Generate a summary report of the batch processing"""
        summary_path = os.path.join(self.output_dir, 'summary', 'processing_summary.json')

        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': len(results_summary),
            'successful': sum(1 for r in results_summary if r['status'] == 'success'),
            'failed': sum(1 for r in results_summary if r['status'] == 'error'),
            'total_detections': sum(r.get('num_detections', 0) for r in results_summary),
            'results': results_summary
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"\nSummary saved to {summary_path}")
        self.log(f"Total images processed: {summary['total_images']}")
        self.log(f"Successful: {summary['successful']}")
        self.log(f"Failed: {summary['failed']}")
        self.log(f"Total detections: {summary['total_detections']}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch processing for teeth instance segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with YOLO only:
  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./models/yolov8x.pt

  # Process with both YOLO and UNet:
  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./models/yolov8x.pt --unet_weights ./models/unet_model.h5

  # Specify output directory:
  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./models/yolov8x.pt --output_dir ./results

  # Use CPU instead of GPU:
  python batch_process_teeth.py --input_dir ./test_images --yolo_weights ./models/yolov8x.pt --device cpu
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--yolo_weights', type=str, required=True,
                        help='Path to YOLOv8 model weights (.pt file)')
    parser.add_argument('--unet_weights', type=str, default=None,
                        help='Path to UNet model weights (.h5 file, optional)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Base output directory (default: output)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use: 0 for GPU, cpu for CPU (default: 0)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    if not os.path.exists(args.yolo_weights):
        print(f"Error: YOLO weights file does not exist: {args.yolo_weights}")
        return

    if args.unet_weights and not os.path.exists(args.unet_weights):
        print(f"Error: UNet weights file does not exist: {args.unet_weights}")
        return

    # Create processor and run
    processor = TeethBatchProcessor(
        yolo_weights=args.yolo_weights,
        unet_weights=args.unet_weights,
        output_base_dir=args.output_dir,
        device=args.device
    )

    processor.process_batch(args.input_dir)


if __name__ == '__main__':
    main()
