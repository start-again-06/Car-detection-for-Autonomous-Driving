## YOLO Object Detection with TensorFlow/Keras

A comprehensive computer vision project that implements the YOLO (You Only Look Once) object detection pipeline using TensorFlow and Keras. The project focuses on real-time object detection by combining a pretrained YOLO model with custom post-processing steps such as confidence filtering, IoU computation, and Non-Max Suppression.

## Features
- End-to-end YOLO object detection pipeline
- Bounding box filtering using objectness and class probabilities
- Intersection over Union (IoU) computation
- Non-Maximum Suppression (NMS) to remove duplicate detections
- Inference on real-world images
- Modular and educational implementation

## Model & Framework
- Model: YOLO (v2-style)
- Framework: TensorFlow 1.x with Keras backend
- Task: Real-time object detection
- Input Shape: 608 × 608 × 3 (RGB)
- Output: Bounding boxes, class labels, confidence scores

## Core Components
- yolo_filter_boxes: Filters predicted boxes based on confidence and class probabilities  
- iou: Computes Intersection over Union between bounding boxes  
- yolo_non_max_suppression: Applies Non-Maximum Suppression  
- yolo_eval: End-to-end YOLO post-processing pipeline  
- predict: Runs inference and saves output images  

## Dataset & Classes
- Dataset: COCO (pretrained)
- Class labels loaded from coco_classes.txt
- Anchor boxes loaded from yolo_anchors.txt

## Evaluation & Testing
- Unit tests for iou, yolo_filter_boxes, and yolo_non_max_suppression
- Tested on intersecting, non-intersecting, and edge-case bounding boxes
- Session-based evaluation using tf.Session()

## Dependencies
- Python 3.x
- TensorFlow 1.x
- Keras
- NumPy
- Pillow
- Matplotlib

## References
- You Only Look Once: Unified, Real-Time Object Detection – Redmon et al.
- YOLO9000: Better, Faster, Stronger – Redmon & Farhadi
- YAD2K: Yet Another Darknet 2 Keras

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.
