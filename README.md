# 🎯 YOLO Object Detection with Keras & TensorFlow

This project demonstrates how to implement the YOLO (You Only Look Once) object detection model in TensorFlow/Keras. It covers the model architecture, preprocessing pipeline, filtering strategies, Non-Max Suppression (NMS), and inference on real-world images.

---

## 📌 Overview

- **Framework**: TensorFlow 1.x / Keras (with backend session)
- **Model**: YOLO (v2-style)
- **Purpose**: Real-time object detection on images
- **Input Shape**: 608x608 RGB images
- **Output**: Bounding boxes, class labels, confidence scores

---

## 🧠 Core Components

- `yolo_filter_boxes`: Filters boxes based on objectness score & class probabilities
- `iou`: Calculates the Intersection Over Union (IoU) of two bounding boxes
- `yolo_non_max_suppression`: Applies Non-Maximum Suppression to reduce duplicate boxes
- `yolo_eval`: End-to-end evaluation pipeline combining all YOLO post-processing steps
- `predict`: Loads, processes, and runs inference on a given image

---

## 🗂️ Project Structure

├── model_data/

│   ├── coco_classes.txt         # List of class names
│   ├── yolo_anchors.txt         # Predefined anchor boxes
│   └── yolo.h5                  # Pretrained YOLO weights

│

├── images/                      # Input test images
├── out/                         # Output folder for results
├── yolo_utils.py               # Image preprocessing, visualization, utility functions
├── yad2k/                      # YOLO architecture code
└── main.py / notebook.ipynb

---

## 🛠️ Preprocessing & Model Flow

1. Load class names and anchor boxes
2. Load pretrained YOLO model (`yolo.h5`)
3. Run predictions using model
4. Apply filtering and non-max suppression
5. Draw predicted boxes on images and save results

---

## 📊 Output Example

- For each test image, the output consists of:
  - **Number of boxes detected**
  - **Box coordinates**
  - **Class predictions**
  - **Overlayed image with bounding boxes and labels**

---

## 🖼️ Image Prediction Usage

To run detection on a new image:

```python
out_scores, out_boxes, out_classes = predict(sess, "your_image.jpg")

Saves result to out/your_image.jpg

Outputs class names and scores in the console

## ✅ Evaluation Tools

Unit test prints for iou, yolo_filter_boxes, and yolo_non_max_suppression

Session-based evaluation using tf.Session()

----

## 🧪 Sample Evaluations

iou() tested on intersecting, non-intersecting, vertex-touching, and edge-touching boxes

yolo_filter_boxes() and yolo_non_max_suppression() tested with synthetic tensors

----

## 📚 References

- 📄 [You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640) — Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
- 📄 [YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242) — Joseph Redmon, Ali Farhadi  
- 💻 [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K) — Allan Zelener  
- 🌐 [Official YOLO Website](https://pjreddie.com/darknet/yolo/) — PJ Reddie  

---

