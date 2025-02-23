import os

# Configuration file for paths and settings

# Paths to models
YOLO_MODEL_PATH = "models/yolo11n-pose.pt"
STGCN_MODEL_PATH = "models/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth"

# Paths for data storage
INPUT_VIDEO_PATH = "data/input_video.mp4"
INPUT_IMAGE_PATH = "data/frame.jpeg"
KEYPOINTS_PATH = "data/keypoints.npy"
OUTPUT_VIDEO_PATH = "output/output_video.mp4"
PREDICTIONS_PATH = "output/predictions.txt"

# Other settings
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for keypoints detection
VISUALIZE = True