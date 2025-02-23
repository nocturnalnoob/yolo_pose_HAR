from ultralytics import YOLO
import numpy as np
import cv2
import config  # Import paths and settings from config.py

# Load YOLOv8 pose model
pose_model = YOLO(config.YOLO_MODEL_PATH)

def get_keypoints(image_path):
    """
    Run YOLO pose estimation on an image and extract keypoints.
    
    :param image_path: Path to the input image.
    :return: Numpy array of keypoints (num_people, 17, 3)
    """
    results = pose_model.predict(image_path)

    if not results:
        print("No results found!")
        return None

    keypoints = results[0].keypoints.cpu().numpy()  # Extract keypoints
    return keypoints

if __name__ == "__main__":
    img_path = config.INPUT_IMAGE_PATH
    keypoints = get_keypoints(img_path)

    if keypoints is not None:
        print("Extracted Keypoints:\n", keypoints)
