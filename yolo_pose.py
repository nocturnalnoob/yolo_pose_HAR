from ultralytics import YOLO
import numpy as np
import config  # Import paths and settings from config.py

class YOLOPose:
    def __init__(self, model_path):
        """
        Initialize the YOLO Pose model.
        :param model_path: Path to the YOLO pose model.
        """
        self.pose_model = YOLO(model_path)

    def detect_keypoints(self, image):
        """
        Run YOLO pose estimation on an image and extract keypoints.
        
        :param image: Input image as a NumPy array (BGR format from OpenCV).
        :return: Numpy array of keypoints (num_people, 17, 3) or None if no detections.
        """
        results = self.pose_model.predict(image)

        if not results or len(results) == 0:
            print("[WARNING] No keypoints detected!")
            return None

        keypoints = results[0].keypoints  # Extract keypoints

        if keypoints is None or keypoints.shape[0] == 0:
            print("[WARNING] No valid keypoints found.")
            return None

        return keypoints.cpu().numpy()  # Convert to NumPy array

if __name__ == "__main__":
    # Test YOLO Pose detection
    yolo_pose = YOLOPose(config.YOLO_MODEL_PATH)
    img_path = config.INPUT_IMAGE_PATH
    import cv2
    frame = cv2.imread(img_path)

    keypoints = yolo_pose.detect_keypoints(frame)

    if keypoints is not None:
        print("Extracted Keypoints:\n", keypoints)
