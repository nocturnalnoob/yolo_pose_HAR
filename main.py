import cv2
import torch
import numpy as np
import config
from yolo_pose import YOLOPose
from utils import normalize_keypoints, visualize_keypoints
from stgcn_inference import STGCNInference

def main():
    # Load YOLO Pose model
    print("[INFO] Loading YOLO Pose model...")
    yolo_pose = YOLOPose(config.YOLO_MODEL_PATH)

    # Load ST-GCN++ model
    print("[INFO] Loading ST-GCN++ model...")
    stgcn = STGCNInference(config.STGCN_MODEL_PATH) # type: ignore

    # Load the video or image
    input_path = config.INPUT_VIDEO_PATH
    if input_path.endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(input_path)
        frames = [frame]  # Single frame for image
    else:
        cap = cv2.VideoCapture(input_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    print(f"[INFO] Processing {len(frames)} frames...")

    for idx, frame in enumerate(frames):
        # Detect keypoints using YOLO Pose
        keypoints = yolo_pose.detect_keypoints(frame)

        if keypoints is None:
            print(f"[WARNING] No keypoints detected in frame {idx}. Skipping...")
            continue

        # Normalize keypoints
        normalized_kp = normalize_keypoints(keypoints, frame.shape)

        # Run ST-GCN++ for action recognition
        action_label = stgcn.predict_action(normalized_kp) # type: ignore
        print(f"[RESULT] Frame {idx}: Action Recognized -> {action_label}")

        # Visualize keypoints (optional)
        if config.VISUALIZE:
            frame = visualize_keypoints(frame, keypoints)
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()
