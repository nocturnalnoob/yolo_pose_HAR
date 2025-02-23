import numpy as np
import config

def normalize_keypoints(keypoints, image_shape):
    """
    Normalize keypoints by scaling them between 0 and 1.
    
    :param keypoints: Numpy array of shape (num_people, 17, 3).
    :param image_shape: Tuple (height, width) of the image.
    :return: Normalized keypoints.
    """
    height, width = image_shape[:2]

    keypoints[:, :, 0] /= width  # Normalize X-coordinates
    keypoints[:, :, 1] /= height  # Normalize Y-coordinates
    return keypoints

def preprocess_keypoints(keypoints, image_shape):
    """
    Prepare keypoints for ST-GCN++ by normalizing and reshaping.

    :param keypoints: Raw keypoints from YOLOv8 Pose.
    :param image_shape: Tuple (height, width) of the image.
    :return: Processed keypoints ready for ST-GCN++.
    """
    if keypoints is None:
        return None

    keypoints = normalize_keypoints(keypoints, image_shape)

    # ST-GCN++ expects shape (num_people, 17, 2), removing confidence scores
    processed_keypoints = keypoints[:, :, :2]  # Remove confidence scores

    return processed_keypoints

if __name__ == "__main__":
    # Example usage
    sample_keypoints = np.array([[[100, 200, 0.9], [150, 250, 0.8], [0, 0, 0.5]]])  # Dummy keypoints
    image_shape = (720, 1280)  # Example image size (height, width)
    
    processed_keypoints = preprocess_keypoints(sample_keypoints, image_shape)
    print("Processed Keypoints:\n", processed_keypoints)
