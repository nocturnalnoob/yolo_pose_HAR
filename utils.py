import cv2
import numpy as np

def normalize_keypoints(keypoints, image_shape):
    """
    Normalize keypoints to be between 0 and 1 for ST-GCN++.
    
    :param keypoints: Numpy array of shape (num_people, 17, 2).
    :param image_shape: Tuple (height, width) of the image.
    :return: Normalized keypoints.
    """
    h, w = image_shape
    keypoints[:, :, 0] /= w  # Normalize X
    keypoints[:, :, 1] /= h  # Normalize Y
    return keypoints

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on an image.

    :param image: OpenCV image.
    :param keypoints: Numpy array of shape (num_people, 17, 2).
    :param color: BGR color tuple for keypoints.
    :return: Image with keypoints drawn.
    """
    for person in keypoints:
        for x, y in person:
            if x > 0 and y > 0:  # Ignore missing keypoints (0,0)
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
    return image

def save_keypoints_to_file(keypoints, filename="keypoints.npy"):
    """
    Save keypoints to a .npy file.
    
    :param keypoints: Numpy array of shape (num_people, 17, 2).
    :param filename: File path to save.
    """
    np.save(filename, keypoints)
    print(f"Keypoints saved to {filename}")

def load_keypoints_from_file(filename="keypoints.npy"):
    """
    Load keypoints from a .npy file.
    
    :param filename: File path to load.
    :return: Loaded keypoints array.
    """
    keypoints = np.load(filename)
    print(f"Loaded keypoints from {filename}")
    return keypoints
