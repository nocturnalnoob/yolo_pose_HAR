import torch
import numpy as np
from config import STGCN_MODEL_PATH
from preprocess import preprocess_keypoints

class STGCNModel:
    def __init__(self, model_path=STGCN_MODEL_PATH):
        """
        Load the ST-GCN++ model.
        
        :param model_path: Path to the pretrained ST-GCN++ model.
        """
        self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.eval()

    def predict(self, keypoints):
        """
        Run inference on the preprocessed keypoints.

        :param keypoints: Numpy array of shape (num_people, 17, 2).
        :return: Predicted action label.
        """
        if keypoints is None or len(keypoints) == 0:
            return "No person detected"

        # Convert numpy array to PyTorch tensor
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_people, 17, 2)

        # Run inference
        with torch.no_grad():
            output = self.model(keypoints_tensor)  # Forward pass

        # Get predicted label
        predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label

if __name__ == "__main__":
    # Example usage
    sample_keypoints = np.random.rand(1, 17, 2)  # Dummy data
    model = STGCNModel()
    
    action_label = model.predict(sample_keypoints)
    print("Predicted Action:", action_label)
