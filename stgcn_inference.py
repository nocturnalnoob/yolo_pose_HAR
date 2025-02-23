from mmaction.apis import init_recognizer
import config

class STGCNInference:
    def __init__(self):
        """
        Initialize ST-GCN++ model using MMAction2.
        """
        self.model = init_recognizer(config.STGCN_MODEL_PATH, device='cpu')

    def predict(self, keypoints):
        """
        Run inference on keypoints.
        :param keypoints: Numpy array of shape (num_people, 17, 2).
        :return: Predicted action label.
        """
        if keypoints is None or len(keypoints) == 0:
            return "No person detected"

        # Convert to proper format and run inference
        result = self.model(keypoints)
        return result['pred_label']  # Extract label

