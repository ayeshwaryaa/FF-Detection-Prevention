import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 🧠 LOGIC FROM MODULE 1.1 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    def __init__(self):
        # Load VGG16 exactly as in your notebook
        self.device = torch.device('cpu') 
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        self.features = None
        self._register_hook()

    def _load_model(self):
        # We take VGG16 and keep only the 'features' part
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = vgg.features
        # Add the Global Average Pooling layer (512 vector)
        return nn.Sequential(features, nn.AdaptiveAvgPool2d((1, 1)))

    def _hook_fn(self, module, input, output):
        self.features = torch.flatten(output, 1)

    def _register_hook(self):
        self.model[-1].register_forward_hook(self._hook_fn)

    def extract(self, image_tensor):
        with torch.no_grad():
            self.model(image_tensor.to(self.device))
        return self.features.cpu().numpy()

# ==========================================
# 🛠️ HELPERS FROM MODULE 2.1 (Detection)
# ==========================================
def get_transform():
    # Standard ImageNet normalization (Matches your config)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def sliding_window(image, window_size, step_size):
    h, w = image.shape[:2]
    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def non_max_suppression(boxes, confidences, threshold):
    # Standard NMS to remove overlapping boxes
    if not boxes: return []
    boxes_array = np.array(boxes)
    
    # Prepare boxes for OpenCV NMS [x, y, w, h] -> [x, y, x2, y2]
    boxes_corners = boxes_array.copy()
    boxes_corners[:, 2] = boxes_array[:, 0] + boxes_array[:, 2]
    boxes_corners[:, 3] = boxes_array[:, 1] + boxes_array[:, 3]
    
    indices = cv2.dnn.NMSBoxes(
        boxes_corners.tolist(), confidences, 0.0, threshold
    )
    if len(indices) == 0: return []
    return [boxes[i] for i in indices.flatten()]