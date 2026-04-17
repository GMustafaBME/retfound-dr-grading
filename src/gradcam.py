"""
Grad-CAM Explainability for DR Grading Model

Generates visual explanations showing which regions
of the fundus image the model focuses on for diagnosis.
"""

import torch
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps highlighting regions the model
    attends to when making DR grade predictions.
    
    Args:
        model: Trained PyTorch model
        target_layer: Layer to compute Grad-CAM for
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index (None = use predicted class)
        
        Returns:
            heatmap: Numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image as numpy array (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W) values in [0, 1]
        alpha: Overlay transparency
    
    Returns:
        overlay: Blended image as numpy array
    """
    colormap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(colormap) * alpha + np.float32(image) * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return overlay
