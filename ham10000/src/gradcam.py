"""
Grad-CAM implementation using Captum.
Produces heatmaps overlaid on original images.
Used for XAI benchmark in Weeks 4-6.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

_SRC  = os.path.dirname(os.path.abspath(__file__))
_HAM  = os.path.dirname(_SRC)
_ROOT = os.path.dirname(_HAM)
for p in [_SRC, _HAM, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from captum.attr import LayerGradCam, LayerAttribution
from model import DermaNet

CLASSES   = ['akiec','bcc','bkl','df','mel','nv','vasc']
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]

def get_last_conv_layer(model: DermaNet):
    """Return the last convolutional layer for ResNet-18."""
    # ResNet-18 backbone: layer4 is the last conv block
    return model.backbone[-3][-1].conv2

def preprocess_image(img_path: str) -> tuple:
    """Load image and return both tensor and original PIL."""
    pil_img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, pil_img

def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize tensor back to displayable image."""
    img = tensor.squeeze(0).cpu().numpy()
    mean = np.array(MEAN)[:, None, None]
    std  = np.array(STD)[:, None, None]
    img  = (img * std + mean).clip(0, 1)
    return img.transpose(1, 2, 0)

def compute_gradcam(
    model:      DermaNet,
    img_tensor: torch.Tensor,
    target_class: int,
    device:     torch.device,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.
    Returns heatmap as numpy array (224, 224), values in [0, 1].
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    layer     = get_last_conv_layer(model)
    grad_cam  = LayerGradCam(model, layer)

    attribution = grad_cam.attribute(
        img_tensor,
        target=target_class,
    )

    # Upsample to input resolution
    upsampled = LayerAttribution.interpolate(
        attribution,
        interpolate_dims=(224, 224),
        interpolate_mode='bilinear',
    )

    heatmap = upsampled.squeeze().cpu().detach().numpy()

    # Normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def overlay_heatmap(
    original_img: np.ndarray,
    heatmap:      np.ndarray,
    alpha:        float = 0.5,
) -> np.ndarray:
    """Overlay a jet colormap heatmap on the original image."""
    import matplotlib.cm as cm
    colored = cm.jet(heatmap)[:, :, :3]  # drop alpha channel
    return (1 - alpha) * original_img + alpha * colored

def visualize_gradcam(
    model:        DermaNet,
    img_path:     str,
    true_class:   int,
    device:       torch.device,
    save_path:    str = None,
    title_extra:  str = "",
) -> dict:
    """
    Full Grad-CAM pipeline for one image.
    Returns dict with prediction, confidence, and heatmap.
    """
    img_tensor, pil_img = preprocess_image(img_path)
    original_np = tensor_to_display(img_tensor)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs  = F.softmax(logits, dim=1).cpu()
        pred   = probs.argmax(1).item()
        conf   = probs[0, pred].item()

    # Compute Grad-CAM for predicted class
    heatmap = compute_gradcam(model, img_tensor, pred, device)
    overlay = overlay_heatmap(original_np, heatmap)

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_np)
        axes[0].set_title(
            f"Original\nTrue: {CLASSES[true_class]}",
            fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay\nPred: {CLASSES[pred]} ({conf:.2f})",
            fontsize=10,
            color='green' if pred == true_class else 'red')
        axes[2].axis('off')

        if title_extra:
            fig.suptitle(title_extra, fontsize=11, y=1.01)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'pred_class':  pred,
        'pred_name':   CLASSES[pred],
        'true_class':  true_class,
        'true_name':   CLASSES[true_class],
        'confidence':  conf,
        'correct':     pred == true_class,
        'heatmap':     heatmap,
        'overlay':     overlay,
    }