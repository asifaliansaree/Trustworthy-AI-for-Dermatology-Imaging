import os, sys, yaml, json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_SRC  = os.path.dirname(os.path.abspath(__file__))
_HAM  = os.path.dirname(_SRC)
_ROOT = os.path.dirname(_HAM)
for p in [_SRC, _HAM, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

CLASSES   = ['akiec','bcc','bkl','df','mel','nv','vasc']
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]

IMG_DIRS = [
    "ham10000/data/HAM10000_images_part_1",
    "ham10000/data/HAM10000_images_part_2",
]

def find_image(image_id: str) -> str:
    for d in IMG_DIRS:
        p = os.path.join(d, image_id + ".jpg")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found: {image_id}")

def load_image(img_path: str):
    """Returns (tensor [1,3,224,224], original PIL, numpy display)."""
    pil = Image.open(img_path).convert('RGB')
    tf  = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    tensor = tf(pil).unsqueeze(0)
    display = np.array(pil.resize((224, 224))) / 255.0
    return tensor, pil, display

def normalize_attr(attr: np.ndarray) -> np.ndarray:
    """Normalize attribution to [0, 1]."""
    attr = np.maximum(attr, 0)
    mx   = attr.max()
    return attr / mx if mx > 0 else attr

def overlay_on_image(display: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """Jet colormap overlay on original image."""
    colored = cm.jet(heatmap)[:, :, :3]
    return (1 - alpha) * display + alpha * colored

def get_prediction(model, tensor, device):
    """Returns (pred_idx, confidence, all_probs)."""
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).cpu()
        pred   = probs.argmax(1).item()
        conf   = probs[0, pred].item()
    return pred, conf, probs[0].numpy()

def remap_resnet_backbone_keys(state_dict: dict) -> dict:
    """
    Remap a checkpoint's ResNet-18 backbone keys from the named-submodule
    format (backbone.conv1, backbone.bn1, backbone.layer1, ...) that older
    versions of DermaNet used, into the nn.Sequential numeric format that
    the current DermaNet.__init__ produces:
        conv1  -> 0
        bn1    -> 1
        layer1 -> 4
        layer2 -> 5
        layer3 -> 6
        layer4 -> 7
    Keys that don't match this pattern (classifier.*, EfficientNet keys,
    etc.) are passed through unchanged. Safe to call on any checkpoint —
    if the keys are already in the numeric format, nothing changes.
    """
    name_to_idx = {
        "conv1":  "0",
        "bn1":    "1",
        "layer1": "4",
        "layer2": "5",
        "layer3": "6",
        "layer4": "7",
    }
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            rest = k[len("backbone."):]
            head, _, tail = rest.partition(".")
            if head in name_to_idx:
                new_key = f"backbone.{name_to_idx[head]}"
                if tail:
                    new_key += f".{tail}"
                remapped[new_key] = v
                continue
        remapped[k] = v
    return remapped

def load_model_and_config(
    ckpt_path="ham10000/checkpoints/baseline_image_only/best_model.pt",
    cfg_path ="ham10000/configs/baseline.yaml",
):
    from model import DermaNet
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = DermaNet(
        num_classes  = 7,
        metadata_dim = 0,
        pretrained   = False,
        dropout      = cfg['model']['dropout'],
        arch         = cfg['model'].get('architecture','resnet18'),
    ).to(device)
    state_dict = remap_resnet_backbone_keys(ckpt['model_state_dict'])
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: epoch {ckpt['epoch']}, "
          f"val_bal_acc={ckpt['val_balanced_accuracy']:.4f}")
    return model, device, cfg

def load_xai_targets(path="ham10000/results/xai_target_cases.json"):
    with open(path) as f:
        return json.load(f)

def save_triple_figure(display, heatmap, overlay, title,
                       pred_name, true_name, conf, save_path,
                       method_name="Grad-CAM"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display)
    axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"{method_name} heatmap", fontsize=10)
    axes[1].axis('off')
    correct = pred_name == true_name
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay | Pred: {pred_name} ({conf:.2f})",
        fontsize=10,
        color='#1D9E75' if correct else '#E24B4A')
    axes[2].axis('off')
    if title:
        fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()