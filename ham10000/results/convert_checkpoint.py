"""
convert_checkpoint.py — one-time fix for the ARCH_REGISTRY refactor.
Remaps a checkpoint trained under the old raw-resnet DermaNet backbone
(self.backbone = resnet18(...); self.backbone.fc = Identity()) to the
current indexed nn.Sequential backbone
(self.backbone = nn.Sequential(*list(resnet18().children())[:-1])).
"""
import torch

CKPT_PATH = 'ham10000/checkpoints/baseline_image_only/best_model.pt'
OUT_PATH  = 'ham10000/checkpoints/baseline_image_only/best_model_remapped.pt'

NAME_TO_INDEX = {
    'conv1': '0',
    'bn1':   '1',
    'layer1': '4',
    'layer2': '5',
    'layer3': '6',
    'layer4': '7',
}

ckpt = torch.load(CKPT_PATH, map_location='cpu')
old_sd = ckpt['model_state_dict']
new_sd = {}

for k, v in old_sd.items():
    if k.startswith('backbone.'):
        rest = k[len('backbone.'):]
        prefix = rest.split('.', 1)[0]
        if prefix in NAME_TO_INDEX:
            rest = NAME_TO_INDEX[prefix] + rest[len(prefix):]
            new_sd[f'backbone.{rest}'] = v
        elif prefix == 'fc':
            # backbone.fc was nn.Identity() — no equivalent param, drop it
            continue
        else:
            raise ValueError(f'Unhandled backbone key: {k}')
    else:
        new_sd[k] = v  # classifier.* etc. unchanged

ckpt['model_state_dict'] = new_sd
torch.save(ckpt, OUT_PATH)
print(f'Remapped {len(old_sd)} -> {len(new_sd)} keys')
print(f'Saved: {OUT_PATH}')