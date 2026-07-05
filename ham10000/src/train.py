import os, sys, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import HAM10000Dataset
from metadata_encoder import MetadataEncoder
from model import DermaNet
import evaluate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg):
    metadata_dim = cfg['model']['metadata_dim']
    encoder = None
    if metadata_dim > 0:
        split_csv = os.path.join(cfg['data']['data_dir'], "HAM10000_split.csv")
        encoder = MetadataEncoder(split_csv)
        assert encoder.dim == metadata_dim, \
            f"config metadata_dim={metadata_dim} but encoder produces {encoder.dim}-d vectors"

    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = HAM10000Dataset(data_dir=cfg['data']['data_dir'], split=split,
                              metadata_encoder=encoder)
        loaders[split] = DataLoader(ds, batch_size=cfg['train']['batch_size'],
                                     shuffle=(split == 'train'),
                                     num_workers=cfg['data'].get('num_workers', 0))
    return loaders


def run_epoch(model, loader, criterion, optimizer, device, use_metadata, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, n_batches = 0.0, 0
    torch.set_grad_enabled(train_mode)
    for batch in loader:
        if use_metadata:
            images, meta, labels = batch
            meta = meta.to(device)
        else:
            images, labels = batch
            meta = None
        images, labels = images.to(device), labels.to(device)
        logits = model(images, meta)
        loss = criterion(logits, labels)
        if train_mode:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); n_batches += 1
    torch.set_grad_enabled(True)
    return total_loss / max(n_batches, 1)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loaders = build_dataloaders(cfg)
    use_metadata = cfg['model']['metadata_dim'] > 0
    print(f"Mode: {'late fusion' if use_metadata else 'image-only baseline'}")

    model = DermaNet(num_classes=cfg['model']['num_classes'],
                      metadata_dim=cfg['model']['metadata_dim'],
                      pretrained=cfg['model']['pretrained'],
                      dropout=cfg['model']['dropout']).to(device)

    class_weights = torch.tensor(
        np.load(os.path.join(cfg['data']['data_dir'], "class_weights.npy")),
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                                  weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    best_val = -float('inf')

    for epoch in range(1, cfg['train']['epochs'] + 1):
        t0 = time.time()
        train_loss = run_epoch(model, loaders['train'], criterion, optimizer, device, use_metadata, True)
        val_loss = run_epoch(model, loaders['val'], criterion, optimizer, device, use_metadata, False)
        y_true, y_pred, y_probs = evaluate.run_inference(model, loaders['val'], device, use_metadata)
        val_bal_acc = evaluate.compute_metrics(y_true, y_pred, y_probs)['balanced_accuracy']
        scheduler.step()
        print(f"Epoch {epoch:3d}/{cfg['train']['epochs']} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_bal_acc {val_bal_acc:.4f} | {time.time()-t0:.1f}s")

        if val_bal_acc > best_val:
            best_val = val_bal_acc
            ckpt = os.path.join(cfg['output']['checkpoint_dir'], 'best_model.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_balanced_accuracy': val_bal_acc, 'config': cfg}, ckpt)
            print(f"  -> new best ({val_bal_acc:.4f}), saved to {ckpt}")

    print(f"Best val balanced accuracy: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline.yaml')
    args = parser.parse_args()
    main(args.config)