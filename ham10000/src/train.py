"""
train.py — production training loop for DermaNet on HAM10000.

Features:
  - Staged fine-tuning (freeze backbone for N epochs)
  - Discriminative learning rates
  - EMA (Exponential Moving Average) of weights
  - Mixed precision (torch.cuda.amp) when CUDA is available
  - Gradient clipping
  - Gradient accumulation
  - CosineAnnealingLR / OneCycleLR / ReduceLROnPlateau
  - Early stopping
  - CSV training log
  - Best + last checkpoint saving
"""
import os, sys, csv, json, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from copy import deepcopy

_THIS  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_THIS)
for _p in [_THIS, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset          import HAM10000Dataset
from model            import build_model
from losses           import build_loss
from metadata_encoder import MetadataEncoder
import evaluate


# ── Reproducibility ───────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── EMA ───────────────────────────────────────────────────────
class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of parameters updated each step as:
        shadow = decay * shadow + (1 - decay) * param
    Using the shadow weights at eval time typically improves accuracy by 0.3–1pp.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into model for evaluation."""
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            m.data.copy_(s.data)


# ── Data loaders ──────────────────────────────────────────────
def build_loaders(cfg: dict, encoder=None) -> dict:
    data_dir = cfg["data"]["data_dir"]
    bs       = cfg["train"]["batch_size"]
    nw       = cfg["data"].get("num_workers", 0)
    img_size = cfg["data"].get("img_size", 224)
    augment  = cfg["data"].get("augment", True)

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = HAM10000Dataset(
            data_dir=data_dir,
            split=split,
            metadata_encoder=encoder,
            img_size=img_size,
            augment=augment,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


# ── Optimizer + scheduler ─────────────────────────────────────
def build_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["train"]
    lr        = train_cfg["lr"]
    wd        = train_cfg.get("weight_decay", 1e-4)

    # Discriminative LR: backbone gets lr/10, classifier gets full lr
    use_disc = cfg["model"].get("freeze_epochs", 0) > 0
    if use_disc:
        param_groups = model.get_parameter_groups(
            lr_backbone=lr / 10,
            lr_classifier=lr,
        )
    else:
        param_groups = model.parameters()

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_cfg = cfg["train"].get("scheduler", "cosine")
    epochs    = cfg["train"]["epochs"]

    if sched_cfg == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif sched_cfg == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["train"]["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    elif sched_cfg == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler '{sched_cfg}'")


# ── Single epoch ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer,
              device, use_meta, train_mode,
              scaler=None, accum_steps=1,
              max_grad_norm=1.0, ema=None):

    model.train() if train_mode else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            if use_meta:
                images, meta, labels = batch
                meta   = meta.to(device)
            else:
                images, labels = batch
                meta = None

            images = images.to(device)
            labels = labels.to(device)

            # Mixed precision forward
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(images, meta)
                loss   = criterion(logits, labels)
                loss   = loss / accum_steps

            if train_mode:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    if ema:
                        ema.update(model)

            total_loss += loss.item() * accum_steps
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────
def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_meta     = cfg["model"].get("metadata_dim", 0) > 0
    use_ema      = cfg["train"].get("use_ema", False)
    accum_steps  = cfg["train"].get("gradient_accumulation", 1)
    max_grad     = cfg["train"].get("max_grad_norm", 1.0)
    freeze_ep    = cfg["model"].get("freeze_epochs", 0)
    early_stop   = cfg["train"].get("early_stopping_patience", 999)
    experiment   = cfg["logging"]["experiment_name"]

    print(f"\nDevice      : {device}")
    print(f"Experiment  : {experiment}")
    print(f"Architecture: {cfg['model'].get('architecture','resnet18')}")
    print(f"Metadata    : {use_meta}")
    print(f"EMA         : {use_ema}")
    print(f"Freeze ep   : {freeze_ep}")

    # ── Encoder ───────────────────────────────────────────────
    encoder = None
    if use_meta:
        split_csv = os.path.join(cfg["data"]["data_dir"], "HAM10000_split.csv")
        encoder   = MetadataEncoder(split_csv)

    # ── Loaders ───────────────────────────────────────────────
    loaders = build_loaders(cfg, encoder)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if freeze_ep > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for first {freeze_ep} epoch(s).")

    # ── EMA ───────────────────────────────────────────────────
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999)) if use_ema else None

    # ── Loss ──────────────────────────────────────────────────
    weights_path = os.path.join(cfg["data"]["data_dir"], "class_weights.npy")
    class_weights = torch.tensor(
        np.load(weights_path), dtype=torch.float32
    ).to(device)
    criterion = build_loss(cfg, class_weights)

    # ── Optimizer + scheduler ─────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer, cfg, steps_per_epoch=len(loaders["train"])
    )
    scaler    = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    sched_name = cfg["train"].get("scheduler", "cosine")

    # ── Dirs ──────────────────────────────────────────────────
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    log_dir  = os.path.join("ham10000", "experiments", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    log_path   = os.path.join(log_dir, "training_log.csv")
    best_ckpt  = os.path.join(ckpt_dir, "best_model.pt")
    last_ckpt  = os.path.join(ckpt_dir, "last_model.pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    best_val     = -float("inf")
    no_improve   = 0
    epochs       = cfg["train"]["epochs"]

    print(f"\nTraining for {epochs} epochs — "
          f"loss={cfg.get('loss',{}).get('name','cross_entropy')}, "
          f"scheduler={sched_name}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Staged fine-tuning: unfreeze backbone after freeze_epochs
        if epoch == freeze_ep + 1 and freeze_ep > 0:
            model.unfreeze_backbone()
            print(f"  [Epoch {epoch}] Backbone unfrozen — full fine-tuning begins.")

        # Train
        tr_loss = run_epoch(
            model, loaders["train"], criterion, optimizer,
            device, use_meta, train_mode=True,
            scaler=scaler, accum_steps=accum_steps,
            max_grad_norm=max_grad, ema=ema,
        )

        # Validate (use EMA weights if enabled)
        if ema:
            # Save the live (currently-training) weights before overwriting
            # the model with the EMA shadow for evaluation.
            live_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)

        val_loss = run_epoch(
            model, loaders["val"], criterion, optimizer,
            device, use_meta, train_mode=False,
        )

        y_true, y_pred, y_probs = evaluate.run_inference(
            model, loaders["val"], device, use_meta
        )
        metrics = evaluate.compute_metrics(y_true, y_pred, y_probs)
        val_bal_acc = metrics["balanced_accuracy"]

        # Restore the live weights after EMA eval so training resumes
        # from where it left off (previously this just re-copied the
        # EMA shadow onto the model again, silently discarding all
        # live training progress after the first eval).
        if ema:
            model.load_state_dict(live_state)

        # Scheduler step
        if sched_name == "plateau":
            scheduler.step(val_bal_acc)
        elif sched_name != "onecycle":
            scheduler.step()
        lr_now = optimizer.param_groups[-1]["lr"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={tr_loss:.4f} | val={val_loss:.4f} | "
            f"bal_acc={val_bal_acc:.4f} | lr={lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )

        # CSV log
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if epoch == 1:
                w.writerow(["epoch","train_loss","val_loss",
                             "val_bal_acc","lr"])
            w.writerow([epoch,
                        round(tr_loss, 4),
                        round(val_loss, 4),
                        round(val_bal_acc, 4),
                        round(lr_now, 6)])

        # Checkpointing
        payload = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state":     optimizer.state_dict(),
            "val_balanced_accuracy": val_bal_acc,
            "config":              cfg,
        }
        torch.save(payload, last_ckpt)

        if val_bal_acc > best_val:
            best_val   = val_bal_acc
            no_improve = 0
            torch.save(payload, best_ckpt)
            print(f"  → New best ({val_bal_acc:.4f}), saved.")
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\nBest val balanced accuracy: {best_val:.4f}")
    print(f"Checkpoints → {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ham10000/configs/baseline.yaml")
    main(p.parse_args().config)