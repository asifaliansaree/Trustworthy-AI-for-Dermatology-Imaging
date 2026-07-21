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
  - Optional WeightedRandomSampler for class-imbalance handling
"""
import os, sys, csv, json, random, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        # BatchNorm running_mean/running_var are buffers, not parameters, so
        # the loop above never touches them. Left alone, eval-time BN stats
        # come from the live (fast-moving) model while the weights come from
        # the smoothed EMA shadow -- an internally inconsistent pairing.
        # Smooth float buffers the same way; integer buffers (e.g.
        # num_batches_tracked) are just copied since averaging doesn't apply.
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            if s.dtype.is_floating_point:
                s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)
            else:
                s.data.copy_(m.data)

    def apply_to(self, model: nn.Module):
        """Copy EMA weights into model for evaluation."""
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            m.data.copy_(s.data)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            m.data.copy_(s.data)


# ── Data loaders ──────────────────────────────────────────────
# Maps the HAM10000 diagnosis codes to the same class indices used
# elsewhere in the pipeline (dataset.py / class_weights.npy ordering).
CLASS_MAP = {"mel": 0, "nv": 1, "bcc": 2, "akiec": 3, "bkl": 4, "df": 5, "vasc": 6}


def build_loaders(cfg: dict, encoder=None) -> dict:
    data_dir = cfg["data"]["data_dir"]
    bs       = cfg["train"]["batch_size"]
    nw       = cfg["data"].get("num_workers", 0)
    img_size = cfg["data"].get("img_size", 224)
    augment  = cfg["data"].get("augment", True)

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = HAM10000Dataset(
            data_dir=data_dir,
            split=split,
            metadata_encoder=encoder,
            img_size=img_size,
            # Never augment val/test — only the training split gets it.
            augment=(augment if split == "train" else False),
        )

    # WeightedRandomSampler — rebalances batches by class frequency.
    # This is a separate on/off switch (cfg["train"]["use_weighted_sampler"])
    # so it can be ablated independently of Focal Loss, per the requested
    # experiment plan (sampler vs. focal, not combined).
    use_sampler = cfg["train"].get("use_weighted_sampler", False)
    if use_sampler:
        split_csv = os.path.join(data_dir, "HAM10000_split.csv")
        df        = pd.read_csv(split_csv)
        train_df  = df[df["split"] == "train"]

        labels       = train_df["dx"].map(CLASS_MAP).values
        class_counts = np.bincount(labels, minlength=len(CLASS_MAP))

        # sampler_power=1.0 -> full inverse-frequency (near-uniform class
        # exposure every epoch, original behavior). Lower values soften the
        # correction: rare classes are still boosted, but not repeated as
        # many times per epoch, which reduces memorization risk on the
        # smallest classes (df: 86, vasc: 114) once combined with a
        # class-weighted loss that's already correcting the same imbalance.
        sampler_power = cfg["train"].get("sampler_power", 1.0)
        sample_weights = 1.0 / np.power(class_counts[labels], sampler_power)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            datasets["train"],
            batch_size=bs,
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
        print("WeightedRandomSampler enabled "
              f"(class counts: {class_counts.tolist()})")
    else:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": train_loader,
        "val": DataLoader(
            datasets["val"], batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"], batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=torch.cuda.is_available(),
        ),
    }


# ── Optimizer + scheduler ─────────────────────────────────────
def build_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["train"]
    lr        = train_cfg["lr"]
    wd        = train_cfg.get("weight_decay", 1e-4)

    # Discriminative LR: backbone gets a lower LR, classifier gets full lr.
    # Previously this was tied to freeze_epochs > 0, which meant it could
    # never be tested on its own -- every run so far either got both
    # (frozen-then-full-LR) or neither. train.backbone_lr_ratio lets you
    # use discriminative LR with freeze_epochs=0 (always-unfrozen), which
    # is the untested combination.
    backbone_lr_ratio = train_cfg.get("backbone_lr_ratio", None)
    use_disc = (
        cfg["model"].get("freeze_epochs", 0) > 0
        or backbone_lr_ratio is not None
    )
    if use_disc:
        ratio = backbone_lr_ratio if backbone_lr_ratio is not None else 0.1
        param_groups = model.get_parameter_groups(
            lr_backbone=lr * ratio,
            lr_classifier=lr,
        )
    else:
        param_groups = model.parameters()

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_cfg = cfg["train"].get("scheduler", "cosine")
    epochs    = cfg["train"]["epochs"]
    warmup_ep = cfg["train"].get("lr_warmup_epochs", 0)

    if sched_cfg == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_ep, 1), eta_min=1e-6
        )
        if warmup_ep > 0:
            # Linear warmup from 10% of base LR up to 100%, then hand off to cosine.
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_ep]
            )
        return cosine
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
              max_grad_norm=1.0, ema=None,
              scheduler=None, step_scheduler_per_batch=False,
              mixup_alpha=0.0):
    """
    scheduler / step_scheduler_per_batch:
        OneCycleLR must be stepped once per optimizer.step() call across
        the ENTIRE training run (not once per epoch) -- its internal
        schedule is defined in terms of total batches, via
        `steps_per_epoch` and `epochs` passed at construction time.
        Previously this scheduler was never stepped at all, so the LR
        stayed frozen at its initial warmup value (max_lr / div_factor)
        for the whole run. Pass step_scheduler_per_batch=True only for
        the onecycle case; cosine/plateau continue to be stepped once
        per epoch in main(), as before.

    mixup_alpha:
        0.0 (default) = off, identical to prior behavior.
        >0 = standard mixup (Zhang et al. 2018), train only. Blends each
        image with another random image in the same batch (lam ~
        Beta(alpha, alpha)) and takes a lam-weighted combination of the
        two per-sample losses. This is the direct fix for memorizing the
        rare classes (df: 86, vasc: 114 train images) once the weighted
        sampler puts them in every batch -- mixup means the model never
        sees the exact same pixels twice, only blends of them, which
        breaks memorization without reducing how often rare classes
        contribute gradient signal. Only applied when train_mode=True;
        validation is always a clean, unmixed forward pass.
    """
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

            do_mixup = train_mode and mixup_alpha > 0.0
            if do_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[perm]
                labels_b = labels[perm]

            # Mixed precision forward
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(images, meta)
                if do_mixup:
                    loss = lam * criterion(logits, labels) + \
                           (1 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
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

                    # OneCycleLR: step once per optimizer step, not per epoch.
                    if step_scheduler_per_batch and scheduler is not None:
                        scheduler.step()

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
    mixup_alpha  = cfg["train"].get("mixup_alpha", 0.0)
    experiment   = cfg["logging"]["experiment_name"]

    print(f"\nDevice      : {device}")
    print(f"Experiment  : {experiment}")
    print(f"Architecture: {cfg['model'].get('architecture','resnet18')}")
    print(f"Metadata    : {use_meta}")
    print(f"EMA         : {use_ema}")
    print(f"Freeze ep   : {freeze_ep}")
    print(f"Mixup alpha : {mixup_alpha}")

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
    step_per_batch = (sched_name == "onecycle")

    # ── Dirs ──────────────────────────────────────────────────
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    log_dir  = os.path.join("ham10000", "experiments", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Checkpoint Directory :", os.path.abspath(ckpt_dir))
    print("Log Directory        :", os.path.abspath(log_dir))
    print("Checkpoint Exists    :", os.path.exists(ckpt_dir))
    print("=" * 60)

    log_path   = os.path.join(log_dir, "training_log.csv")
    best_ckpt  = os.path.join(ckpt_dir, "best_model.pt")
    last_ckpt  = os.path.join(ckpt_dir, "last_model.pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    best_val     = -float("inf")
    no_improve   = 0
    epochs       = cfg["train"]["epochs"]
    smooth_win   = cfg["train"].get("best_metric_smoothing_window", 1)
    bal_acc_hist = []

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
        # Mixup only makes sense once the backbone is unfrozen: during the
        # frozen head-only phase the feature extractor is fixed, so mixed
        # pixel-space inputs just add noise to a classifier that has no
        # capacity to adapt around it. Gating this out fixed a real
        # underfitting regression (v8 run: bal_acc capped at 0.754 with
        # mixup active for all 3 frozen epochs).
        epoch_mixup_alpha = mixup_alpha if epoch > freeze_ep else 0.0
        tr_loss = run_epoch(
            model, loaders["train"], criterion, optimizer,
            device, use_meta, train_mode=True,
            scaler=scaler, accum_steps=accum_steps,
            max_grad_norm=max_grad, ema=ema,
            scheduler=scheduler, step_scheduler_per_batch=step_per_batch,
            mixup_alpha=epoch_mixup_alpha,
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

        # Scheduler step — cosine/plateau step once per epoch here.
        # onecycle is stepped per-batch inside run_epoch() instead, so
        # it is explicitly skipped in this block.
        if sched_name == "plateau":
            scheduler.step(val_bal_acc)
        elif sched_name == "cosine":
            scheduler.step()
        # onecycle: intentionally no epoch-level step -- handled per-batch above.
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
        # ema_state_dict holds the actual weights val_bal_acc was measured
        # on (ema.shadow, applied to `model` above then evaluated).
        # model_state_dict holds the raw, still-training weights (restored
        # via live_state above) — kept so training can resume / so a
        # non-EMA run still saves something meaningful. Previously only
        # model_state_dict was saved, so the EMA-evaluated accuracy in this
        # same payload described weights that were never written to disk.
        payload = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "ema_state_dict":      ema.shadow.state_dict() if ema else None,
            "optimizer_state":     optimizer.state_dict(),
            "val_balanced_accuracy": val_bal_acc,
            "config":              cfg,
        }
        torch.save(payload, last_ckpt)

        if os.path.exists(last_ckpt):
            size = os.path.getsize(last_ckpt) / (1024 * 1024)
            print(f"✓ last_model.pt saved ({size:.2f} MB)")
        else:
            print("✗ ERROR: last_model.pt was NOT saved")

        # Smooth the metric used for "best" selection so a single lucky/unlucky
        # epoch (see the noisy zig-zag in earlier runs) can't get checkpointed
        # as best just because it happened to land on a spike.
        bal_acc_hist.append(val_bal_acc)
        window       = bal_acc_hist[-smooth_win:]
        smoothed_val = sum(window) / len(window)

        if smoothed_val > best_val:
            best_val = smoothed_val
            no_improve = 0

            torch.save(payload, best_ckpt)

            if os.path.exists(best_ckpt):
                size = os.path.getsize(best_ckpt) / (1024 * 1024)
                print(f"✓ best_model.pt saved ({size:.2f} MB)")
            else:
                print("✗ ERROR: best_model.pt was NOT saved")

            print("Checkpoint directory contents:")
            print(os.listdir(ckpt_dir))

            print(
                f"  → New best (raw={val_bal_acc:.4f}, "
                f"smoothed={smoothed_val:.4f}), saved."
            )
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print("\n" + "=" * 60)
    print(f"Best val balanced accuracy (smoothed): {best_val:.4f}")
    print(f"Checkpoint directory: {os.path.abspath(ckpt_dir)}")

    if os.path.exists(ckpt_dir):
        print("\nSaved files:")
        for f in os.listdir(ckpt_dir):
            path = os.path.join(ckpt_dir, f)
            print(f" - {f} ({os.path.getsize(path)/1024/1024:.2f} MB)")
    else:
        print("Checkpoint directory does not exist!")

    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ham10000/configs/baseline.yaml")
    main(p.parse_args().config)