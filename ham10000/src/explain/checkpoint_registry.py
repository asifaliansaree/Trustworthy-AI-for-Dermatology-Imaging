"""
Checkpoint discovery/registry for the explainability pipeline.

Real layout (confirmed from your VS Code Explorer screenshot):
    ham10000/checkpoints/<arch-folder>/fold<N>_best.pt
    ham10000/checkpoints/<arch-folder>/fold<N>_last.pt
    ham10000/checkpoints/<arch-folder>/.cache/          (ignored)
    ham10000/checkpoints/<arch-folder>/.gitattributes    (ignored -- git-lfs pointer, not a checkpoint)

Run this file directly first, before anything else:
    python ham10000/src/explain/checkpoint_registry.py
It prints every discovered checkpoint, which config (if any) it matched
to, and whether the architecture is actually buildable by DermaNet
(checked live against model.py's ARCH_REGISTRY, not a hardcoded copy).

checkpoint_key format used everywhere else in this pipeline:
    "<arch-folder>_fold<N>"   e.g. "convnext-tiny_fold0"
"""
import os
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_ROOT = "ham10000/checkpoints"
CFG_ROOT = "ham10000/configs"
IGNORE_ENTRIES = {".cache", ".gitattributes", ".git"}
FOLD_FILE_RE = re.compile(r"^fold(\d+)_(best|last)\.pt$")

# Explicit arch-folder -> config-stem mapping. Deliberately NOT
# auto-fuzzy-matched for these -- your configs/ has multiple variants per
# architecture (e.g. densenet121_v12recipe.yaml vs
# densenet121_v13_samplerfix.yaml vs densenet121_v12recipe_wd.yaml), and
# guessing wrong here means loading a checkpoint with the WRONG dropout/
# training config, which fails silently (wrong dropout won't crash, it'll
# just be subtly wrong). Confirm/edit these yourself:
ARCH_TO_CONFIG_STEM = {
    "convnext-tiny": "convnext_tiny_v1",
    "densenet121": "densenet121_v12recipe",   # ambiguous: v13_samplerfix and _wd variants also exist -- confirm this is the one actually used for the 30-checkpoint set
    "resnet50": "resnet50_v12recipe",
    "efficientnet-v2-s": None,   # no matching config file found in ham10000/configs/
    "maxvit-tiny": None,          # no matching config file found
    "swin-tiny": None,            # no matching config file found
}

# arch-folder name -> the `architecture` string DermaNet/model.py expects
# (config['model']['architecture']). Used only to sanity-check against
# model.py's live ARCH_REGISTRY below.
ARCH_TO_MODEL_ARCH_NAME = {
    "convnext-tiny": "convnext_tiny",
    "densenet121": "densenet121",
    "resnet50": "resnet50",
    "efficientnet-v2-s": "efficientnet_v2_s",
    "maxvit-tiny": "maxvit_tiny",
    "swin-tiny": "swin_tiny",
}


def _available_configs(cfg_root: str) -> dict:
    if not os.path.isdir(cfg_root):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(cfg_root, f)
        for f in os.listdir(cfg_root) if f.endswith((".yaml", ".yml"))
    }


def _model_arch_registry_keys() -> set:
    """Live-check what model.py actually knows how to build, instead of
    trusting a hardcoded duplicate list that could drift out of sync."""
    try:
        from model import ARCH_REGISTRY
        return set(ARCH_REGISTRY.keys())
    except Exception as e:
        print(f"[WARN] Couldn't import model.ARCH_REGISTRY to sanity-check "
              f"architecture support ({e}). Skipping that check.")
        return set()


def discover_checkpoints(ckpt_root: str = CKPT_ROOT, cfg_root: str = CFG_ROOT) -> dict:
    """
    Returns:
        { checkpoint_key: {
              "ckpt_path": str,
              "ckpt_filename": str,
              "cfg_path": str or None,
              "cfg_stem": str or None,
              "fold": int,
              "arch_folder": str,
              "model_arch_name": str or None,
              "arch_supported": bool,   # True if model.py's ARCH_REGISTRY can build it
          } }
    checkpoint_key is "<arch-folder>_fold<N>", e.g. "convnext-tiny_fold0".
    """
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(
            f"Checkpoint root '{ckpt_root}' does not exist relative to cwd "
            f"({os.getcwd()}). Run scripts from the repo root."
        )

    available_cfgs = _available_configs(cfg_root)
    supported_archs = _model_arch_registry_keys()
    entries = {}

    for arch_folder in sorted(os.listdir(ckpt_root)):
        if arch_folder in IGNORE_ENTRIES:
            continue
        arch_dir = os.path.join(ckpt_root, arch_folder)
        if not os.path.isdir(arch_dir):
            continue

        # Prefer best_model per fold; fall back to last if best is missing.
        fold_files = {}  # fold_num -> {"best": path or None, "last": path or None}
        for fname in os.listdir(arch_dir):
            if fname in IGNORE_ENTRIES:
                continue
            m = FOLD_FILE_RE.match(fname)
            if not m:
                continue
            fold_num, kind = int(m.group(1)), m.group(2)
            fold_files.setdefault(fold_num, {})[kind] = os.path.join(arch_dir, fname)

        cfg_stem = ARCH_TO_CONFIG_STEM.get(arch_folder)
        cfg_path = available_cfgs.get(cfg_stem) if cfg_stem else None
        model_arch_name = ARCH_TO_MODEL_ARCH_NAME.get(arch_folder)
        arch_supported = (model_arch_name in supported_archs) if supported_archs else None

        for fold_num, kinds in sorted(fold_files.items()):
            ckpt_path = kinds.get("best") or kinds.get("last")
            ckpt_filename = "fold*_best.pt" if kinds.get("best") else "fold*_last.pt"
            key = f"{arch_folder}_fold{fold_num}"
            entries[key] = {
                "ckpt_path": ckpt_path,
                "ckpt_filename": os.path.basename(ckpt_path),
                "cfg_path": cfg_path,
                "cfg_stem": cfg_stem,
                "fold": fold_num,
                "arch_folder": arch_folder,
                "model_arch_name": model_arch_name,
                "arch_supported": arch_supported,
            }

    return entries


def print_report(entries: dict) -> None:
    if not entries:
        print(f"No checkpoints found under '{CKPT_ROOT}'. Check the path.")
        return

    print(f"{'checkpoint_key':<28} {'ckpt file':<15} {'config':<28} {'arch buildable?'}")
    print("-" * 90)
    problems = []
    for key, e in sorted(entries.items()):
        cfg_display = os.path.basename(e["cfg_path"]) if e["cfg_path"] else "MISSING"
        arch_display = ("yes" if e["arch_supported"] else
                         "NO (add to model.py ARCH_REGISTRY)" if e["arch_supported"] is False else
                         "unknown")
        print(f"{key:<28} {e['ckpt_filename']:<15} {cfg_display:<28} {arch_display}")
        if e["cfg_path"] is None:
            problems.append(f"{key}: no config matched (arch_folder='{e['arch_folder']}')")
        if e["arch_supported"] is False:
            problems.append(f"{key}: model.py can't build '{e['model_arch_name']}' yet")

    if problems:
        print(f"\n[WARN] {len(problems)} issue(s) to resolve before these checkpoints "
              f"can be loaded:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\nAll discovered checkpoints have a matched config and a supported architecture.")


if __name__ == "__main__":
    print_report(discover_checkpoints())
