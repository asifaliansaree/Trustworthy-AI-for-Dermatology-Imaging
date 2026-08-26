"""
Checkpoint discovery/registry for the explainability pipeline.

Layout: ham10000/checkpoints/<arch-folder>/fold<N>_best.{pt,pth}
(both extensions supported -- MaxViT's checkpoints came down from HF as
.pth while everything else is .pt; no reason to assume every future
architecture will match whatever extension the first few happened to use)

Run this file directly first, before anything else:
    python ham10000/src/explain/checkpoint_registry.py

Design note (changed from earlier versions): architecture support is
checked by reading the `architecture:` field directly out of the matched
config YAML, then checking THAT string against model.py's live
ARCH_REGISTRY -- not a separately hardcoded arch-folder -> arch-name
dict. Two independent guessed mappings (folder name -> config stem,
folder name -> architecture string) drift out of sync with each other
easily, as already happened once. Reading the real YAML content removes
that whole failure mode -- the config file is the actual source of truth
for what architecture string the checkpoint needs.
"""
import os
import re
import sys
import yaml

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_THIS, os.path.dirname(_THIS), os.path.dirname(os.path.dirname(_THIS))]:
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_ROOT = "ham10000/checkpoints"
CFG_ROOT = "ham10000/configs"
IGNORE_ENTRIES = {".cache", ".gitattributes", ".git"}
FOLD_FILE_RE = re.compile(r"^fold(\d+)_(best|last)\.(pt|pth)$")

# Explicit arch-folder -> config-stem mapping. Kept explicit (not
# fuzzy-matched) because configs/ has had multiple differently-named
# variants per architecture at different points (v12recipe vs
# v13_samplerfix, etc.) -- guessing wrong here means loading with the
# WRONG training config, which fails silently. Confirm/edit as your
# configs/ folder changes.
ARCH_TO_CONFIG_STEM = {
    "convnext-tiny": "convnext_tiny_v1",
    "densenet121": "densenet121_v13_samplerfix",
    "resnet50": "resnet50_v12recipe",
    "efficientnet-v2-s": "efficientnet_v2_s",
    "maxvit-tiny": "maxvit_tiny_config",
    "swin-tiny": "swin_t",
}


def _available_configs(cfg_root: str) -> dict:
    if not os.path.isdir(cfg_root):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(cfg_root, f)
        for f in os.listdir(cfg_root) if f.endswith((".yaml", ".yml"))
    }


def _model_arch_registry_keys() -> set:
    try:
        from model import ARCH_REGISTRY
        return set(ARCH_REGISTRY.keys())
    except Exception as e:
        print(f"[WARN] Couldn't import model.ARCH_REGISTRY to sanity-check "
              f"architecture support ({e}). Skipping that check.")
        return set()


def _read_architecture_from_config(cfg_path: str):
    """Read the actual `architecture:` field out of the YAML -- the real
    source of truth, not a guessed folder-name mapping. If it's not
    where expected, print exactly what IS there instead of silently
    returning None, so a structural mismatch is diagnosable in one run
    instead of another round of guessing."""
    fname = os.path.basename(cfg_path)
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] {fname}: YAML failed to parse ({e}). This config's "
              f"checkpoints can't be loaded until the YAML syntax is fixed.")
        return None

    if not isinstance(cfg, dict):
        print(f"[WARN] {fname}: parsed, but isn't a key/value mapping at the top level.")
        return None

    model_section = cfg.get("model")
    if isinstance(model_section, dict) and "architecture" in model_section:
        return model_section["architecture"]

    if isinstance(model_section, dict):
        print(f"[WARN] {fname}: has a 'model' section but no 'architecture' key "
              f"inside it. Keys actually found under 'model': {list(model_section.keys())}")
    else:
        print(f"[WARN] {fname}: no 'model.architecture' path found. "
              f"Full parsed content dumped below so this can get fixed in one pass:")
        print(f"----- {fname} -----")
        print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        print(f"----- end {fname} -----")
    return None


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
              "model_arch_name": str or None,   # read from the YAML itself
              "arch_supported": bool or None,
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
        model_arch_name = _read_architecture_from_config(cfg_path) if cfg_path else None
        arch_supported = (model_arch_name in supported_archs) if (model_arch_name and supported_archs) else None

        for fold_num, kinds in sorted(fold_files.items()):
            ckpt_path = kinds.get("best") or kinds.get("last")
            ext = os.path.splitext(ckpt_path)[1]
            ckpt_filename = f"fold*_best{ext}" if kinds.get("best") else f"fold*_last{ext}"
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

    print(f"{'checkpoint_key':<28} {'ckpt file':<16} {'config':<32} {'architecture':<16} {'buildable?'}")
    print("-" * 110)
    problems = []
    for key, e in sorted(entries.items()):
        cfg_display = os.path.basename(e["cfg_path"]) if e["cfg_path"] else "MISSING"
        arch_name_display = e["model_arch_name"] or "?"
        arch_display = ("yes" if e["arch_supported"] else
                         "NO (add to model.py ARCH_REGISTRY)" if e["arch_supported"] is False else
                         "unknown")
        print(f"{key:<28} {e['ckpt_filename']:<16} {cfg_display:<32} {arch_name_display:<16} {arch_display}")
        if e["cfg_path"] is None:
            problems.append(f"{key}: no config matched (arch_folder='{e['arch_folder']}')")
        elif e["arch_supported"] is False:
            problems.append(f"{key}: model.py can't build '{e['model_arch_name']}' yet (from {os.path.basename(e['cfg_path'])})")

    if problems:
        print(f"\n[WARN] {len(problems)} issue(s) to resolve before these checkpoints "
              f"can be loaded:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\nAll discovered checkpoints have a matched config and a supported architecture.")


if __name__ == "__main__":
    print_report(discover_checkpoints())