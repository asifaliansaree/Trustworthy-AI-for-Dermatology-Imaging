"""
Compares deletion/insertion faithfulness across the three XAI methods
(IG, RISE, FovEx) on the convnext-tiny fold3 pilot set.

Reads:
    ham10000/results/xai/eval/convnext-tiny_fold3/ig_faithfulness.csv
    ham10000/results/xai/eval/convnext-tiny_fold3/rise_faithfulness.csv
    ham10000/results/xai/eval/convnext-tiny_fold3/fovex_faithfulness.csv

IG has two attribution/eval baselines (black, blurred); RISE and FovEx only
have 'blurred'. For the cross-method comparison panels, IG rows are filtered
to eval_baseline == 'blurred' so all three methods are compared on the same
perturbation baseline. A separate panel shows IG's black-vs-blurred ablation
on its own.

Metrics:
    deletion_auc  -- lower is better (faithful attributions delete evidence
                     fast, so predicted-class probability collapses quickly)
    insertion_auc -- higher is better (faithful attributions reveal evidence
                     fast, so predicted-class probability rises quickly)

pointing_game is currently 'not_computed' for every row in the pilot set and
is skipped.

Usage:
    python ham10000/results/plot_faithfulness_comparison.py
    python ham10000/results/plot_faithfulness_comparison.py --key convnext-tiny_fold3
    python ham10000/results/plot_faithfulness_comparison.py \
        --ig path/to/ig.csv --rise path/to/rise.csv --fovex path/to/fovex.csv \
        --out path/to/out_dir
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_COLORS = {
    'ig': '#2a78d6',
    'rise': '#1baf7a',
    'fovex': '#e08a2a',
}
METHOD_LABELS = {
    'ig': 'Integrated Gradients',
    'rise': 'RISE',
    'fovex': 'FovEx',
}
METRICS = [
    ('deletion_auc', 'Deletion AUC (lower = more faithful)', False),
    ('insertion_auc', 'Insertion AUC (higher = more faithful)', True),
]


def load_data(ig_path, rise_path, fovex_path):
    ig = pd.read_csv(ig_path)
    rise = pd.read_csv(rise_path)
    fovex = pd.read_csv(fovex_path)

    for df in (ig, rise, fovex):
        for col in ('deletion_auc', 'insertion_auc'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # keep only the blurred-baseline IG rows for the cross-method comparison
    ig_blurred = ig[ig['eval_baseline'] == 'blurred'].copy()

    combined = pd.concat([ig_blurred, rise, fovex], ignore_index=True)
    return combined, ig


def plot_method_comparison(combined, out_path):
    methods = [m for m in ('ig', 'rise', 'fovex') if m in combined['method'].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (metric, title, _) in zip(axes, METRICS):
        means = [combined.loc[combined['method'] == m, metric].mean() for m in methods]
        stds = [combined.loc[combined['method'] == m, metric].std() for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        bars = ax.bar(labels, means, yerr=stds, capsize=5,
                       color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, mean),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize=9)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    fig.suptitle('Faithfulness comparison — deletion/insertion AUC by XAI method\n'
                  '(convnext-tiny, fold3, eval_baseline=blurred)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class(combined, out_path):
    methods = [m for m in ('ig', 'rise', 'fovex') if m in combined['method'].unique()]
    classes = sorted(combined['true_class'].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(classes))
    width = 0.8 / len(methods)

    for ax, (metric, title, _) in zip(axes, METRICS):
        for i, m in enumerate(methods):
            sub = combined[combined['method'] == m]
            vals = [sub.loc[sub['true_class'] == c, metric].mean() for c in classes]
            ax.bar(x + i * width, vals, width=width,
                   label=METHOD_LABELS[m], color=METHOD_COLORS[m], alpha=0.85)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle('Per-class faithfulness by XAI method (convnext-tiny, fold3)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ig_baseline_ablation(ig, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    baselines = sorted(ig['eval_baseline'].dropna().unique())
    colors = {'black': '#444444', 'blurred': '#2a78d6'}

    for ax, (metric, title, _) in zip(axes, METRICS):
        means = [ig.loc[ig['eval_baseline'] == b, metric].mean() for b in baselines]
        stds = [ig.loc[ig['eval_baseline'] == b, metric].std() for b in baselines]
        bars = ax.bar(baselines, means, yerr=stds, capsize=5,
                       color=[colors.get(b, '#999999') for b in baselines],
                       alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, mean),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_xlabel('eval_baseline', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    fig.suptitle('IG baseline ablation — black vs blurred perturbation', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', default='convnext-tiny_fold3',
                         help="checkpoint key used in ham10000/results/xai/eval/<key>/ (default: convnext-tiny_fold3)")
    parser.add_argument('--ig', default=None, help='path to ig_faithfulness.csv (overrides --key)')
    parser.add_argument('--rise', default=None, help='path to rise_faithfulness.csv (overrides --key)')
    parser.add_argument('--fovex', default=None, help='path to fovex_faithfulness.csv (overrides --key)')
    parser.add_argument('--out', default='ham10000/results/figures',
                         help='output directory for figures (default: ham10000/results/figures)')
    args = parser.parse_args()

    eval_dir = os.path.join('ham10000', 'results', 'xai', 'eval', args.key)
    ig_path = args.ig or os.path.join(eval_dir, 'ig_faithfulness.csv')
    rise_path = args.rise or os.path.join(eval_dir, 'rise_faithfulness.csv')
    fovex_path = args.fovex or os.path.join(eval_dir, 'fovex_faithfulness.csv')

    os.makedirs(args.out, exist_ok=True)

    combined, ig_full = load_data(ig_path, rise_path, fovex_path)

    plot_method_comparison(combined, os.path.join(args.out, 'faithfulness_method_comparison.png'))
    plot_per_class(combined, os.path.join(args.out, 'faithfulness_per_class.png'))
    plot_ig_baseline_ablation(ig_full, os.path.join(args.out, 'faithfulness_ig_baseline_ablation.png'))

    print(f"Saved figures to {args.out}/:")
    print("  faithfulness_method_comparison.png")
    print("  faithfulness_per_class.png")
    print("  faithfulness_ig_baseline_ablation.png")


if __name__ == '__main__':
    main()
