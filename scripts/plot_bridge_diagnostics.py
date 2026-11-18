"""从真实检查点和测试特征缓存绘制传输机制诊断图。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.feature_adapter import ResidualFeatureAdapter
from model.operator import DomainOperator

TASK_NAMES = ("AID", "NWPU-RESISC45", "PatternNet", "RSI-CB256", "UCM")
CLASS_NAMES = ("airport", "bridge", "forest", "sparse_residential")
COLORS = ("#345B82", "#B6603A", "#32775D", "#8B5692")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=230, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fit_diagnostics(checkpoint, output_dir):
    operators = [op for op in checkpoint["operators"] if op["bw_before"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.75), layout="constrained")
    labels = [f'{TASK_NAMES[op["task_id"]]}  [{op["bridge_count"]}]' for op in operators]
    for ax, field, title in zip(
        axes, ("bw", "mean"), ("Covariance discrepancy", "Relative class-center discrepancy")
    ):
        changes = [
            100 * (op[f"{field}_after"] / op[f"{field}_before"] - 1)
            for op in operators
        ]
        positions = range(len(operators))
        colors = ["#B6603A" if value > 0 else "#345B82" for value in changes]
        ax.barh(list(positions), changes, color=colors, height=0.40)
        ax.axvline(0, color="#636B70", linewidth=0.8)
        ax.set_yticks(list(positions), labels if ax is axes[0] else [])
        if ax is axes[1]:
            ax.set_yticks(list(positions), [""] * len(operators))
        ax.invert_yaxis()
        ax.set_xlim(-47, 10)
        ax.set_xlabel("Relative change (%)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for y, value in zip(positions, changes):
            x = value + (-1 if value < 0 else 1)
            ax.text(x, y, f"{value:+.2f}", va="center",
                    ha="right" if value < 0 else "left", fontsize=8)
    save(fig, output_dir, "bridge_fit_diagnostics")


def load_adapter(checkpoint, task_id):
    cfg = checkpoint["config"]["feature_adapter"]
    adapter = ResidualFeatureAdapter(
        checkpoint["feature_dim"], cfg["bottleneck_dim"], cfg["stages"]
    )
    prefix = f"feature_adapters.{task_id}."
    adapter.load_state_dict({
        key[len(prefix):]: value for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    })
    return adapter.eval()


def sample_features(pack, class_ids, per_class, generator):
    selected = []
    for cid in class_ids:
        idx = torch.nonzero(pack["labels"] == cid, as_tuple=True)[0]
        if len(idx) < per_class:
            raise ValueError(f"Class {cid} has fewer than {per_class} test images")
        selected.append(idx[torch.randperm(len(idx), generator=generator)[:per_class]])
    return pack["features"][torch.cat(selected)].float()


@torch.inference_mode()
def plot_cached_tsne(checkpoint, cache_dir, output_dir, seed):
    names = checkpoint["registry"]["id_to_class"]
    ids = [names.index(name) for name in CLASS_NAMES]
    history_pack = load(cache_dir / "00_AID_test.pt")
    current_pack = load(cache_dir / "01_NWPU_test.pt")
    generator = torch.Generator().manual_seed(seed)
    per_class = 36
    history_raw = sample_features(history_pack, ids, per_class, generator)
    current_raw = sample_features(current_pack, ids, per_class, generator)
    history_op, current_op = map(DomainOperator.from_state_dict, checkpoint["operators"])
    history = F.normalize(history_op.apply(load_adapter(checkpoint, 0)(history_raw)), dim=-1)
    before = F.normalize(load_adapter(checkpoint, 1)(current_raw), dim=-1)
    after = F.normalize(current_op.apply(before), dim=-1)
    # 两幅图共享一次投影，历史点在两个面板中使用相同坐标。
    features = torch.cat((history, before, after)).numpy()
    pca = PCA(n_components=40, random_state=seed).fit_transform(features)
    embedded = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        random_state=seed,
    ).fit_transform(pca)
    count = len(history)
    historical, native, canonical = (
        embedded[:count], embedded[count:2 * count], embedded[2 * count:]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), sharex=True, sharey=True)
    for ax, current, title in zip(
        axes, (native, canonical), ("(a) Before transport", "(b) After transport")
    ):
        for index, color in enumerate(COLORS):
            group = slice(index * per_class, (index + 1) * per_class)
            ax.scatter(*historical[group].T, marker="o", facecolors="none",
                       edgecolors=color, s=19, linewidths=0.8)
            ax.scatter(*current[group].T, marker="^", color=color, s=14, alpha=0.74)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set(xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values():
            spine.set_color("#B8C0C6")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=color,
                   label=name.replace("_", " ").capitalize(), markersize=5)
        for color, name in zip(COLORS, CLASS_NAMES)
    ]
    handles.extend([
        plt.Line2D([], [], marker="o", linestyle="none", color="#45494D",
                   markerfacecolor="none", label="AID"),
        plt.Line2D([], [], marker="^", linestyle="none", color="#45494D",
                   label="NWPU-RESISC45"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, output_dir, "bridge_transport_tsne")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="提供缓存路径时同时绘制 t-SNE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fit_diagnostics(load(args.checkpoint_dir / "task_04_UCM.pt"), args.output_dir)
    if args.cache_dir:
        plot_cached_tsne(
            load(args.checkpoint_dir / "task_01_NWPU.pt"),
            args.cache_dir, args.output_dir, args.seed,
        )


if __name__ == "__main__":
    main()
