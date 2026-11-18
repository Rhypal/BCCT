"""生成消融实验图。使用 --demo 查看布局，使用 --input 读取实测结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE = "#2A597A"
ORANGE = "#A6512F"
GRAY = "#65717B"
LIGHT_GRAY = "#DCE3E8"

DEMO = {
    "transport": [
        {"variant": "完整模型", "ratio": 0.68},
        {"variant": "A1 混合协方差", "ratio": 0.86},
        {"variant": "A2 原生 PCA", "ratio": 0.94},
        {"variant": "A3 仅均值拟合", "ratio": 0.90},
    ],
    "boundary": [
        {"variant": "完整模型", "old_to_new": 1.6, "novel_accuracy": 93.7},
        {"variant": "B2 类中心", "old_to_new": 2.9, "novel_accuracy": 94.2},
    ],
    "inference": [
        {"variant": "完整模型", "recurring": 95.1, "novel": 94.6},
        {"variant": "C1 仅规范空间", "recurring": 92.8, "novel": 94.2},
        {"variant": "C2 单一路径", "recurring": 94.3, "novel": 92.9},
    ],
}


def configure_fonts():
    fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in (
        "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "DengXian", "SimHei", "PingFang SC", "Arial Unicode MS",
    ):
        if name in fonts:
            plt.rcParams.update({"font.family": name, "font.size": 10, "pdf.fonttype": 42})
            return
    raise RuntimeError("未找到中文字体。请安装微软雅黑、思源黑体或 Noto Sans CJK SC。")


def plot_transport(ax, rows):
    ys = range(len(rows))
    ratios = [row["ratio"] for row in rows]
    ax.axvline(1.0, color=GRAY, linewidth=1, linestyle="--", label="传输前")
    ax.scatter(ratios, ys, color=BLUE, marker="o", zorder=3)
    ax.set_yticks(list(ys), [row["variant"] for row in rows])
    ax.invert_yaxis()
    ax.margins(x=0.13, y=0.14)
    ax.set_xlabel("分布距离比（传输后 / 传输前）")
    ax.set_title("A  留出桥类", loc="left", fontweight="bold", pad=27)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False,
              fontsize=8, borderaxespad=0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)


def plot_boundary(ax, rows):
    styles = [(BLUE, "o"), (ORANGE, "s"), (GRAY, "^"), (BLUE, "D")]
    for index, row in enumerate(rows):
        color, marker = styles[index % len(styles)]
        ax.scatter(
            row["old_to_new"], row["novel_accuracy"],
            color=color, marker=marker, s=48, label=row["variant"], zorder=3,
        )
    ax.margins(x=0.19, y=0.24)
    ax.set_xlabel("旧类误判为新类的比例（%）")
    ax.set_ylabel("当期新类准确率（%）")
    ax.set_title("B  旧新类边界", loc="left", fontweight="bold", pad=27)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=8, borderaxespad=0)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)


def plot_inference(ax, rows):
    ys = range(len(rows))
    recurring = [row["recurring"] for row in rows]
    novel = [row["novel"] for row in rows]
    ax.hlines(ys, [min(a, b) for a, b in zip(recurring, novel)],
              [max(a, b) for a, b in zip(recurring, novel)], color=GRAY, linewidth=1)
    ax.scatter(recurring, ys, marker="o", s=38, color=BLUE, label="重复类", zorder=3)
    ax.scatter(novel, ys, marker="D", s=34, color=ORANGE, label="新类", zorder=3)
    ax.set_yticks(list(ys), [row["variant"] for row in rows])
    ax.invert_yaxis()
    ax.margins(x=0.15, y=0.25)
    ax.set_xlabel("最终准确率（%）")
    ax.set_title("C  双空间推理", loc="left", fontweight="bold", pad=27)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              frameon=False, fontsize=8, borderaxespad=0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="用模拟数据绘制示意图")
    source.add_argument("--input", type=Path, help="使用实测结果 JSON")
    parser.add_argument("--panel", choices=("A", "B", "C"), help="仅导出一个面板")
    parser.add_argument("--output", required=True, type=Path, help="输出 PNG、PDF 或 SVG")
    args = parser.parse_args()

    configure_fonts()
    data = DEMO if args.demo else json.loads(args.input.read_text(encoding="utf-8-sig"))
    panels = {
        "A": ("transport", plot_transport),
        "B": ("boundary", plot_boundary),
        "C": ("inference", plot_inference),
    }
    selected = [args.panel] if args.panel else list(panels)
    fig, axes = plt.subplots(1, len(selected), figsize=(4.4 * len(selected), 3.45), squeeze=False)
    for name, ax in zip(selected, axes[0]):
        key, plot = panels[name]
        if not data.get(key):
            parser.error(f"面板 {name} 需要非空的 {key} 数据")
        plot(ax, data[key])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    if args.demo:
        fig.suptitle("模拟数据 · 仅用于示意", fontsize=12, color=GRAY, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if args.demo else 0.99), w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
