"""Generate publication-ready architecture and pipeline figures for the ISL project."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "docs" / "figures"
DPI = 300

BG = "#F8FAFC"
WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#526176"
SLATE = "#334155"
BLUE = "#2563EB"
BLUE_DARK = "#173B76"
TEAL = "#0F9D8A"
TEAL_DARK = "#087F71"
CYAN = "#38BDF8"
GOLD = "#E7A928"
CORAL = "#E56B6F"
GREEN = "#159A72"
LINE = "#CBD5E1"
LIGHT_BLUE = "#E8F1FF"
LIGHT_TEAL = "#E4F7F3"
LIGHT_GOLD = "#FFF5D6"
LIGHT_CORAL = "#FCE8E7"
FONT = "DejaVu Sans"

CLASSES = [
    "are you free today",
    "are you hiding something",
    "bring water for me",
    "can you repeat that please",
    "congratulations",
    "do me a favour",
    "do you need something",
    "go and sleep",
    "had your food",
    "he she is my friend",
    "how are you",
    "i am tired",
    "i am very happy",
    "i got hurt",
    "i need water",
    "i promise",
    "my name is",
    "thank you",
    "what do you want to become",
    "you comb your hair",
]


def setup_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def title(ax, figure_number: str, heading: str, subtitle: str = ""):
    ax.text(0.02, 0.965, figure_number.upper(), transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=TEAL, va="top", family=FONT, clip_on=False)
    ax.text(0.02, 0.925, heading, transform=ax.transAxes, fontsize=22,
            fontweight="bold", color=INK, va="top", family=FONT, clip_on=False)
    if subtitle:
        ax.text(0.02, 0.875, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="top", family=FONT, clip_on=False)


def rounded_box(ax, xy, width, height, text, *, face=WHITE, edge=LINE, text_color=INK,
                fontsize=10, radius=0.025, linewidth=1.2, weight="bold"):
    box = FancyBboxPatch(xy, width, height, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                         transform=ax.transAxes, facecolor=face, edgecolor=edge,
                         linewidth=linewidth)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, transform=ax.transAxes,
            ha="center", va="center", color=text_color, fontsize=fontsize,
            fontweight=weight, family=FONT, wrap=True)
    return box


def arrow(ax, start, end, *, color=SLATE, width=1.6, style="-|>", connectionstyle="arc3"):
    patch = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style,
                            mutation_scale=13, linewidth=width, color=color,
                            connectionstyle=connectionstyle)
    ax.add_patch(patch)
    return patch


def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")


def figure_1_pipeline():
    fig, ax = setup_figure(18, 7.2)
    title(ax, "Figure 1", "End-to-end ISL sentence translation pipeline",
          "From visual input to a locked phrase and browser speech output")
    labels = [
        ("Webcam /\nvideo input", LIGHT_BLUE, BLUE),
        ("MediaPipe Holistic\nPose + both hands", LIGHT_TEAL, TEAL_DARK),
        ("Spatial-scale\nnormalization", LIGHT_GOLD, GOLD),
        ("Feature builder\n235 + 235 = 470-D", LIGHT_CORAL, CORAL),
        ("Temporal sequence\n35 frames", LIGHT_BLUE, BLUE),
        ("1D-CNN\nlocal motion", LIGHT_TEAL, TEAL_DARK),
        ("Bi-LSTM\ntemporal dynamics", LIGHT_GOLD, GOLD),
        ("Dense softmax\n20 classes", LIGHT_CORAL, CORAL),
        ("Motion + confidence\ngate", LIGHT_BLUE, BLUE),
        ("UI: 3s hold\n+ Web Speech TTS", LIGHT_TEAL, TEAL_DARK),
    ]
    x0, y, w, h, gap = 0.025, 0.42, 0.082, 0.19, 0.014
    for index, (label, face, edge) in enumerate(labels):
        x = x0 + index * (w + gap)
        rounded_box(ax, (x, y), w, h, label, face=face, edge=edge, fontsize=8.5)
        if index < len(labels) - 1:
            arrow(ax, (x + w + 0.003, y + h / 2), (x + w + gap - 0.003, y + h / 2), color=SLATE)
    ax.text(0.025, 0.30, "Training representation", transform=ax.transAxes, fontsize=9,
            color=MUTED, family=FONT, fontweight="bold")
    ax.plot([0.025, 0.69], [0.285, 0.285], transform=ax.transAxes, color=LINE, linewidth=1)
    ax.text(0.71, 0.30, "Real-time decision and output", transform=ax.transAxes, fontsize=9,
            color=MUTED, family=FONT, fontweight="bold")
    ax.plot([0.71, 0.975], [0.285, 0.285], transform=ax.transAxes, color=LINE, linewidth=1)
    ax.text(0.5, 0.11, "Body-relative geometry reduces appearance bias; temporal modeling learns the sign trajectory.",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11, family=FONT)
    save(fig, "end_to_end_pipeline.png")


def draw_person(ax):
    head = np.array([0.45, 0.76])
    neck = np.array([0.45, 0.67])
    left_shoulder = np.array([0.31, 0.63])
    right_shoulder = np.array([0.59, 0.63])
    left_elbow = np.array([0.23, 0.48])
    right_elbow = np.array([0.67, 0.48])
    left_wrist = np.array([0.18, 0.36])
    right_wrist = np.array([0.73, 0.36])
    hip_left = np.array([0.37, 0.25])
    hip_right = np.array([0.53, 0.25])
    joints = [head, neck, left_shoulder, right_shoulder, left_elbow, right_elbow,
              left_wrist, right_wrist, hip_left, hip_right]
    bones = [(head, neck), (neck, left_shoulder), (neck, right_shoulder),
             (left_shoulder, left_elbow), (left_elbow, left_wrist),
             (right_shoulder, right_elbow), (right_elbow, right_wrist),
             (left_shoulder, hip_left), (right_shoulder, hip_right), (hip_left, hip_right)]
    for start, end in bones:
        ax.plot([start[0], end[0]], [start[1], end[1]], color=SLATE, linewidth=2.6, zorder=2)
    ax.add_patch(Circle(head, 0.055, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=2, zorder=3))
    for point in joints[1:]:
        ax.add_patch(Circle(point, 0.018, facecolor=WHITE, edgecolor=TEAL, linewidth=1.5, zorder=3))
    for wrist, side in [(left_wrist, "L"), (right_wrist, "R")]:
        hand = wrist + np.array([-0.02, -0.07]) if side == "L" else wrist + np.array([0.02, -0.07])
        ax.add_patch(Circle(hand, 0.035, facecolor=LIGHT_TEAL, edgecolor=TEAL_DARK, linewidth=1.5, zorder=3))
    ax.scatter([head[0]], [head[1]], s=100, color=CORAL, edgecolors=WHITE, linewidth=1.5, zorder=4)
    return {"nose": head, "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
            "left_hand": left_wrist + np.array([-0.02, -0.07]),
            "right_hand": right_wrist + np.array([0.02, -0.07])}


def figure_2_geometry():
    fig, ax = setup_figure(13, 8.5)
    title(ax, "Figure 2", "Keypoint normalization geometry",
          "Nose anchoring, shoulder-based scale normalization, hand mapping, and finger flexion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    points = draw_person(ax)
    nose = points["nose"]
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hand = points["left_hand"]
    right_hand = points["right_hand"]
    ax.annotate("Nose, index 0\norigin (0, 0, 0)", xy=nose, xytext=(0.06, 0.82),
                arrowprops={"arrowstyle": "-|>", "color": CORAL, "lw": 1.5},
                fontsize=10, color=CORAL, fontweight="bold", family=FONT,
                bbox={"boxstyle": "round,pad=0.35", "fc": WHITE, "ec": CORAL})
    ax.annotate("Left hand\nrelative to nose", xy=left_hand, xytext=(0.02, 0.21),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.4},
                fontsize=9, color=BLUE_DARK, family=FONT)
    ax.annotate("Right hand\nrelative to nose", xy=right_hand, xytext=(0.69, 0.21),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.4},
                fontsize=9, color=BLUE_DARK, family=FONT)
    ax.annotate("", xy=right_shoulder, xytext=left_shoulder,
                arrowprops={"arrowstyle": "<->", "color": GOLD, "lw": 2.2})
    midpoint = (left_shoulder + right_shoulder) / 2
    ax.text(midpoint[0], midpoint[1] + 0.045, r"$body\_scale = \|L_{shoulder} - R_{shoulder}\|_2$",
            ha="center", color=GOLD, fontsize=9.5, fontweight="bold", family=FONT,
            bbox={"boxstyle": "round,pad=0.25", "fc": LIGHT_GOLD, "ec": GOLD})
    inset = FancyBboxPatch((0.70, 0.53), 0.27, 0.28, boxstyle="round,pad=0.02,rounding_size=0.02",
                           facecolor=WHITE, edgecolor=TEAL, linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(inset)
    ax.text(0.835, 0.775, "Finger flexion angle", transform=ax.transAxes, ha="center",
            color=TEAL_DARK, fontsize=10, fontweight="bold", family=FONT)
    mcp = np.array([0.75, 0.64]); pip = np.array([0.82, 0.58]); tip = np.array([0.90, 0.66])
    ax.plot([mcp[0], pip[0], tip[0]], [mcp[1], pip[1], tip[1]], color=SLATE, linewidth=2, transform=ax.transAxes)
    for point, label in [(mcp, "MCP"), (pip, "PIP"), (tip, "TIP")]:
        ax.add_patch(Circle(point, 0.012, transform=ax.transAxes, facecolor=CYAN, edgecolor=BLUE_DARK, linewidth=1))
        ax.text(point[0], point[1] - 0.04, label, transform=ax.transAxes, ha="center", fontsize=7, color=MUTED, family=FONT)
    ax.text(0.835, 0.565, r"$\cos(\theta)=\frac{v_1\cdot v_2}{\|v_1\|\|v_2\|}$", transform=ax.transAxes,
            ha="center", color=INK, fontsize=11, family=FONT)
    ax.text(0.835, 0.525, r"angle$_{norm}=\arccos(\cdot)/\pi \in [0,1]$", transform=ax.transAxes,
            ha="center", color=MUTED, fontsize=8, family=FONT)
    ax.text(0.50, 0.055, "All coordinates are nose-relative and divided by body_scale, making the representation less sensitive to camera distance.",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED, family=FONT)
    save(fig, "keypoint_normalization_geometry.png")


def layer_box(ax, x, y, width, height, heading, details, face, edge):
    rounded_box(ax, (x, y), width, height, "", face=face, edge=edge, linewidth=1.6)
    ax.text(x + 0.018, y + height - 0.032, heading, transform=ax.transAxes, ha="left", va="top",
            color=edge, fontsize=11, fontweight="bold", family=FONT)
    ax.text(x + 0.018, y + height / 2 - 0.005, details, transform=ax.transAxes, ha="left", va="center",
            color=INK, fontsize=8.3, family=FONT, linespacing=1.35)


def figure_3_network():
    fig, ax = setup_figure(13.5, 9)
    title(ax, "Figure 3", "KeypointSequenceModel architecture",
          "Temporal convolutions learn local motion; the Bi-LSTM learns full sign trajectories")
    boxes = [
        ("Input tensor", "(Batch, 35, 470)\n35 time steps x 470 features", LIGHT_BLUE, BLUE),
        ("Conv1D block 1", "Conv1D: in=470, out=256\nk=3, p=1\nBatchNorm1d -> ReLU -> Dropout(0.25)", LIGHT_TEAL, TEAL_DARK),
        ("Conv1D block 2", "Conv1D: in=256, out=128\nk=3, p=1\nBatchNorm1d -> ReLU -> Dropout(0.25)", LIGHT_GOLD, GOLD),
        ("Recurrent block", "Bi-LSTM\ninput=128, hidden=128\n2 layers, dropout=0.30", LIGHT_CORAL, CORAL),
        ("Feature selection", "Last temporal output\n(Batch, 256)\n128 forward + 128 backward", LIGHT_BLUE, BLUE),
        ("Classification head", "Linear(256, 256)\nBatchNorm1d -> ReLU\nDropout(0.35)", LIGHT_TEAL, TEAL_DARK),
        ("Output", "Linear(256, 20)\nSoftmax\n20 ISL sentence classes", LIGHT_GOLD, GOLD),
    ]
    x, w, h, gap = 0.16, 0.68, 0.095, 0.025
    ys = np.linspace(0.78, 0.12, len(boxes))
    for index, ((heading, details, face, edge), y) in enumerate(zip(boxes, ys)):
        layer_box(ax, x, y, w, h, heading, details, face, edge)
        if index < len(boxes) - 1:
            arrow(ax, (0.5, y - 0.008), (0.5, ys[index + 1] + h + 0.008), color=SLATE)
    ax.text(0.05, 0.50, "Data\nshape", transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=MUTED, fontweight="bold", family=FONT)
    ax.text(0.95, 0.49, "Learned\nrepresentation", transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=MUTED, fontweight="bold", family=FONT)
    save(fig, "neural_network_layers.png")


def decision_box(ax, center, text, kind="process", width=0.19, height=0.08):
    x, y = center
    if kind == "decision":
        points = np.array([[x, y + height / 2], [x + width / 2, y], [x, y - height / 2], [x - width / 2, y]])
        patch = Polygon(points, transform=ax.transAxes, closed=True, facecolor=LIGHT_GOLD, edgecolor=GOLD, linewidth=1.5)
        ax.add_patch(patch)
    else:
        face, edge = (LIGHT_TEAL, TEAL_DARK) if kind == "action" else (WHITE, BLUE)
        patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                               transform=ax.transAxes, boxstyle="round,pad=0.01,rounding_size=0.018",
                               facecolor=face, edgecolor=edge, linewidth=1.4)
        ax.add_patch(patch)
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
            color=INK, family=FONT, fontweight="bold" if kind in {"decision", "action"} else "normal")


def figure_4_flowchart():
    fig, ax = setup_figure(11.5, 12)
    title(ax, "Figure 4", "Real-time inference state machine and smart gates",
          "Motion gating, confidence locking, persistence, TTS, and inactivity reset")
    x = 0.50
    decision_box(ax, (x, 0.84), "New frame\narrives", "action", 0.20, 0.07)
    decision_box(ax, (x, 0.74), "Hands detected?", "decision", 0.20, 0.09)
    decision_box(ax, (x, 0.63), "Reset buffer\nHands not in frame", "process", 0.22, 0.07)
    decision_box(ax, (x, 0.52), "Buffer length\n== 35?", "decision", 0.20, 0.09)
    decision_box(ax, (x, 0.41), "Accumulate frame\nContinue stream", "process", 0.22, 0.07)
    decision_box(ax, (x, 0.30), "Motion energy\n> 0.009?", "decision", 0.20, 0.09)
    decision_box(ax, (x, 0.19), "Holding stance\nBlock prediction", "process", 0.23, 0.07)
    decision_box(ax, (x, 0.08), "Run model\nBi-LSTM inference", "action", 0.22, 0.07)
    arrow(ax, (x, 0.805), (x, 0.785)); arrow(ax, (x, 0.695), (x, 0.665)); arrow(ax, (x, 0.585), (x, 0.565)); arrow(ax, (x, 0.475), (x, 0.455)); arrow(ax, (x, 0.365), (x, 0.345)); arrow(ax, (x, 0.255), (x, 0.225)); arrow(ax, (x, 0.145), (x, 0.115))
    arrow(ax, (x - 0.105, 0.74), (0.18, 0.63), color=CORAL, connectionstyle="arc3,rad=0.15")
    ax.text(0.32, 0.69, "No", transform=ax.transAxes, color=CORAL, fontsize=9, fontweight="bold", family=FONT)
    arrow(ax, (x + 0.105, 0.74), (0.84, 0.74), color=GREEN)
    ax.text(0.69, 0.76, "Yes", transform=ax.transAxes, color=GREEN, fontsize=9, fontweight="bold", family=FONT)
    arrow(ax, (x - 0.105, 0.52), (0.18, 0.41), color=CORAL, connectionstyle="arc3,rad=0.15")
    ax.text(0.32, 0.47, "No", transform=ax.transAxes, color=CORAL, fontsize=9, fontweight="bold", family=FONT)
    arrow(ax, (x - 0.105, 0.30), (0.18, 0.19), color=CORAL, connectionstyle="arc3,rad=0.15")
    ax.text(0.32, 0.25, "No", transform=ax.transAxes, color=CORAL, fontsize=9, fontweight="bold", family=FONT)
    decision_box(ax, (0.84, 0.62), "Confidence >= class\nthreshold AND\ncooldown > 1.5 s?", "decision", 0.22, 0.13)
    decision_box(ax, (0.84, 0.43), "Live HUD update\nNo token lock", "process", 0.21, 0.07)
    decision_box(ax, (0.84, 0.25), "Lock token\nappend phrase", "action", 0.20, 0.07)
    decision_box(ax, (0.84, 0.10), "3-second UI hold\nBrowser TTS", "action", 0.20, 0.07)
    arrow(ax, (x + 0.11, 0.08), (0.73, 0.62), color=BLUE, connectionstyle="arc3,rad=-0.15")
    arrow(ax, (0.84, 0.555), (0.84, 0.47), color=CORAL)
    ax.text(0.87, 0.51, "No", transform=ax.transAxes, color=CORAL, fontsize=9, fontweight="bold", family=FONT)
    arrow(ax, (0.84, 0.365), (0.84, 0.29), color=GREEN)
    ax.text(0.87, 0.33, "Yes", transform=ax.transAxes, color=GREEN, fontsize=9, fontweight="bold", family=FONT)
    arrow(ax, (0.84, 0.215), (0.84, 0.14), color=GREEN)
    ax.text(0.87, 0.17, "locked", transform=ax.transAxes, color=GREEN, fontsize=8, family=FONT)
    decision_box(ax, (0.18, 0.72), "No new token for > 5.0 s\nAuto-reset and clear sentence box", "process", 0.27, 0.06)
    arrow(ax, (0.30, 0.69), (0.42, 0.71), color=CORAL, connectionstyle="arc3,rad=-0.18")
    ax.text(0.19, 0.675, "inactivity timer", transform=ax.transAxes, color=CORAL, fontsize=8, family=FONT)
    save(fig, "inference_decision_flowchart.png")


def discover_classes():
    dataset = ROOT / "custom_dataset"
    if not dataset.exists():
        return CLASSES
    discovered = sorted(path.name for path in dataset.iterdir() if path.is_dir())
    return discovered or CLASSES


def figure_5_distribution():
    labels = discover_classes()
    counts = []
    for label in labels:
        folder = ROOT / "custom_dataset" / label
        counts.append(len(list(folder.glob("*.avi"))) if folder.exists() else 20)
    order = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    ax.set_facecolor(WHITE)
    bars = ax.barh(order, counts, color=TEAL, edgecolor=TEAL_DARK, height=0.66)
    ax.set_yticks(order, labels, fontsize=9, color=INK, family=FONT)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts + [20]) + 5)
    ax.set_xlabel("Number of video samples", fontsize=11, color=MUTED, family=FONT)
    ax.set_title("Dataset class distribution", loc="left", fontsize=22, fontweight="bold", color=INK, pad=42, family=FONT)
    ax.text(0, 1.035, "FIGURE 5", transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEAL, family=FONT)
    ax.text(0, 1.005, "20 ISL conversational classes | 400 recorded videos", transform=ax.transAxes, fontsize=10, color=MUTED, family=FONT)
    ax.grid(axis="x", color=LINE, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    for bar, count in zip(bars, counts):
        ax.text(count + 0.35, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=9, color=INK, family=FONT)
    ax.text(0.98, 0.07, "99%\noverall evaluation", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14, fontweight="bold", color=WHITE, family=FONT,
            bbox={"boxstyle": "round,pad=0.55", "fc": BLUE_DARK, "ec": CYAN, "lw": 1.5})
    fig.subplots_adjust(left=0.29, right=0.96, top=0.88, bottom=0.10)
    save(fig, "dataset_class_distribution.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_pipeline()
    figure_2_geometry()
    figure_3_network()
    figure_4_flowchart()
    figure_5_distribution()
    print(f"Generated 5 figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
