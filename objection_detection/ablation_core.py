from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────
def pick_device() -> str:
    if torch.cuda.is_available():
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[device] Apple Silicon MPS")
        return "mps"
    print("[device] WARNING: no GPU found, using CPU")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Config objects
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_AUG: dict[str, float] = {
    "mosaic":    1.0,
    "mixup":     0.0,
    "degrees":   0.0,
    "translate": 0.1,
    "scale":     0.5,
    "fliplr":    0.5,
}


@dataclass
class RunConfig:
    label: str
    weights: str                        # e.g. 'yolov8n.pt', 'yolov8s.pt', or a path
    optimizer: str = "SGD"              # 'SGD' | 'AdamW' | 'Adam' | 'auto'
    lr0: float = 0.01
    aug: dict[str, float] = field(default_factory=dict)  # overrides on top of DEFAULT_AUG
    extra: dict[str, Any] = field(default_factory=dict)  # any other YOLO .train() kwargs

    def resolved_aug(self) -> dict[str, float]:
        return {**DEFAULT_AUG, **self.aug}

    def arch_name(self) -> str:
        stem = Path(self.weights).stem.lower()
        for size in ["n", "s", "m", "l", "x"]:
            if stem.endswith(f"yolov8{size}") or stem.endswith(f"v8{size}"):
                return f"YOLOv8{size}"
        return stem

    def aug_label(self) -> str:
        return "Enhanced" if self.aug else "Default"


@dataclass
class FixedParams:
    data: str                           
    epochs: int = 15
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    seed: int = 42
    patience: int = 100
    pretrained: bool = True
    verbose: bool = False
    device: str = field(default_factory=pick_device)
    amp: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_yolo_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "data":       self.data,
            "epochs":     self.epochs,
            "imgsz":      self.imgsz,
            "batch":      self.batch,
            "workers":    self.workers,
            "seed":       self.seed,
            "patience":   self.patience,
            "pretrained": self.pretrained,
            "verbose":    self.verbose,
            "device":     self.device,
            "exist_ok":   True,
        }
        if self.amp is not None:
            kwargs["amp"] = self.amp
        kwargs.update(self.extra)
        return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset bootstrap helpers
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_voc2007(save_dir: str | Path) -> str:
    import subprocess
    import xml.etree.ElementTree as ET
    from ultralytics import settings as ul_settings

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = save_dir / "voc2007.yaml"

    datasets_dir = Path(ul_settings["datasets_dir"])
    datasets_dir.mkdir(parents=True, exist_ok=True)
    voc_root = datasets_dir / "VOC"
    images_dir = voc_root / "images"
    labels_dir = voc_root / "labels"

    if (images_dir / "test2007").exists() and (labels_dir / "test2007").exists():
        n_test = len(list((images_dir / "test2007").glob("*.jpg")))
        print(f"[voc] Reusing cached VOC2007 at {voc_root} ({n_test} test images)")
    else:
        print(f"[voc] Downloading VOC2007 to {voc_root} ...")
        voc_root.mkdir(parents=True, exist_ok=True)
        urls = [
            "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        ]
        tmp = voc_root / "_tmp"
        tmp.mkdir(exist_ok=True)
        for url in urls:
            fname = tmp / Path(url).name
            if not fname.exists():
                print(f"  curl {url}")
                rc = subprocess.call([
                    "curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
                    "-o", str(fname), url
                ])
                if rc != 0 or not fname.exists():
                    raise RuntimeError(f"Failed to download {url}")
            print(f"  extract {fname.name}")
            subprocess.check_call(["tar", "-xf", str(fname), "-C", str(tmp)])

        devkit = tmp / "VOCdevkit" / "VOC2007"
        if not devkit.exists():
            raise RuntimeError(f"VOC2007 devkit not found at {devkit}")

        print("[voc] Converting annotations to YOLO format ...")
        VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]
        cls_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}

        def convert_split(split_name: str, image_set_file: str):
            (images_dir / split_name).mkdir(parents=True, exist_ok=True)
            (labels_dir / split_name).mkdir(parents=True, exist_ok=True)

            with open(devkit / "ImageSets" / "Main" / image_set_file) as f:
                ids = [line.strip() for line in f if line.strip()]

            for img_id in ids:
                src_jpg = devkit / "JPEGImages" / f"{img_id}.jpg"
                dst_jpg = images_dir / split_name / f"{img_id}.jpg"
                if not dst_jpg.exists() and src_jpg.exists():
                    dst_jpg.write_bytes(src_jpg.read_bytes())

                xml_path = devkit / "Annotations" / f"{img_id}.xml"
                if not xml_path.exists():
                    continue
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find("size")
                w = float(size.find("width").text)
                h = float(size.find("height").text)

                lines = []
                for obj in root.findall("object"):
                    if obj.find("difficult") is not None and int(obj.find("difficult").text) == 1:
                        continue  # skip "difficult" annotations, standard practice
                    cls = obj.find("name").text
                    if cls not in cls_to_idx:
                        continue
                    bb = obj.find("bndbox")
                    xmin = float(bb.find("xmin").text)
                    ymin = float(bb.find("ymin").text)
                    xmax = float(bb.find("xmax").text)
                    ymax = float(bb.find("ymax").text)
                    # YOLO format: class cx cy w h (all normalized 0..1)
                    cx = ((xmin + xmax) / 2.0) / w
                    cy = ((ymin + ymax) / 2.0) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    lines.append(f"{cls_to_idx[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                (labels_dir / split_name / f"{img_id}.txt").write_text("\n".join(lines))
            print(f"  {split_name}: {len(ids)} images")

        convert_split("train2007", "train.txt")
        convert_split("val2007",   "val.txt")
        convert_split("test2007",  "test.txt")

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[voc] Done. Cached at {voc_root}")

    voc2007 = {
        "path": str(voc_root),
        "train": ["images/train2007", "images/val2007"],
        "val":   ["images/test2007"],
        "names": {
            0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
            5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
            10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
            15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
        },
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(voc2007, f, sort_keys=False)
    print(f"[voc] Wrote {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def run_single(config: RunConfig, fixed: FixedParams, save_dir: str | Path) -> dict[str, Any]:
    from ultralytics import YOLO

    save_dir = Path(save_dir)
    print(f"\n{'='*60}\nRunning: {config.label}")
    print(f"  weights={config.weights} | optimizer={config.optimizer} (lr0={config.lr0})")
    print(f"  aug overrides={config.aug or '(none)'}")
    print(f"{'='*60}")

    model = YOLO(config.weights)

    train_kwargs = {
        **fixed.to_yolo_kwargs(),
        **config.resolved_aug(),
        "optimizer": config.optimizer,
        "lr0":       config.lr0,
        "name":      config.label,
        "project":   str(save_dir),
        **config.extra,
    }

    t_start = time.time()
    model.train(**train_kwargs)
    train_time = round((time.time() - t_start) / 60, 2)

    val_result = model.val(
        data=fixed.data,
        project=str(save_dir),
        name=f"{config.label}_val",
        exist_ok=True,
        verbose=False,
    )

    return {
        "Config":         config.label,
        "Arch":           config.arch_name(),
        "Aug":            config.aug_label(),
        "Optimizer":      config.optimizer,
        "LR":             config.lr0,
        "mAP50":          float(val_result.box.map50),
        "mAP50-95":       float(val_result.box.map),
        "Precision":      float(val_result.box.mp),
        "Recall":         float(val_result.box.mr),
        "Train_Time_min": train_time,
    }


def run_ablation(
    configs: list[RunConfig],
    fixed: FixedParams,
    save_dir: str | Path,
    resume: bool = True,
) -> pd.DataFrame:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    partial_path = save_dir / "results_partial.csv"

    completed: list[dict[str, Any]] = []
    done_labels: set[str] = set()
    if resume and partial_path.exists():
        prev = pd.read_csv(partial_path)
        completed = prev.to_dict("records")
        done_labels = set(prev["Config"].tolist())
        print(f"[resume] Found {len(done_labels)} previously-completed config(s): {sorted(done_labels)}")

    for i, cfg in enumerate(configs, 1):
        if cfg.label in done_labels:
            print(f"\n[{i}/{len(configs)}] SKIP (already done): {cfg.label}")
            continue
        print(f"\n[{i}/{len(configs)}]")
        row = run_single(cfg, fixed, save_dir)
        completed.append(row)
        pd.DataFrame(completed).to_csv(partial_path, index=False)
        print(f"  -> mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}  time={row['Train_Time_min']} min")

    df = pd.DataFrame(completed)
    df.to_csv(save_dir / "results_final.csv", index=False)
    print(f"\n[done] {len(df)} configs complete. Saved to {save_dir / 'results_final.csv'}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — adapts to whatever is in the DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, save_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("[plots] Empty DataFrame, nothing to plot.")
        return

    _plot_all_configs_bars(df, save_dir, plt, mpatches)
    _plot_single_factor(df, save_dir, plt)
    _plot_tradeoff(df, save_dir, plt, mpatches)
    print(f"[plots] Saved to {save_dir}")


def _arch_color(arch: str) -> str:
    palette = {
        "YOLOv8n": "#4C72B0", "YOLOv8s": "#DD8452",
        "YOLOv8m": "#55A868", "YOLOv8l": "#C44E52", "YOLOv8x": "#8172B3",
    }
    return palette.get(arch, "#888888")


def _plot_all_configs_bars(df, save_dir, plt, mpatches):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("COCO-Style Evaluation: All Configurations",
                 fontsize=14, fontweight="bold")

    labels = df["Config"].str.replace(r"C\d+_", "", regex=True)
    colors = [_arch_color(a) for a in df["Arch"]]
    x = np.arange(len(labels))

    for ax, metric in zip(axes, ["mAP50", "mAP50-95"]):
        bars = ax.bar(x, df[metric], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(df[metric].max() * 1.15, 0.1))
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)

    unique_archs = sorted(df["Arch"].unique())
    legend_handles = [mpatches.Patch(color=_arch_color(a), label=a) for a in unique_archs]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.95))
    plt.tight_layout()
    plt.savefig(save_dir / "plot_all_configs.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_single_factor(df, save_dir, plt):
    factors = ["Arch", "Aug", "Optimizer"]
    pairs = _find_single_factor_pairs(df, factors)
    if not pairs:
        return

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("Single-Factor Ablation (mAP50-95)", fontsize=13, fontweight="bold")
    axes = axes[0]

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    metric = "mAP50-95"

    for ax, (factor, pair) in zip(axes, pairs.items()):
        a_row, b_row = pair
        a_label = f"{a_row[factor]}\n({a_row['Config'].split('_')[0]})"
        b_label = f"{b_row[factor]}\n({b_row['Config'].split('_')[0]})"
        ax.bar([a_label, b_label], [a_row[metric], b_row[metric]],
               color=palette[:2], edgecolor="black")
        other = [f for f in factors if f != factor]
        held = ", ".join(f"{f}={a_row[f]}" for f in other)
        ax.set_title(f"{factor}\n({held})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_dir / "plot_single_factor.png", dpi=150, bbox_inches="tight")
    plt.close()


def _find_single_factor_pairs(df: pd.DataFrame, factors: list[str]) -> dict[str, tuple]:
    result: dict[str, tuple] = {}
    rows = df.to_dict("records")
    for factor in factors:
        for i, r1 in enumerate(rows):
            for r2 in rows[i + 1:]:
                diffs = [f for f in factors if r1[f] != r2[f]]
                if diffs == [factor]:
                    a, b = sorted([r1, r2], key=lambda r: str(r[factor]))
                    result[factor] = (a, b)
                    break
            if factor in result:
                break
    return result


def _plot_tradeoff(df, save_dir, plt, mpatches):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Accuracy vs Training Time Tradeoff", fontsize=13, fontweight="bold")
    marker_map = {"Default": "o", "Enhanced": "s"}

    for _, row in df.iterrows():
        label_short = row["Config"].split("_", 1)[-1]
        ax.scatter(row["Train_Time_min"], row["mAP50-95"],
                   color=_arch_color(row["Arch"]),
                   marker=marker_map.get(row["Aug"], "o"),
                   s=140, zorder=3, edgecolors="black", linewidths=0.6)
        ax.annotate(label_short, (row["Train_Time_min"], row["mAP50-95"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)

    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)

    unique_archs = sorted(df["Arch"].unique())
    unique_augs = sorted(df["Aug"].unique())
    legend_handles = [mpatches.Patch(color=_arch_color(a), label=a) for a in unique_archs]
    for aug in unique_augs:
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker_map.get(aug, "o"),
                       color="w", markerfacecolor="gray",
                       label=f"{aug} aug", markersize=10)
        )
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "plot_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame, metric: str = "mAP50-95") -> str:
    lines = ["=" * 60, "ABLATION ANALYSIS SUMMARY", "=" * 60]

    if len(df) == 0:
        return "\n".join(lines + ["(no data)"])

    for factor in ["Arch", "Aug", "Optimizer"]:
        if factor not in df.columns or df[factor].nunique() < 2:
            continue
        groups = df.groupby(factor)[metric].mean().sort_values()
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in groups.items())
        delta = groups.iloc[-1] - groups.iloc[0]
        lines.append(f"\n[{factor}]  {parts}  (spread: {delta:+.4f})")

    best = df.loc[df[metric].idxmax()]
    lines.append(
        f"\nBest config: {best['Config']}  "
        f"({best['Arch']}, {best['Aug']} aug, {best['Optimizer']})"
    )
    lines.append(f"  mAP50:    {best['mAP50']:.4f}")
    lines.append(f"  mAP50-95: {best['mAP50-95']:.4f}")
    lines.append(f"  Time:     {best['Train_Time_min']:.1f} min")

    baseline_rows = df[df["Config"].str.contains("Baseline", case=False, na=False)]
    if len(baseline_rows) > 0:
        baseline = baseline_rows.iloc[0]
        lines.append(f"\nImprovement over baseline ({baseline['Config']}):")
        lines.append(
            f"  mAP50:    {baseline['mAP50']:.4f} -> {best['mAP50']:.4f}  "
            f"({(best['mAP50']-baseline['mAP50']):+.4f})"
        )
        lines.append(
            f"  mAP50-95: {baseline['mAP50-95']:.4f} -> {best['mAP50-95']:.4f}  "
            f"({(best['mAP50-95']-baseline['mAP50-95']):+.4f})"
        )

    return "\n".join(lines)
