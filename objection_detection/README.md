# YOLOv8 Training Optimization — Ablation Study

2D Object Detection, Option 1. YOLOv8 on Pascal VOC (2007+2012 trainval, ~16.5k images, 20 classes). Colab A100.

## Result

**C8_All** (YOLOv8s + AdamW + mixup 0.1 + rotation ±10°) beat the baseline by **+0.0315 mAP50-95** and **+0.0393 mAP50**, at 22% more training time.

| Config | Arch | Aug | Opt | mAP50 | mAP50-95 | Time (min) |
|---|---|---|---|---|---|---|
| C1_Baseline | n | Default | SGD | 0.7943 | 0.5791 | 17.6 |
| C3_Aug | n | Enhanced | SGD | 0.7835 | 0.5467 | 17.8 |
| C4_Opt | n | Default | AdamW | 0.8100 | 0.5955 | 17.9 |
| **C8_All** | **s** | **Enhanced** | **AdamW** | **0.8336** | **0.6106** | **21.4** |

Full discussion and single-factor analysis in Section 8 of the notebook.

## Files

- `ablation_slim.ipynb` — main notebook, outputs embedded
- `ablation_core.py` — config dataclasses, training loop with crash-resume, plots, analysis
- `requirements.txt`

## Run

Open the notebook on GitHub to read. To re-run: upload both `.ipynb` and `.py` to Google Drive, open in Colab with an A100, Runtime → Run all. ~85 min from scratch; seconds if `results_partial.csv` is present (`resume=True`).
