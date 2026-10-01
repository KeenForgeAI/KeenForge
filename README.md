# KeenForge 2.0 — The AI Engineer for Your Own Datasets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/KeenForgeAI/KeenForge?style=social)](https://github.com/KeenForgeAI/KeenForge)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-blue)](README.zh-CN.md)

> **CVPR 2026 Demo Track** · Formerly AutoLabel Pro

**KeenForge** is a local-first desktop app that turns a folder of images into a trained
object-detection model — without writing a single line of code. Import images, let it
**deduplicate and cluster** them, label a handful of cluster representatives, and the model
trains itself in the background. Your data never leaves your computer.

<p align="center">
  <img src="assets/demo.gif" width="800" alt="KeenForge Demo">
</p>

---

## What's New in 2.0

2.0 is not just a bigger 1.0 — it replaces "label everything" with a **closed active-learning loop**.

- 🧹 **Smart Data Cleaning** — 3-layer dedup (perceptual hash → pixel-level MAE → rotation/mirror/±2px-shift alignment) plus cosine clustering that automatically picks the most informative representatives.
- 🔁 **Cluster-Guided Active Learning** — label the cluster representatives (typically **15–50 images per round**, not thousands) → auto-train → compare → **re-cluster at finer granularity** → repeat. Every round, the model's weak spots become visible.
- 🔍 **Model Compare Mode** — run your trained model on *already-labeled* images. Predictions appear as **cyan dashed boxes with confidence**; human labels stay green. **Edit to accept** — un-reviewed predictions can never overwrite ground truth.
- ⚖️ **Bring-Your-Own Weights** — resume from any `.pt` / `.pth`; strategy presets (transfer / recursive / from scratch).
- ⚡ **Non-Blocking Training** — training runs in a separate process; keep labeling, clustering, and reviewing while it trains.
- 📊 **Training Summary** — per-class bounding-box statistics, mAP over rounds, model files.
- 🌏 **Full zh / en UI** — switch language at runtime.
- 🛡️ **Robust I/O** — decodes images by content, so mislabeled extensions (`.png` files that are really JPEG) just work.

---

## How It Works (the loop)

```
      ┌──────────────────────────────────────────────────────────┐
      │                                                          │
  [cluster images] → [label cluster reps] → [auto-train] → [compare & correct]
      ↑                                                          │
      └──────────── re-cluster (finer granularity) ←─────────────┘
```

Label the representatives of each cluster, not the whole dataset. The model trains, predicts the
next clusters, and you only correct what it gets wrong. This is how you get a strong niche-domain
model from a fraction of the annotation budget.

<p align="center">
  <img src="assets/workflow.png" width="720" alt="KeenForge workflow">
</p>

---

## Why KeenForge?

| | Roboflow | Label Studio | **KeenForge 2.0** |
|---|---|---|---|
| **Deployment** | Cloud (data uploaded) | Docker (complex setup) | **Double-click exe** |
| **Data Privacy** | ❌ Images leave your machine | ⚠️ Self-hosted but heavy | ✅ **100% local, offline** |
| **Data Cleaning** | Paid tier | ❌ None | ✅ **Built-in dedup + clustering** |
| **Active Learning** | ❌ None | ❌ None | ✅ **Cluster-guided loop** |
| **AI Assistance** | Paid tier | Manual config | ✅ **YOLO-World zero-shot** |
| **Auto-Training** | Paid GPU credits | ❌ None | ✅ **Auto-triggers on cluster-rep quota** |
| **Model Ownership** | ❌ Locked to platform | ✅ Open source | ✅ **You own your model** |
| **Price** | $79–$249/mo | Free (self-host) | **Free & Open Source** |

---

## What Can You Do With It?

- **Industrial QC**: detect scratches, dents, or soldering defects on a production line
- **PCB / materials**: inspect surface defects where public models simply don't exist
- **Safety Monitoring**: detect helmets, vests, or unsafe behavior in workplace footage
- **Research**: build custom datasets for papers — export to COCO, VOC, or YOLO
- **Agriculture**: identify crop diseases or count livestock from drone photos
- **Anything you can see**: if you can draw a box around it, KeenForge can learn to find it

---

## Quick Start (3 minutes)

### Option 1: Download EXE (Windows, recommended)

1. Go to [Releases](https://github.com/KeenForgeAI/KeenForge/releases) and download the latest `KeenForge` build
2. Double-click to launch
3. Load your image folder
4. Run **Deduplicate** and **Cluster** from the toolbar
5. Label the cluster representatives → the model auto-trains on the round quota
6. Turn on **Model Compare** to review and correct predictions → repeat

### Option 2: One-click launcher (small download, auto-installs the environment)

Download the `KeenForge-2.0.0-source.zip` from [Releases](https://github.com/KeenForgeAI/KeenForge/releases), unpack it, then:

- **Windows**: double-click `run_windows.bat`
- **Linux / macOS**: `./run_unix.sh`

The launcher creates a local `.venv`, installs the dependencies automatically on first run
(a few minutes), and then starts KeenForge. This keeps the download small — the heavy
PyTorch/Ultralytics stack is fetched on your machine instead of being bundled.

### Option 3: Manual (from source)

```bash
# 1. Clone
git clone https://github.com/KeenForgeAI/KeenForge.git
cd KeenForge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python src/main.py
```

**Prerequisites**: Python 3.8+ · CUDA optional (works on CPU, GPU strongly recommended for training)

---

## Core Features

### 3-Layer Deduplication

1. **Perceptual hash** (PDQ) coarse screen
2. **Pixel-level MAE** confirmation (rejects look-alike uniform backgrounds)
3. **Geometric matching** — 8 dihedral transforms + **±2px translation alignment**

Real-world result on GC10-DET (2,306 images): exactly **13 true duplicate groups**, including a pair
that only differs by a 2-pixel shift (MAE 10.14 → **2.57** after alignment).

### Cosine Clustering & Representatives

Images are embedded and clustered (hnswlib, cosine). KeenForge picks one representative per cluster
to cover ~80% of the data, clamped to **15–50 representatives per round** — a coverage-driven quota
instead of a rigid percentage.

### Zero-Shot Bootstrap (YOLO-World)

Type any object name — `"welding defect"`, `"safety helmet"`, `"ripe tomato"` — and KeenForge finds it
immediately. Zero-shot recall has a ceiling (~85–90% in our tests); the first training round breaks
through it.

### Model Zoo

| Model | Use case |
|---|---|
| YOLOv8n/s/m/l/x | General object detection |
| YOLO11n/s/m/l/x | Latest YOLO architecture |
| RT-DETR-l/x | Real-time transformer detection |
| YOLO-World v2 (s/m) | Zero-shot open-vocabulary |
| Faster-RCNN | High accuracy, slower |
| RetinaNet | Dense object scenes |
| SSD300 | Lightweight, fast |

### Format Support

- **Import**: COCO JSON, Pascal VOC XML
- **Export**: COCO JSON, Pascal VOC XML, YOLO TXT
- **Training output**: standard Ultralytics `.pt` weights

---

## Datasets for Niche Domains

Target detection is easy to *showcase* on COCO and hard to *deploy* everywhere else. We curate and
publish **corrected, ready-to-use** industrial datasets — deduplicated, re-split, format-fixed:

**GC10-DET · NEU-DET · DeepPCB · PKU-Market-PCB · TXL-PBC · Raccoon**

→ [huggingface.co/KeenForgeAI](https://huggingface.co/KeenForgeAI) · index: [KeenForgeData](https://github.com/KeenForgeAI/KeenForgeData)

> If you download one of these datasets, KeenForge is the tool they were cleaned with.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` / `D` | Previous / Next image (auto-saves) |
| `R` | Toggle Draw / Edit mode |
| `J` | Move to Junk folder |
| `Ctrl+Z` | Undo |
| `Ctrl+C/V` | Copy / Paste boxes |
| `Ctrl+D` | Duplicate selected box |
| `Delete` | Remove selected box |
| `Middle-click + drag` | Pan canvas |
| `Ctrl + Scroll` | Zoom |

---

## Roadmap 2.x

- [x] Dataset deduplication (3-layer)
- [x] Cosine clustering + representative selection
- [x] Cluster-guided active learning loop
- [x] Model compare mode (edit-to-accept)
- [x] Bring-your-own weights
- [ ] SAM-powered segmentation (polygon masks)
- [ ] Adaptive fine-tuning tuned for YOLO
- [ ] One-click API deployment
- [ ] `pip install` support

**Want to help?** Check the [Issues](https://github.com/KeenForgeAI/KeenForge/issues) tab for `good first issue` tasks.

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Push and open a Pull Request

**Especially needed**:
- UI/UX improvements
- New model integrations
- Documentation translations (Chinese ↔ English)
- Bug reports from your own niche datasets

---

## Citation

If KeenForge helps your research, please cite:

```bibtex
@misc{keenforge2026,
  title={KeenForge: No-Code Desktop Tool for Expert-Led Object Detection Model Training},
  author={Lu Gan and Xi Li},
  year={2026},
  howpublished={\url{https://github.com/KeenForgeAI/KeenForge}},
  note={CVPR 2026 Demo Track}
}
```

---

## License

MIT License — Copyright (c) 2026 KeenForgeAI. Free for open source and commercial use. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b><br>
  <sub>Built with ❤️ by <a href="https://github.com/KeenForgeAI">KeenForgeAI</a></sub>
</p>
