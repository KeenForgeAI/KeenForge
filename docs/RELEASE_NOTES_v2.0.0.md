# KeenForge 2.0.0

> Released: 2026-10-01 · Branch: `main` · Tag: `v2.0.0`

KeenForge 2.0 turns the "label everything, then train" workflow into a **closed
cluster-guided active-learning loop**. It is still a fully local, offline, no-code
desktop app — your images never leave your machine.

## Highlights

- 🧹 **3-layer dataset deduplication** — perceptual hash → pixel-level MAE → dihedral
  transforms + **±2px translation alignment**. Correctly flags near-duplicates that
  differ only by a 2-pixel shift (MAE 10.14 → 2.57 on a real NEU-DET pair).
- 🔁 **Cluster-guided active learning** — cosine clustering picks the most informative
  representatives (typically **15–50 per round**). Label those, the model trains in the
  background, then you compare and correct — and the cycle repeats at finer granularity.
- 🔍 **Model Compare Mode** — run your trained model on *already-labeled* images.
  Predictions are **cyan dashed boxes with confidence**; human labels stay green.
  **Edit to accept** — un-reviewed predictions can never overwrite ground truth.
- ⚖️ **Bring-your-own weights** — resume from any `.pt` / `.pth`; transfer / recursive /
  from-scratch strategies.
- ⚡ **Non-blocking training** — training runs in a separate process; keep labeling while it runs.
- 📊 **Training summary** — per-class bounding-box statistics, per-round mAP, model files.
- 🌏 **Full zh / en UI** — switch language at runtime.
- 🛡️ **Robust image I/O** — decodes by content, so mislabeled extensions (`.png` files
  that are really JPEG) just work.
- 🧠 **Model zoo** — YOLOv8 n/s/m/l/x, YOLO11 n/s/m/l/x, RT-DETR-l/x, YOLO-World v2,
  Faster-RCNN, RetinaNet, SSD300.

## Import / Export

- **Import**: COCO JSON, Pascal VOC XML
- **Export**: COCO JSON, Pascal VOC XML, YOLO TXT
- **Training output**: standard Ultralytics `.pt`

## Install / Run

### Run from source (recommended for now)

```bash
git clone https://github.com/KeenForgeAI/KeenForge.git
cd KeenForge
pip install -r requirements.txt
python src/main.py
```

Windows users who build an executable can find the PyInstaller entry point in
`entry.py` (the spec is kept local).

### Windows bundle

This release includes a Windows onedir build (`KeenForge/`). Unpack and run
`KeenForge.exe`. It bundles a CUDA-enabled PyTorch, so it is large (~4 GB unpacked)
and requires an NVIDIA GPU for fast training (it still runs on CPU, slowly).

## Datasets

Corrected, ready-to-use industrial datasets (deduped, re-split, format-fixed):

GC10-DET · NEU-DET · DeepPCB · PKU-Market-PCB · TXL-PBC · Raccoon
→ https://huggingface.co/KeenForgeAI

## Known issues

- The Windows bundle reports training mAP as `0` when `torchmetrics` is not present in
  the build environment. Run from source with `pip install torchmetrics` for mAP.
- The bundled build uses the brute-force cosine path for semantic dedup (hnswlib is not
  bundled); this is fast for the target dataset sizes (a few thousand images) but slower
  at 10k+ scale.
- The bundled build depends on the environment it was produced in; for a reproducible
  setup prefer running from source with `requirements.txt`.

## Roadmap 2.x

- [ ] SAM-powered segmentation (polygon masks)
- [ ] Adaptive fine-tuning tuned for YOLO
- [ ] One-click API deployment
- [ ] `pip install` support

## Citation

```bibtex
@misc{keenforge2026,
  title={KeenForge: No-Code Desktop Tool for Expert-Led Object Detection Model Training},
  author={Lu Gan and Xi Li},
  year={2026},
  howpublished={\url{https://github.com/KeenForgeAI/KeenForge}},
  note={CVPR 2026 Demo Track}
}
```
