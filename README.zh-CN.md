# KeenForge 2.0 — 你自己的数据集，你自己的「AI 工程师」

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/KeenForgeAI/KeenForge?style=social)](https://github.com/KeenForgeAI/KeenForge)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

> **CVPR 2026 Demo Track — 已接收** · 前身 AutoLabel Pro

**KeenForge** 是一款本地优先的桌面软件，能把一个装满图片的文件夹，变成一个训练好的目标检测模型 —— 全程不写一行代码。导入图片，让它**自动去重、聚类**，你只标注少量「簇代表」，模型就在后台自我训练。**数据永远不出你的电脑。**

<p align="center">
  <img src="assets/gifs/01-hero.gif" width="820" alt="KeenForge 2.0：从图片到训练好的模型">
</p>

---

## 为什么做 KeenForge

2024 年我们在复现论文时发现了一件很少被讨论的事：很多 SOTA 数字，是在**有严重质量问题**的公开数据集上算出来的。被广泛使用的 **BCCD** 血细胞数据集就是典型 —— 照片模糊、红细胞重叠无法区分、标注框缺失。

我们重建了一个经过整理的 **1,260 张**数据集（发表于 *Scientific Data*, 2025）。在原始 BCCD 上，模型上限不到 **92%**；在整理后的版本上能到 **98%+**。**数据集质量决定模型上限。**

接着我们撞上第二个瓶颈：**标注**。X-AnyLabeling 这类工具能帮你「画框」，但并不能真正**加快**标注。在我们的实际流程里，先标 ~250 张 → 训练 YOLOv8n → 自动标注剩余 ~1,000 张 → 人工复核一遍。真正省时间的是这个**人机循环**。

KeenForge 的前身是 **AutoLabel Pro**（CVPR 2026 Demo Track）。2.0 想做得更多：不只是标注工具，而是一个**「AI 工程师」**，帮专家快速标完数据集的同时，训练出最优模型。

> 📄 我们的数据集质量研究：[*Scientific Data* (2025)](https://www.nature.com/articles/s41597-025-05980-z)

---

## 2.0 新在哪

2.0 不是「更大的 1.0」，而是把「标完所有图」换成了**闭环主动学习**。

- 🧹 **智能数据清洗** — 四层去重（SHA-256 → 感知哈希 → 像素 MAE + ±2px 平移对齐 → 语义余弦）+ 聚类，自动挑出信息量最大的代表图。
- 🔁 **聚类引导的主动学习** — 只标注簇代表（每轮 **10–50 张**，不是几千张）→ 自动训练 → 模型比对 → **更细粒度重聚类** → 循环。
- 🔍 **模型比对模式** — 让模型去跑**已有标注**的图片。预测框显示为**青色虚线 + 置信度**，人工框保持绿色。**编辑即采纳** —— 未经确认的预测永远不会覆盖你的标注。
- ⚖️ **自选权重续训** — 从任意 `.pt` / `.pth` 接着训；迁移 / 递归 / 从零三种策略。
- ⚡ **非阻塞训练** — 训练跑在独立进程里，训练期间你照样能标注、聚类、复核。
- 📊 **训练总结** — 各类别标注框统计、逐轮 mAP、模型文件一览。
- 🌏 **中英双语界面** — 运行时随时切换。
- 🛡️ **鲁棒的读图** — 按文件内容识别格式，扩展名写错的图（`.png` 其实是 JPEG）也能正常打开。

---

## 工作原理（闭环）

```mermaid
flowchart LR
    A["导入图片"] --> B["四层去重"]
    B --> C["余弦聚类"]
    C --> D["标注簇代表<br/>每轮 10–50 张"]
    D --> E["后台自动训练"]
    E --> F["模型比对<br/>青色 = AI 预测"]
    F --> G["编辑即采纳"]
    G --> C
```

你标注的是**每簇的代表**，而不是整个数据集。模型训练后会预测下一批簇，你只需修正它做错的部分 —— 用**一小部分标注预算**换来一个小众领域强模型。

---

## 实际效果

### 1. 先清洗数据

<p align="center">
  <img src="assets/gifs/02-deduplicate.gif" width="820" alt="数据去重">
</p>

精确、感知、几何、语义四层重复自动找出 —— 包括一对只差 **2 像素**平移的图（对齐后 MAE 10.14 → 2.57）。

### 2. 标代表，而不是标全部

<p align="center">
  <img src="assets/gifs/03-cluster.gif" width="820" alt="聚类引导的代表选择">
</p>

余弦聚类挑出最具代表性的样本，每轮只需标注 **10–50 张**。

### 3. 后台训练

<p align="center">
  <img src="assets/gifs/04-training.gif" width="820" alt="后台训练">
</p>

训练在独立进程运行 —— 你可以继续标注、聚类、复核。

### 4. 比对、修正、循环

<p align="center">
  <img src="assets/model-comparison.png" width="900" alt="青色预测框的模型比对">
</p>

**青色虚线框**是带置信度的模型预测，**绿色框**是人工标注。编辑即采纳 —— 预测永远不会覆盖你的标注。图中：10 张人工标注、868 张待标，模型已能以 0.82–0.97 的置信度给出 RBC/WBC 框。

---

## 实测结果

在我们的血细胞数据集（**1,275 张 · 3 类 · 23,745 个框**）上：12 轮，第 6 轮达到最佳 **mAP50 = 0.993**。完整的「标注 → 训练 → 再标注」循环都记录在训练总结里。

<p align="center">
  <img src="assets/training-summary.png" width="900" alt="训练总结：12 轮，最佳 mAP 0.993">
</p>

---

## 为什么选 KeenForge？

| | Roboflow | Label Studio | **KeenForge 2.0** |
|---|---|---|---|
| **部署** | 云端（数据要上传） | Docker（配置复杂） | **一键启动器** |
| **数据隐私** | ❌ 图片离开你的电脑 | ⚠️ 自托管但笨重 | ✅ **100% 本地离线** |
| **数据清洗** | 付费层 | ❌ 无 | ✅ **内置四层去重 + 聚类** |
| **主动学习** | ❌ 无 | ❌ 无 | ✅ **聚类引导闭环** |
| **AI 辅助** | 付费层 | 手动配置 | ✅ **YOLO-World 零样本** |
| **自动训练** | 付费 GPU 额度 | ❌ 无 | ✅ **簇代表配额达标即触发** |
| **模型归属** | ❌ 被平台锁死 | ✅ 开源 | ✅ **模型属于你** |
| **价格** | $79–$249/月 | 免费（自托管） | **免费开源** |

---

## 你能用它做什么？

- **工业质检**：检测产线上的划痕、凹坑、焊点缺陷
- **PCB / 材料**：检测公开模型根本不存在的表面缺陷
- **医疗 / 显微**：细胞、颗粒、组织区域的计数与分类
- **安全监控**：识别安全帽、反光衣、危险行为
- **科研**：为论文构建自定义数据集 —— 导出 COCO / VOC / YOLO
- **只要你看得见**：能画框的东西，KeenForge 就能学会找它

---

## 快速开始（3 分钟）

### 方式一：一键启动器（小体积，自动安装环境）

到 [Releases](https://github.com/KeenForgeAI/KeenForge/releases) 下载 `KeenForge-2.0.0-source.zip`，解压后：

- **Windows**：双击 `run_windows.bat`
- **Linux / macOS**：`./run_unix.sh`

启动器会在本地创建 `.venv`，首次运行时**自动安装依赖**（几分钟），然后启动 KeenForge。
这样下载包很小 —— 体积大的 PyTorch/Ultralytics 在你机器上现场安装，而不是打进包里。

然后：

1. 加载图片文件夹
2. 在工具栏点 **数据除重** 和 **相似分组**
3. 标注聚类代表 → 达到本轮配额后自动训练
4. 勾选 **模型比对（cyan）** 复核并修正预测 → 循环

### 方式二：手动从源码运行

```bash
git clone https://github.com/KeenForgeAI/KeenForge.git
cd KeenForge
pip install -r requirements.txt
python src/main.py
```

**环境要求**：Python 3.8+ · CUDA 可选（CPU 可跑，训练强烈建议 GPU）。
Windows 上 `pip install torch` 默认装 CPU 版；NVIDIA GPU 用户请用
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`。

---

## 核心功能

### 四层去重

1. **字节层** — SHA-256 精确重复。
2. **感知层** — PDQHash 256-bit（pHash 64-bit 回退），覆盖 **8 种二面体朝向**，用 Hamming 距离比较。
3. **几何层** — 像素级 MAE（8 种变换取最小值）+ **±2px 平移对齐**。
4. **语义层** — MobileNetV3 / ResNet50 特征，L2 归一化后余弦相似度。

GC10-DET（2,306 张）实测：精确命中 **13 组真重复**，其中包含一对只差 2 像素平移的图（对齐后 MAE 10.14 → **2.57**）。

### 余弦聚类与代表选择

图片被向量化后聚类。KeenForge 每簇选 1 个代表，每轮代表数钳制在 **10–50** 张 —— 覆盖率驱动的配额，而不是死板的百分比。

### 零样本起步（YOLO-World）

输入任意物体名 —— `"焊接缺陷"`、`"安全帽"`、`"熟番茄"` —— 立刻找出目标，无需预训练。零样本召回有天花板（我们实测约 85–90%），第一轮训练就能突破。

### 模型库

| 模型 | 用途 |
|---|---|
| YOLOv8n/s/m/l/x | 通用目标检测 |
| YOLO11n/s/m/l/x | 最新 YOLO 架构 |
| RT-DETR-l/x | 实时 Transformer 检测 |
| YOLO-World v2 (s/m) | 零样本开放词表 |
| Faster-RCNN | 高精度、较慢 |
| RetinaNet | 密集目标场景 |
| SSD300 | 轻量、快速 |

### 格式支持

- **导入**：COCO JSON、Pascal VOC XML
- **导出**：COCO JSON、Pascal VOC XML、YOLO TXT
- **训练产物**：标准 Ultralytics `.pt` 权重

---

## 面向小众领域的数据集

目标检测在 COCO 上好演示，却在别处难落地。我们整理并发布了**修正版、开箱即用**的数据集 —— 去重、重划分、修格式：

**NEU-DET · GC10-DET · DeepPCB · PKU-Market-PCB · TXL-PBC · MagneticTile · PlantDoc · PCB_AoI · Raccoon**

→ [huggingface.co/KeenForgeAI](https://huggingface.co/KeenForgeAI) · 索引：[KeenForgeData](https://github.com/KeenForgeAI/KeenForgeData)

> 如果你下载了其中任一数据集，KeenForge 就是清洗它们所用的工具。

---

## 快捷键

| 按键 | 操作 |
|---|---|
| `A` / `D` | 上一张 / 下一张（自动保存） |
| `R` | 切换 绘制 / 编辑 模式 |
| `J` | 移入 Junk 文件夹 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+C/V` | 复制 / 粘贴框 |
| `Ctrl+D` | 复制选中的框 |
| `Delete` | 删除选中的框 |
| `中键拖拽` | 平移画布 |
| `Ctrl + 滚轮` | 缩放 |

---

## 2.x 路线图

- [x] 数据集四层去重
- [x] 余弦聚类 + 代表选择
- [x] 聚类引导的主动学习闭环
- [x] 模型比对模式（编辑即采纳）
- [x] 自选权重续训
- [ ] SAM 分割（多边形掩码）
- [ ] 面向 YOLO 的自适应 finetune
- [ ] 一键 API 部署
- [ ] 支持 `pip install`

**想参与？** 去 [Issues](https://github.com/KeenForgeAI/KeenForge/issues) 找 `good first issue`。

---

## 社区与贡献

我们希望 KeenForge 由**真正在标注小众数据集的人**一起塑造。

- ⭐ 觉得有用就点个 **Star**
- 🐛 **报 Bug** —— 尤其是用你自己的数据集/格式；一个能复现的样本对我们极有价值
- 💡 **想法与问题** —— 开 Discussion 或 Issue
- 🔧 **参与开发** —— UI/UX、新模型接入、文档翻译（中 ↔ 英）

1. Fork 本仓库
2. 建分支：`git checkout -b feature/your-feature`
3. 提交你的改动
4. 推送并开 Pull Request

---

## 引用

如果 KeenForge 对你有帮助，请引用：

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

## 许可证

MIT License — Copyright (c) 2026 KeenForgeAI. 开源与商用均免费。详见 [LICENSE](LICENSE)。

---

<p align="center">
  <b>⭐ 如果它帮到了你，请给个 Star！</b><br>
  <sub>由 <a href="https://github.com/KeenForgeAI">KeenForgeAI</a> 用 ❤️ 打造</sub>
</p>
