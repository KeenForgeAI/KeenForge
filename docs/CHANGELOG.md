# KeenForge 代码变更记录（CHANGELOG）

> 用途：每次代码修改的**文字记录**，便于追溯「改了什么、为什么改、改在哪几行」。
> 规则：每次修改都追加一条记录，并同步 git commit（commit message 与记录标题一致）。

---

## 2026-10-01

### 构建：修复 PyInstaller 打包与冻结版启动崩溃（hnswlib 相关）+ 冻结版 onedir 化

**背景**：发布 2.0 需要重新打包 exe。原 `KeenForge.spec`（onefile）在本机环境下有两个真实阻塞：① 构建时就崩溃；② 即便绕过构建，打出的 exe 启动即段错误。

**问题 1 – 构建阶段崩溃**（PyInstaller 隔离子进程退出码 3221225477）
- 根因：PyInstaller 的 `find_binary_dependencies()` 用**同一个隔离子进程**依次 `__import__` 所有收集到的包。包列表里 `torch` 排在本地包之前，而 `src/core/__init__.py` → `data_cleaner.py` 在模块顶层 `import hnswlib`。实测 `import torch; import hnswlib` 段错误、`import hnswlib; import torch` 正常 —— 即 Intel OpenMP（torch 的 libiomp5md）与 hnswlib 加载顺序冲突。
- 修复（`src/core/data_cleaner.py`）：把顶层 `import hnswlib` 改为**惰性导入** —— 用 `importlib.util.find_spec("hnswlib")` 探测 `HAS_HNSWLIB`（不真正加载），真正用到时（`_hnsw_dedup`）才 `import hnswlib`。PyInstaller 仍通过字节码静态发现该 import。

**问题 2 – 冻结版启动段错误**
- 现象：开发版（源码）正常，打出的 exe 一启动就 access violation。
- 定位：给 `entry.py` 顶部加 `faulthandler.enable()`，栈显示崩在 `entry.py` 的 `import hnswlib` —— 冻结 bundle 里加载 hnswlib 原生扩展直接段错误（与 bundle 内的 `mkl_rt.1.dll` / `torch/lib/libiomp5md.dll` 冲突）。
- 修复（`KeenForge.spec`）：把 `hnswlib` 加入 `excludes`，从 bundle 中剔除。`data_cleaner` 检测不到 hnswlib 时自动回退到 **`_bruteforce_cosine_dedup`**（`features @ features.T`，目标数据集规模 2000~3000 张下秒级，完全够用）。开发版仍可享受 hnswlib 加速。

**问题 3 – onefile 体积与可诊断性**
- onefile 产物 2.6GB，每次启动解压到临时目录，且崩溃后临时目录被清理无法诊断。
- `KeenForge.spec` 改为 **onedir**（`EXE(exclude_binaries=True)` + `COLLECT`），启动更快、可直接查看 bundle 内容。产物 `dist/KeenForge/`（约 4.3GB，含大量当前 conda 环境带入的无关包，体积裁剪另行处理）。

**其它**
- `KeenForge.spec` 的 `hiddenimports` 补齐 2.0 新增模块（`cleaning_dialog` / `training_dialog` / `data_cleaner` / `app_id` / `coco_matcher` / `image_io`）与可选后端（`clip` / `timm` / `imagehash` / `squarify`）。
- `upx` 由 True 改为 False（本机未装 UPX，此前一直是空操作，显式关闭以免误导）。
- 已知（另行处理）：`torchmetrics` 在本环境缺失 → 训练 mAP 恒为 0；`requirements.txt` 与实际环境版本不一致；`KeenForge.spec` 被 `.gitignore` 的 `*.spec` 忽略，未纳入版本管理。

**验证**：`py_compile` 退出码 0；onedir 构建 Exit 0；`dist/KeenForge/KeenForge.exe` 启动后窗口标题 `KeenForge: 智能标注与自动推理`、`Responding=True`、stdout/stderr 均 0 字节、无 crash。

**改动文件**：`src/core/data_cleaner.py`、`entry.py`（`KeenForge.spec` 已改但被 gitignore）

---

## 2026-09-30

### 发布：版本号统一到 2.0.0 + README 2.0（中英双版）

**背景**：准备发布 KeenForge 2.0（分支 `dev-clustering-v2`，领先 `origin/main` 30 个 commit）。但仓库里仍有三处写着 1.0，且 `README.md` 还是 v1 的内容（roadmap 里"去重 in progress"其实早已完成）。发版前必须先统一版本标识。

**A. 版本号统一**（3 处）：
- `src/__init__.py`：`__version__ = "1.0.0"` → `"2.0.0"`
- `src/utils/app_id.py`：`APP_ID = "com.keenforgeai.keenforge.v1.0"` → `"com.keenforgeai.keenforge.v2.0"`（Windows AppUserModelID。注意：改 AUMID 会让旧的 `v1.0` 注册表项残留，无害，但任务栏图标/名称需要**重启应用**后才会按新 ID 注册。）
- `CITATION.cff`：`version: 1.0.0` → `2.0.0`，`date-released: 2026-09-20` → `2026-09-30`
- `KeenForge.spec` 无版本字段（EXE 未设 `version=`），无需改。窗口标题 `APP_TITLE` 本就不含版本号，无需改。
- 全仓扫描 `*.py / *.cff / *.spec / *.txt / *.md` 确认无残留 `1.0.0` / `keenforge.v1.0`。

**B. README 2.0**：
- `README.md` 重写为 2.0 版：新增 "What's New in 2.0"、闭环示意图段落、对比表补「Data Cleaning / Active Learning」两行、把 "Auto-triggers after 15 labels" 更正为「簇代表配额达标即触发（通常每轮 15–50 张）」、Roadmap 勾掉已完成的去重/聚类、新增「Niche-domain Datasets」段落链到 HuggingFace 组织。
- 新增 `README.zh-CN.md` 全中文版；两版头部互加语言切换徽章。

**验证**：`python -m py_compile src/__init__.py src/utils/app_id.py src/config.py src/ui/main_window.py` 退出码 0；清 `src/**/__pycache__`；`app_stderr.log` 为 0 字节。

**改动文件**：`src/__init__.py`、`src/utils/app_id.py`、`CITATION.cff`、`README.md`、`README.zh-CN.md`（新增）、`src/config.py`、`src/ui/main_window.py`（后者两文件为上一次会话遗留的训练总结布局调整，本次一并提交）

---

## 2026-09-23

### 功能：训练总结新增「标注框数」统计

**背景**：训练总结此前只报告轮次 / mAP / 用时 / 模型文件，看不到标注规模。用户希望补一个标注框数统计，以便判断「这个数据集一共标了多少个框、各类分布如何」。

**实现**（`src/ui/main_window.py`、`src/config.py`）：
- 新增 `MainWindow._count_boxes_by_class(img_list)`：逐个读 YOLO txt（`cls cx cy w h`），累计总框数与各类框数，返回 `(total, [(类名, 框数), ...])` 且按框数降序；类名取自 `self.classes`，越界回退 `class_<id>`；标签文件缺失或行格式损坏时静默跳过，不抛异常。
- `_build_training_summary_html()`：头部新增一行「标注框: 共 N 个 · 覆盖 K 类 · 平均每图 X 个」；轮次表格下方新增「各类别标注框统计」表（类别 / 框数 / 占比）。统计口径与头部「数据集: 共 N 张图片」保持一致，同样基于 `self._full_image_list or self.image_list`。
- `src/config.py`：新增中英文案 `SUM_BOXES` / `SUM_BOXES_TITLE` / `SUM_TH_CLASS` / `SUM_TH_BOXES` / `SUM_TH_SHARE`（zh + en 各 5 条）。

**验证**：`py_compile` 两个改动文件退出码 0；清 `src/**/__pycache__` 后重启应用（stderr 为空）；用真实 `MainWindow` 方法 + 桩对象跑无头测试全过 —— i18n 键在 zh+en 均存在且 `.format()` 正常；真实数据（`txlbc-test`，1275 图）统计出 23,745 框 / 3 类（RBC 21050 · WBC 1984 · Platelets 711）；原有内容零回归（最佳模型行、轮次区间行仍在）；`<tr>` 计数 = 13 + 1 + 类别数；中英双语渲染均正常且无未替换的 i18n 键泄漏。

**已知限制**：统计依赖 YOLO txt 标签与 `get_label_path`，因此只对「扁平 `images/` + `labels/`、标签与图片同名」的布局生效；嵌套划分目录（`images/train/`）或图片名带 `_test` 而后缀标签不带（如 DeepPCB-corrected）的数据集统计为 0 —— 这是 `get_label_path` 的既有行为，本次未改动。

**改动文件**：`src/ui/main_window.py`、`src/config.py`

### 功能：右侧图片列表新增「查找图片」筛选框

**背景**：数据集动辄上千张（如 inclusion 2266 张、GC10-DET 2254 张），右侧文件列表只能靠滚动定位，想复核某一张指定图片（例如 `inclusion_img_02_425392000_00984.jpg`）非常费劲。用户希望能在右侧任务栏直接按名字查找图片。

**实现**（`src/ui/main_window.py`、`src/config.py`）：
- 右侧列表面板顶部（统计标签之上）新增 `self.txt_find` 搜索框：占位提示「查找图片…（回车跳到第一张匹配）」，带一键清除按钮。
- 新增 `_apply_file_filter(text)`：按**文件名子串**（大小写不敏感）**隐藏**不匹配的行；配套新增 `_visible_count()`、`_jump_to_first_match()`（回车选中第一张匹配）。
- 关键设计：用 **`setHidden` 隐藏而不是删除行**，因此行号始终与 `self.image_list` 一一对应 —— `change_image(index)`、`move_to_junk()` 的 `takeItem(current_index)+pop`、聚类重排等所有按下标工作的逻辑都无需改动、零回归风险。过滤期间临时 `blockSignals`，避免打字时误触发切图与推理。
- `populate_file_list()`：给每个 `QListWidgetItem` 通过 `setData(Qt.UserRole, path)` 写入真实路径（用于匹配），并在重建列表后**自动重新应用**当前筛选（切换数据集 / 除重 / 聚类后筛选状态不丢）。
- `_update_list_stats()`：筛选生效时在统计后追加「· 筛选 X/Y 张」，实时显示命中数量。
- `src/config.py`：新增中英文案 `PH_FIND_IMAGE` / `TIP_FIND_IMAGE` / `FMT_LIST_FILTER`（zh + en 各 3 条）。

**验证**：`py_compile` 退出码 0；用真实 `MainWindow` 方法 + 桩对象跑无头测试 23 项全过（子串 / 大小写 / 仅匹配文件名、隐藏计数、统计文案、回车跳转、清除恢复、行号↔image_list 对应、中英键位齐全）。
**改动文件**：`src/ui/main_window.py`、`src/config.py`


### 功能：首轮训练支持加载「自选模型」权重

**背景**：训练设置对话框的模型下拉框只能选内置基础模型（yolov8n/yolo11n/rtdetr…）或本数据集历史轮次的模型，无法从别处拿一个自己训练好的 `.pt` 接着训（例如跨数据集迁移或复用外部权重）。

**实现**（`src/ui/training_dialog.py`、`src/ui/main_window.py`、`src/config.py`）：
- `TrainingSetupDialog` 模型行右侧新增「浏览…」按钮（`_on_browse_model`）：用 `QFileDialog`（`*.pt *.pth`）挑任意权重文件，默认打开本数据集 `training_runs/loop_train/weights`；选中后以 `insertItem(0, "自选: <文件名>", <绝对路径>)` 写入下拉框并选中，重复选同一文件不会产生重复项。
- `get_params()` 沿用 `currentData() or currentText()`：自选项返回**绝对路径**，普通项返回模型名，调用方契约不变。
- `MainWindow.start_background_training()`：
  - 新增 `custom_model_path` 判定 —— 仅当所选值以 `.pt/.pth` 结尾**且是已存在的文件**时才算「自选」，避免把 `best_xxx.pt` 这类裸文件名误判成路径。
  - `base_model_key` 改为从 basename 派生（自选时），保证轮次产物命名 `best_<key>_r<n>_m<mmmm>.pt` 仍然规范。
  - `params_model_path` 优先取自选路径，并在策略分支（transfer/recursive/scratch）之后统一覆盖 `model_to_use = custom_model_path` —— **无论哪种策略、YOLO 还是经典模型，都用用户选的那个文件开训**。
- `src/config.py`：新增中英文案 `TD_BTN_BROWSE` / `TD_TIP_BROWSE` / `TD_CUSTOM_PREFIX`（zh + en 各 3 条）。

**验证**：`py_compile` 三个改动文件退出码 0；无头测试（真实方法 + 桩）—— 对话框契约 11 项全过（浏览写绝对路径、get_params 返回路径、重复选不重复、取消不改选、与已有轮次模型共存），训练解析 9 项全过（自选 .pt/.pth 命中、基础模型仍落到项目根 `yolov8n.pt`、历史轮次模型仍命中 weights 目录、回归无泄漏）。
**已知限制**：经典 Torchvision 模型（ssd/faster/retina）的训练分支本就无法从 `.pth` 恢复权重（会从 torchvision 默认权重重建），选自选 `.pth` 时同样受此限制；本功能主要服务 YOLO `.pt` 自选权重。
**改动文件**：`src/ui/training_dialog.py`、`src/ui/main_window.py`、`src/config.py`

### 改进：查找框默认折叠为 🔍 图标 + 修复：训练中聚类误用全量图片（1256→1055）

**背景**：
1. 上一版「查找图片」是**常驻**的输入框，不用查找时也会占掉右侧面板一行；用户希望默认只留一个 🔍 图标，点了才展开输入框。
2. 在训练**进行中**点工具栏「聚类」做第二轮分组时，对话框显示「将参与聚类的图片: 1256 张」（全量），而不是「1055 张」（尚未标注的图）；必须等训练跑完再点才是 1055。

**改动一：查找框折叠为 🔍 图标**（`src/ui/main_window.py`）
- 右侧面板顶部改为一行 `QHBoxLayout`：统计标签（拉伸权重 1）+ `self.btn_find`（`QToolButton`，文本 `🔍`，`checkable`，`NoFocus`）；输入框 `self.txt_find` 移到该行下方，初始 `setVisible(False)`。
- 新增 `_toggle_find_box()`：点 🔍 → 展开 + `setFocus()` + `selectAll()`；再点一次 → 收起。
- 新增 `_collapse_find_box()`：收起时**一并清空筛选**（清空会触发 `_apply_file_filter("")` 让所有行恢复显示），避免「输入框已隐藏但列表仍被过滤」的困惑状态。
- 新增 `Esc` 快捷键（`QShortcut` + `QKeySequence(Qt.Key_Escape)`，`setContext(Qt.WidgetShortcut)` 仅焦点在输入框时生效）：同样走收起 + 清空。
- `retranslate_ui()`：切换中英文时同步刷新 🔍 的 tooltip 与输入框的占位文案（沿用 `hasattr` 守卫风格）。
- 新增导入：`QShortcut`（QtWidgets）、`QKeySequence`（QtGui）。
- 过滤本体（隐藏行而非删行、行号↔`self.image_list` 一一对应、统计文案「· 筛选 X/Y 张」、回车跳转）完全沿用上一版，**未改动**。

**改动二：训练中聚类应只针对未标注图片**（`src/ui/main_window.py`）
- **根因**：`open_cluster_dialog()` 用 `self._has_trained_model()` 判断「是否已进入下一轮」。而 `_has_trained_model()` 只看磁盘上有没有 `training_runs/loop_train/weights/best_*.pt` —— 这个文件是**训练结束时**才写出的，所以训练进行中判定为 False，代码走 `base = list(self.image_list)`（全量 1256 张）。
- **修复**：改用「**标注状态**」作为判据 —— 遍历 `self._full_image_list`（为空时回退 `self.image_list`）统计已存在 label 文件的图片数；只要 `labeled > 0` 或 `_has_trained_model()` 为真，就按「下一轮」处理，只聚类未标注图片。
- 顺带修正：`_full_image_list` 为空时不再把 `remaining` 算成空列表而误弹「训练总结」。
- 训练前（一张未标注）行为不变（聚类全量）；训练结束后行为不变。**只有训练进行中这一段时间从 1256 修正为 1055。**

**验证**：
- `python -m py_compile src/ui/main_window.py` 退出码 0。
- 无头测试 `test_find_toggle.py`（真实 `MainWindow` + offscreen，**23 项全过**）：默认收起 / 🔍 是放大镜字符 / 可勾选 / 初始未勾选 / `txt_find.isHidden()` / 点击展开 / 输入 `cat` 命中 2 张且文案「筛选 2/3 张」/ 再次点击收起且清空且恢复 3 行 / Esc 收起且清空 / `retranslate_ui()` 不崩且刷新文案。
- 无头测试 `test_cluster_gate.py`（真实 `open_cluster_dialog` + 桩对象 + 假 `CleaningDialog` 捕获 base，**10 项全过**）：全未标注 → base=6；**训练中（2 张已标注、磁盘无 `best_*.pt`）→ base=4（修复前为 6）**；训练结束 → base=4（与训练中一致）；全部标注 → 弹训练总结不弹对话框；仅剩 1 张 → 弹提示；`_full_image_list` 为空 → 回退且 base=4。
- 重启应用（`launch_debug.py`）后 `app_stderr.log` 为 0 字节、无 crash 日志；`git status` 干净。
- 测试脚本会把 `QSettings` 快照并**原样恢复**，不污染用户真实窗口状态（已核对注册表 `dataset_root` / `current_index` / `model_path` 与测试前一致）。

**改动文件**：`src/ui/main_window.py`、`src/config.py`

### 优化：顶部工具栏重排 + 加大字体 + 按钮重命名

**背景**：工具栏字体只有 12px 偏小，10 个按钮挤成一排没有逻辑分组；顺序也跟实际工作流不符。用户要求按「加载数据集 → 除重 → 聚类 → 标签设置 → 加载模型 → 训练总结 → 导入标签 → 导出标签 → 语言」的顺序排，并把字体调大。

**改动**（`src/ui/main_window.py`、`src/config.py`）：
- **顺序重排**：按用户指定的工作流顺序组织为 4 组，组间加分隔线：
  1. 加载数据集
  2. 数据除重 + 相似分组（数据预处理）
  3. 标签设置 + 加载模型 + 训练总结（标注/模型）
  4. 导入标签 + 导出标签 + 语言（数据 I/O 与设置）
- **字体加大**：工具栏按钮字号 `12px → 14px`，padding `4px 10px → 6px 12px`，圆角 `4px → 6px`，统一 `PointingHandCursor`；背景改为更柔和的 `#f5f6f8`，hover 变 `#e4e9f0`、按下/选中变蓝色高亮 `#d0e4ff`，分隔线变细 `#d0d4da`。
- **按钮重命名**（`src/config.py` zh + en 同步）：
  - `BTN_DEDUP`: `除重` → `数据除重`（更明确只做重复移除，不动聚类）
  - `BTN_CLUSTER`: `聚类` → `相似分组`（更准确 —— 它把相似图片分组，每组选 1 张代表标注，不是单纯的聚类算法）
  - `TIP_CLUSTER`: 提示同步改为「按相似度把图片分组，每组选 1 张代表优先标注」
  - `BTN_LABEL_SETUP`: `标签设置 (类别 + AI 辅助)` → `标签设置`（删掉括号，更简洁）
  - `BTN_SUMMARY`: 英文 `Summary` → `Training Summary`（与「训练总结」对齐）
- `retranslate_ui()` 中工具栏文案刷新顺序改为与显示顺序一致（原来刷新顺序和实际位置对不上，中英切换时按钮短暂显示旧文案）。

**验证**：
- `python -m py_compile src/ui/main_window.py src/config.py` 退出码 0。
- 无头测试 `test_toolbar.py`（真实 `MainWindow` + offscreen，**15 项全过**）：工具栏实际顺序 = 9 按钮 + 4 分隔线、按钮类型正确、`font-size: 14px`、hover/pressed/separator 样式齐全、英文文案与顺序正确、`Label Setup` 无括号、`retranslate_ui()` 后中英顺序与文案一致。
- 重启应用（`launch_debug.py`）后 `app_stderr.log` 为 0 字节、无 crash 日志。

**改动文件**：`src/ui/main_window.py`、`src/config.py`

### 优化：左侧栏清理死代码 + 训练监控卡片放大 + 底部信息分组

**背景**：左侧栏 `setup_ui` 里构建了完整的「模型设置 + 策略单选 + 参数表」区块（`model_group` / `lbl_mode` / 3 个单选 / `param_group`），但训练设置已挪进 `TrainingSetupDialog`，这些控件全部 `setVisible(False)` 藏起来——约 80 行死代码，既拖慢启动又让后来者分不清「到底哪个是模型选择器」。另外监控指标卡片偏小（数值 15px / 标题 10px）、日志只有 90px 高、底部 `tips`/`data_source`/`infer_result` 三个信息标签样式不统一。

**改动**（`src/ui/main_window.py`）：
- **清理隐藏控件死代码**：`model_group` / `lbl_mode` / `radio_transfer` / `radio_recursive` / `radio_scratch` / `param_group`（含 `lbl_model_title`/`lbl_prompt_title`/`lbl_epochs_title`/`lbl_freeze_title` 等附属标签）不再挂载进 `left_layout`，也不再做 `setVisible(False)`。但**保留对象本体**（`self.combo_yolo_model`/`spin_epochs`/`spin_freeze`/`radio_*`/`txt_prompt`）——它们的 `.currentText()`/`.value()`/`.isChecked()` 被 `check_trigger_training`、`start_background_training`、`closeEvent`（`base_model_index` 持久化）、策略预设等大量引用，当默认值用，删对象会连锁崩溃。
- `retranslate_ui()` 同步加 `hasattr` 守卫：这些控件已从属性里消失，语言切换时不能再访问。
- **训练监控卡片放大**：`metric_val_*` 数值 `15px → 18px`，新增 `QLabel#metric_title { font-size: 11px }`；`QProgressBar` `min/max-height 18px → 22px`、字号 `11px → 12px`；`log_text.setMaximumHeight 90 → 120`。
- **底部信息统一分组**：`lbl_data_source` + `lbl_infer_result` + `tips_label` 收进 `self.info_group`（`QGroupBox`），统一留白/间距；给两个裸标签补上 `objectName`（`lbl_data_source` / `lbl_infer_result`）并在动态样式表里配字号/颜色，不再靠行内 `setStyleSheet` 散写。

**验证**：
- `python -m py_compile src/ui/main_window.py` 退出码 0。
- 无头测试 `test_sidebar.py`（真实 `MainWindow` + offscreen，**30 项全过**）：被保留对象仍在（`combo_yolo_model`/`spin_epochs`/`spin_freeze`/`radio_recursive` 默认值不变）、死控件已消失（`model_group`/`param_group`/各附属标签均无 `hasattr`）、`log_text` 限高 120、样式表含 18px/22px、`info_group` 存在且三个信息标签都在其内、`retranslate_ui` 中英切换不崩、监控标题正确刷新。
- 重启应用（`launch_debug.py`）后 `app_stderr.log` 为 0 字节、无 crash 日志。

**改动文件**：`src/ui/main_window.py`

### 修复：训练中配额已满却弹训练框（点了没反应）→ 改为提示一次 + 记住 + 训练结束自动弹

**背景**：用户在训练进行中完成了本轮全部代表标注，配额已满。此时 `check_trigger_training()` 会**无条件**弹出 `TrainingSetupDialog`；但 `start_background_training()` 第一行是 `if self.is_training: return`（静默返回），导致用户填完整个对话框点「确定」后什么都不会发生，且无任何提示。另外 `check_trigger_training()` 在 `1128`（存标注）、`1756`（D 键跳图）两处被调，训练中每存一次标注就可能再弹一次，用户不处理就一直弹。

**改动**（`src/ui/main_window.py`、`src/config.py`）：
- 新增状态标志 `self._pending_round_ready`（`__init__` 初始化为 `False`，`load_dataset_from_path` 同步清零）。
- `check_trigger_training()`：配额已满后、`_restore_full_list()` **之前**插入 `is_training` 门检。若仍在训练 → 置 `_pending_round_ready=True`、弹**一次** `MSG_ROUND_WAIT_TRAINING` 提示（去重靠标志位，后续同帧/同轮内不再弹）、状态栏显示 `FMT_ROUND_WAIT_TRAINING`，然后 `return`；**不**恢复全量列表，不打扰用户正在进行的标注。
- `check_training_status` 的 `status == "success"` 分支：在 `update_dynamic_styles()` 后检测 `_pending_round_ready`，为真则清零并用 `QTimer.singleShot(350, self.check_trigger_training)` 把训练设置框**延迟到下一轮事件循环再弹出**，避免在 1s 轮询回调里嵌套 `exec_()` 导致 Qt/matplotlib 崩溃（沿用 `_open_next_round_cleaning` 的同款写法）。
- `src/config.py` 新增中英文案 `MSG_ROUND_WAIT_TRAINING`（告知「第{0}轮标注已完成，但第{1}轮模型仍在训练中…你的标注已保存」）与 `FMT_ROUND_WAIT_TRAINING`（状态栏「第{0}轮已就绪 · 等待当前训练结束后自动开始」），zh + en 各 2 条。
- 训练失败 / 用户手动 `stop_training` 时 `_pending_round_ready` **保留** —— 标志只在「成功结束」分支被消费；失败/停止属于非预期中断，下一轮就绪状态理应依然有效（用户下次正常触发时会重新判断）。

**验证**：
- `python -m py_compile src/ui/main_window.py src/config.py` 退出码 0。
- 无头测试 `test_defer_training.py`（真实 `check_trigger_training` + 桩对象 + 假 `TrainingSetupDialog`/`QMessageBox`，**12 项全过**）：空闲时配额满 → 正常弹对话框；**训练中配额满 → 只弹一次 `MSG_ROUND_WAIT_TRAINING`，不弹训练框，置 `_pending_round_ready`，状态栏显示「已就绪 · 等待…自动开始」**；重复调用不再二次弹提示；训练成功分支消费标志并 `singleShot(350)` 重新触发 `check_trigger_training`；训练结束后再调 → 训练设置框正常弹出。
- 重启应用（`launch_debug.py`）后 `app_stderr.log` 为 0 字节、无 crash 日志。

**改动文件**：`src/ui/main_window.py`、`src/config.py`

---

## 2026-09-21

### 改进：去重像素门限增加小平移对齐（抓「同图平移几像素」的重复）

**背景**：用软件扫 NEU-DET 时发现 `scratches_232` / `scratches_247`（旋转 90° 的近似重复）被正确报出，但同类中还有一对 `pitted_surface_171` / `pitted_surface_292`（同图平移 2px）**漏报**。

**根因**：`DedupEngine._pair_mae()` 只做 8 种二面体变换，**不做平移对齐**。平移过的副本在 no-shift 下 MAE 偏高（pitted 对在 128×128 下 10.14 > 阈值 6.0）被拒；而平移对齐后只有 2.57。

**修复**（`src/core/data_cleaner.py`）：
- `_pair_mae(path_a, path_b, shift_range=0)` 新增 `shift_range` 参数：>0 时对 b 做 ±shift_range 像素的整数平移搜索，取最优匹配。
- `DedupEngine.__init__` 新增 `shift_search_px=2`、`shift_search_factor=3.0`。
- `_hash_scan` 改为两段门限：no-shift MAE 已过阈值 → 直接判重；否则若落在 `shift_search_factor × 阈值` 邻域内（≤18）才付出平移搜索的代价，避免全量扫描变慢。

**实测**：`pitted_surface_171` vs `pitted_surface_292` → no-shift 10.14 → shift2 **2.57**（现在会被判为重复）；随机同类对仍为 ~50，无误报。

**改动文件**：`src/core/data_cleaner.py`

---

### 修复：移除零检出时的状态栏提示（用户不需要）

**现象**：上一版在推理成功但 0 个预测框时，会在左下角状态栏常驻显示「未检测到任何目标（0 个预测框）」，用户觉得多余、干扰视线。
**修复**：`on_inference_result` 恢复为原始的静默早退——把「空推理」与「图片尺寸为 0」合并成一行 `if img_w == 0 or img_h == 0 or not predictions: return`，不再写状态栏。
**改动文件**：`src/ui/main_window.py`（`MSG_NO_DETECTIONS` 键保留在 `src/config.py`，不再被引用）。

---

### 修复：模型比对模式下「无标签图片」的预测框被画成人工标注 + 零检出无提示

**现象**：① 在「模型比对」模式下打开一张**没有标签**的图片（如 `inclusion_img_02_425392000_00984.jpg`，全目录 2266 张中有 12 张无标签），模型预测框被画成了**绿色实线**，和人工标注完全一样，用户无法区分，误以为「没有 AI 画框」；② 推理成功但一个目标都没检出时，`on_inference_result` **静默 return**，界面上没有任何提示。

**根因**（`src/ui/main_window.py` → `on_inference_result`）：AI 框的判定条件写成了「比对模式 **且** 图片有标签」：
```python
_labeled = (... os.path.exists(self.get_label_path(...)))
is_ai = _compare_mode and _labeled
```
无标签图片 `_labeled=False` → `is_ai=False` → 走普通预标注的绿色样式。但「模型比对」的语义应当是：**只要开了比对，模型画的一切都是「模型的意见」**，与图片自身有没有标签无关。

（实测证据：用用户训练的 `best_yolo11n_r1_m0804.pt` 在该图上跑，确实检出 2 个框，conf 0.4228 / 0.2611，均高于 `CONF_STANDARD=0.25` —— 所以不是阈值问题、也不是模型没检出，纯粹是样式判定写错。）

**修复**：
- `is_ai` 判定改为**只看比对模式**：`is_ai=bool(getattr(self, '_compare_mode', False))`。语义变为：比对模式 ON → 模型框一律青色虚线且永不写入标签；比对模式 OFF + AI 预标注 ON → 绿色实线普通预标注（行为不变）。
- 新增 `MSG_NO_DETECTIONS` 提示：`predictions` 为空时在状态栏显示「未检测到任何目标（0 个预测框）」，不再静默返回。`img_w/img_h == 0` 的早退保持不变。

**改动文件**：`src/ui/main_window.py`、`src/config.py`（新增 `MSG_NO_DETECTIONS` 中英文案）。

---


### 修复：导入 Pascal VOC 丢失类别名（按首次出现分配编号 + load_existing_labels 用数字编造类别名）

**现象**：用户导入 Pascal VOC 数据集后类别名被破坏 —— 除了拼写错误的 `10_yaozhed`，还混进了垃圾类别 `d`；`data.yaml` 变成 `0: person, 1: '1', 2: '2', ... 9: '9'`，真实类别名全部丢失。

**根因**（两个 bug + 一处缺失）：
1. `src/core/label_converter.py` `from_pascal_voc()`：类别索引按**原始类别字符串首次出现的顺序**分配（`temp_classes = list(self.classes)` 后逐个 append 新名字）。索引↔名字映射既不稳定，又以原始字符串为键 —— 拼写错误的名字和垃圾名都会各自变成一个类别。
2. `src/ui/main_window.py` `load_existing_labels()`：当标签的类别索引超出当前类别表时，用 `self.classes.append(str(len(self.classes)))` **按索引编造类别名**，这正是 `data.yaml` 出现 `1:'1' … 9:'9'` 的来源。
3. VOC 导入成功后**没有把发现的类别名写回 `data.yaml`**，类别名无法持久化。

**修复**：
- `load_existing_labels()`：不再用 `str(len(self.classes))` 编造，改为写入明确标记的占位名 `class_{索引}`（保留原 while 守卫），并加注释：绝不按索引编造类别名，用户需在「标签设置」里修正。
- VOC 导入处理器 `import_voc_labels()`：导入成功后把返回的新类别名 `extend` 到 `self.classes`，并调用 `save_data_yaml()` 将真实类别名写入 `data.yaml`；若 `converter.skipped_classes` 非空（无效类别名），额外弹窗提示。
- `from_pascal_voc()`：新增轻量校验 —— 类别名需匹配 `^[A-Za-z0-9_\-]+$` 且长度 ≥ 2；不合格的名字（如 `d`、`?`、空串）收集进新增属性 `self.skipped_classes` 上报调用方，**不再静默生成类别**。返回签名保持 `(count, new_classes)` 不变，新增属性向后兼容。

**验证**：`py_compile` 三个改动文件退出码 0；单文件加载 `label_converter.py` 实测：`person`→0、`10_yaozhed`→1，`d`/`?`/空串进入 skipped 且不生成标签文件；重启应用后 `app_stderr.log` 0 字节、无 `keenforge_crash.log`。

**改动文件**：`src/core/label_converter.py`、`src/ui/main_window.py`、`src/config.py`（新增翻译键 `MSG_SKIPPED_CLASSES` 中英各一）。

---

### 功能：训练收尾改为非阻塞（去掉自动弹窗）+ 新增「模型比对」模式

**背景**：用户两点诉求 —— ① 训练完成后会自动弹出聚类向导，而训练本身可能很久（GC10-DET 第 1 轮 2253 张跑了 **56.8 分钟**），用户希望训练期间能并行做聚类和标注；② 已部署的模型对**已有标注**的图片完全不预测，因为 `change_image` 只在没有标签文件时才跑推理，用户想用模型给自己的标注做交叉复核。

**A. 训练收尾改为非阻塞**（`src/ui/main_window.py` → `check_training_status` 的 success 分支）

- **删掉** `self._show_next_round_cleaning_dialog()` —— 聚类向导不再自动弹出。需要时点工具栏「聚类」即可（`open_cluster_dialog` 本来就支持「有训练模型时只聚类未标注图片」）。
- 训练报告从**模态弹窗** `QMessageBox.information` 改为**状态栏 + 训练日志**；完整报告仍可点工具栏「训练总结」查看。
- `_restore_full_list()` 改为**只在 `_round_is_active`（轮次模式）时**才调用 —— 不再无条件重写 `image_list` 并刷新列表，你正在做的聚类视图不会被冲掉。
- 说明：训练本来就跑在独立的 `multiprocessing.Process` 里，UI 全程可响应；`is_training` 也不阻塞标注 / 翻页 / 聚类。真正打断你的只有上面两个模态弹窗 + 一次列表重写。

**B. 新增「模型比对」模式**

- 左侧栏新增复选框 **「模型比对（青色虚线）」**，与「AI 预标注辅助」并列；状态随数据集持久化到 `annotation_settings.json` 的 `compare_mode`。
- 开启后，`change_image` 对**已有标注的图片也会跑一遍模型**：预测框以**青色虚线**绘制，人工框仍是绿色实线。
- 预测框**显示置信度**（如 `2_hanfeng 0.87`）。为此 `inference_engine.py` 的预测元组由 5 个元素扩展为 6 个 `[cls_id, x, y, w, h, conf]`；`model_wrappers.py` 的经典 / ONNX 两条路径同步补上置信度；`on_inference_result` 对旧的 5 元素元组仍兼容（`pred[1:5]` + 缺省 conf=1.0）。

**C. 数据安全：预测框永不污染人工标注（关键）**

- `BoxItem` 新增 `is_ai` / `conf` 属性；`add_box()` 新增 `is_ai` / `conf` 参数，并**返回创建的框**。
- `save_current_annotation()` 增加过滤：`if getattr(item, 'is_ai', False): continue` —— **未审核的模型预测永远不会写入标签文件**。
- **编辑即采纳**：移动或缩放预测框时（`itemChange` 的 `ItemPositionChange` 分支 + `setRect()` 覆写）自动调用 `promote_to_human()`，`is_ai` 置 False、颜色转绿，从此变成可保存的人工标注 —— 延续人机回环「模型推理 → 人工审核修改」的语义。
- 切换图片时预测框随 `load_pixmap()` 的 `scene.clear()` 自动清除，不会残留。
- 颜色方案：人工=绿色实线，模型预测=青色 `#00BCD4` 虚线，选中=红色（互不冲突；虚线保证灰度截图或色盲用户也能区分）。

**改动文件**：`src/ui/main_window.py`、`src/ui/canvas.py`、`src/core/inference_engine.py`、`src/core/model_wrappers.py`、`src/config.py`。

---

## 2026-09-20

### 修复：去重引擎误删（感知哈希过松 + 保留者不看标注）

**背景**：在 GC10-DET 上实测发现，软件的「除重」把**有标注、内容并不相同**的图片当成重复删掉了。对 40 张被删图片逐像素复核：只有 **13 张是真重复**（逐像素相同，或 90°/180°/270° 旋转 / 镜像相同），另外 27 张中 **19 张**与最近似图的像素平均误差高达 6~21（明显不是同一张图）。

**根因**（`src/core/data_cleaner.py`）：
1. **感知哈希阈值对均匀背景图过松** —— `hash_threshold=5` 是 PDQ 哈希的汉明距离阈值。钢板表面检测图大面积是均匀灰度背景，PDQ 哈希天然接近，汉明距离很容易 ≤5 → 误判。
2. **保留者选择完全不看标注** —— 三层去重的「保留者」都只是 `paths[0]`（扫描顺序里第一个遇到的）。`DuplicateGroup.representative` 的注释写着 *best-quality / highest-resolution image*，但**代码里从未实现任何质量排序**；而且 `DedupEngine` 只接收图片路径，根本不知道标签存在，无法倾向于保留有标注的那张。

**修复**：
- `DedupEngine.__init__` 新增 `pixel_mae_threshold: Optional[float] = 6.0`：哈希命中后增加**第二道像素级确认** —— 两张图在 8 种二面体变换下的最小像素平均误差（0-255）必须 ≤ 阈值才算重复。只在哈希已命中的配对上做（不遍历全量），所以开销可忽略；设为 `None` 可退回旧行为。
- `DedupEngine.scan` 新增 `keep_priority: Optional[Callable[[str], int]] = None`：调用方可传入「保留优先级」，数值越大越可能被保留。新增模块级辅助函数 `_reorder_group_keep_priority()` 按优先级重排每组的 `paths`（`paths[0]` 是保留者、`paths[1:]` 是待删），并把 `scores` / `reasons` 同步重排、修正 `reasons[0]`。排序是**稳定**的且给原保留者一个 tie-break 加成，所以 L3（旋转/翻转）组原本「保留 original 朝向」的选择不会被破坏。
- 修正 `DuplicateGroup.representative` 的误导性注释。

**实测**（GC10-DET 原始 2306 张）：
- 修复前后引擎都报 **13 组重复**，与逐像素复核的真重复数一致
- `keep_priority` 三项单元测试通过：有标注副本被提升为保留者；同优先级时保留 L3 原有朝向选择；L1 组的 reasons 数量不变
- 扫描耗时无变化（像素确认只在哈希命中时触发）

**⚠️ 未解之谜**：GC10-DET 的 `_removed_duplicates/` 里有 40 张，但引擎只能解释其中 **15 张**（13 组重复的 13 个副本 + 2 个被一并勾选的保留者），另外 **25 张不是去重引擎判出来的**。全代码库只有 `main_window._apply_cleaning_removals` 一条路径会写 `_removed_duplicates`，而它只能删「界面列表里被勾选的组」的 `paths[1:]`（引擎只报 13 组，最多删 13 张）。来源待查。

**⚠️ 另一个发现（本次未改）**：`blur_threshold = 100.0`（Laplacian 方差）对这批低对比灰度图完全不合理 —— 2306 张里有 **1622 张（70%）** 被判为「模糊」。该列表只提示、不自动删除，但基本失去参考价值。

**改动文件**：`src/core/data_cleaner.py`。

---

### 功能：加载数据集时不再强制弹出「标签设置」；未分组图片恢复常规显示

**背景**：用户反馈两点 —— ① 加载数据集时**自动弹出标签设置对话框**，但很多时候用户只是想**先看看数据集**，并不想马上配置；② 右侧文件列表里**未参与聚类的图片**被显示成 `[未分组] xxx.jpg` 且是灰色，看起来像「异常状态」，其实只是没做聚类而已。

**改动**（`src/ui/main_window.py`）：

**A. 标签设置改为按需触发**（`load_dataset_from_path`）
- 删除加载流程里自动 `LabelSetupDialog(...).exec_()` 的整段（原第 1129-1136 行）。
- 类别仍然正常加载：有 `data.yaml` 就从里面读，没有就默认 `["person"]`；`_load_ai_settings()` 照常执行（AI 辅助开关 + 训练轮次）。
- 用户进入界面后随时可以点工具栏的**「标签设置」**按钮（`open_label_setup_dialog`）去配置类别和 AI 辅助 —— 该路径**未改动**。

**B. 未分组图片恢复常规显示**（`populate_file_list`）
- 原：`item.setText(self.tr_text("FMT_FILE_UNGROUPED").format(fname))` + `item.setForeground(QColor("#999"))` → 显示成 `[未分组] xxx.jpg` 且灰色。
- 新：只 `item.setText(fname)`，**不加前缀、不设灰色**，用列表默认颜色。
- 聚类相关的两种状态（`FMT_FILE_REP` 代表图、`FMT_FILE_AI` 待标注）**保持不变** —— 它们代表真实的聚类语义。
- 效果：**没做过聚类时，文件列表和引入聚类之前完全一样**；只有真正做了聚类，才会看到带簇号的前缀和颜色。

**不改动**：聚类算法、`FMT_FILE_REP` / `FMT_FILE_AI` 的显示、`open_label_setup_dialog`、数据清洗（除重 / 聚类）按钮。

**改动文件**：`src/ui/main_window.py`。

## 2026-09-18

### 修复：切换数据集后图片卡死、A/D 无法翻页（导航失效）

**现象**：设置完标签进入标注页面后，图片卡在第一张不能翻到下一张；键盘 A/D 无效，点右侧文件列表也不换图。

**根因**（`src/ui/main_window.py`）：`self.canvas.setFocus()` 被放在了 `_effective_prompt()` 的 `return` **之后**（第 1480 行），是**永远执行不到的死代码**。

- 早先重构把 `change_image()` 里的内联 prompt 逻辑抽成 `_effective_prompt()` 方法，新方法定义被插在 `change_image()` 结尾那句 `self.canvas.setFocus()` **之前**，但没有把 setFocus 一起挪回 `change_image()` 末尾 → 键盘焦点再也不会落到画布上。
- 于是 `change_image()` 跑完后焦点停在别的控件（如右侧文件列表）。`QListWidget` 会把可打印字符吃掉做首字母搜索，A/D 根本传不到 `MainWindow.keyPressEvent` → 图片永远不翻页。

**修复**：把 `self.canvas.setFocus()` 移到 `change_image()` 方法体**末尾**（`if/else` 之后），恢复「每次换图都把键盘焦点交回画布」的原设计；同时删除 `_effective_prompt()` 里的死代码。

**验证**：AST 断言 —— `setFocus` 是 `change_image()` 的最后一条语句，且 `_effective_prompt()` 中不再含该调用。

---

### 修复：切换数据集后左侧栏仍显示上一个数据集的训练数据

**现象**：加载新数据集、设置完标签进入标注页后，左侧「训练监控」面板仍显示上一个数据集（Raccoon-clean）的训练数据：轮次、mAP50、Loss、Epoch、进度条、耗时、状态徽章、训练日志。

**根因**（`src/ui/main_window.py`）：**代码库里根本不存在数据集状态重置函数**。`load_dataset_from_path()` 只重置了 `dataset_root / image_list / _full_image_list / _round_* / classes / _ai_assist_enabled / train_round / train_loop_count / verified_count_since_train` 和 `lbl_info`、`lbl_total_stats` 两个标签；下面这些**训练监控状态从头到尾没人清**：

`total_images_in_model`、`current_batch_count`、`last_model_name`、`_active_train_model`、`last_epochs`、`last_map`、`train_start_time`、`is_training`、`_badge_state`/`lbl_badge`、`lbl_map`、`lbl_loss`、`lbl_epoch`、`train_progress`、`lbl_timer`、`log_text`。

其中 `log_text.clear()` 只在训练**开始**时调用（`check_training_status` 的 `start` 分支），所以旧日志会一直留着。

另有一处**数据源不一致**：`lbl_total_stats` 在 `load_dataset_from_path`（第 1114 行）和 `_refresh_total_stats`（第 980 行）用 `total_ann`（当前数据集已标注张数），但 `retranslate_ui`（第 173-175 行）用的是 `total_images_in_model`（上一次训练的张数，只在训练成功时才赋值）。所以切数据集后只要**切换中英文**，或删一张图触发 `_refresh_total_stats`，就会把上一个数据集的训练张数又显示出来。

**修复**：
- 新增 `_reset_monitor_state()`：清零全部训练监控属性，并把面板恢复到初始外观（进度条归零、计时器「就绪」、mAP/Loss 归 0.000、Epoch 0/0、徽章「就绪」、训练按钮复位、清空训练日志）。
- `load_dataset_from_path()` 在确认图片列表非空后立即调用它（清洗向导之前）。
- `retranslate_ui()` 改为调用 `_refresh_total_stats()`，与另外两处统一用当前数据集的已标注张数；`_refresh_total_stats()` 里 `train_round` 改用 `getattr(..., 0)` 兜底。

**改动文件**：`src/ui/main_window.py`（`change_image`、`_effective_prompt`、`_reset_monitor_state`（新增）、`load_dataset_from_path`、`retranslate_ui`、`_refresh_total_stats`）。

---
### 修复：画布空白、图片显示不出来（文件扩展名与实际格式不符）

**现象**：加载 `TXL-PBC\images-all` 后，右侧文件列表正常列出 1256 张图，但**画布一片空白**，点哪张都不显示。

**排查**：
- 磁盘数据完好：`images-all` 1256 张、`images\{train,val,test}` 878+252+126、`Raccoon-clean` 193 张，回收站里也没有相关图片。
- 用 Qt 直接解码这 1256 张 png：**只有 100 张能解码，1156 张 `QImage`/`QPixmap` 返回 null（0x0）**。
- 对比文件头找到根因：失败文件前 12 字节是 `ff d8 ff e0 00 10 4a 46 49 46`（**JPEG / JFIF**），成功文件是 `89 50 4e 47 0d 0a 1a 0a`（PNG）。
- 即：**这批文件是改了扩展名的 JPEG**（内容 JPEG、文件名 `.png`），占比 1156/1256 ≈ 92%。
- 原始数据（桌面 `lucy_files\...\TXL_PBC\images_all`，1440 张）**全部是 `.jpg` 且格式正确**，说明是在数据集整理阶段（1440 → 1256）被统一改名成 `.png` 时引入的。

**为什么现在才暴露**：Qt 的 `QImageReader` **优先按扩展名**选解码器（`decideFormatFromContent` 默认 False），`.png` 扩展名 + JPEG 内容 → PNG 解码器失败 → `QPixmap` 为 null → `change_image()` 里 `if pixmap.isNull(): return` 提前返回 → 画布保持空白，而文件列表照样显示文件名。而 OpenCV（`cv2.imread`）、PIL、浏览器都会**嗅探内容**，所以之前训练、除重、上传统计全都「看起来正常」。

**修复**：新增 `src/utils/image_io.py`，提供 `load_qpixmap(path)`：
- `QImageReader.setDecideFormatFromContent(True)` —— 让 Qt **按文件内容**判断格式，忽略错误的扩展名；
- `setAutoTransform(True)` —— 顺带支持 EXIF 旋转；
- 解码失败返回 null QPixmap（调用方原本就有 `isNull()` 保护）。

`main_window.py` 两处 `QPixmap(path)` 改为 `load_qpixmap(path)`：`change_image()`（主显示路径）、`check_training_status()` 的 success 分支（训练后重新预标注）。

**实测**（`images-all` 全量 1256 张）：修复前 `QPixmap(path)` 可加载 **100 / 1256**；修复后 `load_qpixmap(path)` 可加载 **1256 / 1256**。

**改动文件**：`src/utils/image_io.py`（新增）、`src/ui/main_window.py`（导入 + 2 处调用）。

**⚠️ 遗留（数据集侧，未改）**：`TXL-PBC\images-all`、`TXL-PBC\images\{train,val,test}` 以及**已发布到 HuggingFace / ModelScope 的 `KeenForgeAI/TXL-PBC-corrected`** 都含这批扩展名不符的文件（train 878 张中 808 张、val 252 中 236 张、test 126 中 112 张）。软件侧已容错，但数据集本身建议修正（改名为 `.jpg` 或重新编码）后重新发布。

---
### 功能：数据清洗改为「按需触发」，并拆分为「除重」「聚类」两个独立按钮

**背景/需求**：加载新数据集时软件会**强制弹出**清洗向导（除重 → 聚类 → 预览），用户无法先观察数据集原有的标签标注情况。用户要求：
1. 加载数据集时**不再自动弹**清洗向导；
2. 工具栏的「数据清洗」拆成**两个独立按钮**：「除重」和「聚类」，由用户自己选择，不再强迫用户做清洗。

**A. 去掉加载时的强制清洗**（`src/ui/main_window.py` `load_dataset_from_path`）
- 删除原先「无 `cluster_map.json` 就自动 `CleaningDialog(...).exec_()`」的整段（原第 1109-1124 行）。
- 现在加载数据集只做：解析图片 → 重置训练监控 → 标签设置对话框 → 填充列表 → 加载模型。清洗完全由用户从工具栏主动触发。
- 副作用：不再有「加载即生成 `cluster_map.json`」的行为，该文件现在只在用户主动点「聚类」后生成。

**B. 工具栏拆分**（`src/ui/main_window.py`）
- `tb_clean`（数据清洗）→ 拆成 `tb_dedup`（除重）与 `tb_cluster`（聚类）两个 `QToolButton`，分别接 `open_dedup_dialog` / `open_cluster_dialog`，各带 tooltip。
- `open_cleaning_dialog()` 拆成两个方法：
  - `open_dedup_dialog()`：`CleaningDialog(..., mode="dedup")`，只跑除重，完成后调 `_apply_cleaning_removals()`。
  - `open_cluster_dialog()`：`CleaningDialog(..., mode="cluster")`，只跑聚类；有训练模型时只聚类**未标注**的图片（沿用原跟进轮逻辑），完成后调 `_apply_diverse_order()` + `_save_cluster_map_from_dialog()`。
- `retranslate_ui()` 同步刷新两个新按钮的文案与 tooltip。

**C. `CleaningDialog` 增加真正的模式开关**（`src/ui/cleaning_dialog.py`）
- 新增 `mode` 参数（`"full"` / `"dedup"` / `"cluster"`，默认 `"full"` 保持原三页向导）。
  - 注：原有的 `wizard_mode` 参数**是死代码**（第 148 行赋值后从未被读取），这次不再依赖它。
- `"cluster"`：内部把 `start_step` 抬到 1，即跳过除重页，走「聚类设置 → 预览」两页（等价于原跟进轮流程）。
- `"dedup"`：只显示第 1 页（除重），导航栏隐藏「上一步/下一步」，直接显示「完成除重」；`_on_ok()` 增加校验——除重扫描后必须点过「确认移除选中」或「跳过，保留全部」才能完成，否则提示 `CLN_MSG_FIRST_REMOVE`。
- 窗口标题按模式区分：`CLN_TITLE_DEDUP` / `CLN_TITLE_CLUSTER` / `CLN_TITLE`。

**D. 新增翻译键**（`src/config.py`，中英各一份）
`BTN_DEDUP`（除重/Deduplicate）、`BTN_CLUSTER`（聚类/Cluster）、`TIP_DEDUP`、`TIP_CLUSTER`、`CLN_TITLE_DEDUP`、`CLN_TITLE_CLUSTER`、`CLN_STEP1_ONLY`、`CLN_BTN_FINISH_DEDUP`。

**不改动**：
- 训练完成后的「下一轮清洗」弹窗（`_open_next_round_cleaning`）仍自动弹——那是训练后的跟进流程，不是加载流程。
- 「重新聚类并开始标注」按钮、除重/聚类的实际算法与结果处理逻辑（`_apply_cleaning_removals` / `_apply_diverse_order` / `_save_cluster_map_from_dialog`）。

**验证**：AST 断言 —— `load_dataset_from_path` 不再构造 `CleaningDialog`；`tb_clean` / `open_cleaning_dialog` 已无残留引用；`tb_dedup` / `tb_cluster` 在 `setup_ui` 创建并在 `retranslate_ui` 刷新；`CleaningDialog.__init__` 含 `mode` 参数且 `_update_nav_buttons` 有 dedup 分支；8 个新翻译键在中英字典各出现 1 次。`py_compile` 三个文件退出码 0。

**改动文件**：`src/ui/main_window.py`、`src/ui/cleaning_dialog.py`、`src/config.py`。

**备注**：`CleaningDialog` 的 `wizard_mode` 参数现已完全无用（保留仅为向后兼容），后续可清理。

---

## 2026-09-17

### 修复：数据清洗第 1 步「除重」无重复图片时卡死

**现象**：在 Raccoon-clean 数据集测试除重，未发现重复图片，点「下一步」弹出「请先确认移除或跳过」，但「移除」「跳过」两个按钮都无法点击，流程卡死。

**根因**（`src/ui/cleaning_dialog.py`）：
- `_on_duplicates_ready()` 在 `total_remove == 0`（无重复）时，把「确认移除」「跳过」两个按钮都 `setEnabled(False)`（原第 617-618 行）。
- 但 `self._removed_paths` 仍为 `None`（`_start_scan()` 第 521 行设为 None，之后无人再改）。
- 点「下一步」→ `_go_next()` 判断 `self._removed_paths is None` → 弹「CLN_MSG_FIRST_REMOVE」拦截。
- 结果：两个按钮被禁用（无法点），又被「请先确认移除或跳过」拦截（必须点），死锁。

**修复**：无重复图片时自动 `self._removed_paths = []`（等价于「跳过，保留全部」），使「下一步」可直接通过。与文案 `CLN_NO_DUP`「未发现重复图片，可直接进入下一步」一致。

**改动位置**：`src/ui/cleaning_dialog.py` 第 620-623 行（新增 4 行）：

```python
        self.btn_apply_remove.setEnabled(total_remove > 0)
        self.btn_skip_remove.setEnabled(total_remove > 0)

        # 无重复图片时自动跳过（_removed_paths 置空），否则「下一步」会被
        # 「请先确认移除或跳过」拦截，而两个按钮又都被禁用，导致卡死。
        if total_remove == 0:
            self._removed_paths = []
```

**影响范围**：仅 `cleaning_dialog.py` 一个文件；不影响有重复图片的正常路径（有重复时按钮正常启用，用户仍可手动选择移除或跳过）。

---

### 修复：横图超出画布左右两侧（按高度适配忽略宽度）

**现象**：标注 raccoon181 等横图（宽>高）时，图片超出左右画布边界，标注框看起来会移出画布两侧；上一轮测试同问题导致多张图框坐标误判。

**根因**（`src/ui/canvas.py`）：
- `_fit_image()` 默认 `ZOOM_FIT_HEIGHT` 模式用 `scale_factor = (vh - 8) / self.image_h` 只按高度适配，完全忽略宽度。
- 横图缩放后宽度超过视口宽度 → 左右溢出。
- `setHorizontalScrollBarPolicy(AlwaysOff)` 禁用了水平滚动条，溢出部分无法滚动查看；代码注释误认为「FIT_HEIGHT 使整图在视口内」，只对高度成立、对宽度不成立。
- Raccoon 数据集绝大多数为横图（如 259x194、275x183、640x426 等）。

**修复**：在 `ZOOM_FIT_HEIGHT` 分支加宽度上限（contain 适配）。竖图行为不变，横图改为按宽度适配，整图始终完整可见。

**改动位置**：`src/ui/canvas.py` `_fit_image()` 第 386-389 行（新增 4 行）：

```python
                scale_factor = (vh - 8) / self.image_h
                # 横图(宽>高)只按高度适配会溢出左右画布边界，且水平滚动条已禁用；
                # 这里加一个宽度上限，确保整图始终完整可见（contain 适配）
                if scale_factor * self.image_w > vw:
                    scale_factor = (vw - 8) / self.image_w
```

**影响范围**：仅 `canvas.py` 一个文件、`_fit_image()` 一个方法；画框/移动/缩放/保存的坐标钳制逻辑未改动。

---

### 修复：YOLO-World 推理失败（缺 clip 依赖）

**现象**：AI 协助开时自动加载 `yolov8s-worldv2.pt`，但推理报错 `ModuleNotFoundError: No module named 'clip'`。

**根因**：YOLO-World 零样本文本编码需要 OpenAI CLIP 包（`import clip`）。该包未安装；ultralytics 尝试用 `uv pip install git+https://github.com/ultralytics/CLIP.git` 自动安装，但 ①`uv` 因 site-packages 里畸形的 `pyodbc-4.0.0_unsupported.dist-info` 报错 ②git+github 被墙，双重失败。

**修复**：`pip install openai-clip`（走清华镜像，直连快）；CLIP 权重 ViT-B/32（338MB）已自动下载缓存到 `D:\AutoLabel-Pro\weights\clip`。已实测 `set_classes(['raccoon'])` 推理成功。

**改动文件**：`requirements.txt` 新增 `openai-clip>=1.0.1`（依赖声明）。

---

### 功能：AI 协助自动用 World 模型 + 隐藏左侧模型/提示词 + 工具栏模型菜单

**背景**：用户首轮用零样本 World 模型标注 → 训练 → 迭代模型；提示词应与类别自动同步（设了 raccoon 应变成 raccoon）。

**改动**：
- **隐藏左侧栏** `model_group`（模型选择器 + YOLO-World 提示词框）。
- `_effective_prompt()` 改为**永远返回数据集类别**（提示词自动 = 类别）。
- `check_and_load_local_model()` 新增中间分支：无训练模型 + AI 协助开 → 自动加载 `yolov8s-worldv2`（零样本）；否则才用基础模型。
- `restore_state()` 修复：AI 协助开时不再用旧基础模型覆盖自动选的 World 模型。
- 工具栏「加载模型」按钮 → **下拉菜单**：①加载自己的模型（列出已训练轮次模型 + 从文件选择）②基础模型子菜单 ③World 模型子菜单。
- 模型列表提取为模块常量 `BASE_MODELS`/`WORLD_MODELS`/`CLASSIC_MODELS`/`ALL_MODELS`。
- 新增辅助方法：`_apply_loaded_model` / `_resolve_world_model_path` / `_list_trained_models` / `_select_and_load_model` / `_build_model_menu`。

**改动文件**：`src/ui/main_window.py`、`src/config.py`（新增 BTN_MODEL_OWN/OWN_FILE/BASE/WORLD、MSG_NO_TRAINED_MODEL 翻译键）。

**⚠️ 注意**：World 模型文件 `yolov8s-worldv2.pt` 本地还没有，首次加载会触发 ultralytics 自动下载（走 GitHub，可能慢或被墙）。

**不改动**：训练逻辑、训练对话框、聚类、画布、数据清洗、保存逻辑。

---

### 修复：滚轮缩小标注框越界（上轮修复的构造函数错误）+ 新增左侧栏 AI 辅助开关

**现象**：上轮修复后放大不再越界，但**缩小反而越界**、且框变成全屏大小后无法移动。

**根因**：上轮修复把 `QRectF(l, t, r, b)` 当成了「左上/右下」构造，但 Qt 的 4 参构造实际是 `(x, y, 宽, 高)`——右/下边界值被当成了宽/高，导致框被错误放大。

**修复**：改用两点构造 `QRectF(QPointF(左上), QPointF(右下))`，每条边独立钳制，放大/缩小都正确停在图片边界。

**新增功能**：左侧栏加「AI 预标注辅助」复选框（与标签设置对话框同步）：
- `main_window.py`：新增 `chk_ai_assist` + `_on_ai_assist_toggled`（开启时若需自动切 World 模型并刷新当前图）+ `_sync_ai_assist_checkbox`（对话框改完后同步）。
- `config.py`：新增 LBL_AI_ASSIST / TIP_AI_ASSIST 翻译键。

**改动文件**：`src/ui/canvas.py`、`src/ui/main_window.py`、`src/config.py`。

---

### 修复：滚轮放大标注框会超出画布

**现象**：选中标注框后滚轮放大，框会超出图片/画布边界。

**根因**（`src/ui/canvas.py` `_scale_box`）：边界钳制用 `moveLeft/moveRight` 等**平移**操作（各保持对边不动）。当框持续放大到**比图还宽**时，左右钳制互相冲突：框 [-25,125]（图宽100）→ `moveLeft(0)` 变 [0,150] → `moveRight(100)` 又变 [-50,100]，左侧越界且越滚越严重。

**修复**：改为直接对四条边取边界值构造矩形（`max(0,left)/min(image_w,right)` 等），每条边独立钳制，框宽超图宽时自动压回 [0, image_w]×[0, image_h]。

**改动文件**：`src/ui/canvas.py`（仅 `_scale_box` 钳制逻辑）。

---

### 修复：标注框线条粗细不一致

**现象**：不同图片（缩放不同）下标注框线条有的粗有的细。

**根因**（`src/ui/canvas.py`）：
- 框轮廓用 `max(1, int(round(2.0 / s)))` 场景单位笔宽，整数取整导致不同缩放下屏幕宽度 1.5~3px 波动；
- 十字辅助线同样量化；
- 画框时的黄色虚线框用固定 2 场景单位（缩放下 1~6px 波动）。

**修复**：统一改用 cosmetic pen（`setCosmetic(True)` + `setWidthF`），笔宽直接是设备像素，任何缩放下都精确 2px（未选中）/ 3px（选中）。涉及：框轮廓、十字辅助线、黄色绘制虚线框。

**改动文件**：`src/ui/canvas.py`（仅画笔构造）。

---

### 修复：World 模型零样本漏检（置信度阈值偏高）

**现象**：部分明显的 raccoon 识别不出来。

**排查**：对 20 张 Raccoon-clean 图实测不同 conf 阈值（每图应有 raccoon）：conf=0.15 漏 4 张、0.10 漏 3 张、0.05 漏 3 张、0.01 漏 2 张。提示词换用 a raccoon / raccoon animal 等无明显提升。

**根因**：World 零样本 conf 阈值 0.15 偏高；且零样本「raccoon」本身有 ~85-90% 的召回天花板（部分 raccoon 无论如何调低阈值也无法命中）。

**修复**：`inference_engine.py` 提取阈值常量 `CONF_ZERO_SHOT = 0.05`（原 0.15）、`CONF_STANDARD = 0.25`，World 分支改用 `CONF_ZERO_SHOT`。

**说明**：零样本只是首轮“起手”，召回上限有限；标注一批后训练得到的模型召回会显著提升。

---

### 修复：World 模型推理仅第一张成功 + 检测误报 person（两个 bug）

**现象**：重开数据集、开 AI 协助后，第一张图能正确识别 raccoon，但后续图推理失败；且偶尔提示「检测到相似目标 person」。

**根因 1（设备错配）**：`set_classes` 首次调用时模型在 CPU，CLIP 文本编码器被缓存到 CPU；`predict(device='cuda')` 把模型（含 CLIP 子模块）搬到 GPU，但 CLIP 的 `self.device` 属性仍是陈旧的 `cpu`；第二次 `set_classes` 时 token 按 cpu 移、CLIP 模型已在 cuda → `RuntimeError: cuda:0 and cpu`。

**根因 2（person 误报）**：World 模型 `set_classes(['raccoon'])` 后检测返回 `cls_id=0`；但 `on_inference_result` 对非自定义模型用 `COCO_CLASSES[0]`（= 'person'）映射，导致 raccoon 被显示成 person。

**修复**：
- `inference_engine.py`：`set_classes` 前先 `self.model.to(self.device)`，确保 CLIP 编码器建在正确设备；新增 `is_world_model` 属性。
- `main_window.py` `on_inference_result`：World 模型改用模型自身 `names` 映射（与自定义模型同逻辑），不再用 COCO_CLASSES。

**改动文件**：`src/core/inference_engine.py`、`src/ui/main_window.py`。已实测连续两次 `set_classes+predict` 均正常、cls 映射为 raccoon。
