# 实现与调试记录

## 关系分析配置

- `config/relations.yaml` 保存关系名称、定义，以及 `prompt.task`、`prompt.guidelines`。修改关系或判断原则无需改动 Python。
- `scripts/relations.py` 保留固定输出协议和少量结果检查；主流程按“读配置与检测结果 → 生成查询 → 保存模型输入 → 调用模型 → 保存结果”顺序展开。
- 默认配置根据脚本位置定位，不依赖当前工作目录。可通过 `--config 路径` 切换配置；显式路径相对运行命令时的目录解析。
- 省略 `--relations` 时按 YAML 中的顺序查询全部关系；指定时填写配置中的关系名称，不再通过两阶段参数解析动态生成 `choices`。
- 配置约定包含 `relations` 映射，以及 `prompt.task`、`prompt.guidelines` 文本。直接使用 `yaml.safe_load()` 读取，不做逐项类型、空值或多余字段检查；检测 JSON 同样信任 `detect.py` 的输出格式。
- 文件、YAML、API 和 JSON 解析错误直接显示原始 traceback，不再包装异常或生成 `error.json`。只保留对象对有效性、模型正常结束、查询编号覆盖完整，以及 `yes/no/unknown` 状态检查。模型只返回查询编号对应的状态和理由，程序按请求顺序关联原有对象 ID 和关系名称。
- PyYAML 会将未加引号的 `on/off/yes/no` 解析为布尔值，这类关系名称必须加引号，例如 `"on"`。
- 每次运行在独立输出子目录保存 `config.yaml`（实际加载内容的快照）、最终 `prompt.txt` 和 `request.json`；请求记录包含配置来源路径。快照可直接作为后续 `--config` 输入。
- API 密钥继续放在 `.env`，不要放进 YAML。PyYAML 已作为直接依赖加入 `pyproject.toml` 和 `uv.lock`；dry-run 需要 PyYAML，但不需要 API 密钥或 OpenAI 客户端。

### 使用

```bash
uv run python scripts/relations.py \
  --config config/relations.yaml \
  --detections outputs/detect_025/test_objects.json \
  --pair 1 2 \
  --relations left_of right_of on \
  --output outputs/relations_025 \
  --dry-run
```

省略 `--config` 即使用项目默认配置；去掉 `--dry-run` 发起真实 API 调用。新增关系只需在 YAML 的 `relations` 中添加定义。

如果环境中的 `ALL_PROXY`、`all_proxy` 使用不支持的 `socks://`，可在命令前加 `env -u ALL_PROXY -u all_proxy`，保留已有的 HTTP/HTTPS 代理。

### 简化后的验证记录

- 在项目外目录运行 CLI：默认配置定位正确，无 API 密钥的 dry-run 成功；配置快照及最终提示词仍正常保存。
- 自定义配置：新增关系、替换任务说明、显式关系选择、默认关系顺序及配置快照复用均验证通过；重复指定的查询仍会去重。
- 检测 JSON 中的相对图片路径仍以该 JSON 所在目录解析，并已实际读取图片、生成 Base64 数据。
- 不存在的对象 ID 和自身对象对仍被拒绝。缺失配置文件、错误 YAML 显示原始 traceback；`--help` 不读取配置。
- 查询答案按编号关联，不受模型返回键顺序影响；缺失或额外编号和非法状态被拒绝，最终结果按请求顺序保存。
- 真实 API 调用成功，JSON/CSV 结果一致，记录保存在 `outputs/relations_025/test_objects_20261026_171600_260105/`。
- 使用 `--max-tokens 1` 发起真实调用，模型以 `length` 结束；程序在解析内容前直接抛出“响应未正常完成”异常，保留原始响应但不生成关系结果或 `error.json`。

## 图片批处理入口

`scripts/main_image.py` 默认按编号处理 `data/images/img01_*` 到 `img10_*`，不处理 `test.png`。YOLO 权重为 `models/yolo11n.pt`，视觉关系模型从 `.env` 的 `QWEN_MODEL` 读取，不再写死模型名称。

```bash
uv run python scripts/main_image.py
```

只分析指定图片，或者调整检测置信度、关系查询批次大小：

```bash
uv run python scripts/main_image.py --images 1 2 --conf 0.25 --batch-size 16
```

默认输入、权重、配置、`.env` 和输出位置均根据项目位置确定；显式指定的路径相对当前工作目录解析。API 密钥和地址继续取自 `.env`。关系模型的选择优先级为 `--model` > 当前进程的 `QWEN_MODEL` > `.env` 的 `QWEN_MODEL`（`load_dotenv` 不覆盖已有环境变量）。YOLO 权重通过 `--yolo-model` 修改，不由 `QWEN_MODEL` 控制；新的关系模型需要支持图片输入和当前 API 请求参数。

输出统一放在 `outputs/image_main/`，不创建 `output/`：

```text
outputs/image_main/
  config.yaml
  summary.csv
  img01_laptop_cup/
    original.png
    detected.png
    objects.json
    request.json
    relations.json
    relations.csv
```

- `objects.json` 包含对象 ID、类别、置信度、原图坐标系中的边界框，以及原图和编号检测图路径。
- 对 n 个对象生成 n×(n−1) 个有向对象对，分别查询配置中的全部关系，不按类别过滤。默认四种关系对应 4×n×(n−1) 项查询。
- 提示词列出查询编号 `q1、q2…` 与完整查询的对应关系，并提供同编号的 JSON 结果模板；模型只填写状态和理由，程序关联已知的对象 ID 和关系名称。这样不要求模型重复抄写固定信息，避免把整数 ID 改写为类别名。
- 默认每批 16 项查询、最多输出 4096 tokens，最后合并所有批次。`relations.json` 记录各批实际模型名称、耗时和用量，保留 yes/no/unknown 全部结果及理由；批处理不保存提示词或原始响应，也不记录相应路径。独立运行 `relations.py` 时仍保留原有调试文件。
- 不足两个对象时不请求关系模型，保存 `status: no_pairs`、空关系列表和仅有表头的关系 CSV。
- `summary.csv` 每完成一张图更新一次，记录本次图片的对象数、有向对象对数、查询数、批次数和三种状态数量。重跑同名图片会覆盖对应结果；仅选择部分图片运行时，汇总只包含本次选择，其他图片目录不删除。
- YOLO 和 OpenAI 客户端均只加载一次，图片及 API 批次顺序执行。异常直接显示 traceback，不自动重试，也不跳过失败图片。
- 复用 `detect.detect_image()`、`relations.request_relations()` 和 `relations.save_relations()`；两个原脚本的命令行及原有输出文件命名保持可用。

### 批处理验证

- 第 1 张图单独运行成功，检测到 2 个对象，两个方向共 8 项关系；已查看带编号检测图并核对关系结果。
- 两个原脚本的真实命令均执行成功，YOLO 输出可继续直接交给关系脚本使用。
- 将 API 地址设为不可达地址，在项目外目录运行真实 YOLO 推理：分别用置信度 1.0、0.9 得到 0、1 个对象，均未发送 API 请求，空 JSON/CSV 及汇总正确。
- 批次大小为 0 时在推理前明确报错；临时验证数据自动清理。
- 首次十图运行在第 3 张图发现模型只回答 12/16 项；添加完整模板后，同一批请求通过，但后续另一批出现把整数对象 ID 改为 `"chair"` 的问题。最终改为查询编号协议，已对发生 ID 错误的 16 项查询发起真实请求并验证全部返回；没有忽略错误、放宽完整性检查或增加自动重试。
- 新协议下原关系脚本真实调用通过；已验证返回键顺序颠倒时答案仍关联正确，缺失/额外编号和非法状态仍报错。对外的关系 JSON/CSV 字段不变。
- 此前查询编号协议下完整十图运行成功：共 43 个检测对象、960 项关系查询、63 个真实 API 批次；当时已核对查询覆盖、JSON/CSV、图片、原始响应和汇总计数。本次修正没有重新运行十图。
- `img09_cat_sofa.jpg` 仅检测到 1 个 cat，因此没有可查询对象对，输出空关系结果并标记 `no_pairs`，不属于执行失败。已查看第 1、3、10 张编号检测图。

### 标签与输出精简修正

- `draw_object()` 移除黑色描边和彩色文字的两次叠加，改为黑底白字、单次绘制。
- `main_image.py` 不再创建 `prompts/`、`responses/`；已清理历史生成的 18 个目录，并从 9 个已有关系文件中移除过时路径，保留原关系答案和检测信息。
- 本次仅运行 `--images 1`：从 `.env` 读取 `qwen3.8-flash`，检测到 2 个对象，8 项关系查询完成，JSON/CSV 一致；已查看新检测图，文字无重复。
- 已检查所有已有图片目录，不再含上述目录或路径引用；没有对其他图片重新推理或绘图。`summary.csv` 现在仅记录本次图片 1 的测试，其他图片保留此前生成的检测图。

## 视频检测与 ByteTrack 跟踪

`scripts/main_video.py` 只做 YOLO 检测和 ByteTrack 跨帧关联，不调用 Qwen、不读取 `.env`，也不生成提示词或响应目录。

```bash
# 默认完整处理 video01；不会自动处理其他视频
uv run python scripts/main_video.py

# 只处理第一个视频的前 75 帧
uv run python scripts/main_video.py \
  --source data/videos/video01_book_place.mp4 \
  --max-frames 75

# 摄像头入口；需要实际摄像头和桌面环境
uv run python scripts/main_video.py --source 0 --show
```

- 默认使用 `models/yolo11n.pt`，可通过 `--model` 更换检测权重；这里的 `--model` 不是 Qwen 模型名称。
- 模型只加载一次，OpenCV 逐帧读取，调用 `model.track(frame, tracker="bytetrack.yaml", persist=True)`，读取 `result.boxes.id`。`persist=True` 保留相邻帧的轨迹，编号不由 `enumerate()` 产生。
- 每次运行只打开一个视频源并新建模型/跟踪器；ID 只在当前视频内有意义，可能跳号，不保证离开画面再出现仍沿用原 ID。
- 默认检测置信度为 0.1，保留低分检测供 ByteTrack 关联；使用 Ultralytics 自带 `bytetrack.yaml`，没有另写跟踪算法或自动调参。
- 复用 `detect.draw_object()` 绘制黑底白字标签，绘图时把轨迹 ID 传给 `object_id`，保存的数据字段仍为 `track_id`；不叠加 `result.plot()`。
- 只保存当前帧已确认且匹配到的轨迹；没有跟踪对象时仍写一帧记录，`objects` 为空，不编造 ID。
- 默认不打开窗口，适合无桌面环境；`--show` 打开预览，按 q 正常结束并保存汇总。摄像头预览速度受采集、推理和显示共同影响，不承诺所有设备都实时。
- 默认用输入帧率保存视频；无法读取帧率时需显式提供 `--fps`。该参数只控制输出播放帧率，不提高推理速度。

输出位于 `outputs/video_main/<视频名>/`，摄像头使用 `camera_<编号>`：

```text
outputs/video_main/video01_book_place/
  tracked.mp4
  tracks.jsonl
  summary.json
```

- `tracked.mp4` 是带跟踪 ID、类别及置信度的画面，不包含原音频；重跑同一来源会覆盖这三个文件。
- `tracks.jsonl` 每行对应一帧，含从 1 开始的 `frame_index`、`timestamp_seconds` 和对象列表。文件时间戳来自输入视频的解码时间；摄像头使用本次处理开始后的采集时刻。对象包含 `track_id`、类别、置信度和原图坐标下的 `bbox_xyxy`。
- `summary.json` 记录帧数、画面尺寸、输入/输出 FPS、输出时长、处理耗时、平均处理 FPS、轨迹数和各轨迹的首末帧、出现帧数。轨迹摘要的 `class_name` 是首次出现时的类别，逐帧类别以 JSONL 为准。
- 处理 FPS 包含读取帧、检测跟踪、绘制、视频和 JSONL 写入，不包含模型加载；轨迹数量不等于真实物体数量，类别正确性也不由 ByteTrack 保证。

### 视频验证记录

- 本次仅使用 `video01_book_place.mp4` 的开头，主验证处理前 75 帧；没有运行 video02、video03 或完整视频。
- 输出视频可完整解码 75 帧，720×1366、25 FPS、3 秒；JSONL 恰有 75 行，时间戳为 0 至 2.96 秒、间隔 0.04 秒，视频和汇总计数一致。
- 观察到 7 条轨迹，ID 1、3 在全部 75 帧持续出现；已检查每帧无重复 ID，轨迹首末帧与出现次数一致，并查看标注视频预览。
- 本次平均处理速度约 32.08 FPS，高于源视频的 25 FPS，仅代表当前环境和这段短片。
- 当前 YOLO 权重存在误分类和类别跳变：同一跟踪区域出现 book/knife/laptop、dining table/cake/pizza 等类别变化。未用类别平滑掩盖检测结果，也不能据此断言 ID 对应的类别一定正确。
- 另用同一视频前 5 帧、置信度 1.0 验证空轨迹分支：逐帧对象列表为空、无虚构 ID；显式输出 20 FPS 时，输入时间戳仍保持 25 FPS 的间隔。临时输出已清理。
- `--max-frames 0` 在打开视频前报错。摄像头和桌面窗口预览入口已实现，但本次没有进行硬件或窗口测试。
