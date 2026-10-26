# 机器人视觉感知场景信息理解实验
注：该文件由我本人维护，AI 请在 AGENT.md 文件中纂写调试记录！！！

## 项目说明
**任务内容：**

利用 RGB 图像或短视频，完成常见物体的类别识别与二维定位，调用视觉语言模型分析物体间的空间、语义关系；视频增加目标跟踪与关系变化分析，输出结构化场景信息，并与人工标注对照评价。

**主要工具和技术：**

`YOLO + ByteTrack`：目标检测与跟踪

`OpenCV`：图像视频处理

`通义视觉模型 API`：物体关系理解

`Visual Genome`：图像关系数据

`Charades`：室内活动视频


## 使用方法

### 图像部分
使用 `unset all_proxy ALL_PROXY` 修正代理问题。

1. 调用 YOLOv11n 模型实现目标检测：
```bash
uv run python scripts/detect.py \
  --source data/images/test.png \
  --model models/yolo11n.pt \
  --conf 0.25 \
  --output outputs/detect_025
```
其中，`model` 是调用模型；`source` 是检测对象；`conf` 是置信度；`project` 是检测结果存放位置

2. 调用 qwen3.8-flash 来判定对象关系：
```bash
uv run python scripts/relations.py \
  --detections outputs/detect_025/test_objects.json \
  --pair 1 2 \
  --relations left_of right_of on \
  --output outputs/relations_025
```
其中，detections 用于

3. 批量处理所有图像:
```bash
uv run python scripts/main_image.py
```

4. 处理特定图像：
```bash
uv run python scripts/main_image.py \
  -- images 1
```

### 视频部分
使用 `unset all_proxy ALL_PROXY` 修正代理问题。


