# 机器人视觉感知场景信息理解实验

## 项目说明
**任务内容：**
利用 RGB 图像或短视频，完成常见物体的类别识别与二维定位，调用视觉语言模型分析物体间的空间、语义关系；视频增加目标跟踪与关系变化分析，输出结构化场景信息，并与人工标注对照评价。

**主要工具和技术：**
`YOLO + ByteTrack`：目标检测与跟踪
`OpenCV`：图像视频处理
`通义视觉模型 AP`I：物体关系理解
`Visual Genome`：图像关系数据
`Charades`：室内活动视频

注：该文件由我本人维护，AI 请在 AGENT.md 文件中纂写调试记录。


## 使用方法

调用 YOLOv11n 模型实现目标检测：
```bash
uv run yolo predict \
  model=models/yolo11n.pt \
  source=data/images/test.png \
  conf=0.25 \
  save=True \
  project=/home/long753/projects/robot/task_3/outputs \
  name=detect_025
```
其中，model 指定调用模型；source 指定检测对象；conf 为置信度；project 为检测结果存放位置

