#!/usr/bin/env python3
"""单张图片检测：保存原图、带对象 ID 的图片及结构化 JSON。"""

import argparse
import json
from pathlib import Path

import cv2
import ultralytics
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--model", default="yolo11n.pt", help="模型权重名称或路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--device", default=None, help="例如 0 或 cpu；默认自动选择")
    parser.add_argument("--output", type=Path, default=Path("outputs/detect"),
                        help="输出目录，相对于运行命令时的目录")
    args = parser.parse_args()
    return args


def draw_object(image, obj):
    """框和文字中的 ID 与 JSON 中的 object_id 保持一致。"""
    x1, y1, x2, y2 = [round(v) for v in obj["bbox_xyxy"]]
    palette = [(0, 180, 0), (255, 100, 0), (0, 140, 255), (180, 0, 180)]
    color = palette[(obj["object_id"] - 1) % len(palette)]
    label = (f'ID {obj["object_id"]}: {obj["class_name"]} '
             f'{obj["confidence"]:.2f}')
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape[:2]
    scale = min(0.6, max(width, height) / 1000)
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 1)
    label_x = max(0, min(x1, width - text_w - 8))
    label_y = min(height - baseline - 4, max(text_h + 4, y1 - 8))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # 黑底白字，标签只绘制一次。
    cv2.rectangle(image, (label_x - 2, label_y - text_h - 3),
                  (label_x + text_w + 4, label_y + baseline + 2),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(image, label, (label_x, label_y), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def detect_image(model, source, output, conf=0.25, iou=0.7, imgsz=640,
                 device=None, filenames=None):
    """检测并保存图片与对象信息；model 由调用方加载，可供多张图片复用。"""
    source = Path(source).expanduser().resolve()
    if model.task != "detect":
        raise ValueError("请使用目标检测权重，例如 yolo11n.pt")
    result = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,  # 由本脚本显式保存，输出路径不受 runs_dir 设置影响。
    )[0]

    objects = []
    if result.boxes is not None:
        # 将结果移到 CPU，转换为可写入 JSON 的 Python 数字。
        for object_id, box in enumerate(result.boxes.cpu(), start=1):
            class_id = int(box.cls.item())
            objects.append({
                "object_id": object_id,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })

    original = result.orig_img
    annotated = original.copy()
    for obj in objects:
        draw_object(annotated, obj)

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if filenames is None:
        filenames = (f"{source.stem}_original.png",
                     f"{source.stem}_annotated.png",
                     f"{source.stem}_objects.json")
    original_path, annotated_path, json_path = [output / name for name in filenames]

    for path, image in [(original_path, original), (annotated_path, annotated)]:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"图片保存失败：{path}")

    height, width = original.shape[:2]
    data = {
        "source": str(source),
        "original_image": str(original_path),
        "annotated_image": str(annotated_path),
        "image_width": width,
        "image_height": height,
        "bbox_format": "xyxy_pixels",
        "model": str(model.ckpt_path),
        "ultralytics_version": ultralytics.__version__,
        "confidence_threshold": conf,
        "nms_iou_threshold": iou,
        "imgsz": imgsz,
        "speed_ms": result.speed,
        "object_count": len(objects),
        "objects": objects,
    }
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return data


def main():
    args = parse_args()
    model = YOLO(args.model)
    data = detect_image(model, args.source, args.output,
                        conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        device=args.device)
    print(f"检测到 {data['object_count']} 个物体")
    print(f"原图：{data['original_image']}")
    print(f"编号图：{data['annotated_image']}")
    print(f"对象 JSON：{Path(args.output).resolve() / (Path(args.source).stem + '_objects.json')}")


if __name__ == "__main__":
    main()

