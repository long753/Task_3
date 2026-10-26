#!/usr/bin/env python3
"""YOLO + ByteTrack：逐帧检测并跟踪视频或摄像头中的对象。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from detect import draw_object

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "data/videos/video01_book_place.mp4"),
                        help="视频路径，或摄像头编号，例如 0")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/video_main")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="例如 cpu 或 0")
    parser.add_argument("--show", action="store_true", help="显示实时画面，按 q 退出")
    parser.add_argument("--max-frames", type=int, help="只处理前 N 帧，省略则处理到结束")
    parser.add_argument("--fps", type=float, help="输出视频帧率；默认使用输入帧率")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames 必须大于 0")
    return args


def main():
    # 1. 一次加载模型，打开单个视频源；每次运行都有独立的跟踪状态。
    args = parse_args()
    model = YOLO(str(args.model.expanduser()))
    is_camera = args.source.isdecimal()
    source = int(args.source) if is_camera else str(Path(args.source).expanduser().resolve())
    name = f"camera_{source}" if is_camera else Path(source).stem
    output = args.output.expanduser().resolve() / name
    cap = cv2.VideoCapture(source)
    writer = None
    frame_count = 0
    track_history = {}
    started = perf_counter()

    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = args.fps if args.fps is not None else source_fps
        if fps <= 0:
            raise ValueError("无法获得有效帧率，请用 --fps 指定输出帧率")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "tracks.jsonl").open("w", encoding="utf-8") as tracks:
            while args.max_frames is None or frame_count < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = (perf_counter() - started if is_camera
                             else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("无法创建输出视频")

                # 2. persist=True 保留前一帧的轨迹，由 ByteTrack 分配 ID。
                result = model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                    device=args.device, save=False, verbose=False,
                )[0]
                frame_count += 1
                objects = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for box in boxes.cpu():
                        track_id = int(box.id.item())
                        class_id = int(box.cls.item())
                        obj = {
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": float(box.conf.item()),
                            "bbox_xyxy": box.xyxy[0].tolist(),
                        }
                        objects.append(obj)

                        # 3. 复用图片绘图函数，但编号来自轨迹，不是本帧检测顺序。
                        draw_object(frame, {**obj, "object_id": track_id})
                        if track_id not in track_history:
                            track_history[track_id] = {
                                "track_id": track_id,
                                "class_name": obj["class_name"],
                                "first_frame": frame_count,
                                "last_frame": frame_count,
                                "frames_seen": 0,
                            }
                        history = track_history[track_id]
                        history["last_frame"] = frame_count
                        history["frames_seen"] += 1

                # 4. 每帧立即写入，未确认轨迹或没有对象时保存空列表。
                tracks.write(json.dumps({
                    "frame_index": frame_count,
                    "timestamp_seconds": timestamp,
                    "objects": objects,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                writer.write(frame)
                if frame_count % 25 == 0:
                    speed = frame_count / (perf_counter() - started)
                    print(f"已处理 {frame_count} 帧，当前 {len(objects)} 个对象，"
                          f"平均 {speed:.1f} FPS", flush=True)
                if args.show:
                    cv2.imshow("YOLO + ByteTrack", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError("未读取到任何视频帧")

    # 5. 汇总实际处理速度和各条轨迹；轨迹数量不等于真实物体数量。
    elapsed = perf_counter() - started
    summary = {
        "source": source,
        "source_kind": "camera" if is_camera else "video",
        "model": str(model.ckpt_path),
        "tracker": "bytetrack.yaml",
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "output_fps": fps,
        "output_duration_seconds": frame_count / fps,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_count / elapsed,
        "track_count": len(track_history),
        "tracks": list(track_history.values()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"完成：{frame_count} 帧，{len(track_history)} 条轨迹，"
          f"平均 {summary['processing_fps']:.1f} FPS")
    print(f"输出目录：{output}")


if __name__ == "__main__":
    main()