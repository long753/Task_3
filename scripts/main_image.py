#!/usr/bin/env python3
"""依次检测 img01—img10，再调用 Qwen 判断全部有向对象对的关系。"""

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from ultralytics import YOLO
import yaml

from detect import detect_image
from relations import (
    build_prompt, image_data_url, load_config, request_relations,
    save_relations, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=int, choices=range(1, 11),
                        default=list(range(1, 11)), help="图片编号，默认 1—10")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/images")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/image_main")
    parser.add_argument("--yolo-model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--model", help="覆盖 .env 中的 QWEN_MODEL")
    parser.add_argument("--config", type=Path, default=ROOT / "config/relations.yaml")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="例如 cpu 或 0")
    parser.add_argument("--batch-size", type=int, default=16, help="每次请求的关系查询数")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    return args


def main():
    # 1. 一次加载配置、YOLO 和 API 客户端。
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    model = args.model or os.environ["QWEN_MODEL"]
    config = load_config(args.config.expanduser())
    detector = YOLO(str(args.yolo_model.expanduser()))
    input_dir = args.input_dir.expanduser().resolve()
    images = [next(input_dir.glob(f"img{number:02d}_*")) for number in args.images]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    summary = []
    with OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=args.timeout, max_retries=0,
    ) as client:
        for index, source in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] {source.name}", flush=True)

            # 2. 检测物体，保存带编号的检测图和位置 JSON。
            image_output = output / source.stem
            scene = detect_image(
                detector, source, image_output, conf=args.conf, device=args.device,
                filenames=("original.png", "detected.png", "objects.json"),
            )
            objects = scene["objects"]

            # 3. 每个不同对象对都查询两个方向，再展开配置中的全部关系。
            queries = []
            for subject in objects:
                for obj in objects:
                    if subject["object_id"] == obj["object_id"]:
                        continue
                    for predicate in config["relations"]:
                        queries.append({
                            "subject_id": subject["object_id"],
                            "predicate": predicate,
                            "object_id": obj["object_id"],
                        })
            write_json(image_output / "request.json", {
                "source": str(source), "objects": objects, "queries": queries,
                "model": model, "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
            })

            # 4. 分批发送原图、编号图和查询，合并各批返回结果。
            all_relations = []
            batches = []
            if queries:
                image_content = [
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(scene[field]))}}
                    for field in ("original_image", "annotated_image")
                ]
                for start in range(0, len(queries), args.batch_size):
                    batch = queries[start:start + args.batch_size]
                    number = len(batches) + 1
                    print(f"  第 {number} 批：{len(batch)} 项关系", flush=True)
                    prompt = build_prompt(objects, batch, config)
                    result = request_relations(
                        client, model, prompt, image_content, batch,
                        max_tokens=args.max_tokens,
                    )
                    all_relations.extend(result.pop("relations"))
                    batches.append({
                        "batch": number, "query_count": len(batch),
                        **result,
                    })

            # 5. 不足两个物体时查询为空，直接保存空关系结果，不请求模型。
            save_relations(image_output, {
                "status": "ok" if queries else "no_pairs",
                "source": str(source), "objects": objects,
                "model_requested": model, "batches": batches,
                "relations": all_relations,
            })
            summary.append({
                "image": source.name, "object_count": len(objects),
                "pair_count": len(objects) * (len(objects) - 1),
                "query_count": len(queries), "batch_count": len(batches),
                "yes": sum(r["status"] == "yes" for r in all_relations),
                "no": sum(r["status"] == "no" for r in all_relations),
                "unknown": sum(r["status"] == "unknown" for r in all_relations),
                "status": "ok" if queries else "no_pairs",
            })
            with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            print(f"  {len(objects)} 个物体，{len(all_relations)} 项关系已保存", flush=True)
    print(f"全部完成：{output}")


if __name__ == "__main__":
    main()