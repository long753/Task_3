#!/usr/bin/env python3
"""读取 detect.py 输出，调用通义视觉 API 判断指定对象对的关系。"""

import argparse
import base64
import csv
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

RELATIONS = {
    "left_of": "主体的框中心在客体框中心左侧，按画面坐标判断",
    "right_of": "主体的框中心在客体框中心右侧，按画面坐标判断",
    "on": "主体放在客体表面并由客体支撑，仅仅处于上方不算",
    "holding": "主体通过手部抓握或托持客体，仅靠近不算",
}
STATUSES = {"yes", "no", "unknown"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True,
                        help="detect.py 生成的 *_objects.json")
    parser.add_argument("--pair", nargs=2, type=int, action="append", required=True,
                        metavar=("SUBJECT_ID", "OBJECT_ID"),
                        help="指定有方向的对象对；可以重复此参数")
    parser.add_argument("--relations", nargs="+", choices=RELATIONS,
                        default=["left_of", "right_of", "on", "holding"])
    parser.add_argument("--output", type=Path, default=Path("outputs/relations"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=None, help="覆盖 .env 中的 QWEN_MODEL")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true",
                        help="只检查输入、生成提示词，不调用 API")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("--max-tokens 和 --timeout 必须为正数")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def load_scene(path, pairs, relation_types):
    scene = json.loads(path.read_text(encoding="utf-8"))
    objects = scene.get("objects")
    if not isinstance(objects, list):
        raise ValueError("检测 JSON 必须包含 objects 列表")
    if not all(isinstance(obj, dict) and type(obj.get("object_id")) is int
               for obj in objects):
        raise ValueError("每个对象必须包含整数 object_id")
    ids = [obj["object_id"] for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("检测结果中 object_id 重复")
    queries = []
    for subject_id, object_id in pairs:
        if subject_id == object_id or subject_id not in ids or object_id not in ids:
            raise ValueError(f"无效对象对 {subject_id}, {object_id}；可用 ID：{ids}")
        for predicate in relation_types:
            query = {"subject_id": subject_id, "predicate": predicate,
                     "object_id": object_id}
            if query not in queries:
                queries.append(query)
    images = []
    for field in ["original_image", "annotated_image"]:
        image = Path(scene[field]).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"找不到图片：{image}")
        images.append(image)
    return objects, queries, images


def build_prompt(objects, queries):
    definitions = {q["predicate"]: RELATIONS[q["predicate"]] for q in queries}
    return (
        "你正在进行机器人场景关系判断。第一张是原图，第二张是同一画面的编号图。\n"
        "编号图只用于定位对象，结合原图判断关系。对象名称可能有检测错误；"
        "如果无法对应到实际物体，请返回 unknown。不得增加对象或更改 ID。\n"
        "只回答指定 queries 中的查询，每个查询恰好返回一次。\n"
        "关系定义：" + json.dumps(definitions, ensure_ascii=False) + "\n"
        "status 只能为 yes、no、unknown。只有画面证据明确时才返回 yes 或 no；"
        "遮挡、无法确认身份或证据不足时返回 unknown。不能仅凭类别常识判断。\n"
        "reason 用一句简短中文描述可见证据，不推测未观察到的动作。\n"
        "只输出 JSON 对象，顶层只有 relations 列表，不要 Markdown 代码块。\n"
        "每一项包含 subject_id（整数）、predicate（字符串）、"
        "object_id（整数）、status（字符串）、reason（字符串）。\n"
        "对象列表：" + json.dumps(objects, ensure_ascii=False) + "\n"
        "queries：" + json.dumps(queries, ensure_ascii=False)
    )


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise ValueError(f"不支持的图片格式：{path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def validate_response(text, queries):
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        raise ValueError("响应必须包含 relations 列表")
    expected = {(q["subject_id"], q["predicate"], q["object_id"]) for q in queries}
    found = {}
    for item in data["relations"]:
        if not isinstance(item, dict):
            raise ValueError("relations 中每一项必须为对象")
        if type(item.get("subject_id")) is not int or type(item.get("object_id")) is not int:
            raise ValueError("响应中的对象 ID 必须为整数")
        if not isinstance(item.get("predicate"), str):
            raise ValueError("predicate 必须为字符串")
        key = (item["subject_id"], item["predicate"], item["object_id"])
        if key not in expected or key in found:
            raise ValueError(f"响应包含未请求或重复的关系：{key}")
        if not isinstance(item.get("status"), str) or item["status"] not in STATUSES:
            raise ValueError("status 必须为 yes / no / unknown")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("每项关系必须包含非空 reason")
        found[key] = {**dict(zip(("subject_id", "predicate", "object_id"), key)),
                      "status": item["status"], "reason": item["reason"]}
    if set(found) != expected:
        raise ValueError("响应缺少部分请求的关系")
    return [found[(q["subject_id"], q["predicate"], q["object_id"])] for q in queries]


def main():
    args = parse_args()
    # dry-run 仅依赖标准库，便于先核对输入和提示词。
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(args.env_file.expanduser(), override=False)
    model = args.model or os.getenv("QWEN_MODEL") or "qwen3-vl-plus"
    base_url = (os.getenv("QWEN_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    detections = args.detections.expanduser().resolve()
    objects, queries, images = load_scene(detections, args.pair, args.relations)
    prompt = build_prompt(objects, queries)

    # 每次运行使用独立子目录，保留不同参数和失败记录。
    run_name = detections.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.expanduser().resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    request = {
        "detections_json": str(detections),
        "original_image": str(images[0]),
        "annotated_image": str(images[1]),
        "objects": objects,
        "queries": queries,
        "model_requested": model,
        "base_url": base_url,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "dry_run": args.dry_run,
    }
    write_json(output / "request.json", request)
    print(f"请求 {len(queries)} 项关系；结果目录：{output}", flush=True)
    if args.dry_run:
        print("输入检查通过；已生成 prompt.txt 和 request.json，未调用 API。")
        return

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("请在项目 .env 中配置 DASHSCOPE_API_KEY")
    from openai import OpenAI
    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url",
                        "image_url": {"url": image_data_url(image)}})
    client = OpenAI(api_key=api_key, base_url=base_url,
                    timeout=args.timeout, max_retries=0)
    start = perf_counter()
    phase = "api_error"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            temperature=0,
            max_tokens=args.max_tokens,
        )
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        write_json(output / "raw_response.json", response.model_dump(mode="json"))
        phase = "validation_error"
        choice = response.choices[0]
        text = choice.message.content or ""
        (output / "raw_content.txt").write_text(text, encoding="utf-8")
        if choice.finish_reason != "stop":
            raise ValueError(f"响应未正常完成：{choice.finish_reason}")
        relations = validate_response(text, queries)
    except Exception as exc:
        write_json(output / "error.json", {
            "status": phase, "error_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        })
        raise SystemExit(f"{phase}：{exc}；详情见 {output / 'error.json'}") from None
    finally:
        client.close()

    write_json(output / "relations.json", {
        "status": "ok",
        "detections_json": str(detections),
        "objects": objects,
        "model_requested": model,
        "model_returned": response.model,
        "elapsed_ms": elapsed_ms,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "relations": relations,
    })
    with (output / "relations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "predicate", "object_id", "status", "reason"])
        writer.writeheader()
        writer.writerows(relations)
    print(json.dumps(relations, ensure_ascii=False, indent=2))
    print("已保存关系 JSON / CSV、提示词、原始响应和调用耗时。")


if __name__ == "__main__":
    main()

