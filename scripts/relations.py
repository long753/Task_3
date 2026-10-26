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

import yaml

STATUSES = {"yes", "no", "unknown"}


def load_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "relations.yaml",
        help="关系定义与提示词 YAML；默认使用项目 config/relations.yaml",
    )
    parser.add_argument("--detections", type=Path, required=True,
                        help="detect.py 生成的 *_objects.json")
    parser.add_argument("--pair", nargs=2, type=int, action="append", required=True,
                        metavar=("SUBJECT_ID", "OBJECT_ID"),
                        help="指定有方向的对象对；可以重复此参数")
    parser.add_argument("--relations", nargs="+",
                        help="要判断的关系名称；省略时使用配置中的全部关系")
    parser.add_argument("--output", type=Path, default=Path("outputs/relations"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=None, help="覆盖 .env 中的 QWEN_MODEL")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true",
                        help="只生成配置快照和提示词，不调用 API")
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def build_prompt(objects, queries, config):
    definitions = {q["predicate"]: config["relations"][q["predicate"]] for q in queries}
    indexed_queries = {
        f"q{i}": query for i, query in enumerate(queries, start=1)
    }
    template = {"relations": {
        key: {"status": "", "reason": ""} for key in indexed_queries
    }}
    return "\n".join([
        config["prompt"]["task"].strip(),
        config["prompt"]["guidelines"].strip(),
        "不得增加对象或更改 ID。",
        f"queries 共 {len(queries)} 项，按查询编号逐项填写 status 和 reason，不得省略。",
        "查询编号必须原样保留；无法判断也要填写 unknown，不需要返回对象 ID 或关系名称。",
        "根据 queries 中 subject_id → object_id 的方向判断，不能颠倒主体与客体。",
        "关系定义：" + json.dumps(definitions, ensure_ascii=False),
        "status 只能为 " + "、".join(sorted(STATUSES)) + "。",
        "只输出 JSON 对象，顶层只有 relations 映射，不要 Markdown 代码块。",
        "relations 的键是查询编号，每个值只包含 status 和一句中文 reason。",
        "对象列表：" + json.dumps(objects, ensure_ascii=False),
        "queries：" + json.dumps(indexed_queries, ensure_ascii=False),
        "请填写以下 JSON 模板：" + json.dumps(template, ensure_ascii=False),
    ])


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def validate_response(text, queries):
    answers = json.loads(text)["relations"]
    keys = [f"q{i}" for i in range(1, len(queries) + 1)]
    if set(answers) != set(keys):
        raise ValueError("模型返回的查询编号与请求不一致")
    relations = []
    for key, query in zip(keys, queries):
        answer = answers[key]
        if answer["status"] not in STATUSES:
            raise ValueError("status 必须为 yes / no / unknown")
        relations.append({
            **query, "status": answer["status"], "reason": answer["reason"],
        })
    return relations


def request_relations(client, model, prompt, image_content, queries,
                      max_tokens=2048, *, output=None):
    """发送一次视觉查询；仅独立关系脚本需要保存原始响应。"""
    start = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt}, *image_content,
        ]}],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
        temperature=0,
        max_tokens=max_tokens,
    )
    elapsed_ms = round((perf_counter() - start) * 1000, 2)
    if output is not None:
        write_json(output / "raw_response.json", response.model_dump(mode="json"))
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"响应未正常完成：{choice.finish_reason}")
    text = choice.message.content
    if output is not None:
        (output / "raw_content.txt").write_text(text, encoding="utf-8")
    return {
        "model_requested": model,
        "model_returned": response.model,
        "elapsed_ms": elapsed_ms,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "relations": validate_response(text, queries),
    }


def save_relations(output, data):
    write_json(output / "relations.json", data)
    with (output / "relations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "predicate", "object_id", "status", "reason"])
        writer.writeheader()
        writer.writerows(data["relations"])


def main():
    # 1. 读取参数、实验配置和检测结果。
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    relation_types = args.relations or list(config["relations"])
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(args.env_file.expanduser(), override=False)
    model = args.model or os.getenv("QWEN_MODEL")
    base_url = (os.getenv("QWEN_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    detections = args.detections.expanduser().resolve()
    scene = json.loads(detections.read_text(encoding="utf-8"))
    objects = scene["objects"]

    # 2. 将对象对和关系名称组合成查询。
    ids = [obj["object_id"] for obj in objects]
    queries = []
    for subject_id, object_id in args.pair:
        if subject_id == object_id or subject_id not in ids or object_id not in ids:
            raise ValueError(f"无效对象对 {subject_id}, {object_id}；可用 ID：{ids}")
        for predicate in relation_types:
            query = {"subject_id": subject_id, "predicate": predicate,
                     "object_id": object_id}
            if query not in queries:
                queries.append(query)

    # 检测 JSON 中的相对图片路径以该 JSON 所在目录为基准。
    images = []
    for field in ("original_image", "annotated_image"):
        image = Path(scene[field]).expanduser()
        if not image.is_absolute():
            image = detections.parent / image
        images.append(image.resolve())
    prompt = build_prompt(objects, queries, config)

    # 3. 保存本次配置与最终提示词，便于核对模型输入。
    run_name = detections.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.expanduser().resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    request = {
        "detections_json": str(detections),
        "config_path": str(config_path),
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
        print("已生成配置快照、prompt.txt 和 request.json，未调用 API。")
        return

    # 4. 把提示词和两张图片一起发送给视觉模型。
    api_key = os.environ["DASHSCOPE_API_KEY"]
    from openai import OpenAI
    image_content = [
        {"type": "image_url", "image_url": {"url": image_data_url(image)}}
        for image in images
    ]
    with OpenAI(api_key=api_key, base_url=base_url,
                timeout=args.timeout, max_retries=0) as client:
        result = request_relations(
            client, model, prompt, image_content, queries,
            max_tokens=args.max_tokens, output=output,
        )

    # 5. 将模型返回的关系保存为 JSON 和 CSV。
    save_relations(output, {
        "status": "ok",
        "detections_json": str(detections),
        "objects": objects,
        **result,
    })
    print(json.dumps(result["relations"], ensure_ascii=False, indent=2))
    print("已保存关系 JSON / CSV、提示词、原始响应和调用耗时。")


if __name__ == "__main__":
    main()