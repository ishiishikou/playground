#!/usr/bin/env python3
import base64
import io
import json
import os
import statistics
import time
import urllib.request

from PIL import Image, ImageDraw

BASE_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
OUT = os.environ.get("RESULT_JSON", "vlm_sop_cache_result.json")

SOP = """You are a visual SOP compliance checker for an industrial workstation.
Your task is deliberately narrow and deterministic. Judge only the visible condition in the current observation image.

Standard operating procedure:
1. Inspect only the current observation image; do not infer hidden or previous actions.
2. The target compliance condition is the presence of a large solid RED SQUARE near the center of the image.
3. A BLUE square is not compliant.
4. A red circle is not compliant.
5. Other shapes or colors are not compliant unless the red square itself is visibly present.
6. Ignore the image background.
7. Ignore small decorative marks and borders.
8. Do not use prior images to answer the current image.
9. Treat every image as an independent observation.
10. Do not infer intent, sequence, worker identity, or safety state beyond the specified visual condition.
11. If the red square is clearly present, output PASS.
12. If it is absent, output FAIL.
13. If a blue square is present instead, output FAIL.
14. The decision must depend on the current image pixels.
15. Do not reuse a previous image's decision.
16. The textual SOP is fixed across observations.
17. The observation image changes on every request.
18. Return exactly one label and no explanation.
19. Valid labels are PASS and FAIL only.
20. This benchmark measures whether the fixed textual SOP prefix can be reused while the changing image is still processed freshly.

Output contract: exactly PASS or FAIL."""

def image_b64(label: str, variant: int) -> str:
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    # Vary geometry slightly so every image is byte-distinct.
    off = (variant % 4) * 4
    box = (64 + off, 64, 192 + off // 2, 192)
    if label == "PASS":
        d.rectangle(box, fill=(230, 20 + variant, 20))
    else:
        d.rectangle(box, fill=(20, 60 + variant * 3, 230))
    # Unique tiny corner marker outside the target region.
    d.rectangle((5 + variant, 5, 9 + variant, 9), fill=(variant * 17 % 255, 120, 80))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def post(path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw else {}

def erase_slot():
    req = urllib.request.Request(BASE_URL + "/slots/0?action=erase", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def infer(label: str, variant: int, cache_prompt: bool):
    img = image_b64(label, variant)
    payload = {
        "model": "vlm-sop",
        "messages": [
            {"role": "system", "content": SOP},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Current independent SOP observation image:"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                    {"type": "text", "text": "Apply the fixed SOP to this image. Return the required label."},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 4,
        "cache_prompt": cache_prompt,
        "id_slot": 0,
        "grammar": 'root ::= "PASS" | "FAIL"',
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", payload)
    wall_ms = (time.perf_counter() - t0) * 1000
    text = resp["choices"][0]["message"].get("content") or ""
    got = text.strip().strip('"').strip()
    timings = resp.get("timings", {})
    return {
        "expected": label,
        "got": got,
        "correct": got == label,
        "variant": variant,
        "wall_ms": wall_ms,
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
    }

def run_mode(name: str, cache_prompt: bool):
    erase_slot()
    # Prime with a byte-distinct image. This is recorded separately and excluded
    # from steady-state comparison.
    prime = infer("PASS", 90 if cache_prompt else 80, cache_prompt)
    labels = ["PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL"]
    rows = [infer(label, i + (20 if cache_prompt else 0), cache_prompt) for i, label in enumerate(labels)]
    return {"name": name, "cache_prompt": cache_prompt, "prime": prime, "rows": rows}

def summarize(mode):
    rows = mode["rows"]
    walls = [r["wall_ms"] for r in rows]
    prompt_ms = [r["prompt_ms"] for r in rows if isinstance(r.get("prompt_ms"), (int, float))]
    cache_n = [r["cache_n"] for r in rows if isinstance(r.get("cache_n"), (int, float))]
    return {
        "correct": sum(r["correct"] for r in rows),
        "n": len(rows),
        "first_ms": walls[0],
        "total_ms": sum(walls),
        "mean_ms": statistics.mean(walls),
        "median_ms": statistics.median(walls),
        "decisions_per_s": len(rows) / (sum(walls) / 1000.0),
        "mean_prompt_ms": statistics.mean(prompt_ms) if prompt_ms else None,
        "mean_cache_n": statistics.mean(cache_n) if cache_n else None,
        "cache_n_each": cache_n,
    }

def main():
    v1 = run_mode("V1_no_prompt_cache", False)
    v2 = run_mode("V2_fixed_prefix_cache", True)
    result = {
        "benchmark": "vlm_sop_changing_images_prefix_cache",
        "fixed_sop_chars": len(SOP),
        "images_per_mode": 8,
        "all_measured_images_byte_distinct": True,
        "v1": v1,
        "v2": v2,
        "summary": {"v1": summarize(v1), "v2": summarize(v2)},
    }
    a = result["summary"]["v1"]
    b = result["summary"]["v2"]
    result["comparison"] = {
        "v1_over_v2_speedup": a["total_ms"] / b["total_ms"] if b["total_ms"] else None,
        "latency_reduction_pct": (1 - b["total_ms"] / a["total_ms"]) * 100 if a["total_ms"] else None,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
# workflow trigger: benchmark definition unchanged
