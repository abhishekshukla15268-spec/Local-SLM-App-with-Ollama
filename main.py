"""
Ollama LLM Workbench — Professional Backend
============================================
Endpoints:
  GET  /                  → Serve UI
  GET  /models            → List available models
  GET  /system-stats      → Live CPU, RAM, VRAM, process memory
  GET  /memory            → Raw Ollama /api/ps output
  GET  /report            → Load last saved study JSON

  POST /chat              → Single-model chat with full metrics + quality
  POST /compare           → Run prompt across ALL models in parallel
  POST /benchmark         → Single-model detailed benchmark
  POST /extract           → Pydantic-validated JSON extraction with retry
  POST /temperature-test  → Same prompt at N temperatures, one model
  POST /compare-temp      → model × temperature matrix (all combos)
  POST /study             → Full model comparison study with report
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import psutil
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Ollama LLM Workbench", version="2.0.0")

BASE_DIR = os.path.dirname(__file__)


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "templates", "chat.html"))


# ─── Ollama Client ────────────────────────────────────────────────────────────

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODELS: list[str] = ["tinyllama:latest", "llama3.2:1b", "gemma:2b"]

# ─── Benchmark Prompts ────────────────────────────────────────────────────────

TEST_PROMPTS: list[str] = [
    "Explain what a neural network is in simple terms.",
    "Write a Python function to reverse a string.",
    "What are the main differences between TCP and UDP?",
    "Summarize the concept of supply and demand in economics.",
    "Write a haiku about programming.",
    "Explain the difference between a stack and a queue.",
    "What is the time complexity of binary search?",
    "Describe how HTTPS works in 3 sentences.",
    "Write a SQL query to find duplicate rows in a table.",
    "What are the SOLID principles in software engineering?",
    "Explain the CAP theorem simply.",
    "What is the difference between concurrency and parallelism?",
    "Write a regular expression to validate an email address.",
    "What is a REST API?",
    "Explain how garbage collection works.",
    "What is the difference between a process and a thread?",
    "Describe the MVC design pattern.",
    "What is Docker and why is it useful?",
    "Explain the concept of recursion with an example.",
    "What is the difference between SQL and NoSQL databases?",
    "Write a short explanation of how DNS works.",
    "What are microservices and when should you use them?",
    "Explain what a hash table is and its average time complexity.",
    "What is CI/CD?",
    "Describe the observer design pattern.",
    "What is the difference between HTTP GET and POST?",
    "Explain what an API gateway does.",
    "What is eventual consistency?",
    "Write a Python list comprehension to filter even numbers.",
    "Explain the concept of idempotency in APIs.",
    "What is a load balancer and why is it important?",
    "Describe the publish-subscribe messaging pattern.",
    "What are environment variables and why are they used?",
    "Explain the difference between authentication and authorization.",
    "What is rate limiting and why is it needed?",
]

# ─── Request / Response Models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    prompt: str
    model_name: str = "llama3.2:1b"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7


class CompareRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7


class ExtractRequest(BaseModel):
    prompt: str
    model_name: str = "llama3.2:1b"
    temperature: float = 0.0


class TemperatureTestRequest(BaseModel):
    prompt: str
    model_name: str = "llama3.2:1b"
    temperatures: list[float] = [0.0, 0.3, 0.7, 1.0]


class CompareTempRequest(BaseModel):
    """Cross-product: run every (model, temperature) combination."""
    prompt: str
    models: list[str] = MODELS
    temperatures: list[float] = [0.0, 0.3, 0.7, 1.0]
    system_prompt: str = "You are a helpful assistant."


class StudyRequest(BaseModel):
    num_prompts: int = 10


class ExtractedData(BaseModel):
    summary: str
    key_points: list[str]
    sentiment: str
    confidence: float


# ─── Quality Metrics ──────────────────────────────────────────────────────────


def quality_metrics(text: str) -> dict:
    """Compute response quality heuristics."""
    if not text or not text.strip():
        return {
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "unique_word_ratio": 0.0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0,
        }

    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    unique_words = set(words)

    word_count = len(words)
    unique_count = len(unique_words)
    char_count = len(text)
    sentence_count = max(len(sentences), 1)
    avg_word_len = round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    lexical_diversity = round(unique_count / word_count, 3) if word_count else 0.0

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "unique_word_ratio": lexical_diversity,
        "avg_word_length": avg_word_len,
        "lexical_diversity": lexical_diversity,
    }


# ─── Core Engine ──────────────────────────────────────────────────────────────


def query_model_with_metrics(
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
) -> dict:
    """Stream a response and collect detailed benchmarking + quality metrics."""
    start = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []
    token_count = 0

    try:
        stream = ollama_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(content)
                token_count += 1

        end = time.perf_counter()
        total_latency = round(end - start, 3)
        ttft = round(first_token_time - start, 3) if first_token_time else total_latency
        gen_time = end - (first_token_time or end)
        tps = round(token_count / gen_time, 2) if gen_time > 0 else 0.0

        response_text = "".join(chunks)
        q = quality_metrics(response_text)

        return {
            "model": model_name,
            "response": response_text,
            "metrics": {
                "total_latency_s": total_latency,
                "time_to_first_token_s": ttft,
                "tokens_generated": token_count,
                "tokens_per_second": tps,
                "generation_time_s": round(gen_time, 3),
                "temperature": temperature,
            },
            "quality": q,
        }

    except Exception as e:
        return {
            "model": model_name,
            "error": str(e),
            "metrics": {
                "total_latency_s": round(time.perf_counter() - start, 3),
                "temperature": temperature,
            },
            "quality": {},
        }


def extract_json_with_retry(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> dict:
    """Extract structured JSON, validate with Pydantic, retry once on failure."""
    schema_prompt = (
        "You are a data extraction assistant. Respond ONLY with valid JSON "
        "matching this schema:\n"
        '{"summary":"string","key_points":["string"],'
        '"sentiment":"positive|negative|neutral|mixed",'
        '"confidence":0.0-1.0}\n'
        "No markdown fences, no extra text. Just the raw JSON object."
    )

    attempts: list[dict] = []
    for attempt_num in range(max_retries + 1):
        actual_prompt = prompt
        if attempt_num > 0:
            actual_prompt = (
                "Your previous response was not valid JSON. "
                "Respond ONLY with a raw JSON object, nothing else.\n\n"
                f"Original request: {prompt}"
            )

        result = query_model_with_metrics(model_name, actual_prompt, schema_prompt, temperature)

        if "error" in result:
            attempts.append({"attempt": attempt_num + 1, "error": result["error"]})
            continue

        raw = result["response"].strip()
        try:
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)
            validated = ExtractedData(**parsed)
            attempts.append(
                {
                    "attempt": attempt_num + 1,
                    "status": "valid",
                    "raw": raw,
                    "validated": validated.model_dump(),
                    "metrics": result["metrics"],
                    "quality": result.get("quality", {}),
                }
            )
            return {
                "success": True,
                "data": validated.model_dump(),
                "attempts": attempts,
                "model": model_name,
            }
        except (json.JSONDecodeError, ValidationError) as e:
            attempts.append(
                {
                    "attempt": attempt_num + 1,
                    "status": "invalid",
                    "raw": raw,
                    "validation_error": str(e),
                    "metrics": result["metrics"],
                }
            )

    return {"success": False, "data": None, "attempts": attempts, "model": model_name}


def get_model_memory() -> dict:
    """Query Ollama /api/ps for memory usage of currently loaded models."""
    try:
        resp = httpx.get("http://localhost:11434/api/ps", timeout=5.0)
        data = resp.json()
        mem: dict = {}
        for m in data.get("models", []):
            name = m.get("name", "")
            size_bytes = m.get("size", 0)
            vram_bytes = m.get("size_vram", 0)
            mem[name] = {
                "size_mb": round(size_bytes / (1024 * 1024), 1),
                "size_vram_mb": round(vram_bytes / (1024 * 1024), 1),
                "size_bytes": size_bytes,
                "size_vram_bytes": vram_bytes,
            }
        return mem
    except Exception:
        return {}


def get_system_stats() -> dict:
    """Return live system resource metrics using psutil."""
    cpu_percent = psutil.cpu_percent(interval=0.2)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info()

    model_memory = get_model_memory()

    return {
        "cpu": {
            "percent": cpu_percent,
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
        },
        "ram": {
            "total_mb": round(vm.total / 1024 / 1024, 1),
            "used_mb": round(vm.used / 1024 / 1024, 1),
            "available_mb": round(vm.available / 1024 / 1024, 1),
            "percent": vm.percent,
        },
        "swap": {
            "total_mb": round(swap.total / 1024 / 1024, 1),
            "used_mb": round(swap.used / 1024 / 1024, 1),
            "percent": swap.percent,
        },
        "process": {
            "rss_mb": round(proc_mem.rss / 1024 / 1024, 1),
            "vms_mb": round(proc_mem.vms / 1024 / 1024, 1),
        },
        "ollama_models": model_memory,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def generate_report(
    study_results: dict,
    prompts: list[str],
    memory_samples: dict | None = None,
) -> dict:
    """Aggregate per-model stats into a rich comparison report."""
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_prompts": len(prompts),
        "models": {},
    }

    for model, results in study_results.items():
        ok = [r for r in results if "error" not in r]
        errs = [r for r in results if "error" in r]

        if ok:
            lats = [r["metrics"]["total_latency_s"] for r in ok]
            ttfts = [r["metrics"]["time_to_first_token_s"] for r in ok]
            tps_list = [r["metrics"]["tokens_per_second"] for r in ok]
            toks = [r["metrics"]["tokens_generated"] for r in ok]

            # Latency percentiles
            sorted_lats = sorted(lats)
            p50 = sorted_lats[len(sorted_lats) // 2]
            p95_idx = min(int(math.ceil(len(sorted_lats) * 0.95)) - 1, len(sorted_lats) - 1)
            p95 = sorted_lats[p95_idx]

            # Quality aggregation
            quality_ok = [r for r in ok if r.get("quality")]
            avg_word_count = 0.0
            avg_lex_diversity = 0.0
            avg_quality_score = 0.0
            if quality_ok:
                avg_word_count = round(
                    sum(r["quality"].get("word_count", 0) for r in quality_ok) / len(quality_ok), 1
                )
                avg_lex_diversity = round(
                    sum(r["quality"].get("lexical_diversity", 0) for r in quality_ok)
                    / len(quality_ok),
                    3,
                )
                # Composite quality score: normalize diversity + relative word count (0-1 scale)
                avg_quality_score = round(avg_lex_diversity * 100, 1)

            model_data: dict = {
                "total_runs": len(results),
                "successful": len(ok),
                "errors": len(errs),
                "success_rate_pct": round(len(ok) / len(results) * 100, 1),
                "avg_latency_s": round(sum(lats) / len(lats), 3),
                "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3),
                "avg_tokens_per_second": round(sum(tps_list) / len(tps_list), 2),
                "avg_tokens_generated": round(sum(toks) / len(toks), 1),
                "min_latency_s": round(min(lats), 3),
                "max_latency_s": round(max(lats), 3),
                "p50_latency_s": round(p50, 3),
                "p95_latency_s": round(p95, 3),
                "min_tps": round(min(tps_list), 2),
                "max_tps": round(max(tps_list), 2),
                "quality": {
                    "avg_word_count": avg_word_count,
                    "avg_lexical_diversity": avg_lex_diversity,
                    "quality_score": avg_quality_score,
                },
            }
        else:
            model_data = {
                "total_runs": len(results),
                "successful": 0,
                "errors": len(errs),
                "success_rate_pct": 0.0,
            }

        # Memory from per-prompt samples
        if memory_samples and model in memory_samples:
            samples = memory_samples[model]
            if samples:
                mem_vals = [s["size_mb"] for s in samples]
                vram_vals = [s["size_vram_mb"] for s in samples]
                model_data["memory"] = {
                    "avg_memory_mb": round(sum(mem_vals) / len(mem_vals), 1),
                    "peak_memory_mb": round(max(mem_vals), 1),
                    "avg_vram_mb": round(sum(vram_vals) / len(vram_vals), 1),
                    "peak_vram_mb": round(max(vram_vals), 1),
                    "samples": len(samples),
                }

        report["models"][model] = model_data

    return report


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/models")
def list_models():
    return {"models": MODELS}


@app.get("/system-stats")
def system_stats():
    return get_system_stats()


@app.get("/memory")
def get_memory():
    """Get raw Ollama model memory usage."""
    try:
        resp = httpx.get("http://localhost:11434/api/ps", timeout=5.0)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Chat & Benchmark ──────────────────────────────────────────────────────────


@app.post("/chat")
def chat(req: ChatRequest):
    return query_model_with_metrics(req.model_name, req.prompt, req.system_prompt, req.temperature)


@app.post("/benchmark")
def benchmark(req: ChatRequest):
    return query_model_with_metrics(req.model_name, req.prompt, req.system_prompt, req.temperature)


@app.post("/compare")
def compare(req: CompareRequest):
    """Run prompt across all models in PARALLEL for speed."""

    def _run(model: str) -> dict:
        return query_model_with_metrics(model, req.prompt, req.system_prompt, req.temperature)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {executor.submit(_run, m): m for m in MODELS}
        # Preserve original order
        ordered = {m: None for m in MODELS}
        for future in as_completed(futures):
            model = futures[future]
            ordered[model] = future.result()
        results = [ordered[m] for m in MODELS]

    return {"prompt": req.prompt, "results": results}


# ── Extraction ────────────────────────────────────────────────────────────────


@app.post("/extract")
def extract(req: ExtractRequest):
    return extract_json_with_retry(req.model_name, req.prompt, req.temperature)


# ── Temperature Experiments ───────────────────────────────────────────────────


@app.post("/temperature-test")
def temperature_test(req: TemperatureTestRequest):
    """Run same prompt at multiple temperatures on one model (sequential)."""
    results = [
        query_model_with_metrics(req.model_name, req.prompt, "You are a helpful assistant.", t)
        for t in req.temperatures
    ]
    return {"prompt": req.prompt, "model": req.model_name, "results": results}


@app.post("/compare-temp")
def compare_temp(req: CompareTempRequest):
    """
    Run every (model × temperature) combination in parallel.
    Returns a matrix: { model: { temperature: result } }
    """
    combos: list[tuple[str, float]] = [
        (m, t) for m in req.models for t in req.temperatures
    ]

    cell_results: dict[str, dict[str, dict]] = {m: {} for m in req.models}

    def _run(model: str, temp: float) -> tuple[str, float, dict]:
        result = query_model_with_metrics(model, req.prompt, req.system_prompt, temp)
        return model, temp, result

    with ThreadPoolExecutor(max_workers=min(len(combos), 8)) as executor:
        futures = [executor.submit(_run, m, t) for m, t in combos]
        for future in as_completed(futures):
            model, temp, result = future.result()
            cell_results[model][str(temp)] = result

    return {
        "prompt": req.prompt,
        "models": req.models,
        "temperatures": req.temperatures,
        "matrix": cell_results,
    }


# ── Study ─────────────────────────────────────────────────────────────────────


@app.get("/prompts")
def get_prompts():
    return {"prompts": TEST_PROMPTS, "count": len(TEST_PROMPTS)}


@app.post("/study")
def run_study(req: StudyRequest):
    prompts = TEST_PROMPTS[: req.num_prompts]
    study_results: dict[str, list] = {}
    memory_samples: dict[str, list] = {}

    for model in MODELS:
        model_results: list = []
        memory_samples[model] = []
        for p in prompts:
            result = query_model_with_metrics(model, p, "You are a helpful assistant.", 0.0)
            model_results.append(result)
            snap = get_model_memory()
            if model in snap:
                memory_samples[model].append(snap[model])
        study_results[model] = model_results

    report = generate_report(study_results, prompts, memory_samples)
    path = os.path.join(BASE_DIR, "study_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


@app.get("/report")
def get_report():
    path = os.path.join(BASE_DIR, "study_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": "No report found. Run a study first."}
