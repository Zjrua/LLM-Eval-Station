#!/usr/bin/env python3
"""
Inference Benchmark: 推理延迟/吞吐/内存实测工具
适用于 Apple Silicon (MPS) 或 CUDA 环境下的大模型性能测试。

Usage:
  python3 infer_benchmark.py --provider ollama --model qwen3.5:27b --prompt "Hello"
  python3 infer_benchmark.py --provider openai --base-url http://localhost:8000/v1 --model my-model
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import psutil
import requests


@dataclass
class InferenceResult:
    """单次推理结果"""
    tok_s: float = 0.0          # tokens/second
    ttft_ms: float = 0.0        # time to first token (ms)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0      # total latency
    memory_gb: float = 0.0       # RSS at end of inference


@dataclass
class BenchmarkReport:
    """汇总报告"""
    model: str = ""
    provider: str = ""
    prompt: str = ""
    rounds: int = 0
    results: List[InferenceResult] = field(default_factory=list)

    @property
    def avg_tok_s(self) -> float:
        return statistics.mean(r.tok_s for r in self.results) if self.results else 0

    @property
    def avg_ttft_ms(self) -> float:
        return statistics.mean(r.ttft_ms for r in self.results) if self.results else 0

    @property
    def avg_memory_gb(self) -> float:
        return statistics.mean(r.memory_gb for r in self.results) if self.results else 0

    @property
    def std_tok_s(self) -> float:
        return statistics.stdev(r.tok_s for r in self.results) if len(self.results) > 1 else 0

    def summary(self) -> str:
        lines = [
            f"Model: {self.model}",
            f"Provider: {self.provider}",
            f"Rounds: {self.rounds}",
            f"",
            f"Avg Speed:    {self.avg_tok_s:.2f} tok/s (±{self.std_tok_s:.2f})",
            f"Avg TTFT:     {self.avg_ttft_ms:.2f} ms",
            f"Avg Memory:   {self.avg_memory_gb:.1f} GB",
        ]
        return "\n".join(lines)


def _get_memory_gb() -> float:
    proc = psutil.Process(os.getpid())
    # Use RSS of the process (approximate for subprocess)
    return proc.memory_info().rss / (1024 ** 3)


def run_single(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> InferenceResult:
    """Run a single inference request and measure performance."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0

    try:
        resp = requests.post(url, json=payload, timeout=120, stream=True)
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                usage = chunk.get("usage")

                if delta.get("content") and first_token_time is None:
                    first_token_time = time.perf_counter()

                if usage:
                    completion_tokens = usage.get("completion_tokens", 0)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"  Error: {e}")

    end = time.perf_counter()
    total_ms = (end - start) * 1000

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else total_ms

    elapsed_s = max(end - start, 0.001)
    tok_s = completion_tokens / elapsed_s if completion_tokens > 0 else 0

    return InferenceResult(
        tok_s=tok_s,
        ttft_ms=ttft_ms,
        completion_tokens=completion_tokens,
        latency_ms=total_ms,
        memory_gb=_get_memory_gb(),
    )


def run_benchmark(
    base_url: str,
    model: str,
    prompt: str,
    rounds: int = 3,
    max_tokens: int = 256,
) -> BenchmarkReport:
    """Run multiple rounds and aggregate results."""
    report = BenchmarkReport(model=model, provider=base_url, prompt=prompt, rounds=rounds)

    for i in range(rounds):
        print(f"  Round {i+1}/{rounds}...", end=" ", flush=True)
        result = run_single(base_url, model, prompt, max_tokens)
        report.results.append(result)
        print(f"{result.tok_s:.2f} tok/s, TTFT {result.ttft_ms:.0f}ms, "
              f"{result.completion_tokens} tokens, {result.memory_gb:.1f}GB")

    return report


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base-url", default=None, help="Custom OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", default="请用100字介绍机器学习的核心概念。", help="Test prompt")
    parser.add_argument("--rounds", type=int, default=3, help="Number of benchmark rounds")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=None, help="Report context length")
    args = parser.parse_args()

    if args.provider == "ollama":
        base_url = args.base_url or "http://localhost:11434/v1"
    else:
        base_url = args.base_url or "http://localhost:8000/v1"

    print(f"=== LLM Inference Benchmark ===")
    print(f"Provider: {args.provider}")
    print(f"Model:    {args.model}")
    print(f"URL:      {base_url}")
    print(f"Rounds:   {args.rounds}")
    print()

    report = run_benchmark(
        base_url=base_url,
        model=args.model,
        prompt=args.prompt,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
    )

    print()
    print(report.summary())

    # Save JSON report
    out = {
        "model": report.model,
        "provider": report.provider,
        "rounds": report.rounds,
        "avg_tok_s": round(report.avg_tok_s, 2),
        "avg_ttft_ms": round(report.avg_ttft_ms, 2),
        "std_tok_s": round(report.std_tok_s, 2),
        "avg_memory_gb": round(report.avg_memory_gb, 1),
        "results": [asdict(r) for r in report.results],
    }
    out_file = f"/tmp/benchmark_{args.model.replace('/', '_').replace(':', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    main()
