#!/usr/bin/env python3
"""
Parallel Unlimited-OCR test: N workers each loading the model independently.
Each worker OCRs a subset of pages from the PDF.

Usage:
  python3 /tmp/uocr_parallel.py --workers 4 --start-page 1 --end-page 20
  python3 /tmp/uocr_parallel.py --workers 8 --start-page 1 --end-page 40
  python3 /tmp/uocr_parallel.py --workers 4 --all    # all 484 pages
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from multiprocessing import Process, Queue, Value
from pathlib import Path

import fitz  # pymupdf
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.path.expanduser("~/.cache/modelscope-cache/PaddlePaddle/Unlimited-OCR")
PDF_PATH = os.path.expanduser("~/tmp/概率论与数理统计教程 第三版.pdf")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

# ── MPS monkey-patch (same as MCP server) ──
if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _orig_tensor_cuda = torch.Tensor.cuda
    def _cuda_to_mps(self, device=None, non_blocking=False, memory_format=None):
        if str(self.device).startswith("mps"):
            return self
        return self.to("mps")
    torch.Tensor.cuda = _cuda_to_mps


class OCRWorker:
    """Per-process model wrapper (loads model once)."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is not None:
            return
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_safetensors=True, dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            self.model = self.model.eval().cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.model = self.model.eval().to("mps")
        else:
            self.model = self.model.eval()
        t1 = time.time()
        print(f"  [Worker {self.worker_id}] Model loaded in {t1-t0:.1f}s", flush=True)

    def ocr_page(self, image_path: str, prompt: str = "<image>OCR. ") -> str:
        self._load_model()
        out = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=tempfile.mkdtemp(prefix=f"uocr_w{self.worker_id}_"),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=4096,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
            eval_mode=True,
        )
        return (out or "").strip()


def render_page(pdf: fitz.Document, page_idx: int, out_dir: str) -> str:
    """Render PDF page to PNG, return path."""
    page = pdf[page_idx]
    pix = page.get_pixmap(dpi=200)
    path = os.path.join(out_dir, f"p{page_idx:04d}.png")
    pix.save(path)
    return path


def worker_process(worker_id: int, pages: list, result_queue: Queue,
                   page_labels: dict, done_counter):
    """Process in child: load model, OCR assigned pages, push results."""
    try:
        worker = OCRWorker(worker_id)
        for i, (page_num, pdf_idx) in enumerate(pages):
            label = page_labels.get(page_num, f"p{page_num}")
            # Each worker renders its own pages (no shared write contention)
            pdf = fitz.open(PDF_PATH)
            image_path = render_page(pdf, pdf_idx, tempfile.mkdtemp(prefix=f"uocr_w{worker_id}_render_"))
            pdf.close()

            t0 = time.time()
            text = worker.ocr_page(image_path)
            elapsed = time.time() - t0

            result_queue.put({
                "page": page_num,
                "label": label,
                "text": text,
                "elapsed": elapsed,
                "worker": worker_id,
            })

            # Cleanup rendered image
            try:
                os.unlink(image_path)
            except OSError:
                pass

        done_counter.value += 1  # signal completion
    except Exception as e:
        print(f"\n  [Worker {worker_id}] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--start-page", type=int, default=1, help="First page number (1-indexed)")
    parser.add_argument("--end-page", type=int, default=20, help="Last page number (inclusive)")
    parser.add_argument("--all", action="store_true", help="OCR all 484 pages")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Page number mapping (PDF page numbers, usually page_idx = page_num - 1)
    if args.all:
        page_numbers = list(range(1, 485))
    else:
        page_numbers = list(range(args.start_page, args.end_page + 1))

    num_pages = len(page_numbers)
    if num_pages == 0:
        print("No pages to OCR.")
        return

    # Labels not needed for bulk, but we can mark unknown
    page_labels = {pn: f"p{pn}" for pn in page_numbers}

    # Split pages among workers
    chunk_size = math.ceil(num_pages / args.workers)
    chunks = []
    for w in range(args.workers):
        chunk = page_numbers[w * chunk_size : (w + 1) * chunk_size]
        if chunk:
            # (page_number, pdf_index) — PDF is 0-indexed
            chunks.append([(pn, pn - 1) for pn in chunk])

    print(f"Pages: {num_pages}, Workers: {args.workers}, Chunk size: ~{chunk_size}")

    result_queue = Queue()
    done_counter = Value("i", 0)

    processes = []
    t_start = time.time()

    for w_id, chunk in enumerate(chunks):
        p = Process(target=worker_process, args=(w_id, chunk, result_queue, page_labels, done_counter))
        p.start()
        processes.append(p)

    # Collector: read results as they come in
    results = {}
    completed_workers = set()
    last_report = time.time()

    try:
        while len(completed_workers) < len(chunks):
            try:
                msg = result_queue.get(timeout=2)
                pn = msg["page"]
                results[pn] = msg
                elapsed = msg["elapsed"]
                now = time.time()
                print(f"  p{pn} {msg['label']}  w{msg['worker']}  {len(msg.get('text',''))}chars  {elapsed:.1f}s  [{len(results)}/{num_pages}]", flush=True)
            except Exception:
                # Check for stalled workers
                now = time.time()
                if now - last_report > 30:
                    # check which processes are alive
                    live = sum(1 for p in processes if p.is_alive())
                    print(f"  [status] {len(results)}/{num_pages} pages, {live}/{len(chunks)} workers alive", flush=True)
                    last_report = now
                # Check if any process died unexpectedly
                for p in processes:
                    if not p.is_alive() and p.pid not in completed_workers:
                        p.join(timeout=0.1)
                        completed_workers.add(p.pid)

        # Ensure all results collected
        while not result_queue.empty():
            try:
                msg = result_queue.get(timeout=1)
                results[msg["page"]] = msg
            except Exception:
                break

    except KeyboardInterrupt:
        print("\nInterrupted, saving partial results...")
        for p in processes:
            p.terminate()

    total_time = time.time() - t_start

    # Wait for processes
    for p in processes:
        p.join(timeout=5)

    # Build output
    pages_done = sorted(results.keys())
    output = {}
    for pn in pages_done:
        output[f"p{pn}"] = {
            "label": results[pn].get("label", f"p{pn}"),
            "text": results[pn].get("text", ""),
            "elapsed": results[pn].get("elapsed", 0),
            "worker": results[pn].get("worker", 0),
        }

    # Stats
    total_chars = sum(len(v.get("text", "")) for v in output.values())
    elapsed_times = [results[pn]["elapsed"] for pn in pages_done]
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    throughput = len(pages_done) / total_time if total_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Workers: {args.workers}")
    print(f"Pages: {len(pages_done)}/{num_pages}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Avg per page: {avg_time:.1f}s")
    print(f"Throughput: {throughput:.2f} pages/s ({throughput*60:.1f} pages/min)")
    print(f"Total chars: {total_chars}")
    print(f"{'='*60}")

    # Save
    out_path = args.output or f"/tmp/uocr_parallel_w{args.workers}_p{args.start_page}-{args.end_page}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
