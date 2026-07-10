#!/usr/bin/env python3
"""
Unlimited-OCR MCP Server (stdio transport)
Replaces paddleocr-vl-mcp-server with the new baidu/PaddlePaddle/Unlimited-OCR model.

Same tool names/signatures as the old server (backward compatible for skills):
  - ocr_image, ocr_image_path, ocr_table, ocr_formula, ocr_chart, ocr_spotting

Key adaptations for Apple Silicon (MPS):
  - Monkey-patches torch.Tensor.cuda() -> .to("mps") because the upstream
    modeling_unlimitedocr.py hardcodes .cuda() in its infer() and infer_multi().
  - Model loaded once on first call (lazy, ~3.4GB bf16).
  - Uses eval_mode=True so the model returns the decoded text directly
    (no need to read a file back from output_path).
  - Defaults to "gundam" mode (base_size=1024, image_size=640, crop_mode=True),
    which is the recommended config for single-image OCR in the upstream README.
"""

import base64
import os
import tempfile
import warnings
from typing import Optional

# Suppress noisy upstream warnings (autocast, attention mask, etc.)
warnings.filterwarnings("ignore", category=UserWarning)

# ── MPS compatibility patch (must run BEFORE transformers/imports) ────────────
# The upstream modeling_unlimitedocr.py hardcodes `.cuda()` in ~15 places.
# We redirect .cuda() to .to("mps") so the model works on Apple Silicon.
import torch

if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _orig_tensor_cuda = torch.Tensor.cuda

    def _cuda_to_mps(self, device=None, non_blocking=False, memory_format=None):
        if str(self.device).startswith("mps"):
            return self
        return self.to("mps")

    torch.Tensor.cuda = _cuda_to_mps

from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "OCR_MODEL_PATH",
    "/Users/zjrua/.cache/modelscope-cache/PaddlePaddle/Unlimited-OCR",
)
# Pin to local-only once downloaded (skip the network round-trip)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# transformers 4.46+ warns about torch_dtype; the upstream code uses the old name
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ── Model cache (lazy, single-load) ───────────────────────────────────────────

_model = None
_tokenizer = None


def _get_model():
    """Lazy load Unlimited-OCR onto MPS (Apple Silicon). Returns (model, tokenizer)."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            _model = _model.eval().cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _model = _model.eval().to("mps")
        else:
            _model = _model.eval()
    return _model, _tokenizer


# ── Prompt templates ──────────────────────────────────────────────────────────
# Unlimited-OCR uses a single prompt format: "<image>{task}." (note trailing period
# and a single space). The old PaddleOCR-VL server took a categorical "task" string
# (OCR:/Table Recognition:/etc.) — we map each to the closest Unlimited-OCR prompt
# so the same tool still does the right thing.

PROMPT_MAP = {
    "ocr":              "OCR this image.",            # generic text recognition
    "table":            "Convert this table to markdown.",  # table structure
    "formula":          "Convert this formula to LaTeX.",   # math recognition
    "chart":            "Parse this chart and extract the data.",
    "spotting":         "Detect text regions in the image.",  # bounding boxes
    "seal":             "Recognize the seal in this image.",
    # Free-form fallback: pass through if user provides explicit prompt
}

# ── Core OCR function ─────────────────────────────────────────────────────────

def _ocr(image_input: str, prompt: str, max_tokens: int = 4096) -> str:
    """
    Run Unlimited-OCR on an image. Returns the decoded text.

    Args:
        image_input: Either a filesystem path, or a "data:image/...;base64,..." URL.
        prompt: Task prompt. Can be either:
                - a categorical key (ocr/table/formula/chart/spotting/seal) — mapped
                  to the matching Unlimited-OCR prompt
                - a full prompt string starting with "<image>..."
        max_tokens: Generation length cap (up to 32768 supported by model).
    """
    model, tokenizer = _get_model()

    # Normalize prompt
    pl = prompt.strip().lower()
    if pl in PROMPT_MAP:
        full_prompt = f"<image>{PROMPT_MAP[pl]} "
    elif prompt.lstrip().startswith("<image>"):
        full_prompt = prompt
    else:
        # bare string — wrap as OCR task
        full_prompt = f"<image>OCR. "

    # Resolve data URL → temp file
    is_temp = False
    if image_input.startswith("data:"):
        img_bytes = base64.b64decode(image_input.split(",", 1)[1])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            image_input = f.name
            is_temp = True

    try:
        # Use the upstream .infer() method with eval_mode=True (returns text directly).
        # gundam config: best for single-image document parsing per upstream README.
        out = model.infer(
            tokenizer,
            prompt=full_prompt,
            image_file=image_input,
            output_path=tempfile.mkdtemp(prefix="uocr_"),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=max_tokens,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=False,
            eval_mode=True,
        )
        return (out or "").strip()
    finally:
        if is_temp:
            try:
                os.unlink(image_input)
            except OSError:
                pass


# ── MCP Server (tool names mirror the old paddleocr-vl server) ───────────────

mcp = FastMCP("unlimited-ocr")


@mcp.tool()
def ocr_image(
    image_base64: str,
    prompt: str = "OCR:",
    max_tokens: int = 4096,
) -> str:
    """Run OCR on a base64-encoded image using Unlimited-OCR (MPS, Apple Silicon).

    Args:
        image_base64: Base64-encoded image data (PNG/JPEG).
        prompt: Task prompt. Use "OCR:" for text, "Table Recognition:" for tables,
                "Formula Recognition:" for formulas, "Chart Recognition:" for charts,
                "Spotting:" for text spotting, "Seal Recognition:" for seals.
                (Old paddleocr-vl vocabulary — mapped internally to Unlimited-OCR prompts.)
        max_tokens: Maximum tokens to generate (up to 32768).

    Returns:
        Recognized text content (with optional <|det|>...<|/det|> layout tags).
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, prompt, max_tokens)


@mcp.tool()
def ocr_image_path(
    image_path: str,
    prompt: str = "OCR:",
    max_tokens: int = 4096,
) -> str:
    """Run OCR on a local image file using Unlimited-OCR (MPS, Apple Silicon).

    Args:
        image_path: Absolute path to a local image file.
        prompt: Task prompt. Use "OCR:" for text, "Table Recognition:" for tables,
                "Formula Recognition:" for formulas, "Chart Recognition:" for charts,
                "Spotting:" for text spotting, "Seal Recognition:" for seals.
        max_tokens: Maximum tokens to generate (up to 32768).

    Returns:
        Recognized text content.
    """
    return _ocr(image_path, prompt, max_tokens)


@mcp.tool()
def ocr_table(
    image_base64: str,
    max_tokens: int = 8192,
) -> str:
    """Extract table from an image using Unlimited-OCR.

    Args:
        image_base64: Base64-encoded image data containing a table.
        max_tokens: Maximum tokens to generate.

    Returns:
        Table content in markdown or text format.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, "Table Recognition:", max_tokens)


@mcp.tool()
def ocr_formula(
    image_base64: str,
    max_tokens: int = 4096,
) -> str:
    """Extract math formulas from an image using Unlimited-OCR.

    Args:
        image_base64: Base64-encoded image data containing formulas.
        max_tokens: Maximum tokens to generate.

    Returns:
        LaTeX-formatted formulas.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, "Formula Recognition:", max_tokens)


@mcp.tool()
def ocr_chart(
    image_base64: str,
    max_tokens: int = 4096,
) -> str:
    """Extract chart content from an image using Unlimited-OCR.

    Args:
        image_base64: Base64-encoded image data containing a chart.
        max_tokens: Maximum tokens to generate.

    Returns:
        Chart description and data.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, "Chart Recognition:", max_tokens)


@mcp.tool()
def ocr_seal(
    image_base64: str,
    max_tokens: int = 2048,
) -> str:
    """Recognize seal/stamp text from an image using Unlimited-OCR.

    Args:
        image_base64: Base64-encoded image data containing a seal/stamp.
        max_tokens: Maximum tokens to generate.

    Returns:
        Seal/stamp text content.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, "Seal Recognition:", max_tokens)


@mcp.tool()
def ocr_spotting(
    image_base64: str,
    max_tokens: int = 4096,
) -> str:
    """Detect and recognize text positions (spotting) from an image using Unlimited-OCR.

    Args:
        image_base64: Base64-encoded image data.
        max_tokens: Maximum tokens to generate.

    Returns:
        Text spotting results with positions.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, "Spotting:", max_tokens)


if __name__ == "__main__":
    mcp.run(transport="stdio")
