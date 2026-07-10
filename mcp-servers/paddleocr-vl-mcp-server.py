#!/usr/bin/env python3
"""
PaddleOCR-VL-1.6 MCP Server (stdio transport)
Uses mlx_vlm to run OCR on images locally on Apple Silicon.
Model: PaddlePaddle/PaddleOCR-VL-1.6 (1.93GB, bf16)
"""

import base64
import os
import tempfile
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "OCR_MODEL_PATH",
    "/Users/zjrua/.cache/paddleocr-vl-models/PaddleOCR-VL-1.6",
)

# ─── Model cache (lazy, main thread only — MLX GPU streams are thread-local) ─

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        from mlx_vlm import load
        _model, _processor = load(MODEL_PATH, trust_remote_code=True)
    return _model, _processor


# ─── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP("paddleocr-vl")


def _ocr(image_input: str, prompt: str, max_tokens: int = 2048) -> str:
    """Core OCR function."""
    model, processor = _get_model()
    from mlx_vlm import generate

    formatted = (
        f"<|begin_of_sentence|>User: "
        f"<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        f"{prompt}\nAssistant:\n"
    )

    is_temp = False
    if image_input.startswith("data:"):
        # base64 data URL → temp file
        img_bytes = base64.b64decode(image_input.split(",", 1)[1])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            image_input = f.name
            is_temp = True

    try:
        result = generate(
            model, processor,
            prompt=formatted,
            image=image_input,
            max_tokens=max_tokens,
            verbose=False,
        )
        return result.text
    finally:
        if is_temp:
            os.unlink(image_input)


@mcp.tool()
def ocr_image(
    image_base64: str,
    prompt: str = "OCR:",
    max_tokens: int = 2048,
) -> str:
    """Run OCR on a base64-encoded image using PaddleOCR-VL-1.6 (MLX, Apple Silicon).

    Args:
        image_base64: Base64-encoded image data (PNG/JPEG).
        prompt: Task prompt. Use "OCR:" for text, "Table Recognition:" for tables,
                "Formula Recognition:" for formulas, "Chart Recognition:" for charts,
                "Spotting:" for text spotting, "Seal Recognition:" for seals.
        max_tokens: Maximum tokens to generate.

    Returns:
        Recognized text content.
    """
    data_url = f"data:image/png;base64,{image_base64}"
    return _ocr(data_url, prompt, max_tokens)


@mcp.tool()
def ocr_image_path(
    image_path: str,
    prompt: str = "OCR:",
    max_tokens: int = 2048,
) -> str:
    """Run OCR on a local image file using PaddleOCR-VL-1.6 (MLX, Apple Silicon).

    Args:
        image_path: Absolute path to a local image file.
        prompt: Task prompt. Use "OCR:" for text, "Table Recognition:" for tables,
                "Formula Recognition:" for formulas, "Chart Recognition:" for charts,
                "Spotting:" for text spotting, "Seal Recognition:" for seals.
        max_tokens: Maximum tokens to generate.

    Returns:
        Recognized text content.
    """
    return _ocr(image_path, prompt, max_tokens)


@mcp.tool()
def ocr_table(
    image_base64: str,
    max_tokens: int = 4096,
) -> str:
    """Extract table from an image using PaddleOCR-VL-1.6.

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
    max_tokens: int = 2048,
) -> str:
    """Extract math formulas from an image using PaddleOCR-VL-1.6.

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
    max_tokens: int = 2048,
) -> str:
    """Extract chart content from an image using PaddleOCR-VL-1.6.

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
    max_tokens: int = 1024,
) -> str:
    """Recognize seal/stamp text from an image using PaddleOCR-VL-1.6.

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
    max_tokens: int = 2048,
) -> str:
    """Detect and recognize text positions (spotting) from an image using PaddleOCR-VL-1.6.

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
