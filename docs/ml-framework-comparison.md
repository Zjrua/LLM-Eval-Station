# MLX Framework Research: Deploying Qwen3.5 on Apple Silicon
## Research Date: April 14, 2026

---

## 1. What is MLX and How Does It Compare to llama.cpp?

### MLX Overview
- **Developer**: Apple (ml-explore team)
- **GitHub Stars**: ~25,400 (MLX core), ~4,740 (mlx-lm)
- **Latest mlx-lm Version**: v0.31.2 (released April 7, 2026)
- **Language**: Python-first (with Metal backend), NumPy-like API
- **License**: MIT

MLX is Apple's array framework designed specifically for Apple Silicon (M1/M2/M3/M4 chips). It provides:
- Unified CPU/GPU memory model (no explicit memory transfers needed)
- Lazy evaluation with compilation optimization
- Metal shader-based compute for GPU acceleration
- Automatic differentiation for training

### MLX vs llama.cpp on Apple Silicon

| Feature | MLX (mlx-lm) | llama.cpp |
|---------|-------------|-----------|
| Language | Python | C/C++ |
| Backend | Apple Metal (via Python) | Apple Metal (via ggml) |
| Memory Model | Unified (NumPy-like) | Explicit buffer management |
| Quantization | Affine, MXFP4, NVFP4, MXFP8, mixed-bit | GGUF (Q2_K through Q8_0) |
| Fine-tuning | Yes (LoRA, full, QLoRA) | Limited |
| API Server | Built-in OpenAI-compatible | Built-in (llama-server) |
| Model Ecosystem | HuggingFace Hub (auto-convert) | GGUF format |
| Distributed Inference | Yes (multi-GPU) | Yes |
| Speculative Decoding | Yes | Yes |
| Tool Calling | Yes (via chat templates) | Varies |

### Key Advantages of MLX:
1. Python ecosystem integration (LangChain, etc.)
2. Auto-conversion from HuggingFace
3. First-class fine-tuning support
4. Apple-optimized unified memory
5. Rapid active development by Apple

### Key Advantages of llama.cpp:
1. Lower-level C/C++ control
2. Cross-platform portability
3. Mature GGUF quantization ecosystem
4. Smaller runtime overhead
5. Wider community

---

## 2. MLX-Compatible Qwen3.5 - FULLY SUPPORTED

mlx-lm has dedicated model files:
- mlx_lm/models/qwen3_5.py (Dense)
- mlx_lm/models/qwen3_5_moe.py (MoE)

Qwen3.5 uses a novel **hybrid attention mechanism**:
- Every 4th layer: standard full (softmax) attention
- Other 3 layers: GatedDeltaNet (linear attention with gated delta rule)
- Supports MoE variants
- Distributed sharding for multi-chip systems

To convert any Qwen3.5 model:
  pip install mlx-lm
  mlx_lm.convert --model Qwen/Qwen3.5-7B-Instruct -q

---

## 3. Quantization Options

### Standard Modes:
| Mode | Bits | Group Size | Description |
|------|------|-----------|-------------|
| affine | 4 | 64 | Standard (most common) |
| mxfp4 | 4 | 32 | Apple MX FP4 format |
| nvfp4 | 4 | 16 | NVIDIA FP4 format |
| mxfp8 | 8 | 32 | Apple MX FP8 format |

### Mixed-Bit Recipes:
| Recipe | Low/High | Notes |
|--------|----------|-------|
| mixed_2_6 | 2/6 | Aggressive compression |
| mixed_3_4 | 3/4 | Balanced (like Q4_K_M) |
| mixed_3_6 | 3/6 | Quality-focused |
| mixed_4_6 | 4/6 | High quality |

MoE models auto-use 8-bit for gate layers.

Commands:
  mlx_lm.convert --model Qwen/Qwen3.5-7B -q --q-mode affine
  mlx_lm.convert --model Qwen/Qwen3.5-7B -q --quant-predicate mixed_3_4
  mlx_lm.convert --model Qwen/Qwen3.5-7B -q --q-mode mxfp8

---

## 4. Performance Comparison (M1 Ultra)

All three ultimately use the same Metal GPU backend on Apple Silicon.

| Framework | 7B tok/s | 70B tok/s | Notes |
|-----------|---------|----------|-------|
| MLX | ~40-55 | ~8-12 | Unified memory advantage |
| llama.cpp | ~35-50 | ~7-11 | Mature Metal kernels |
| Ollama | ~30-45 | ~6-10 | API layer overhead |

Qwen3.5 hybrid attention should give faster prefill speed than pure attention models.

Built-in benchmarking:
  mlx_lm.benchmark --model mlx-community/Qwen3.5-7B-4bit --prompt-tokens 512 --generation-tokens 1024

---

## 5. MLX Server Setup (OpenAI-Compatible)

  pip install mlx-lm
  mlx_lm.server --model mlx-community/Qwen3.5-7B-4bit

Default: localhost:8080

Supported: /v1/chat/completions, /v1/models, streaming, tool calling,
all sampling params, logprobs, speculative decoding, multi-turn.

Drop-in OpenAI client:
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

Note: Server has only basic security - not recommended for production as-is.

---

## 6. Community & Benchmarks

- MLX: 25k+ stars, mlx-lm: 4.7k+ stars
- Active Apple development (35+ PRs per release)
- mlx-community HF org: thousands of pre-converted models
- mlx-my-repo Space: convert any HF model to MLX
- Community reports MLX matches/exceeds llama.cpp on Apple Silicon

---

## 7. Tool Calling - YES, FULLY SUPPORTED

11 tool parsers available: function_gemma, gemma4, glm47, json_tools,
kimi_k2, longcat, minimax_m2, mistral, pythonic, qwen3_coder

Auto-detection from chat template:
- Qwen-style (with tool_call.name) -> json_tools parser
- Qwen Coder (XML function=) -> qwen3_coder parser

Full streaming tool call support with OpenAI-compatible format.
Multi-turn tool conversations supported.

---

## Summary

Quick start:
  pip install mlx-lm
  mlx_lm.convert --model Qwen/Qwen3.5-7B-Instruct -q --quant-predicate mixed_3_4
  mlx_lm.server --model mlx_model --port 8080

Choose MLX for: Apple Silicon + Python + fine-tuning + easiest setup
Choose llama.cpp for: cross-platform + GGUF ecosystem + low-level control

Qwen3.5 notes: Hybrid attention natively supported, MoE quantization handled,
KV cache fix in v0.31.2, should be efficient for long-context workloads.