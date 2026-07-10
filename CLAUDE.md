# LLM Eval Workbench

Apple Silicon (M1 Ultra) 上大模型本地部署、性能评测与工具链集成的实验项目。

## 项目背景

从实际使用中积累的本地部署经验和评测工具：
- MLX / llama.cpp / Ollama 三框架横向选型
- Qwen3.5-27B 控制变量实测数据
- Unlimited-OCR VLM (3.34B) 的 MCP Server 封装
- 多进程并行 OCR 脚本（4-8 worker，已验证 2.7x-4.1x 加速）
- OCR JSON → LaTeX/Markdown 转换器（已用于 484 页教材 OCR）
- 模型能力 9 维度 × 6 角色 加权评分框架

## 关键文件

- `docs/ml-framework-comparison.md` — MLX vs llama.cpp vs Ollama 对比
- `docs/qwen35-mlx-benchmark.md` — Qwen3.5-27B 实测数据
- `docs/model-evaluation-guide.md` — 评测体系完整文档
- `mcp-servers/unlimited-ocr-mcp-server.py` — Unlimited-OCR MCP Server（主力）
- `scripts/ocr/parallel_ocr.py` — 并行 OCR
- `scripts/ocr/ocr_to_latex.py` — OCR → LaTeX 转换
- `tools/benchmark-weights.yaml` — 评测集权重配置

## 注意

- OCR 模型路径: `~/.cache/modelscope-cache/PaddlePaddle/Unlimited-OCR`（6.21GB）
- MPS patch 是因为上游 modeling_unlimitedocr.py 硬编码 `.cuda()`
- 评测分数数据需从各模型官方技术报告手动收集，脚本只做加权计算
