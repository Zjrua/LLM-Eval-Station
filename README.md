# LLM Eval Workbench

Apple Silicon (M1 Ultra) 上的大模型本地部署、性能评测与工具链集成实验。

## 项目结构

```
llm-eval-workbench/
├── docs/                          # 研究文档
│   ├── ml-framework-comparison.md  # MLX vs llama.cpp vs Ollama 横向选型
│   ├── qwen35-mlx-benchmark.md     # Qwen3.5-27B MLX vs GGUF 跨框架实测
│   └── model-evaluation-guide.md   # 模型评测体系：9维度 × 6角色加权评分
├── mcp-servers/                   # MCP Server 封装
│   ├── unlimited-ocr-mcp-server.py  # Unlimited-OCR (3.34B VLM) → MCP
│   ├── paddleocr-vl-mcp-server.py    # PaddleOCR-VL-1.6 → MCP (旧版)
│   └── README.md                     # MCP Server 架构说明
├── scripts/
│   ├── ocr/                         # OCR 相关脚本
│   │   ├── parallel_ocr.py          # 多进程并行 OCR（4-8 worker）
│   │   ├── ocr_to_latex.py          # OCR JSON → LaTeX/Markdown 转换器
│   │   └── clean_tex.py             # LaTeX artifact 清洗工具
│   └── benchmark/                   # 性能基准测试
│       └── infer_benchmark.py      # 推理延迟/吞吐/内存基准测试脚本
├── tools/                          # 辅助工具
│   └── benchmark-weights.yaml      # 评测集权重配置（可被脚本读取）
├── CLAUDE.md                        # AI Agent 项目说明
└── .gitignore
```

## 核心内容

### 1. 推理框架横向选型

在 M1 Ultra 上对比 **MLX**、**llama.cpp (GGUF)**、**Ollama** 三个主流推理框架：

| 维度 | MLX | llama.cpp | Ollama |
|------|-----|-----------|--------|
| 语言 | Python | C/C++ | Go+llama.cpp |
| 后端 | Apple Metal | Apple Metal (ggml) | llama.cpp 封装 |
| 量化格式 | affine/MXFP4/MXFP8 | GGUF (Q2_K~Q8_0) | GGUF |
| 微调支持 | LoRA/QLoRA | 有限 | 无 |
| Python 生态 | ✅ 原生 | ❌ | ❌ |

### 2. Qwen3.5-27B 跨框架实测

三轮控制变量测试（M1 Ultra, 2026-04-14）：

| 条件 | GGUF (Ollama Q4_K_M) | MLX (4bit) |
|------|----------------------|------------|
| 256K context | 12.24 tok/s, 42GB | 25.43 tok/s, 15GB |
| 32K context (公平) | **16.56 tok/s**, 24GB | **10.54 tok/s**, 15GB |
| 3次并行验证 | 16.46-16.63 tok/s | 9.89-11.21 tok/s |

**结论**：GGUF 生成速度 +50%+，MLX 首token延迟 -33%、内存省 40%。

### 3. Unlimited-OCR MCP Server

将 [Unlimited-OCR](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR)（3.34B VLM，OmniDocBench SOTA）封装为 [MCP](https://modelcontextprotocol.io/) Server：

- **7 个工具**：文本/表格/公式/图表/印章/文字定位
- **Apple Silicon MPS 适配**：monkey-patch `torch.Tensor.cuda()` → `.to("mps")`
- **Lazy 加载**：首次调用时才加载模型（~3.4GB bf16）
- **兼容旧接口**：工具签名与原 PaddleOCR-VL server 完全一致

### 4. 模型评测体系

9 维度（综合/编程/数学/长文本/指令遵循/工具调用/科学/中文/对话）× 6 角色（Coder/Planner/Writer/Researcher/Reviewer/Doubter）加权评分框架。

## 硬件环境

- **M1 Ultra** (128GB Unified Memory)
- **macOS 26.5.2**
- Python 3.12 + PyTorch (MPS)
- MLX v0.31.x, Ollama, llama.cpp

## License

MIT

## Author

[张钧瑞](https://github.com/Zjrua) — 哈尔滨工业大学应用统计
