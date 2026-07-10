# Qwen3.5-27B MLX vs GGUF 跨框架实测

> 测试日期：2026-04-14 · 硬件：Apple M1 Ultra (128GB UMA) · 来源：飞书实测报告

## 实验设计

用 Qwen3.5-27B（27B Dense）做三轮控制变量测试，在 MLX (LM Studio, 4bit) 和 GGUF (Ollama, Q4_K_M) 之间对比。

## 结果

| 测试轮次 | 条件 | Ollama (GGUF Q4_K_M, ~17GB) | LM Studio (MLX 4bit, ~15GB) | 分析 |
|---------|------|-----------------------------|---------------------------|------|
| Round 1 | 256K context（各自身back） | 12.24 tok/s, TTFT 0.64s, 42GB | 25.43 tok/s, TTFT 3.28s, ~15GB | MLX 速度 2×，但 Ollama 背着22GB KV cache |
| Round 2 | 32K context（公平对比） | **16.56 tok/s**, TTFT 1.04s, 24GB | **10.54 tok/s**, TTFT 0.70s, ~15GB | GGUF 生成速度 +57%，MLX TTFT -33%，内存省 40% |
| Round 3 | 3次并行验证 | 16.46-16.63 tok/s | 9.89-11.21 tok/s | GGUF 生成稳定领先 50%+，MLX TTFT 整体更优 |

## KV Cache 内存参考（27B 模型）

| Context Length | 内存占用 |
|---------------|---------|
| 4K | 18GB |
| 32K | 24GB |
| 256K | 42GB |

> GGUF 量化本身不减少 KV cache（机制决定）。

## 关键发现

1. **GGUF（Ollama）生成速度领先 50%+** — 稳定生成场景更优
2. **MLX 首token延迟更低（-33%）** — 交互式场景更优
3. **MLX 内存效率更高（-40%）** — 适合多模型共存
4. **MLX Python 生态更优** — 适合定制化服务和微调

## 选型决策

Embedding 服务选用 MLX 的理由：
- Embedding 更关注首token延迟（TTFT 决定单句编码速度）
- MLX Python 生态对定制化 Embedding 服务更必要
- 统一内存模型在多模型 LRU 管理（0.6B + 8B 并存）下优势明显
- Embedding 模型无 KV cache 瓶颈，生成速度劣势不成立
