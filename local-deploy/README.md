# 本地模型部署与监控 (local-deploy)

基于 Apple MLX / oMLX 的本地 LLM 推理服务监控系统。

## 背景

AI Agent 系统需要本地化部署 Embedding 模型（中文语义检索）和轻量 LLM（辅助视觉理解），
以避免云端 API 数据外泄与计费成本。Apple Silicon（M1 Ultra）上通过 MLX + oMLX 方案部署。

## 文件说明

| 文件 | 功能 |
|------|------|
| `omlx_client.py` | 异步 HTTP 客户端，封装 OMLX 服务（192.168.192.2:8000），通过 ZeroTier 跨网络访问 |
| `omlx_monitor.py` | FastAPI 路由：健康检查 / 模型列表 / GPU 状态 / 延迟测试 |
| `app.py` | 独立部署入口：`uvicorn app:app --host 0.0.0.0 --port 8080` |
| `test_omlx.py` | pytest 测试（8 个用例，含离线/在线 mock 场景） |

## API 端点

```
GET /api/omlx/health   → {"status": "online", "reachable": true, "response_ms": 45.2}
GET /api/omlx/models   → {"status": "online", "count": 2, "models": [...]}
GET /api/omlx/status   → GPU 和服务器状态
GET /api/omlx/latency  → 3 次采样平均延迟
```

## 技术栈

- **推理框架**: Apple MLX (mlx-lm v0.31.2, mlx-embeddings)
- **推理服务器**: oMLX 0.4.1 (OpenAI 兼容 API)
- **模型**: Qwen3-Embedding (0.6B/8B 4bit-DWQ), Qwen3.5-9B
- **网络**: ZeroTier 虚拟网络 (192.168.192.0/24)
- **后端**: Python 3.11+, FastAPI, httpx, pytest

## 框架选型对比

| 维度 | MLX | llama.cpp (GGUF) | Ollama |
|------|-----|-----------------|--------|
| 量化精度 | affine/MXFP4/NVFP4/mixed-bit | Q4_K_M/Q5_K_M | 受限 |
| Python 生态 | 原生 API + LoRA | C++ 绑定 | 不可编程 |
| UMA 利用 | 直接 Metal/MPS | Metal 后端 | 间接 |
| 新架构适配 | 快（Qwen3 原生支持）| 中等 | 慢 |

结论：MLX + oMLX 兼顾精度、可编程性和统一内存利用率。

## 运行

```bash
# 安装依赖
pip install fastapi uvicorn httpx

# 启动监控服务
uvicorn app:app --host 0.0.0.0 --port 8080

# 运行测试
pytest test_omlx.py -v
```
