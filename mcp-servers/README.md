# MCP Server 封装

将本地 VLM/OCR 模型封装为 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) Server，
通过 stdio transport 与 AI Agent 集成。

## 架构

```
AI Agent (Hermes) ←stdio→ MCP Server ←MPS→ OCR/VLM Model
```

## 可用 Server

### Unlimited-OCR MCP Server（主力）

**模型**：[PaddlePaddle/Unlimited-OCR](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR) (3.34B VLM)

**工具列表**：
| 工具 | 功能 |
|------|------|
| `ocr_image` | base64 图片 → 文本识别 |
| `ocr_image_path` | 本地图片路径 → 文本识别 |
| `ocr_table` | 图片 → Markdown 表格 |
| `ocr_formula` | 图片 → LaTeX 公式 |
| `ocr_chart` | 图片 → 图表数据提取 |
| `ocr_seal` | 图片 → 印章文字识别 |
| `ocr_spotting` | 图片 → 文字定位 (带坐标) |

**特性**：
- Apple Silicon MPS 适配（monkey-patch `.cuda()` → `.to("mps")`）
- Lazy 加载（首次调用才加载模型，~3.4GB bf16）
- 向后兼容 PaddleOCR-VL 的工具签名
- `gundam` 模式（base_size=1024, crop_mode=True）

**依赖**：
```
pip install torch transformers mcp
```

### PaddleOCR-VL MCP Server（旧版）

**模型**：PaddlePaddle/PaddleOCR-VL-1.6 (1.93GB, bf16)

已被 Unlimited-OCR 替代，保留作为参考和回退方案。

## 配置示例（Hermes Agent）

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  unlimited-ocr:
    command: python3
    args: ["/path/to/unlimited-ocr-mcp-server.py"]
    transport: stdio
```

## 启动测试

```bash
# 测试 MCP Server 是否正常
python3 mcp-servers/unlimited-ocr-mcp-server.py

# 通过 Hermes CLI 验证
hermes tools list | grep ocr
```

## 已知限制

1. MPS 上 model.generate() 是 autoregressive 的，GPU 利用率不满（CPU 瓶颈在 token-by-token 解码）
2. 上游 `modeling_unlimitedocr.py` 硬编码 `.cuda()` 约 15 处，需要 monkey-patch
3. batch OCR 时多进程各自加载模型，内存占用 = worker × model_size
