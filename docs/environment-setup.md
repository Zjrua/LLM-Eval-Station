# 环境配置 (uv)

本项目用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖。
Python 固定为 3.12（见 `.python-version`），依赖锁定在 `uv.lock`。

## 快速开始

```bash
# 1. 装全部依赖 (核心 + 权重学习 + 测试), 自动创建 .venv 并下载 Python 3.12
uv sync --group learning --group dev

# 2. 跑测试 (应全绿)
uv run pytest tests/ -v

# 3. 合成数据 sanity check (验证权重学习 pipeline 数学正确)
uv run python scripts/learn_weights.py --source synthetic --n 2000
```

## 网络问题排查 (国内常见)

uv 下载 Python 3.12 默认走 GitHub（`github.com/astral-sh/python-build-standalone`），
国内可能连不上。项目已配清华 PyPI 镜像（见 `uv.toml`），但 **Python 发行版下载源需要单独配代理**。

### 方案 A: 直连 GitHub (能访问 github.com)

无需额外配置，直接：
```bash
uv sync --group learning --group dev
```

### 方案 B: 用 GitHub 代理 (不能直连但能用 ghfast.top / ghproxy)

设环境变量后再 sync（PowerShell 语法）：
```powershell
$env:UV_PYTHON_INSTALL_MIRROR = "https://ghfast.top"
uv sync --group learning --group dev
```

Bash 语法：
```bash
export UV_PYTHON_INSTALL_MIRROR="https://ghfast.top"
uv sync --group learning --group dev
```

备选代理：`https://ghproxy.com`、`https://mirror.ghproxy.com`

> 注：`UV_PYTHON_INSTALL_MIRROR` 是 uv 官方支持的环境变量，会把
> `https://github.com/astral-sh/python-build-standalone/releases/download/...`
> 替换为 `https://<mirror>/https://github.com/astral-sh/python-build-standalone/releases/download/...`。

### 方案 C: GitHub 和代理都不通 (只能用 PyPI 镜像)

这是国内最常见的封锁场景。uv 下 Python 走 GitHub (python-build-standalone) 会失败,
但**清华有独立的 Python 解释器镜像站**, 可绕过 GitHub 手动装 Python, 再让 uv 复用。

**步骤 1: 从清华镜像下载 Python 3.12 安装包**
```bash
# 清华 Python 镜像 (与 PyPI 镜像不同, 是独立的解释器镜像)
# https://mirrors.tuna.tsinghua.edu.cn/python/
curl -L -o python-3.12.9-amd64.exe \
  "https://mirrors.tuna.tsinghua.edu.cn/python/3.12.9/python-3.12.9-amd64.exe"
```
或用浏览器访问 https://mirrors.tuna.tsinghua.edu.cn/python/3.12.9/ 手动下载。

**步骤 2: 静默安装到独立目录 (不污染系统 PATH)**
```bash
# PowerShell 或 Git Bash
./python-3.12.9-amd64.exe /passive \
  "TargetDir=$env:LOCALAPPDATA\Programs\Python312" \
  InstallAllUsers=0 PrependPath=0 Include_test=0 Include_doc=0 \
  Include_launcher=0 InstallLauncherAllUsers=0
```
> 注意: 用 `/passive` 不要用 `/quiet` — `/quiet` 在无桌面会话(如远程/容器)会失败。

**步骤 3: 让 uv 复用这个 Python**
```bash
# 指向刚装的 3.12 (每次 sync 要带 --python, 或设 UV_PYTHON 环境变量)
uv sync --group learning --group dev \
  --python "$LOCALAPPDATA/Programs/Python312/python.exe"

# 一劳永逸: 写入环境变量 (PowerShell)
$env:UV_PYTHON = "$env:LOCALAPPDATA\Programs\Python312\python.exe"
# 或写入用户环境变量持久化
[Environment]::SetEnvironmentVariable("UV_PYTHON", $env:UV_PYTHON, "User")
```

**验证**:
```bash
uv run python --version   # 应输出 Python 3.12.9
```

> 这套方案已在受限网络环境实测通过 (清华镜像下载 25MB/1秒, 静默安装 11秒, uv sync 成功)。
> 关键认知: **PyPI 镜像只托管包, 不托管 Python 解释器**。Python 本体在清华的
> `mirrors.tuna.tsinghua.edu.cn/python/` (独立镜像站), 不在 `pypi.tuna.tsinghua.edu.cn`。

## 依赖组说明

依赖按用途分组（PEP 735 `dependency-groups`）：

| 组 | 用途 | 安装命令 |
|---|---|---|
| (核心) | 评分引擎 scoring/profiles/models | `uv sync` |
| `learning` | 权重学习子模块 (numpy/jax/numpyro) | `uv sync --group learning` |
| `dev` | 测试 (pytest) | `uv sync --group dev` |
| `fastapi` | local-deploy 监控服务 | `uv sync --group fastapi` |

可组合：`uv sync --group learning --group dev`

**注意**：不装 `learning` 组时，核心评分功能（`scripts/evaluate.py` 默认手工权重）仍可用。
`learning` 组只在跑 `scripts/learn_weights.py` 或对应测试时需要。

## 常用命令

```bash
# 加新依赖
uv add <package>                          # 加到核心
uv add --group learning <package>         # 加到 learning 组

# 更新依赖
uv lock --upgrade                         # 重新解析锁定
uv sync                                   # 按锁文件同步

# 跑任意命令 (自动用 .venv)
uv run python scripts/evaluate.py
uv run pytest tests/test_learning_data.py -v
uv run python scripts/learn_weights.py --source synthetic
```

## 验证清单

环境配好后，依次跑这三条，每条都应成功：

### 1. 依赖装好
```bash
uv run python -c "import numpy, sklearn, jax, numpyro, pytest; print('依赖 OK')"
```
**预期输出**：`依赖 OK`

### 2. 测试全绿
```bash
uv run pytest tests/ -v 2>&1 | tail -20
```
**预期**：约 48 个测试全部 PASSED。
`test_learning_model.py` 的 NUTS 采样测试可能慢（每个几秒），总共 1-2 分钟。

### 3. 权重学习 pipeline 可跑
```bash
uv run python scripts/learn_weights.py --source synthetic --n 2000
```
**预期**：打印 sanity check 报告，MAE < 0.05，判定"✓ 通过"，并生成 `data/learned_weights.json`。

## 失败排查

### `uv sync` 报 TLS handshake / connect 错误
→ Python 下载源不通，用上面方案 B 或 C。

### jax/numpyro 安装失败 (Windows wheel 问题)
单独装指定版本：
```bash
uv add --group learning "jax==0.4.30" "numpyro==0.14.0"
```

### NUTS 采样测试超时或 R-hat > 1.05
增大采样参数：
```bash
uv run python scripts/learn_weights.py --source synthetic --n 2000 \
    --warmup 800 --samples 1500 --prior-sigma 0.3
```

### `test_learning_model.py` 全部 skipped
说明 `learning` 组没装。确认：
```bash
uv sync --group learning --group dev
```

## 实测验证记录

本环境的配置已在受限网络（GitHub 和代理全不通，仅清华镜像可用）下实测通过：
- 方案 C 从清华镜像下载 Python 3.12.9（25MB，1秒）
- 静默安装到 `$LOCALAPPDATA/Programs/Python312`（11秒）
- `uv sync --group learning --group dev` 成功，62 个包全部解析安装
- `uv run pytest tests/` **49/49 全部通过**
- `uv run python scripts/learn_weights.py --source synthetic` 完整跑通：
  sanity check MAE 0.0388 通过，5/6 profile USE_LEARNED（Planner 正确 FALLBACK）
- `uv run python scripts/evaluate.py --weights auto` 生成带 CI 的报告

所有步骤在沙箱内已验证，你按上面方案 A/B/C 任一配置好网络即可直接复现。
