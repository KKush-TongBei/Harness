# Harness 多模态虚假信息检测 MVP

基于 Harness 控制论框架的 **图+文整体新闻真伪检测** 最小可用系统（PoC）。输入为新闻配图 + 标题/正文/来源，输出 verdict 与 issue_type。

- **Qwen3-VL-2B**（:8000）— 视觉理解与融合判决
- **TruFor**（:8001）— 图像伪造检测传感器
- **Mock RAG**（:8002）— 本地关键词知识库检索

## 快速开始

### 1. 安装 Harness 依赖

```bash
cd /Applications/School/computer/Harness
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 准备模型权重

**Qwen3-VL-2B**（`/Applications/School/Qwen-VL-master`）：

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B
```

**TruFor**（`/Applications/School/computer/temporary/TruFor/test_docker`）：

```bash
bash docker_build.sh   # 自动下载权重到 weights/
```

### 3. 启动三个微服务（各开一终端）

```bash
# Terminal 1 — Mock RAG
cd /Applications/School/computer/Harness
source .venv/bin/activate
uvicorn mocks.rag_server:app --host 127.0.0.1 --port 8002

# Terminal 2 — Qwen
cd /Applications/School/Qwen-VL-master
QWEN_CPU=true uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 3 — TruFor
cd /Applications/School/computer/temporary/TruFor/test_docker/src
TRUFOR_DEVICE=cpu uvicorn api_server:app --host 127.0.0.1 --port 8001
```

### 4. 健康检查

```bash
bash scripts/start_services.sh
```

### 5. 生成测试用例并运行

```bash
python scripts/generate_test_cases.py

# 图+文新闻条目（从 metadata 加载）
python scripts/run_pipeline.py --case-id military_demo_01

# 手动指定标题/正文
python scripts/run_pipeline.py \
  --image data/test_cases/military_demo_01.jpg \
  --headline "某军区举行公开日" \
  --body "主战坦克向民众展示" \
  --source "新华社"

# Baseline 对照
python scripts/run_pipeline.py --case-id military_demo_01 --baseline

# A/B 消融实验（18 条均含 headline/body/source）
python scripts/run_ab_test.py
```

### issue_type 说明

| issue_type | 含义 |
|------------|------|
| `matching` | 图文一致、来源可信 |
| `manipulated_image` | 图像被篡改 |
| `text_image_mismatch` | 标题/正文与画面不符 |
| `new_text_old_image` | 新文旧图（移花接木） |
| `misleading_text` | 图真但标题/正文造谣或夸大 |
| `fabricated_both` | 图文均伪造 |

结果输出至 `outputs/` 和 `outputs/ab_test/summary.csv`。

### 离线自检

```bash
python scripts/validate_offline.py
python scripts/test_fusion_policy.py
python scripts/test_news_post.py
```

更新 `data/knowledge_base.json` 或 `mocks/rag_server.py` 后，需 **重启 Mock RAG 服务** 以加载新配置。

## 架构

```
输入（图+标题/正文/来源）→ Step1 图文理解 → Step2 RAG → Step3 TruFor → Step4 融合 → issue_type + verdict
```

## 目录结构

```
harness/          # 五组件：context, tools, state, loop, evaluation
mocks/            # Mock RAG FastAPI
data/             # 知识库 + 测试用例
scripts/          # 启动、流水线、消融脚本
outputs/          # 运行结果（gitignore）
```

## 分支

实验代码在 `harness-test` 分支。
