# Harness 多模态虚假信息检测 MVP

基于 Harness 控制论框架的多模态虚假信息检测最小可用系统（PoC）。通过 HTTP 微服务集成：

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

# 单张图 Harness 流程
python scripts/run_pipeline.py --image data/test_cases/military_demo_01.jpg

# Baseline 对照
python scripts/run_pipeline.py --image data/test_cases/military_demo_01.jpg --baseline

# A/B 消融实验
python scripts/run_ab_test.py
```

结果输出至 `outputs/` 和 `outputs/ab_test/summary.csv`。

### 离线自检

```bash
python scripts/validate_offline.py
python scripts/test_fusion_policy.py
```

更新 `data/knowledge_base.json` 或 `mocks/rag_server.py` 后，需 **重启 Mock RAG 服务** 以加载新配置。

## 架构

```
输入图片 → [上下文管理] → Step1 Qwen 描述 → Step2 RAG 锚点
         → Step3 TruFor 分数 → Step4 Qwen 融合 → 评估接口 → 报告
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
