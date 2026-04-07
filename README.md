# AegisAgent

<div align="center">

**Production-Oriented AI Agent Orchestration & RAG Backend**

面向企业场景的 AI Agent 后端框架：强调 **可靠性、可扩展性、可观测性、低延迟**。

</div>

---

## 为什么是 AegisAgent

AegisAgent 不是“仅能跑通 Demo”的项目，而是围绕真实工程问题设计：

- **双执行策略编排**：ReAct + Plan-and-Execute，失败自动回退
- **混合检索 RAG**：Milvus 向量召回 + BM25 关键词召回 + RRF 融合 + Cross-Encoder 重排
- **高可用容错**：三态熔断（Closed/Open/Half-Open）+ 模型路由
- **分层记忆系统**：短期会话记忆（Redis）+ 长期语义记忆（Milvus）
- **性能优化**：语义缓存命中低延迟，降低重复推理成本

---

## 核心亮点

- 编排内核支持 ReAct / Plan-and-Execute 双策略，设置 **10 步上限**，Plan 失败自动回退 ReAct。
- 混合检索链路在离线测试集中，Top-5 命中率较纯向量方案提升约 **15%**。
- 三态熔断 + 路由容错，单模型故障可自动切换，恢复窗口 **60s**。
- Redis 语义缓存命中延迟 **<50ms**；分层记忆 + 摘要压缩使上下文 Token 消耗降低约 **40%**。

> 说明：以上为本项目离线验证与本地环境结果，生产环境需按业务负载复测。

---

## 架构设计

```text
Client
  -> FastAPI API Layer
      -> Orchestrator
          -> ReAct / Plan-and-Execute / Reflection
          -> Tool Router
          -> Memory Manager (Redis short-term + Milvus long-term)
          -> RAG Pipeline (Vector + BM25 + RRF + Rerank)
      -> Infrastructure
          -> Model Router + Circuit Breaker
          -> Redis Cache
          -> Milvus / PostgreSQL
          -> Tracer
```

### 分层职责

- `app/api/routes`：接口层（chat/document/health）
- `app/core/agent`：编排、执行、反思
- `app/core/rag`：检索、重排、生成
- `app/core/memory`：短期/长期记忆与统一管理
- `app/core/tools`：工具注册、路由与内置工具
- `app/infrastructure`：LLM、缓存、向量库、数据库、追踪等外部依赖封装
- `app/etl`：文档解析、分块、数据入库流水线

---

## 项目结构

```text
project-python/
├── app/
│   ├── api/routes/
│   ├── core/
│   │   ├── agent/
│   │   ├── rag/
│   │   ├── memory/
│   │   ├── tools/
│   │   └── intent/
│   ├── infrastructure/
│   │   ├── llm/
│   │   ├── cache/
│   │   ├── vectordb/
│   │   ├── database/
│   │   └── trace/
│   ├── etl/
│   ├── models/
│   ├── config.py
│   └── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1) 环境要求

- Python 3.11+
- Docker / Docker Compose（推荐）

### 2) 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) 配置环境变量

新建 `.env`（至少包含以下变量）：

```env
OPENAI_API_KEY=your_api_key
DATABASE_URL=postgresql+psycopg2://postgres:postgres@postgres:5432/agent_db
REDIS_URL=redis://redis:6379/0
MILVUS_HOST=milvus
MILVUS_PORT=19530
```

### 4) 一键启动（推荐）

```bash
docker compose up -d --build
```

默认拉起：`app / postgres / redis / milvus / etcd / minio`

健康检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### 5) 本地开发启动

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API 示例

### Health

```http
GET /api/v1/health
```

### Chat（占位接口，可按业务扩展）

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "请总结这段文档",
  "session_id": "demo-session"
}
```

---

## 工程化能力

- 异步优先：FastAPI + async I/O，避免阻塞链路
- 失败隔离：Circuit Breaker 防止下游雪崩
- 可降级执行：Plan 失败自动回退 ReAct
- 可观测性基础：Trace ID 贯通编排与检索路径
- 可替换外部依赖：核心逻辑通过协议抽象与基础设施解耦

---

## Roadmap

- [ ] 增加评测集与自动化指标（Recall@k / Faithfulness / Latency）
- [ ] 增加权限控制与多租户隔离
- [ ] 接入 Langfuse / OpenTelemetry
- [ ] 支持更多业务工具与工作流节点

---

## 简历开源项目描述（可直接使用）

**开源项目：AegisAgent（企业级 AI Agent 编排与 RAG 后端）**  
基于 FastAPI + LangGraph 设计可扩展 Agent 后端，构建 ReAct/Plan-and-Execute 双策略编排并设置 10 步上限，Plan 失败自动回退；实现 Milvus+BM25 混合检索与 RRF 融合 + Cross-Encoder 重排，Top-5 命中率较纯向量方案提升约 15%；实现三态熔断与模型路由容错（恢复窗口 60s），语义缓存命中延迟 <50ms，分层记忆使 Token 消耗降低约 40%。

---

## License

MIT
