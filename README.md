# AegisAgent · AegisCode

**把问答、代码任务、长期记忆与技能放在一个可审查的工作台。**

AegisCode 面向个人开发者和企业成员：描述目标，由 Agent 规划、调用工具并保留执行证据；需要写文件、运行代码或调用外部服务时，由用户批准。经过审阅的偏好、约束和任务经验可以跨会话复用。

[安装与启动](#安装与启动) · [模型配置](#模型配置) · [使用方式](#使用方式) · [项目亮点](#项目亮点) · [使用手册](docs/使用手册.md)

![AegisCode 工作台](docs/升级方案/截图/0.2.3/00-浅色空工作台.png)

## 安装与启动

| 方式 | 适用场景 | 需要准备 |
|---|---|---|
| Windows EXE / 安装器 | 日常桌面使用 | Windows；模型服务配置；发布状态见下文 |
| Python 源码 / wheel | 开发者、跨平台本地运行 | Python 3.12+ |
| Docker 个人版 | 简单部署和持久化 | Docker Compose；SQLite 数据卷 |
| Docker 企业版 | PostgreSQL 持久化部署 | Docker Compose；数据库密码；自己的 TLS 入口 |

基础工作台不要求 Redis、Milvus 或 Node.js。使用模型需要可用的模型服务；执行代码需要单独配置 Docker 沙箱。

### Windows 桌面使用

当前版本 **0.2.3**：[版本说明与全部下载](https://github.com/doinb666/AegisAgent/releases/tag/v0.2.3)。

- [Windows 安装器 EXE](https://github.com/doinb666/AegisAgent/releases/download/v0.2.3/AegisCode-Setup.exe)：安装到空目录后双击 AegisCode.exe。
- [Windows 便携 ZIP](https://github.com/doinb666/AegisAgent/releases/download/v0.2.3/AegisCode-portable.zip)：解压并保留完整目录。
- [Python wheel](https://github.com/doinb666/AegisAgent/releases/download/v0.2.3/aegiscode-0.2.3-py3-none-any.whl)：适合 Python 用户。
- [SHA256 校验和](https://github.com/doinb666/AegisAgent/releases/download/v0.2.3/SHA256SUMS.txt)：使用 PowerShell `Get-FileHash 文件名 -Algorithm SHA256` 核对。

如需自行构建，在 Windows 执行：

```powershell
git clone https://github.com/doinb666/AegisAgent.git
cd AegisAgent
python -m venv .venv
.venv/Scripts/python.exe -m pip install ".[dev]"
.venv/Scripts/python.exe scripts/build_windows.py
```

构建后任选一种方式：

- 双击 `dist/AegisCode-Setup.exe`，按安装器提示安装。
- 解压 `dist/AegisCode-portable.zip`，双击其中的 `AegisCode.exe`。保留整个目录及 `_internal`。

EXE 默认仅监听本机，自动打开浏览器；配置与数据默认存放在 `%LOCALAPPDATA%/AegisCode/`。在该目录创建 `.env`，配置模型后重启。支持 `--port`、`--data-dir`、`--config`、`--no-browser`。安装包不包含模型权重，当前构建未做代码签名。

### Python 源码启动

```bash
git clone https://github.com/doinb666/AegisAgent.git
cd AegisAgent
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-harness.txt
Copy-Item .env.example .env
# 填写自己的模型配置，见下文
.venv/Scripts/python.exe -m app.launcher
```

Linux / macOS：

```bash
source .venv/bin/activate
pip install -r requirements-harness.txt
cp .env.example .env
# 填写自己的模型配置，见下文
python -m app.launcher
```

浏览器访问 **http://127.0.0.1:8000**，先创建个人账号再登录。密码至少 8 个字符。8000 已占用时：

```powershell
.venv/Scripts/python.exe -m app.launcher --port 8001
```

wheel 构建与安装：

```bash
python -m pip install ".[dev]"
python scripts/build_wheel.py
# 安装 dist 中实际生成的 aegiscode-版本号-py3-none-any.whl
python -m pip install dist/aegiscode-0.2.3-py3-none-any.whl
aegiscode
```

### Docker 部署

先复制 `.env.example` 为 `.env` 并填写模型配置。

个人版使用 SQLite：

```bash
docker compose -f docker-compose.personal.yml up --build -d
```

企业版使用 PostgreSQL，先在 `.env` 设置自己的 `POSTGRES_PASSWORD`：

```bash
docker compose up --build -d
```

默认访问 http://127.0.0.1:8000，数据通过卷持久化。管理员在「能力与设置」创建企业成员，支持管理员、操作员和只读成员。远程部署通过自己的 TLS 入口访问；多副本部署还需配置共享限流、备份和沙箱隔离。

升级时备份配置与数据，更新源码后重新构建并启动。不要删除数据卷；现有任务出现 `interrupted` 时先核对工具副作用，避免重复执行。

## 模型配置

最小 `.env` 示例：

```dotenv
OPENAI_API_KEY=替换为自己的密钥
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=替换为可用且支持工具调用的模型
AEGIS_DATA_DIR=data
```

支持多个 OpenAI 兼容模型服务，按优先级切换；每个模型路由具有独立熔断状态。具体配置见 [安装与运维](docs/升级方案/06-安装与运维.md)。密钥只放环境变量或本地 `.env`，不要提交到 Git。

未配置模型时仍可登录、管理记忆和资料；执行任务会明确提示失败。模型列表表示已配置，连通性需实际调用验证。

## 使用方式

1. **建立个人偏好**：在「长期记忆」保存回答风格、约束和边界，审阅后启用。再次登录会加载本人有效偏好，任务开始时按需召回相关经验。
2. **提供资料和项目背景**：在「知识库」上传 TXT、Markdown 或文本 PDF；在「项目」保存目标和约束，再创建任务。扫描 PDF 需先 OCR。
3. **执行任务**：描述目标与验收要求，选择直接执行或先规划。查看任务进度、完整工具证据和本人工作区文件；有副作用的操作逐次审批。
4. **复用与改进技能**：任务结束后反馈成功或失败，在「Skills」审查候选经验的来源、适用条件和边界，再启用、修订或退役。可导入导出 SKILL.md 及文本资源包。
5. **恢复长任务**：断开页面连接不会取消后台任务，重新打开后恢复持久事件。需要停止时使用「停止任务」；未知副作用状态需人工核对。

可以从以下任务开始：

```text
根据我的知识库回答问题，列出来源，并指出证据不足的部分。
审查代码中的空输入、异常处理和权限边界，先收集证据再给结论。
将本次失败原因整理成技能候选，写清触发条件、修复步骤和验证方法。
```

Skill 快速体验：[下载示例 SKILL.md](docs/examples/skills/review-python/SKILL.md)，在「Skills」导入并审阅启用。资源只作为文本读取，不会自动执行脚本。

### 多 Agent 和代码工作区

当前主 Agent 可通过工具委派最多两个只读子任务，统一汇总结果；子权限取父权限交集，禁止递归和危险操作，预算为主控汇总预留空间。

- **Fork**：继承有界会话上下文，适合从同一背景分头分析。
- **Agent Team**：独立上下文的受控子任务，由主 Agent 汇总证据。
- **Worktree / 代码副本**：经审批从管理员绑定仓库准备独立目录；Worktree 使用 detached 方式，不自动合并用户分支。

当前仓库准备与子任务读取仍是独立能力；父子代码目录联动、任务依赖图及独立结果验收门正在升级，尚不能作为完整协作编码团队使用。管理员仓库配置、工具调用和操作步骤见 [使用手册](docs/使用手册.md)；新增设计见 [协作升级方案](docs/升级方案/13-协作代码任务与鲁棒性验收.md)。

## 项目亮点

- **受控任务闭环**：工具调用与持久任务结合，支持规划失败回退、默认 10 步预算、反思反馈和全链路 Trace ID。
- **个人长期记忆与技能积累**：成功与失败轨迹提炼为偏好、约束、经验和 Skill 草稿；来源追溯、去重、版本修订和退役形成跨会话复用闭环。候选经审阅后生效，权限规则不参与自进化。
- **分层上下文管理**：大工具结果外置保存，保留摘要与读取引用；近期窗口和结构化历史摘要控制长会话上下文。缓存与 Token 收益以实测为准。
- **中心化多 Agent**：主控保留规划和审批权；子运行只读、有限预算、父权限子集，父停止后回收活跃子，长结果可按引用读取全文。
- **权限与安全边界**：租户与用户隔离、角色控制、工具参数校验、模型风险信号和精确参数人工审批；代码只在配置的 Docker 沙箱执行。
- **外部 MCP 与知识检索**：官方 MCP SDK 接入运维批准的服务；知识库使用 BM25，可选 Milvus 向量、RRF 融合和 Cross-Encoder 重排，依赖故障明确降级。
- **恢复与模型容错**：幂等任务创建、Worker 租约、检查点、取消、SSE 游标重连；多模型路由与三态熔断，60 秒恢复探测窗口。

自进化指推理时的记忆与技能积累，当前不会在线更新模型权重。上述机制不保证每次任务成功，未知副作用不会自动重放。

## 当前技术栈

| 层次 | 技术 |
|---|---|
| 后端与执行内核 | Python 3.12、FastAPI、异步任务执行、原生工具调用 |
| 存储 | SQLAlchemy async、SQLite（个人版）、PostgreSQL（企业版） |
| 模型与外部工具 | OpenAI 兼容 API、MCP SDK、多模型路由与熔断 |
| 可选知识检索 | Milvus、BM25、RRF、Cross-Encoder |
| 工作台 | 同源 HTML / CSS / JavaScript、SSE |
| 安装与隔离 | Docker Compose、Docker 执行沙箱、PyInstaller、Python wheel |

当前主链路采用自主 Harness；LangChain、LangGraph、Redis 不再是当前核心依赖。完整使用方法见 [用户手册](docs/使用手册.md)，安全部署见 [安装与运维](docs/升级方案/06-安装与运维.md)，能力验证与边界见 [验收记录](docs/升级方案/12-多Agent协作补齐与验收.md)。

## 常见问题

- **注册提示字段错误**：账号不能为空，密码至少 8 字符。界面会先校验并显示具体原因；浏览器缓存旧界面时请强制刷新。
- **没有模型或任务失败**：填写模型配置并重启，确认该模型支持工具调用和你的账号权限。
- **无法执行代码**：配置并验证 Docker 沙箱；未配置时拒绝执行，不降级为宿主运行。
- **看不到仓库工具或 MCP**：它们由运维配置与用户授权决定，不能通过普通对话添加任意宿主路径或服务地址。
- **下载或启动安装器**：使用上方 0.2.3 Release 链接并核对 SHA256；便携版不能只复制一个 EXE。8000 已占用时添加 `--port 8001` 启动。

API 文档位于 `/docs`，数据库就绪探针为 `/api/v1/health/ready`。业务 API 需要登录令牌，创建任务需要 `Idempotency-Key`。

包元数据标记 MIT，仓库尚缺 LICENSE 正文；正式使用和分发前请核对权属及许可。
