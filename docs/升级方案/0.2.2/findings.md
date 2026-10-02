# 发现与决策

## 需求

用户要求持续实现、验证、GitHub更新，打开类似Codex的企业工作台，并针对整个目录剔除冗余代码。原身份、沙箱、MCP、记忆、Skill与恢复边界继续有效。

## 代码发现

正式app/main.py只挂harness与health。当前8000为旧Docker服务；8001为本轮正式源码服务，root/data为持久目录。没有根目录.env文件，模型可用性仍须从真实capabilities确认，不从页面外观推断。

旧core/agent、core/memory、infrastructure/cache无外部消费者；部分tools与rag模块只有__init__附带导入。正式KnowledgeService依赖MilvusManager、Reranker与RetrievalResult。ETL隔离worker仍经ETLPipeline使用parser/chunker。旧chat/document未挂载，但用户的独立Gradio入口引用其HTTP路径，保留兼容源文件，不作为已可用产品入口。

## 来源状态

2026-10-01本轮实际请求https://developers.openai.com/codex/app/features/，代理握手超时，trust_env=False直连403；没有取得有效页面或官方截图。布局依据用户指定的任务工作方式与本项目实际数据，不宣称官方功能全量复刻。

find-skills `agent dashboard`返回20个生态候选，没有安装外部技能；本地目录扫描与18项矩阵见09。ui-ux-pro-max设计查询命中Flat Design，但推荐的产品营销页结构不符合登录工作台，未采纳Hero/video方案；取紧凑布局和可访问性建议。

## 用户已有未关联变动

requirements.txt新增gradio；app/gradio_demo.py、docker-compose.gradio.yml、3分钟项目介绍、Trace-ID说明、Gradio启动说明、项目问答汇总未跟踪；docs/assets/README.md原有删除。清理需求不被解释为可覆盖这些修改，分批提交保留它们。

## 2026-10-02 Hermes官方核验与简历边界

用户追加按最新源码改写Agent工程师简历。已读取本地resume-builder及写作、岗位参考；只交付文字，不自动创建简历文件。

直连以下官方文档均HTTP 200，来源为Nous Research，核验日期2026-10-02：

- https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- https://hermes-agent.nousresearch.com/docs/user-guide/features/skills
- https://hermes-agent.nousresearch.com/docs/developer-guide/architecture

Hermes文档说明：MEMORY.md与USER.md分离、会话开始时冻结快照以稳定前缀；Skill按目录元信息、正文、参考资源渐进加载；后台复盘沉淀记忆和Skill，并提供写入审批；会话持久化与Profile隔离。上述是外部产品机制，不代表AegisCode实现了同样后端或效果。

本项目对应实现：assets.py两阶段有界词法召回（500候选、32粗排、8返回），skill_files.py声明式文件包与资源读取，evolution.py有租约和指纹去重的后台草稿提炼，runtime.py工具结果artifact外置与上下文预算，store.py任务租约与副作用未知态。当前记忆事实源是SQL，不是Hermes文件或Redis/Milvus分层记忆。

简历可写受控推理时自进化、Skill版本与CAS冲突保护、精确参数审批、SSE重连与恢复、中心化只读子任务、可选混合RAG；不写在线RL/LoRA、自动技能回放晋升、完善安全防御或已提高推理准确率。Plan实现是预生成JSON计划再进入执行循环，不称复杂动态Plan-and-Execute执行器。Prompt Cache只有稳定前缀设计，未测收益。

LangChain/LangGraph/Redis已从当前主线移除；Claude Code/Hermes为设计参考，不写成直接使用的技术依赖。原15%、40%、50ms缺少本项目基准，不用于简历成果。

## 本机服务恢复

接续时旧8001/8772监听与Docker引擎均不存在；已恢复正式8001（原data保持），另用data/ui-workbench-023启动8772确定性测试模型。readiness分别200。Docker恢复后运行容器重新可见，PyPI的SQLAlchemy索引在容器HTTP200；再次尝试默认构建，结果以实际日志为准。
