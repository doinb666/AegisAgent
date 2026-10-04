# 0.2.4 任务计划

- [x] 父子工作区只读联动、两节点依赖图和独立账本验收。
- [x] OpenAI 兼容、Anthropic、Azure、Ollama 与自定义模型来源。
- [x] 正式工作台迁移为 TypeScript + Vite，保留 HTML/CSS 与 SSE。
- [x] 注册负例、协作界面和完整工作台浏览器验收。
- [x] 清理可重建构建产物和旧验收数据，保留正式数据与发布物。
- [x] 完整 Python 回归、静态检查与浏览器验收。
- [x] 分功能提交并同步 GitHub main。
- [x] 构建并验证 0.2.4 wheel、便携版与 Windows 安装器。
- [x] 发布 0.2.4 Release，更新 README 下载链接。

本计划覆盖 0.2.4 交付，不代表在线训练、自动 PR/合并和大规模集群容灾已经完成。真实模型与 Milvus 质量评测、进一步并发性能优化及旧正式进程重启仍需继续处理；具体边界见 progress.md。

执行约束：保护用户未提交的 Gradio、requirements 和独立文档改动；发布只从已提交 Git 快照构建。
