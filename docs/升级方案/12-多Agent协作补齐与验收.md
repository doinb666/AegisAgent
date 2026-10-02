# 多 Agent 协作补齐与验收

日期：2026-10-02。承接02中已批准的A16协作、安全路径与恢复契约；用户本轮再次授权继续编码、逐项验收后提交GitHub。不扩大子权限、不引入任意Agent通信、不修改数据库schema、不自动合并或推送用户仓库。

## 执行方式与事实

采用模块B增量实施：先复现，后最小修复，专项验收、独立规格、独立质量复核，再原子提交并尝试普通推送。新架构及生产发布另立方案。本轮执行subagent-driven-development；使用项目`.venv/Scripts/python.exe`，临时SQLite和确定性模型不代表真实商业模型效果。

正式8001首页及readiness均HTTP200，服务保持运行；原8000企业容器不覆盖。Git代理7897无监听，直连GitHub的ls-remote也超时，仍继续本地实施；不强推、不索取访问Token。

## 技能路由

find-skills已扫描全局目录与项目目录；项目无`.codex/skills`，`npx skills find 'agent orchestration'`返回20项生态结果，未安装外部技能。本地候选如下：

| 候选技能 | 功能/场景 | 本轮使用与理由 |
|---|---|---|
| find-skills | 查找与匹配能力 | 是，完成语义检索与目录扫描 |
| writing-plans | 拆分实施与验收 | 是，沿已有A16记录模块输入输出 |
| code-refactor | 依赖与边界审计 | 是，复核当前协作实现而非重写 |
| systematic-debugging | 根因与复现 | 是，真实Windows路径及SQLite恢复问题 |
| test-driven-development | 红绿回归 | 是，先验证缺口再修复 |
| code-simplifier | 重复代码与校验收敛 | 是，统一路径策略及协作清理 |
| subagent-driven-development | 实施与两阶段审查 | 是，每模块独立实现及复核 |
| requesting-code-review | 质量审查 | 是，安全和恢复不能只靠自检 |
| verification-before-completion | 查验结果后交付 | 是，逐项记录命令、失败和限制 |
| Git Smart Commit | 按功能提交 | 是，用户已明确授权提交和推送 |
| webapp-testing | 实际浏览器验收 | 是，前端可访问及既有流程复核 |
| frontend-design | 界面结构与样式 | 条件启用，实际修改UI时使用 |
| ui-ux-pro-max | 交互与可访问性 | 条件启用，与前端三组合同时使用 |
| impeccable | 产品体验与一致性 | 条件启用，与前端三组合同时使用 |
| framework-selection | Agent框架选择 | 否，本轮修已有Harness，不换框架 |

Sequential Thinking、DuckDuckGo、Context7、Serena未加载，不冒称MCP调用。采用本地rg、Git、pytest与浏览器工具；所有诊断只用临时数据，不操作正式用户任务。

## 能力现状

已有：主控工具委派、最多两子任务、禁止递归、独立子执行池、同tenant/owner身份、只读子权限及父权限子集校验、显式取消级联、有界Fork上下文、管理员绑定仓库的副本与detached Worktree。

部分：主控汇总的预算、等待与完整结果引用；父异常恢复后的子任务清理；Windows工具/归档路径与预览策略一致性。

未实现：子结果的确定性质量验收门、任务依赖DAG、子只读代码工具与父Worktree联动、任意团队消息及自动Git合并。后二项不属于当前A16要求；其余扩展不能用“Team”名字冒称已具备。

## M1：工具及仓库路径校验

- 文件：`app/harness_tools/workspace.py`、`project.py`、`tests/test_workspace_tool_boundary.py`。
- 输入：本人任务相对路径、管理员仓库归档；输出：合法中文嵌套文件或明确拒绝。
- 已复现：Windows `.git.`/`.git `读取`.git`；ZIP `.git.`覆盖已有Worktree元数据。当前预览的`_path_parts`已严格校验，可复用。
- [x] 红测：初始74项中62项失败；追加ZIP原始反斜杠、NUL与链接目录项3项失败。覆盖真实Windows别名、ADS、设备名、静态junction；普通中文路径保留。
- [x] 最小实现：工具和归档共用片段规则，目录ZIP项处理合法尾斜线；检查中间及工作区目录静态链接，拒绝重解析点，检查ZIP原始名称。不宣称防不可信宿主TOCTOU。
- [x] 专项：root复跑新增测试、workspace inspection、工具与真实临时Git Fork/Worktree：114通过、1项旧Windows符号链接权限跳过（22.99秒）。独立规格91项通过、独立质量79项通过；Ruff与差异检查通过。
- [ ] 提交并普通推送，记录远端核对或具体阻塞。

## M2：父子恢复生命周期

- 文件：`app/harness/store.py`、必要的`runtime.py`及新增恢复测试；不变schema。
- 输入：过期父租约、started delegate、queued/running子；输出：父interrupted，子不得成为孤儿执行。
- 已复现：`claim(child_only=True)`将父恢复为interrupted后仍把其子领取为running。
- [ ] 红测：父终态/过期started、活跃子、完成子、其他owner隔离及双Store竞争。
- [ ] 最小实现：同事务清理终态父活跃子；领取时核对并锁住同作用域父状态，保留started工具未知结果的interrupted语义；运行子在父失效后有界停止。
- [ ] 专项与独立两阶段审查；已有正常并发池、取消和完成检查点复用不回归。
- [ ] 提交并普通推送；PostgreSQL多副本未实测时明确单列。

## M3：委派预算、等待及结果

- 文件：`app/harness_tools/catalog.py`、必要的`app/harness/service.py`、协作回归测试；可提取小的协作模块，避免目录类继续增长。
- 输入：一或两个只读任务、父可用预算与权限、fork/team；输出：真实子终态、简短预览及完整结果引用。
- 已复现：父7步、两子各只用1步仍因预留6步无法汇总；6步时第二子创建预算失败；20秒等待后返回queued但实际取消；5000字子答案只给2000字且没有完整artifact引用；受限父固定四工具被安全拒绝。
- [ ] 红测：7步两子完成可汇总；不足时创建前返回失败且零新增子；权限交集、等待超时真实状态、长答案完整读取、他人拒绝。
- [ ] 最小实现：整批预算预检，为主汇总至少保留一步；按剩余预算分配子步骤，继续遵守两子总量；子工具取父已授权只读目录交集；等待受父时间/工具预算约束，超时先取消后返回实际状态；长答案保存带来源的artifact并返回引用。
- [ ] 专项、主控真实Worker集成及独立两阶段审查；保持失败不伪造成功、控制权不移交、未知副作用不重放。
- [ ] 提交并普通推送。

## 总体验收与发布边界

- [ ] 全套pytest、实际修改范围Ruff、全仓F类、compileall、diff与文档链接检查。
- [ ] 按最终代码验证源码工作台与浏览器；只有包装代码变化或安装产物更新时重新构建并运行验证，不拿旧包当新能力。
- [ ] 更新README、使用手册、进度与提交记录；正式8001加载最新后端前先核对无活动任务，保留原数据。
- [ ] 停本轮独立测试服务，保留正式8001和原8000企业容器、所有持久数据。

基线tools/runtime：45 passed、1 skipped，28.50秒。两只读审查另行点选7项/6项均通过，但缺口未被旧测试覆盖，不能据此把A16判完整通过。真实模型效果、共享限流、灾备、企业共享技能与在线训练仍未完成。
