# Skill修订审阅与反馈隔离验收

日期：2026-10-07。对应[第44号契约](44-Skill修订草稿审阅实施契约.md)，后台修订见[第42号](42-Skill反馈修订与幂等验收.md)。本批为源码增量，公开0.2.9安装包不包含该审阅入口。两项规格问题已关闭，规格、质量复审及主控集成均通过；用户已确认，当前执行中文提交及GitHub同步。

## 本批行为

任务审查提供可选反馈说明，按既有API的4000个Unicode码点限制提交；失败保留说明，同一任务重读或离开视图再返回仍保留说明和有效错误，成功后清空，切换任务或账号清空。旧响应不能清除新一代说明。技能库标注修订草稿、原版本及未验证说明，展开时才读取本人原技能。当前版本、来源和活动状态一致才展示对照；不一致时提示原技能已变化，不伪造生成时原文。

修订建议不自动启用，不替换原技能，不证明失败因果或修复效果。对照采用有界纯文本，原权限仍由服务端判断。现有正文、导入导出、版本恢复和退役入口保留。

## 实际验证进度

- [x] 真实HTTP红例：工具执行、经验提炼、人工启用、真实召回及失败反馈均成功；旧界面缺少修订草稿标识，退出1。
- [x] 实施首轮修订、原内容管理和原技能三模块浏览器合跑退出0，会话5710。真实HTTP与浏览器故障注入分别标注。
- [x] 主控最终前端类型与构建退出0：18模块，app.js为90.66kB、gzip为30.12kB；恢复后再次执行退出0。文件大小不是生产延迟指标。
- [x] 主控检查发现测试夹具格式未通过，实施仅用格式器合并相邻字符串；相关三文件Ruff及格式检查均退出0。
- [x] 4000个表情真实提交并核对持久反馈事件，4001个字符在前端拦截；不能用UTF-16长度替代后端码点计数。
- [x] 实施方深浅390／768／1440、44px、可见焦点与减少动画通过；主控查看真实对照截图，原正文和建议均可见。
- [x] 两项独立规格问题修复后复审通过；独立执行实际eventRow和openRun路径，退出0并核对冻结SHA。
- [x] 独立质量审查通过：TypeScript、Ruff及格式退出0；Vite内存构建与冻结app.js逐字一致；实际组件离线浏览器来源、网络、焦点及过期响应边界退出0，无Critical／Important。
- [x] 主控最终冻结集成与截图复核：会话2002四模块修订／计划／增量输出／文件差异均退出0；源码SHA与下表一致。
- [x] 本轮后台专项重新执行60通过，61.97秒，退出0；不与全量902相加。
- [x] 本机8003重启加载本批源码，首页200、feedback-note存在、数据库ready。重启前只读核对无运行任务，保留原数据库，仅停止已确认的自有进程。
- [ ] 中文提交、GitHub同步及独立远端核对。

## 独立审查发现的问题

| 问题 | 独立证据 | 修复与验收门 |
|---|---|---|
| 完成事件重复投影 | 执行记录先投影为draft_count，再传入只识别draft_assets的提示函数；真实1项草稿显示待核对 | 已关闭：提示先从原事件生成，证据只使用公开投影；完整eventRow确认数量与中文文案，不泄露额外字段 |
| 同任务误清反馈 | openRun对同一Run也无条件清空note与失败说明；点击当前任务或切视图返回都会丢稿 | 已关闭：按任务与账号绑定；同Run重新读取、失败后重新读取及切视图返回保留，迟到结果不清新输入 |

工具名称的额外核查：当前生产SCHEMAS均符合前端ASCII规则；外部MCP工具名位于mcp_call参数，不是运行权限边界中的工具名。本批没有找到真实服务生成含点号或中文权限名的反例，不把假设当作已发生问题。

非阻断Minor：畸形事件中重复的draft_assets ID会重复计数。后端限制最多2项，拒绝重复来源并为每草稿生成独立UUID，正常事件不可达；当前冻结不改动，保留为展示层可选加固。独立离线组件验证不替代主控真实HTTP集成。

## 复现范围与限制

```powershell
npm.cmd run frontend:verify
.venv/Scripts/python.exe scripts/run_ui_acceptance.py --module scripts.skill_revision_browser_acceptance --module scripts.assets_browser_acceptance --module scripts.skill_browser_acceptance
.venv/Scripts/python.exe scripts/run_ui_acceptance.py --module scripts.skill_revision_browser_acceptance --module scripts.plan_browser_acceptance --module scripts.model_output_browser_acceptance --module scripts.file_changes_browser_acceptance
```

验收使用随机本机端口、独立SQLite账号和确定性模型；原文读取及反馈经过真实鉴权API。403／404、网络失败、原版本漂移、畸形元数据、恶意正文和迟到响应另通过明确标注的浏览器故障注入验证，不能据此推断商业模型修订质量。

前三模块回归保留默认配置；四模块组合因计划专项使用20步与20并发夹具，不是生产默认10动作。截图见[截图导览](../截图/技能修订/截图导览.md)，根导览已补第15节并核对实际文件。主控查看1440真实对照；390深色截图无横向溢出，但展开导航与审查使页面较长，截图顶部有滚动裁切，不据此宣称所有正文都在首屏。

后台902项通过、2项平台跳过是第42号完整回归，不能与本批浏览器模块或60项专项相加。真实供应商、生产数据库、Docker新增专项、完整无障碍合规及新版安装包交付本批尚未验证。可见浏览器自动打开受当前认证限制，提供8003访问地址，不把HTTP就绪当作窗口已展示。

## 最终源码冻结

| 文件 | SHA256 |
|---|---|
| `frontend/src/skill-revision-review.ts` | `18C6F80E66DF4FE635D41CEABBFE1BF8F3B486850616192D784ECEF66D1C04D5` |
| `frontend/src/app.ts` | `DE8A30853B373E131AB330EA0E43EF1960893D72647192DFD575FAFD17B45B63` |
| `frontend/src/workspace-ui.ts` | `D8BD56AFE2F8D763A794248ADFF2988C3C10959B9FE0C012C9BDC213A7F832B7` |
| `app/web/index.html` | `A4123A76E1766D51AEB27095DF9B6D5FD1BD303E946F633D4583BAC56DCD401D` |
| `app/web/styles.css` | `C0D349B4619823AD55B46FDEA6628839EF3EB831163AE1C0DE74EA593832F0E2` |
| `app/web/app.js` | `C8D4F4C2AEE8C9AFFEBFFB6AE1C1A526CCA4E35B8F4DB2F413340D9576D1C31F` |
| `scripts/skill_revision_browser_acceptance.py` | `745A0558D7FB2EF9D73088DDFF704578A6C4C18BDD1101BD7A1065552EE030AB` |
| `scripts/run_ui_acceptance.py` | `563876B2AD4D676AF2168429573B3051BF7AA35672234CE78AD8A1F1ABE3F167` |
| `tests/ui_server.py` | `9DAE6AE5114428D1572BB579C2E60AC5CEB0023DABF66912B6938A7FAE964AFB` |

实际清理包括不可达反馈分支、重复截断及网络辅助函数、统一反馈控件重置；保留任务绑定与异步代次边界。
