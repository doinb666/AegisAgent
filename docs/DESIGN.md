# 工作台视觉方案

使用浅色办公工作台：暖灰背景、墨色正文、低饱和绿色强调；中文本地无衬线字体。桌面左导航、中央对话、右侧任务审查；移动端一列，导航可切换。正文最长 72ch。

色彩以 CSS 变量和 OKLCH 定义；按钮、输入、列表共享圆角和 focus 样式。状态包括 idle、loading、running、waiting_approval、completed、failed、cancelled、interrupted 与 reconnecting。审批展开在任务区，内容以纯文本展示。动画仅用于 150–200ms 状态过渡，减少动画设置禁用。

首版使用同源静态 HTML/CSS/JS，避免 EXE 安装依赖 Node。用户输出与工具参数通过 textContent 渲染，禁止不可信 HTML。持久事件以 ID 去重；网络失败重试读取事件，不重发任务创建。
