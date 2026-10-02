# 0.2.3 发现

- GitHub HTTPS 直连恢复；单次 `-c http.proxy=` 的 ls-remote 和 push 成功，main 更新至 aab23df。未改全局代理。
- 日志最后三次 422 均在 `/api/v1/auth/register`；注册按钮未进行表单有效性检查；错误数组被前端隐藏。具体字段输入未回显也未验证。
- 现有 8001 readiness 数据库 up，旧源码服务运行。既有 EXE/wheel 为 0.2.2，尚未更新或公开 Release。
- 浏览器打开工具失败：`failed to start codex app-server: 系统找不到指定的路径`。不将工具失败声称为界面已打开。
- 用户指定 README 突出安装、使用与亮点，不展示开发流水账；参考截图为浅色极简编码工作台。
- 子任务当前没有 file_read，不能读取父代码；委派字符串数组无依赖契约，也没有独立结果证据验收门。
