# TypeScript 前端与目录清理记录

日期：2026-10-03。

## 正式前端

正式界面使用 `frontend/src/*.ts`、Vite、`app/web/index.html` 与 `styles.css`。`npm run frontend:check` 做类型检查，`npm run frontend:build` 生成 FastAPI、Docker、wheel 与 Windows EXE 共用的 `app/web/app.js`。发布构建会先执行两项检查，避免源码和静态产物不一致。Gradio 不属于正式入口。

## 目录用途

| 目录 | 用途 | 是否可整体删除 |
|---|---|---|
| `build/` | PyInstaller、wheel 和分词器构建缓存 | 可；下次构建自动生成 |
| `dist/` | 本地发布产物；`releases/` 保存已发布版本 | 不整体删除；可清理重复的根目录产物 |
| `data/` | 正式账号数据库、任务工作区及本地验收数据 | 不可整体删除；先停服务、备份并识别来源 |
| `scripts/` | 安装、构建、恢复、沙箱和浏览器验收入口 | 不可；发布与验收使用 |
| `tests/` | 权限、租约、幂等、模型协议及协作边界回归 | 不可；生产改动的安全网 |

## 本轮清理

`build/`、`dist/` 根目录重复的旧 EXE／ZIP／wheel、旧安装与浏览器验收副本及 Python 缓存已清理；`dist/releases/v0.2.3/`、正式 `data/aegis.db*` 和当前浏览器验收库保留。两份旧 wheel 验收虚拟环境中的原生 `.pyd` 受当前 Windows 执行边界保护，目录已从约 177 MB 缩减到约 19 MB，但未冒险修改权限或停止无关服务强删。
