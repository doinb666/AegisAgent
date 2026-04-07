# Demo Assets

该目录用于存放 README 中引用的演示资源文件。

## 推荐文件

- `demo.gif`：30-60 秒产品演示（health + chat 流程）

## 建议录制流程

1. 打开 Swagger：`http://127.0.0.1:8000/docs`
2. 调用 `GET /api/v1/health`
3. 调用 `POST /api/v1/chat`
4. 导出 GIF 为 `demo.gif` 并放在本目录

放置后，README 中的下列引用会自动生效：

```md
![AegisAgent Demo](docs/assets/demo.gif)
```
