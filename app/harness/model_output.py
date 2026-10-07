"""持久临时文本帧；网络等待不占数据库事务，工具碎片不会进入事件。"""

from time import monotonic

from app.infrastructure.llm.stream_assembly import MAX_TEXT_CHARS

MAX_FRAMES = 256
MAX_FRAME_CHARS = 4096
FLUSH_CHARS = 512
FLUSH_SECONDS = 0.2


class ModelOutput:
    def __init__(self, worker, run_id, call_id):
        self.worker = worker
        self.run_id = run_id
        self.call_id = call_id
        self.seq = 0
        self.offset = 0
        self.frames = 0
        self.pending = ""
        self.last_flush = monotonic()
        self.streamed = False
        self.retracted = False
        self.preview_truncated = False
        self.characters = 0

    async def receive(self, delta):
        if delta.get("type") not in {"content", "tools"}:
            raise ValueError("模型临时输出类型无效")
        self.streamed = True
        if delta["type"] == "tools":
            self.pending = ""
            await self.retract("tools")
            return
        text = delta["text"]
        if not isinstance(text, str):
            raise ValueError("模型临时输出须为文本")
        self.characters += len(text)
        if self.characters > MAX_TEXT_CHARS:
            raise ValueError("模型临时文本输出超限")
        if self.retracted:
            return
        if self.frames >= MAX_FRAMES:
            self.preview_truncated = True
            return
        self.pending += text
        if (
            self.frames == 0
            or len(self.pending) >= FLUSH_CHARS
            or monotonic() - self.last_flush >= FLUSH_SECONDS
        ):
            await self.flush()

    async def flush(self):
        while self.pending and self.frames < MAX_FRAMES:
            text, self.pending = self.pending[:MAX_FRAME_CHARS], self.pending[MAX_FRAME_CHARS:]
            await self.emit("model_output_delta", {"offset": self.offset, "text": text})
            self.offset += len(text)
            self.frames += 1
            self.last_flush = monotonic()
        if self.pending:
            self.preview_truncated = True
            self.pending = ""

    async def emit(self, event_type, data):
        async with self.worker.active_transaction(self.run_id) as (session, run):
            marker = run.config.get("model_inflight")
            if not marker or marker.get("call_id") != self.call_id:
                raise RuntimeError("模型输出与未完成调用标记不一致")
            self.seq += 1
            run.config = {
                **run.config,
                "model_inflight": {
                    **marker,
                    "seq": self.seq,
                    **({"retracted": True} if event_type == "model_output_retracted" else {}),
                },
            }
            self.worker.store.emit(
                session, run, event_type, {"call_id": self.call_id, "seq": self.seq, **data}
            )

    async def retract(self, reason):
        if not self.retracted:
            await self.emit("model_output_retracted", {"reason": reason})
            self.retracted = True
            self.pending = ""

    def finished_data(self):
        self.seq += 1
        return {
            "call_id": self.call_id,
            "seq": self.seq,
            "reason": "tools" if self.retracted else "answer",
            "characters": self.characters,
            "streamed": self.streamed,
            "preview_truncated": self.preview_truncated,
        }
