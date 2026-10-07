"""真实数据库帧、完整检查点与过期租约恢复，不模拟 Token 切片。"""

import asyncio

import pytest

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Run
from app.harness.runtime import Worker
from app.harness.store import Store
from tests.test_harness_runtime import FakeModel, FakeTools, setup_service, wait_status
from tests.test_model_streaming import chunk, routed


@pytest.mark.asyncio
async def test_complete_tools_execute_only_after_assembly_and_next_answer_commits(
    tmp_path, monkeypatch
):
    from tests.test_model_streaming import Stream

    router, client = routed(monkeypatch, [])

    async def create(**kwargs):
        client.requests.append(kwargs)
        if len(client.requests) == 1:
            client.stream = Stream(
                [
                    chunk("待校验说明"),
                    chunk(
                        tools=[
                            {
                                "index": 0,
                                "id": "read-1",
                                "function": {"name": "read", "arguments": '{"value":'},
                            }
                        ]
                    ),
                    chunk(tools=[{"index": 0, "function": {"arguments": '"证据"}'}}]),
                    chunk(finish="tool_calls"),
                ]
            )
        else:
            client.stream = Stream([chunk("已根据工具证据完成", finish="stop")])
        return client.stream

    monkeypatch.setattr(client, "create", create)
    tools = FakeTools()
    service, principal = await setup_service(tmp_path, router, tools, evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "complete-tool")
        completed = await wait_status(service, principal, run["id"], {"completed", "failed"})
        assert completed["status"] == "completed"
        assert completed["answer"] == "已根据工具证据完成"
        assert tools.executions == 1
        events = await service.events(principal, run["id"])
        finished = [
            event["data"]["reason"] for event in events if event["type"] == "model_output_finished"
        ]
        assert finished == ["tools", "answer"]
        assert [
            event["data"]["reason"] for event in events if event["type"] == "model_output_retracted"
        ] == ["tools"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancel_retracts_in_terminal_transaction_and_closes_stream(tmp_path, monkeypatch):
    release = asyncio.Event()
    router, client = routed(
        monkeypatch, [chunk("取消前片段"), release, chunk("禁止提交", finish="stop")]
    )
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "cancel-stream")
        async with asyncio.timeout(3):
            while not any(
                event["type"] == "model_output_delta"
                for event in await service.events(principal, run["id"])
            ):
                await asyncio.sleep(0.01)
        assert (await service.cancel(principal, run["id"]))["status"] == "cancelled"
        async with asyncio.timeout(1):
            while not client.stream.closed:
                await asyncio.sleep(0.01)
        events = await service.events(principal, run["id"])
        output = [event for event in events if event["type"].startswith("model_output_")]
        assert output[-1]["type"] == "model_output_retracted"
        assert output[-1]["data"]["reason"] == "failed"
        assert (await service.get_run(principal, run["id"]))["answer"] is None
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
async def test_checkpoint_failure_rolls_back_complete_messages_and_marker_clear(
    tmp_path, monkeypatch
):
    router, client = routed(monkeypatch, [chunk("完整响应", finish="stop")])
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path, evolution_enabled=False), router, FakeTools()
    )
    await service.store.initialize()
    user = await service.register("checkpoint-atomic", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    worker = Worker(service)
    created = await service.create_run(principal, "任务", "atomic")
    await service.store.claim(worker.id)
    run = await worker.load(created["id"])
    try:
        response = await worker.model_call(run, run.messages, public_output=True, step=1)
        original = service.store.emit

        def fail_checkpoint(session, current, event_type, data):
            if event_type == "model_output_finished":
                raise RuntimeError("模拟检查点提交失败")
            return original(session, current, event_type, data)

        monkeypatch.setattr(service.store, "emit", fail_checkpoint)
        with pytest.raises(RuntimeError, match="提交失败"):
            await worker.checkpoint(
                run.id,
                [*run.messages, {"role": "assistant", "content": response.content}],
                1,
                "model",
                {},
            )
        async with service.store.sessions() as session:
            current = await session.get(Run, run.id)
            assert current.config["model_inflight"]
            assert len(current.messages) == 2
            assert current.step == 0
        assert not any(
            event["type"] in {"model", "model_output_finished"}
            for event in await service.events(principal, run.id)
        )
    finally:
        await service.store.close()
        await router.aclose()


@pytest.mark.asyncio
async def test_nonstream_fallback_reports_actual_characters_without_delta(tmp_path, monkeypatch):
    router, client = routed(monkeypatch, [], streaming=False)
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "complete-only")
        completed = await wait_status(service, principal, run["id"], {"completed"})
        events = await service.events(principal, run["id"])
        assert not any(event["type"] == "model_output_delta" for event in events)
        finished = next(
            event["data"] for event in events if event["type"] == "model_output_finished"
        )
        assert finished["streamed"] is False
        assert finished["characters"] == len(completed["answer"])
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_delta", "after_delta", "before_checkpoint"])
async def test_crash_boundaries_keep_marker_and_other_store_interrupts(
    tmp_path, monkeypatch, phase
):
    release = asyncio.Event()
    if phase == "before_checkpoint":
        values = [chunk("完整", finish="stop")]
    elif phase == "before_delta":
        values = [release]
    else:
        values = [chunk("暂存"), release]
    router, client = routed(monkeypatch, values)
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path, evolution_enabled=False), router, FakeTools()
    )
    await service.store.initialize()
    user = await service.register("crash-boundary", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    worker = Worker(service)
    created = await service.create_run(principal, "任务", phase)
    assert await service.store.claim(worker.id) == created["id"]
    run = await worker.load(created["id"])
    task = asyncio.create_task(worker.model_call(run, run.messages, public_output=True, step=1))
    try:
        if phase == "before_checkpoint":
            await task
        else:
            async with asyncio.timeout(3):
                while True:
                    events = await service.events(principal, run.id)
                    target = (
                        "model_output_started" if phase == "before_delta" else "model_output_delta"
                    )
                    if any(event["type"] == target for event in events):
                        break
                    await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with service.store.sessions.begin() as session:
            current = await session.get(Run, run.id)
            assert current.config["model_inflight"]
            assert len(current.messages) == 2
            current.lease_until = 0
        other = Store(service.settings)
        try:
            assert await other.claim("recovery") is None
        finally:
            await other.close()
        assert (await service.get_run(principal, run.id))["status"] == "interrupted"
        assert len(client.requests) == 1
        assert not any(
            event["type"] == "model_output_finished"
            for event in await service.events(principal, run.id)
        )
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.store.close()
        await router.aclose()


@pytest.mark.asyncio
async def test_delta_persists_before_completion_and_final_checkpoint_clears_marker(
    tmp_path, monkeypatch
):
    release = asyncio.Event()
    router, client = routed(
        monkeypatch, [chunk("<script>第一片</script>"), release, chunk("最终文本", finish="stop")]
    )
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "stream")
        async with asyncio.timeout(3):
            while True:
                events = await service.events(principal, run["id"])
                if any(event["type"] == "model_output_delta" for event in events):
                    break
                await asyncio.sleep(0.01)
        assert (await service.get_run(principal, run["id"]))["answer"] is None
        async with service.store.sessions() as session:
            persisted = await session.get(Run, run["id"])
            assert persisted.config["model_inflight"]["call_id"]
            assert len(persisted.messages) == 2
        other = HarnessService(service.settings)
        try:
            async with other.store.sessions() as session:
                persisted = await session.get(Run, run["id"])
                assert persisted.config["model_inflight"]
            replayed = await other.events(principal, run["id"])
            assert replayed == events
            with pytest.raises(HarnessError) as forbidden:
                await other.events(
                    Principal("unknown-user", principal.tenant_id, "admin"), run["id"]
                )
            assert forbidden.value.status_code == 404
        finally:
            await other.store.close()
        release.set()
        completed = await wait_status(service, principal, run["id"], {"completed", "failed"})
        assert completed["status"] == "completed"
        assert completed["answer"] == "<script>第一片</script>最终文本"
        async with service.store.sessions() as session:
            persisted = await session.get(Run, run["id"])
            assert "model_inflight" not in persisted.config
        events = await service.events(principal, run["id"])
        assert events[0]["type"] == "queued"
        assert any(event["type"] == "model_output_finished" for event in events)
        assert client.stream.closed
    finally:
        release.set()
        await service.close()


@pytest.mark.asyncio
async def test_expired_inflight_marker_interrupts_without_second_generation(tmp_path):
    service, principal = await setup_service(
        tmp_path, FakeModel(delay=10), FakeTools(), evolution_enabled=False
    )
    try:
        for worker in service.workers:
            worker.task.cancel()
        await asyncio.gather(*(worker.task for worker in service.workers), return_exceptions=True)
        run = await service.create_run(principal, "任务", "crash")
        async with service.store.sessions.begin() as session:
            current = await session.get(Run, run["id"])
            current.status, current.lease_owner, current.lease_until = "running", "dead", 0
            current.config = {
                **current.config,
                "model_inflight": {"call_id": "unfinished", "step": 1},
            }
        assert await service.store.claim("new") is None
        interrupted = await service.get_run(principal, run["id"])
        assert interrupted["status"] == "interrupted"
        assert service.model_router.delegate.calls == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_partial_failure_retracts_and_never_commits_answer(tmp_path, monkeypatch):
    router, client = routed(monkeypatch, [chunk("尚未完成"), TimeoutError("模拟中断")])
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "failure")
        failed = await wait_status(service, principal, run["id"], {"failed"})
        assert failed["answer"] is None
        events = await service.events(principal, run["id"])
        assert [
            event["data"]["reason"] for event in events if event["type"] == "model_output_retracted"
        ] == ["failed"]
        assert not any(event["type"] == "model_output_finished" for event in events)
        assert client.stream.closed
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_tool_first_delta_retracts_and_invalid_json_never_executes(tmp_path, monkeypatch):
    router, client = routed(
        monkeypatch,
        [
            chunk("暂存文本"),
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call",
                        "function": {"name": "read", "arguments": '{"value":'},
                    }
                ]
            ),
            chunk(finish="tool_calls"),
        ],
    )
    tools = FakeTools()
    service, principal = await setup_service(tmp_path, router, tools, evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "invalid-tool")
        await wait_status(service, principal, run["id"], {"failed"})
        assert tools.executions == 0
        events = await service.events(principal, run["id"])
        assert [
            event["data"]["reason"] for event in events if event["type"] == "model_output_retracted"
        ] == ["tools"]
        assert '{"value":' not in str(events)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plan", "reflection"])
async def test_internal_calls_never_enable_public_stream(tmp_path, monkeypatch, mode):
    router, client = routed(monkeypatch, [chunk("不应消费", finish="stop")])
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", mode, mode=mode)
        assert (await wait_status(service, principal, run["id"], {"completed", "failed"}))[
            "status"
        ] == "completed"
        assert all(
            "stream" not in request and "on_delta" not in request for request in client.requests
        )
        assert not any(
            event["type"].startswith("model_output_")
            for event in await service.events(principal, run["id"])
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_legacy_fake_has_no_extra_callback_or_output_events(tmp_path):
    service, principal = await setup_service(
        tmp_path, FakeModel(), FakeTools(), evolution_enabled=False
    )
    try:
        run = await service.create_run(principal, "任务", "legacy")
        assert (await wait_status(service, principal, run["id"], {"completed"}))["answer"] == "完成"
        assert not any(
            event["type"].startswith("model_output_")
            for event in await service.events(principal, run["id"])
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_frame_limit_marks_incomplete_preview_and_final_answer_keeps_all_text(
    tmp_path, monkeypatch
):
    ticks = iter(range(1000))
    monkeypatch.setattr("app.harness.model_output.monotonic", lambda: next(ticks))
    # 缩小测试帧预算，保留300个真实增量的压力路径，避免256次磁盘同步拖慢专项。
    monkeypatch.setattr("app.harness.model_output.MAX_FRAMES", 4)
    router, client = routed(monkeypatch, [*[chunk("字") for _ in range(300)], chunk(finish="stop")])
    service, principal = await setup_service(tmp_path, router, FakeTools(), evolution_enabled=False)
    try:
        run = await service.create_run(principal, "任务", "frame-limit")
        completed = await wait_status(
            service, principal, run["id"], {"completed", "failed"}, timeout=10
        )
        assert completed["status"] == "completed"
        assert completed["answer"] == "字" * 300
        events = await service.events(principal, run["id"])
        frames = [event["data"] for event in events if event["type"] == "model_output_delta"]
        assert len(frames) == 4
        assert all(len(frame["text"]) <= 4096 for frame in frames)
        finished = next(
            event["data"] for event in events if event["type"] == "model_output_finished"
        )
        assert finished["preview_truncated"] is True
        assert finished["characters"] == 300
    finally:
        await service.close()
