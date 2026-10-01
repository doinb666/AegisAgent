"""隔离执行服务：只部署在内部网络，独立于公开 API。"""

import asyncio
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Aegis 隔离执行器", docs_url=None, redoc_url=None)
LIMIT = asyncio.Semaphore(2)


class Execution(BaseModel):
    workspace_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    code: str = Field(min_length=1, max_length=12000)


def execute_container(request: Execution) -> dict:
    import docker
    from docker.types import LogConfig

    host_root = os.environ.get("AEGIS_SANDBOX_HOST_ROOT", "")
    if not host_root or not Path(host_root).is_absolute():
        raise RuntimeError("执行器未配置 Docker 宿主工作区绝对路径")
    visible_root = Path(os.environ.get("AEGIS_SANDBOX_VISIBLE_ROOT", host_root)).resolve()
    local = (visible_root / request.workspace_id).resolve()
    if not local.is_relative_to(visible_root) or not local.is_dir():
        raise ValueError("工作区不存在或越界")
    host_path = str(Path(host_root) / request.workspace_id)
    timeout = min(120, max(1, int(os.environ.get("AEGIS_SANDBOX_TIMEOUT_SECONDS", "30"))))
    client = docker.from_env(timeout=timeout + 5)
    container = None
    try:
        container = client.containers.run(
            os.environ.get("AEGIS_SANDBOX_IMAGE", "python:3.12-slim"),
            ["python", "-I", "-c", request.code],
            detach=True,
            user="65534:65534",
            network_disabled=True,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            mem_limit="128m",
            cpu_period=100000,
            cpu_quota=50000,
            pids_limit=64,
            working_dir="/tmp",
            tmpfs={"/tmp": "rw,noexec,nosuid,size=16m,mode=1777"},
            volumes={host_path: {"bind": "/workspace", "mode": "ro"}},
            log_config=LogConfig(type="json-file", config={"max-size": "1m", "max-file": "1"}),
        )
        status = container.wait(timeout=timeout)
        output = container.logs(tail=200).decode("utf-8", errors="replace")[:16000]
        return {"exit_code": status["StatusCode"], "output": output, "isolated": True}
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except docker.errors.APIError:
                pass
        client.close()


@app.post("/execute")
async def execute(request: Execution, authorization: str = Header(default="")):
    token = os.environ.get("AEGIS_SANDBOX_TOKEN", "")
    if not token or not hmac.compare_digest(authorization, "Bearer " + token):
        raise HTTPException(401, "隔离执行服务认证失败")
    async with LIMIT:
        try:
            return await asyncio.to_thread(execute_container, request)
        except Exception as exc:
            raise HTTPException(503, "隔离执行失败或超时，未在宿主重试") from exc
