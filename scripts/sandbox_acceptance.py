"""使用真实 Docker 验证执行器边界；只挂载专门的测试工作区。"""

import json
import os
from pathlib import Path

from app.sandbox_service import Execution, execute_container


def main():
    root = Path("data/sandbox-acceptance").resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace_id = "a" * 64
    workspace = root / workspace_id
    workspace.mkdir(exist_ok=True)
    os.environ["AEGIS_SANDBOX_HOST_ROOT"] = str(root)
    os.environ["AEGIS_SANDBOX_VISIBLE_ROOT"] = str(root)
    os.environ["AEGIS_SANDBOX_TIMEOUT_SECONDS"] = "5"
    code = (
        "import os,socket,json\n"
        "checks={'non_root':os.getuid()!=0,'host_secret_absent':not os.path.exists('/app/.env')}\n"
        "try:\n open('/workspace/should-not-write.txt','w').write('blocked')\n"
        " checks['workspace_read_only']=False\n"
        "except OSError: checks['workspace_read_only']=True\n"
        "try:\n socket.create_connection(('1.1.1.1',443),timeout=1)\n"
        " checks['network_blocked']=False\n"
        "except OSError: checks['network_blocked']=True\n"
        "open('/tmp/result.txt','w').write('temporary')\nchecks['temporary_write']=True\n"
        "print(json.dumps(checks))\n"
    )
    result = execute_container(Execution(workspace_id=workspace_id, code=code))
    assert result["exit_code"] == 0, result
    checks = json.loads(result["output"])
    assert all(checks.values()), checks
    assert not (workspace / "should-not-write.txt").exists()
    try:
        execute_container(Execution(workspace_id=workspace_id, code="while True: pass"))
        raise AssertionError("无限循环未超时")
    except Exception as error:
        assert not isinstance(error, AssertionError)
    print(
        json.dumps(
            {"sandbox": "真实Docker", "checks": checks, "timeout": "通过"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
