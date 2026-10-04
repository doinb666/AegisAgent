"""协作入口、模型来源与配置生成器验收；拦截全部业务请求。"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", default="docs/升级方案/截图/0.2.4")
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("仅允许本机服务")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors, submitted = [], []
    cap = {
        "models": ["same-model"],
        "model_routes": [
            {
                "id": "route-first",
                "label": "主服务",
                "model": "same-model",
                "provider": "custom",
                "priority": 0,
            },
            {
                "id": "route-backup",
                "label": "备用服务",
                "model": "same-model",
                "provider": "anthropic",
                "priority": 1,
            },
        ],
        "tools": ["delegate", "project_prepare", "file_read"],
        "role": "admin",
        "max_steps": 10,
        "collaboration": {
            "modes": ["fork", "team"],
            "project_modes": ["fork", "worktree"],
        },
    }
    run = {
        "id": "fixture-run",
        "message": "检查",
        "status": "completed",
        "session_id": "fixture-session",
        "step": 1,
        "trace_id": "fixture-trace",
        "answer": "确定性展示",
        "model": "route-backup",
        "collaboration_mode": "fork",
        "project_mode": "worktree",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_api(route):
            path = urlparse(route.request.url).path
            if path.endswith("/auth/me"):
                value = {"bootstrap": {"memories": []}}
            elif path.endswith("/capabilities"):
                value = cap
            elif path.endswith("/workspace/overview"):
                value = {"runs": {"total": 0, "by_status": {}}, "assets": {"total": 0}}
            elif path.endswith("/events"):
                route.fulfill(content_type="text/event-stream", body="")
                return
            elif path.endswith("/files"):
                value = {"files": [], "truncated": False}
            elif path.endswith("/runs"):
                if route.request.method == "POST":
                    submitted.append(route.request.post_data_json)
                    value = run
                else:
                    value = []
            elif path.endswith("/fixture-run"):
                value = run
            else:
                raise AssertionError(f"未预期的业务请求：{path}")
            route.fulfill(json=value)

        page.route("**/api/v1/**", route_api)
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        assert (
            page.evaluate("typeof WorkspaceUI.renderCollaboration") == "function"
        ), "缺少协作状态展示"
        page.evaluate(
            "async () => { state.token='fixture'; "
            "await enter({username:'展示账号',role:'admin'}); }"
        )
        assert page.locator("#model option").count() == 3
        page.locator("#model").select_option("route-backup")
        page.locator(".task-options summary").click()
        page.locator("#collaboration-mode").select_option("fork")
        page.locator("#project-mode").select_option("worktree")
        page.locator("#message").fill("检查代码与约束")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成")
        assert submitted[-1]["model"] == "route-backup"
        assert submitted[-1]["collaboration_mode"] == "fork"
        assert submitted[-1]["project_mode"] == "worktree"
        page.evaluate("""() => {
            addEvent(101,'collaboration_graph',{nodes:[
                {id:'inspect',run_id:'child-a',depends_on:[],status:'completed'},
                {id:'review',run_id:null,depends_on:['inspect'],status:'blocked'}]});
            addEvent(102,'collaboration_acceptance',{run_id:'child-a',status:'rejected',
                reasons:['缺少必需工具证据：file_read <img src=x onerror=alert(1)>'],evidence:[]});
        }""")
        expect(page.locator("#collaboration-review")).to_be_visible()
        expect(page.locator("#collaboration-nodes")).to_contain_text("未通过")
        expect(page.locator("#collaboration-nodes")).to_contain_text("inspect")
        assert page.locator("#collaboration-nodes img").count() == 0
        page.screenshot(path=str(output / "10-协作依赖与验收.png"), full_page=True)
        page.locator('[data-view="settings"]').click()
        page.locator("#model-config-builder summary").click()
        page.locator("#provider-kind").select_option("ollama")
        page.locator("#provider-model").fill("my-local-model")
        page.locator("#provider-label").fill("本机")
        page.locator("#provider-add").click()
        expect(page.locator("#model-config-preview")).to_contain_text("my-local-model")
        assert page.locator("#provider-key-env").is_disabled()
        page.locator("#provider-kind").select_option("azure")
        page.locator("#provider-model").fill("deployment")
        page.locator("#provider-base").fill("https://resource.openai.azure.com")
        page.locator("#provider-key-env").fill("AZURE_MODEL_KEY")
        page.locator("#provider-add").click()
        assert "azure" not in page.locator(
            "#model-config-preview"
        ).inner_text(), "未填版本不能生成Azure配置"
        page.locator("#provider-version").fill("2024-10-21")
        page.locator("#provider-add").click()
        expect(page.locator("#model-config-preview")).to_contain_text("api_version")
        page.screenshot(path=str(output / "11-多来源模型配置.png"), full_page=True)
        page.locator('[data-view="chat"]').click()
        page.evaluate(
            "() => { WorkspaceUI.configureCollaboration("
            "{collaboration:{project_modes:[]}},true); }"
        )
        assert page.locator('#project-mode option[value="worktree"]').is_disabled()
        for width in (390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {"passed": 8, "page_errors": errors, "database_writes": 0},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
