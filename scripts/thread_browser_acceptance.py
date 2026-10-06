"""分页历史、错误重试和过期响应验收；拦截业务 API，不写正式数据。"""

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path("docs/截图/会话历史"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许连接本机服务")
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "id": "thread-demo",
        "session_id": "demo-session",
        "message": "给出空输入的修复建议",
        "answer": (
            "## 修复建议\n\n```python\ndef normalize(items):\n    return items or []\n```\n\n"
            "修改后仍需运行测试确认。"
        ),
        "status": "completed",
        "trace_id": "demo-trace",
        "step": 2,
        "model": "demo-model",
    }
    earlier = {
        "id": "earlier",
        "message": "审查代码时先检查哪些边界？",
        "answer": "先检查空输入、异常类型和路径访问范围。",
        "status": "completed",
    }
    oldest = {
        "id": "oldest",
        "message": "以后请先给结论，再说明证据。",
        "answer": "本会话会按这个顺序回答。跨会话偏好请在长期记忆中审核启用。",
        "status": "completed",
    }
    errors, writes, pending = [], [], []
    controls = {"fail": True, "delay": False, "inject": False}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_api(route):
            parsed = urlparse(route.request.url)
            path = parsed.path
            if route.request.method != "GET":
                writes.append(path)
                route.fulfill(status=405, json={"detail": "验收禁止写入"})
                return
            if path.endswith("/auth/me"):
                value = {"username": "演示账号", "role": "admin", "bootstrap": {"memories": []}}
            elif path.endswith("/capabilities"):
                value = {"models": ["demo-model"], "tools": [], "role": "admin", "max_steps": 10}
            elif path.endswith("/workspace/overview"):
                value = {
                    "runs": {"total": 3, "by_status": {"completed": 3}},
                    "assets": {"total": 0},
                }
            elif path.endswith("/runs"):
                value = [run]
            elif path.endswith("/thread-demo"):
                value = run
            elif path.endswith("/thread"):
                if controls["fail"]:
                    controls["fail"] = False
                    route.fulfill(status=503, json={"detail": "临时不可用，请重试"})
                    return
                if parse_qs(parsed.query).get("before"):
                    value = {"items": [oldest], "has_more": False, "next_before": None}
                    if controls["delay"]:
                        pending.append((route, value))
                        return
                else:
                    item = earlier
                    if controls["inject"]:
                        item = {**earlier, "answer": "<img src=x onerror=alert(1)>"}
                    value = {"items": [item, run], "has_more": True, "next_before": "earlier"}
            elif path.endswith("/events"):
                route.fulfill(content_type="text/event-stream", body="")
                return
            elif path.endswith("/files"):
                value = {"files": [], "truncated": False}
            else:
                route.fulfill(status=404, json={"detail": "未预期的验收请求"})
                return
            route.fulfill(json=value)

        page.route("**/api/v1/**", route_api)
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.evaluate("sessionStorage.setItem('aegis-token','fixture')")
        page.reload()
        expect(page.locator("#shell")).to_be_visible()
        page.locator("#run-list .run-item").click()
        history = page.get_by_role("region", name="之前的会话问答")
        expect(history.get_by_role("button", name="重试加载会话")).to_be_visible()
        expect(page.locator("#answer")).to_contain_text("修复建议")
        history.get_by_role("button").click()
        expect(history).to_contain_text("已加载 1 轮历史问答")
        history.get_by_role("button", name="加载更早的问答").click()
        expect(history).to_contain_text("已加载 2 轮历史问答")
        expect(history.locator(".message.user").first).to_have_text("你" + oldest["message"])
        expect(history.get_by_role("button")).to_be_hidden()
        page.locator("#conversation").evaluate("element => element.scrollTop = 0")
        page.screenshot(path=str(args.output / "01-连续问答.png"), full_page=True)
        page.locator("#conversation").evaluate(
            "element => element.scrollTop = element.scrollHeight"
        )
        page.screenshot(path=str(args.output / "04-当前代码问答.png"), full_page=True)
        page.locator("#theme-toggle").click()
        page.screenshot(path=str(args.output / "02-深色会话.png"), full_page=True)
        controls["inject"] = True
        page.locator("#run-list .run-item").click()
        expect(history).to_contain_text("<img src=x onerror=alert(1)>")
        assert history.locator("img,script").count() == 0
        controls["delay"] = True
        history.get_by_role("button", name="加载更早的问答").click()
        page.wait_for_function(
            "document.querySelector('[aria-label=\"之前的会话问答\"] button').disabled"
        )
        page.locator("#new-task").click()
        assert len(pending) == 1
        pending[0][0].fulfill(json=pending[0][1])
        page.wait_for_load_state("networkidle")
        expect(history).to_have_count(0)
        expect(page.locator("#welcome")).to_be_visible()
        controls.update(delay=False, inject=False)
        page.locator("#run-list .run-item").click()
        expect(history).to_contain_text(earlier["answer"])
        page.set_viewport_size({"width": 390, "height": 844})
        if page.locator("#sidebar-toggle").get_attribute("aria-expanded") == "true":
            page.locator("#sidebar-toggle").click()
        if page.locator("#inspector-toggle").get_attribute("aria-expanded") == "true":
            page.locator("#inspector-toggle").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        expect(page.locator("#message")).to_be_visible()
        page.screenshot(path=str(args.output / "03-移动会话.png"), full_page=True)
        assert not errors, errors
        assert not writes, writes
        browser.close()
    print(
        json.dumps(
            {"checks": 9, "page_errors": errors, "business_writes": writes}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
