"""编码工作台键盘、代码展示与注入负例；业务请求全部拦截，不写数据库。"""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8779")
    parser.add_argument("--output", type=Path, default=Path("docs/升级方案/截图/0.2.5"))
    parser.add_argument(
        "--baseline", action="store_true", help="从 HEAD 加载旧静态资源，验证回归能捕获缺失能力"
    )
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("仅允许本机服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors, writes = [], []
    run = {
        "id": "coding-fixture",
        "session_id": "coding-session",
        "message": "审查空输入并给出修复示例",
        "status": "completed",
        "trace_id": "fixture-trace",
        "step": 2,
        "answer": (
            "## 验证结论\n\n输入为空时先返回 `[]`。\n\n"
            "```python\ndef normalize(items):\n    return items or []\n```\n\n"
            "<img src=x onerror=alert(1)>\n\n"
            "```html\n<script>alert('unsafe')</script>\n```\n\n"
            "```text\n未闭合代码围栏仍应可读"
        ),
        "model": "fixture-model",
        "collaboration_mode": "team",
        "project_mode": None,
    }
    user = {"username": "界面验收账号", "role": "admin"}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            permissions=["clipboard-read", "clipboard-write"],
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_api(route):
            path = urlparse(route.request.url).path
            if route.request.method != "GET":
                writes.append(path)
                raise AssertionError("导航不应触发业务写请求")
            if path.endswith("/auth/me"):
                value = {**user, "bootstrap": {"memories": []}}
            elif path.endswith("/capabilities"):
                value = {"models": ["fixture-model"], "tools": [], "role": "admin", "max_steps": 10}
            elif path.endswith("/workspace/overview"):
                value = {
                    "runs": {"total": 1, "by_status": {"completed": 1}},
                    "assets": {"total": 0},
                }
            elif path.endswith("/runs"):
                value = [run]
            elif path.endswith("/coding-fixture"):
                value = run
            elif path.endswith("/thread"):
                value = {"items": [run], "has_more": False, "next_before": None}
            elif path.endswith("/events"):
                route.fulfill(content_type="text/event-stream", body="")
                return
            elif path.endswith("/files"):
                value = {"files": [], "truncated": False}
            elif path.endswith("/assets"):
                value = []
            else:
                raise AssertionError(f"未预期请求：{path}")
            route.fulfill(json=value)

        page.route("**/api/v1/**", route_api)
        if args.baseline:
            for url_path, file_path, mime in (
                ("/", "app/web/index.html", "text/html"),
                ("/static/app.js", "app/web/app.js", "text/javascript"),
                ("/static/styles.css", "app/web/styles.css", "text/css"),
            ):
                content = subprocess.check_output(
                    ["C:/Program Files/Git/cmd/git.exe", "show", "HEAD:" + file_path]
                )

                def serve_static(body, mime_type):
                    def handler(route):
                        route.fulfill(body=body, content_type=mime_type)

                    return handler

                page.route(args.url.rstrip("/") + url_path, serve_static(content, mime))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.keyboard.press("Control+k")
        expect(page.locator("#quick-actions")).to_be_hidden()
        page.evaluate("sessionStorage.setItem('aegis-token','fixture')")
        page.reload()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#workspace-overview")).to_contain_text("本人任务 1")
        page.screenshot(path=str(args.output / "00-工作台首页.png"), full_page=True)
        page.locator("#sidebar-toggle").click()
        expect(page.locator("#workspace-sidebar")).to_be_hidden()
        page.locator("#sidebar-toggle").click()
        expect(page.locator("#workspace-sidebar")).to_be_visible()
        page.keyboard.press("Control+k")
        expect(page.locator("#command-query")).to_be_focused()
        page.locator("#command-query").fill("不存在的入口<img>")
        expect(page.locator("#command-results button")).to_have_count(0)
        expect(page.locator("#command-count")).to_contain_text("没有")
        page.keyboard.press("Escape")
        expect(page.locator("#quick-actions")).to_be_hidden()
        page.locator("#command-toggle").click()
        page.keyboard.press("Escape")
        expect(page.locator("#command-toggle")).to_be_focused()
        page.keyboard.press("Control+k")
        page.locator("#command-query").fill("长期记忆")
        page.keyboard.press("Enter")
        expect(page.locator("#view-title")).to_have_text("长期记忆")
        page.keyboard.press("Control+k")
        page.locator("#command-query").fill("任务空间")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Enter")
        expect(page.locator("#view-title")).to_have_text("任务空间")
        page.locator("#run-list .run-item").click()
        expect(page.locator("#answer .code-block")).to_have_count(3)
        expect(page.locator("#answer h3")).to_have_text("验证结论")
        expect(page.locator("#answer code").first).to_have_text("[]")
        assert page.locator("#answer img, #answer script").count() == 0
        expect(page.locator("#answer")).to_contain_text("<img src=x onerror=alert(1)>")
        code = page.locator("#answer .code-block").first
        code.get_by_role("button", name="复制代码").click()
        expect(code.get_by_role("button")).to_have_text("已复制")
        copied = page.evaluate("navigator.clipboard.readText()")
        assert copied.replace("\r\n", "\n") == "def normalize(items):\n    return items or []"
        page.evaluate(
            "Object.defineProperty(navigator, 'clipboard', "
            "{value:{writeText:async()=>{throw new Error('denied')}}})"
        )
        code.get_by_role("button").click()
        expect(code.locator(".copy-status")).to_contain_text("请手动复制")
        assert (
            page.evaluate("window.getSelection().toString()")
            == "def normalize(items):\n    return items or []"
        )
        page.screenshot(path=str(args.output / "01-代码线程与审查.png"), full_page=True)
        page.keyboard.press("Control+k")
        page.screenshot(path=str(args.output / "02-快捷跳转.png"), full_page=True)
        page.keyboard.press("Escape")
        page.locator("#theme-toggle").click()
        page.screenshot(path=str(args.output / "03-深色代码线程.png"), full_page=True)
        # 产品展示使用另一个明确的界面夹具，安全负例截图保留作为验收证据。
        clean_run = {
            **run,
            "answer": (
                "## 空输入处理\n\n先统一输入边界，再执行主流程。\n\n"
                "```python\ndef normalize(items):\n    return items or []\n```\n\n"
                "验证用例：空值、空列表与非空列表。实际任务仍需运行测试确认。"
            ),
        }
        page.emulate_media(reduced_motion="reduce")
        page.evaluate("run => window.renderRun(run)", clean_run)
        page.screenshot(path=str(args.output / "05-深色编码体验.png"), full_page=True)
        page.locator("#theme-toggle").click()
        page.screenshot(path=str(args.output / "06-浅色编码体验.png"), full_page=True)
        page.set_viewport_size({"width": 375, "height": 844})
        if page.locator("#sidebar-toggle").get_attribute("aria-expanded") == "true":
            page.locator("#sidebar-toggle").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        expect(page.locator("#message")).to_be_visible()
        page.screenshot(path=str(args.output / "04-移动线程.png"), full_page=True)
        page.reload()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#workspace-sidebar")).to_be_hidden()
        for width, height in ((375, 844), (844, 390)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            expect(page.locator("#command-toggle")).to_be_visible()
        assert not errors, errors
        assert not writes, writes
        browser.close()
    print(
        json.dumps(
            {
                "checks": 18,
                "page_errors": errors,
                "business_writes": writes,
                "evidence": "业务数据为确定性界面夹具，非真实模型效果",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
