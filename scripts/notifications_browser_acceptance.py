"""真实站内通知验收：完成、失败、审批、已读与只读重试。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/站内通知"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许本机独立验收服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.locator("#username").fill("notice-browser-" + str(time.time_ns()))
        page.locator("#password").fill("notice-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#notifications-toggle")).to_be_visible()
        page.locator("#message").fill("计算并检查结果")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        page.locator("#notifications-toggle").click()
        expect(page.locator("#notifications-list")).to_contain_text("任务完成 · 未读")
        page.locator("#notifications-list button").click()
        expect(page.locator("#notifications-panel")).to_be_hidden()
        expect(page.locator("#notifications-count")).to_be_hidden()
        page.locator("#new-task").click()
        page.locator("#message").fill("写文件并核对审批证据")
        page.locator("#send").click()
        expect(page.locator("#approval")).to_be_visible(timeout=20000)
        page.locator("#notifications-toggle").click()
        expect(page.locator("#notifications-list")).to_contain_text("等待审批 · 未读")
        page.screenshot(path=str(args.output / "01-完成与审批通知.png"), full_page=True)
        page.locator("#notifications-list button").first.click()
        expect(page.locator("#approval")).to_be_visible()
        # 打开通知只标记已读，审批仍然需要用户单独决定。
        expect(page.locator("#run-status")).to_have_text("等待你的审批")
        page.locator("#reject").click()
        expect(page.locator("#run-status")).to_have_text("已停止", timeout=20000)
        page.locator("#new-task").click()
        page.locator("#message").fill("验收：模拟模型失败")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("执行失败", timeout=20000)
        page.locator("#notifications-toggle").click()
        expect(page.locator("#notifications-list")).to_contain_text("任务失败 · 未读")
        page.screenshot(path=str(args.output / "02-失败与已读状态.png"), full_page=True)

        def fail_read(route):
            route.fulfill(status=503, json={"detail": "验收故障注入"})

        page.route("**/api/v1/notifications?*", fail_read)
        page.locator("#notifications-refresh").click()
        expect(page.locator("#notifications-status")).to_contain_text("读取失败")
        expect(page.locator("#notifications-list")).to_contain_text("任务失败")
        page.unroute("**/api/v1/notifications?*", fail_read)
        page.locator("#notifications-refresh").click()
        expect(page.locator("#notifications-status")).to_have_text("")
        page.locator("#notifications-close").focus()
        page.keyboard.press("Escape")
        expect(page.locator("#notifications-panel")).to_be_hidden()
        expect(page.locator("#notifications-toggle")).to_be_focused()
        for width in (390, 768):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#notifications-toggle")).to_be_hidden()
        assert not errors, errors
        browser.close()
    print(
        json.dumps({"checks": 10, "page_errors": errors, "fault_injection": 1}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
