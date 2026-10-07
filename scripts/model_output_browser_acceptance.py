"""真实HTTP验证模型增量展示、刷新恢复、工具撤销、失败和取消边界。"""

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/增量输出"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许独立本机验收服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: (errors.append(dialog.message), dialog.dismiss()))
        page.goto(args.url)
        page.locator("#username").fill("output-ui-" + str(time.time_ns()))
        page.locator("#password").fill("model-output-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()

        def submit(message):
            page.locator("#new-task").click()
            page.locator("#message").fill(message)
            page.locator("#send").click()
            expect(page.locator("#model-output")).to_be_visible()
            expect(page.locator("#model-output-body")).to_contain_text("第一段")
            expect(page.locator("#run-status")).to_have_text("正在推进")
            current_id = page.evaluate("sessionStorage.getItem('aegis-last-run')")
            token = page.evaluate("sessionStorage.getItem('aegis-token')")
            current = page.request.get(
                args.url + f"/api/v1/runs/{current_id}",
                headers={"Authorization": "Bearer " + token},
            ).json()
            assert current["status"] == "running" and not current["answer"]
            return current_id

        run_id = submit("验收：流式输出")
        page.screenshot(path=str(args.output / "01-真实片段与未验收状态.png"))
        page.reload()
        expect(page.locator("#answer")).to_contain_text("第三段", timeout=15000)
        expect(page.locator("#model-output")).to_have_count(0)
        expect(page.locator("#conversation img")).to_have_count(0)
        expect(page.locator("#answer")).to_contain_text("<img")
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        runs = page.request.get(args.url + "/api/v1/runs", headers=headers).json()
        assert len(runs) == 1 and runs[0]["id"] == run_id

        terminal_id = submit("验收：流式输出")
        page.route(
            args.url + f"/api/v1/runs/{terminal_id}",
            lambda route: route.fulfill(status=503, json={"detail": "确定性详情故障注入"}),
            times=1,
        )
        expect(page.locator("#notice")).to_contain_text("任务已结束", timeout=15000)
        expect(page.locator("#run-status")).to_have_text("任务完成")
        expect(page.locator("#model-output")).to_have_count(0)
        expect(page.locator("#answer")).to_contain_text("第三段", timeout=15000)

        submit("验收：流式中断")
        expect(page.locator("#run-status")).to_have_text("执行失败", timeout=15000)
        expect(page.locator("#model-output")).to_have_count(0)
        expect(page.locator("#answer")).not_to_contain_text("第一段")

        submit("验收：流式工具")
        expect(page.locator("#answer")).to_have_text("AegisCode计算结果为5", timeout=15000)
        expect(page.locator("#model-output")).to_have_count(0)
        expect(page.locator("#answer")).not_to_contain_text("第一段")

        submit("验收：流式慢任务")
        responsive = 0
        for theme in ("light", "dark"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for width, height in ((1440, 1000), (820, 1100), (390, 844)):
                page.set_viewport_size({"width": width, "height": height})
                expect(page.locator("#model-output-body")).to_contain_text("第一段")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                responsive += 1
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("#cancel").click()
        expect(page.locator("#run-status")).to_have_text("已停止", timeout=15000)
        expect(page.locator("#model-output")).to_have_count(0)

        detached = submit("验收：流式慢任务")
        page.locator("#new-task").click()
        expect(page.locator("#model-output")).to_have_count(0)
        page.wait_for_timeout(1100)
        expect(page.locator("#model-output")).to_have_count(0)
        stopped = page.request.post(
            args.url + f"/api/v1/runs/{detached}/cancel", headers=headers, data={}
        )
        assert stopped.ok
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#model-output")).to_have_count(0)
        assert not errors, errors
        print(
            f"增量输出浏览器验收通过：提前展示、刷新原Run、纯文本、工具撤销、失败、取消、切换与退出；{responsive}组响应式，无页面异常"
        )
        browser.close()


if __name__ == "__main__":
    main()
