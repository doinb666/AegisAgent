"""桌面工作台真实浏览器验收；仅连接本机确定性测试服务。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8772")
    parser.add_argument("--output", type=Path, default=Path("docs/升级方案/截图/0.2.2"))
    arguments = parser.parse_args()
    if urlparse(arguments.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("仅允许本机测试服务")
    arguments.output.mkdir(parents=True, exist_ok=True)
    errors = []
    server_errors = []
    injected_server_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def record_server_error(response):
            if response.status < 500:
                return
            if response.headers.get("x-workbench-error-injection") == "detail-read":
                injected_server_errors.append(response.url)
            else:
                server_errors.append(response.url)

        page.on("response", record_server_error)
        page.goto(arguments.url)
        page.wait_for_load_state("networkidle")
        expect(page.locator("#theme-toggle")).to_be_visible()
        assert page.locator("html").get_attribute("data-theme") == "dark"
        page.locator("#theme-toggle").click()
        assert page.locator("html").get_attribute("data-theme") == "light"
        page.reload()
        page.wait_for_load_state("networkidle")
        assert page.locator("html").get_attribute("data-theme") == "light"
        page.locator("#theme-toggle").click()
        page.locator("#username").fill("workbench-" + str(time.time_ns()))
        page.locator("#password").fill("workbench-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#workspace-overview")).to_contain_text("任务")
        page.locator("#message").fill("写文件并核对预览")
        page.locator("#send").click()
        expect(page.locator("#approval")).to_be_visible(timeout=20000)
        page.locator("#approve").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        page.locator("#files-refresh").click()
        expect(page.locator("#file-list")).to_contain_text("result.txt")
        page.locator("#file-list").get_by_role("button", name="result.txt", exact=False).click()
        expect(page.locator("#file-preview")).to_contain_text("已批准")
        page.screenshot(path=str(arguments.output / "01-深色线程与文件.png"), full_page=True)
        page.locator("#inspector").evaluate("element => element.scrollTop = 0")
        page.screenshot(path=str(arguments.output / "00-深色桌面工作台.png"), full_page=True)
        page.locator("#inspector-toggle").click()
        expect(page.locator("#inspector")).to_be_hidden()
        page.locator("#inspector-toggle").click()
        expect(page.locator("#inspector")).to_be_visible()

        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        ids = []
        for number in range(23):
            response = page.request.post(
                arguments.url + "/api/v1/runs",
                headers={**headers, "Idempotency-Key": f"browser-{number}"},
                data={"message": f"分页检索任务 {number}"},
            )
            assert response.status == 202, response.text()
            run = response.json()
            ids.append(run["id"])
            # Playwright 的 JS 轮询不能依赖 Promise 真值，使用终态 HTTP 证据。
            deadline = time.monotonic() + 20
            while True:
                status = page.request.get(
                    arguments.url + "/api/v1/runs/" + run["id"], headers=headers
                ).json()["status"]
                if status == "completed":
                    break
                assert status not in {"failed", "cancelled", "interrupted"}, status
                assert time.monotonic() < deadline, "分页准备任务未完成"
                page.wait_for_timeout(50)
        page.locator("#refresh").click()
        expect(page.locator("#runs-more")).to_be_visible()
        page.locator("#runs-more").click()
        expect(page.locator("#run-list .run-item")).to_have_count(24)
        page.locator("#run-search").fill("分页检索任务 22")
        expect(page.locator("#run-list .run-item")).to_have_count(1)
        expect(page.locator("#run-list")).to_contain_text("分页检索任务 22")
        page.locator("#run-filter").select_option("failed")
        expect(page.locator("#run-list .run-item")).to_have_count(0)
        page.locator("#run-filter").select_option("completed")
        expect(page.locator("#run-list .run-item")).to_have_count(1)
        page.locator("#run-search").fill("")
        expect(page.locator("#run-list .run-item")).to_have_count(20)

        # 先延迟旧任务详情，再打开新任务，旧响应不能覆盖新线程。
        delayed = []
        pattern = "**/api/v1/runs/" + ids[-1]
        page.route(pattern, lambda route: delayed.append(route))
        page.locator("#run-list .run-item").filter(has_text="分页检索任务 22").click()
        page.wait_for_function("document.querySelector('#run-list') !== null")
        page.locator("#run-list .run-item").filter(has_text="分页检索任务 21").click()
        expect(page.locator(".message.user")).to_contain_text("分页检索任务 21")
        assert delayed, "没有捕获延迟请求"
        for route in delayed:
            route.fulfill(response=route.fetch())
        page.unroute(pattern)
        page.wait_for_load_state("networkidle")
        expect(page.locator(".message.user")).to_contain_text("分页检索任务 21")

        # A切换B的详情读取期间，发送与快捷键都不得沿用A的旧线程。
        delayed_switch = []
        unexpected_posts = []

        def catch_create_during_loading(route):
            if route.request.method == "POST":
                unexpected_posts.append(route.request.post_data_json)
                route.fulfill(status=409, json={"detail": "验收拦截：加载中不应创建任务"})
            else:
                route.continue_()

        page.route("**/api/v1/runs", catch_create_during_loading)
        page.route(pattern, lambda route: delayed_switch.append(route))
        with page.expect_request(pattern):
            page.locator("#run-list .run-item").filter(has_text="分页检索任务 22").click()
        expect(page.locator("#run-status")).to_have_text("正在读取任务…")
        loading_disabled = page.locator("#send").is_disabled()
        page.locator("#message").fill("加载期间不得发入旧线程")
        page.locator("#message").press("Control+Enter")
        page.wait_for_timeout(200)
        assert delayed_switch, "没有捕获B的详情读取请求"
        for route in delayed_switch:
            route.fulfill(status=404, json={"detail": "任务不存在或无权访问"})
        page.unroute(pattern)
        page.unroute("**/api/v1/runs")
        page.wait_for_load_state("networkidle")
        failures = []
        if not loading_disabled:
            failures.append("切换任务详情期间发送按钮仍可用")
        if unexpected_posts:
            failures.append("加载期间快捷键发出创建请求：" + str(unexpected_posts))
        if "任务不存在或无权访问" not in page.locator("#notice").inner_text():
            failures.append("详情404错误提示被吞")
        if page.locator("#run-status").inner_text() == "正在读取任务…":
            failures.append("详情失败后未退出加载状态")
        assert not failures, failures
        expect(page.locator("#send")).to_be_enabled()

        # 可追溯的500故障注入与真实服务错误分开断言，不能永久留在加载态。
        page.route(
            pattern,
            lambda route: route.fulfill(
                status=500,
                headers={"x-workbench-error-injection": "detail-read"},
                json={"detail": "验收注入：详情服务故障"},
            ),
        )
        page.locator("#run-list .run-item").filter(has_text="分页检索任务 22").click()
        expect(page.locator("#notice")).to_contain_text("验收注入：详情服务故障")
        expect(page.locator("#run-status")).to_have_text("任务读取失败")
        expect(page.locator("#send")).to_be_enabled()
        page.unroute(pattern)
        assert injected_server_errors == [arguments.url + "/api/v1/runs/" + ids[-1]]

        # 网络读取失败也必须结束加载，并允许重新打开任务。
        page.route(pattern, lambda route: route.abort("failed"))
        page.locator("#run-list .run-item").filter(has_text="分页检索任务 22").click()
        expect(page.locator("#notice")).to_contain_text("无法读取任务")
        expect(page.locator("#run-status")).to_have_text("任务读取失败")
        expect(page.locator("#send")).to_be_enabled()
        page.unroute(pattern)
        page.locator("#run-list .run-item").filter(has_text="分页检索任务 21").click()
        expect(page.locator(".message.user")).to_contain_text("分页检索任务 21")

        # 同任务视图重复导航不能作废仍在读取的详情或留下永久加载状态。
        repeated_chat = []
        page.route(pattern, lambda route: repeated_chat.append(route))
        with page.expect_request(pattern):
            page.locator("#run-list .run-item").filter(has_text="分页检索任务 22").click()
        expect(page.locator("#send")).to_be_disabled()
        page.locator('[data-view="chat"]').click()
        page.locator('[data-view="chat"]').click()
        assert repeated_chat, "没有捕获重复导航场景的详情读取请求"
        for route in repeated_chat:
            route.fulfill(response=route.fetch())
        page.unroute(pattern)
        expect(page.locator(".message.user")).to_contain_text("分页检索任务 22")
        expect(page.locator("#send")).to_be_enabled()
        expect(page.locator("#chat-view")).to_have_attribute("aria-busy", "false")

        # 页面刷新后能力读取尚未完成，用户切视图应优先于旧任务恢复链。
        delayed_startup = []
        page.route("**/api/v1/capabilities", lambda route: delayed_startup.append(route))
        with page.expect_request("**/api/v1/capabilities"):
            page.reload(wait_until="domcontentloaded")
            expect(page.locator("#shell")).to_be_visible()
        page.locator('[data-view="skills"]').click()
        expect(page.locator("#assets-view")).to_be_visible()
        expect(page.locator("#notice")).to_be_empty()
        assert delayed_startup, "没有捕获启动能力读取请求"
        for route in delayed_startup:
            route.fulfill(response=route.fetch())
        page.unroute("**/api/v1/capabilities")
        page.wait_for_load_state("networkidle")
        expect(page.locator('[data-view="skills"]')).to_have_class("nav-button selected")
        expect(page.locator("#assets-view")).to_be_visible()
        expect(page.locator(".message.user")).to_have_count(0)
        expect(page.locator("#notice")).to_be_empty()

        page.locator('[data-view="settings"]').click()
        expect(page.locator("#capability-list")).to_contain_text("模型")
        page.screenshot(path=str(arguments.output / "02-能力与设置.png"), full_page=True)
        page.locator("#theme-toggle").click()
        page.wait_for_function(
            "() => getComputedStyle(document.querySelector('#new-task')).color "
            "=== getComputedStyle(document.body).color"
        )
        page.screenshot(path=str(arguments.output / "03-浅色设置.png"), full_page=True)
        for width, height in [(390, 844), (768, 1024), (1440, 1000)]:
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
            page.screenshot(path=str(arguments.output / f"04-布局-{width}.png"), full_page=True)

        # 不可信内容必须作为文本；延迟的资产查询不能覆盖当前视图。
        injected = "<img src=x onerror=window.__xss=1>"
        response = page.request.post(
            arguments.url + "/api/v1/assets",
            headers=headers,
            data={"kind": "preference", "name": "文本边界", "content": injected},
        )
        assert response.status == 201, response.text()
        delayed_assets = []

        def hold_first_assets(route):
            if not delayed_assets:
                delayed_assets.append(route)
            else:
                route.continue_()

        page.route("**/api/v1/assets", hold_first_assets)
        page.locator('[data-view="skills"]').click()
        page.locator('[data-view="memories"]').click()
        expect(page.locator("#asset-list")).to_contain_text("文本边界")
        for route in delayed_assets:
            route.fulfill(response=route.fetch())
        page.unroute("**/api/v1/assets")
        page.wait_for_load_state("networkidle")
        expect(page.locator("#asset-list")).to_contain_text(injected)
        assert page.evaluate("window.__xss === undefined")
        assert page.locator("#asset-list img").count() == 0

        viewer_name = "viewer-" + str(time.time_ns())
        response = page.request.post(
            arguments.url + "/api/v1/auth/members",
            headers=headers,
            data={"username": viewer_name, "password": "workbench-test-password", "role": "viewer"},
        )
        assert response.status == 201, response.text()
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        page.locator("#username").fill(viewer_name)
        page.locator("#password").fill("workbench-test-password")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#send")).to_be_enabled()
        page.locator("#message").fill("计算只读任务")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        expect(page.locator("#feedback")).to_be_hidden()
        page.locator('[data-view="memories"]').click()
        expect(page.locator("#add-asset")).to_be_hidden()
        page.locator('[data-view="documents"]').click()
        expect(page.locator("#upload-form")).to_be_hidden()
        page.locator('[data-view="skills"]').click()
        expect(page.locator("#import-skill")).to_be_disabled()
        assert not errors, errors
        assert not server_errors, server_errors
        browser.close()
    print(
        json.dumps(
            {
                "工作台浏览器验收": "通过",
                "页面错误": errors,
                "HTTP服务错误": server_errors,
                "详情500故障注入": len(injected_server_errors),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
