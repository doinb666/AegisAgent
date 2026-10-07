"""验证只读定时计划的创建、暂停、取消、错误恢复与窄屏交互。"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/定时任务"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许独立本机验收服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.locator("#username").fill("schedule-ui-" + str(time.time_ns()))
        page.locator("#password").fill("schedule-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        held_capabilities = []
        page.route(
            "**/api/v1/capabilities",
            lambda route: held_capabilities.append((route, route.fetch())),
            times=1,
        )
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.get_by_role("button", name="定时任务", exact=True).click()
        assert len(held_capabilities) == 1
        route, response = held_capabilities.pop()
        route.fulfill(response=response)
        expect(page.locator("#schedule-list")).to_contain_text("还没有定时计划")
        page.locator("#schedule-add").click()
        page.locator("#schedule-title").fill("   ")
        page.locator("#schedule-message").fill("   ")
        future = datetime.now().astimezone() + timedelta(hours=1)
        page.locator("#schedule-time").fill(future.strftime("%Y-%m-%dT%H:%M"))
        page.locator("#schedule-form").get_by_role("button", name="创建计划").click()
        expect(page.locator("#schedule-error")).to_contain_text("不能只包含空白")
        page.locator("#schedule-title").fill("每日知识库检查")
        page.locator("#schedule-message").fill("检索我的知识库，列出证据不足的内容。")
        page.locator("#schedule-frequency").select_option("interval")
        page.locator("#schedule-interval").fill("3600")
        # 模拟提交已持久化、客户端却只收到错误；原输入和幂等键须用于恢复。
        def lose_create_response(route):
            assert route.fetch().ok
            route.fulfill(status=503, json={"detail": "响应丢失，请重试"})

        page.route("**/api/v1/schedules", lose_create_response, times=1)
        page.locator("#schedule-form").get_by_role("button", name="创建计划").click()
        expect(page.locator("#schedule-error")).to_contain_text("响应丢失")
        expect(page.locator("#schedule-title")).to_have_value("每日知识库检查")
        page.clock.install(time=datetime.now().astimezone() + timedelta(hours=2))
        page.locator("#schedule-form").get_by_role("button", name="创建计划").click()
        row = page.locator(".schedule-row").filter(has_text="每日知识库检查")
        expect(row).to_be_visible()
        expect(row).to_have_count(1)
        page.clock.set_system_time(datetime.now().astimezone())
        page.route(
            "**/api/v1/schedules/*",
            lambda route: route.fulfill(status=409, json={"detail": "计划版本已变更"}),
            times=1,
        )
        row.get_by_role("button", name="暂停", exact=True).click()
        expect(page.locator("#schedules-status")).to_contain_text("版本已变更")
        expect(row).to_contain_text("已启用")
        page.locator("#schedules-refresh").click()
        expect(page.locator("#schedules-status")).to_have_text("")
        row.get_by_role("button", name="暂停", exact=True).click()
        expect(row).to_contain_text("已暂停")

        def old_detail(route):
            data = route.fetch().json()
            data.update(status="active", version=1)
            route.fulfill(json=data)

        page.route("**/api/v1/schedules/*", old_detail, times=1)
        row.get_by_role("button", name="查看触发记录").click()
        expect(page.locator("#schedule-detail")).to_contain_text("暂无触发记录")
        expect(row).to_contain_text("已暂停")

        def old_list(route):
            data = route.fetch().json()
            for item in data:
                item.update(status="active", version=1)
            route.fulfill(json=data)

        page.route("**/api/v1/schedules?*", old_list, times=1)
        page.locator("#schedules-refresh").click()
        expect(page.locator("#schedules-status")).to_have_text("")
        expect(row).to_contain_text("已暂停")
        row.get_by_role("button", name="恢复", exact=True).click()
        expect(row).to_contain_text("已启用")
        row.get_by_role("button", name="查看触发记录").click()
        expect(page.locator("#schedule-detail")).to_contain_text("暂无触发记录")
        page.screenshot(path=str(args.output / "01-计划管理与只读边界.png"))
        row.get_by_role("button", name="取消计划", exact=True).click()
        expect(row).to_contain_text("已取消")
        expect(row.get_by_role("button", name="恢复", exact=True)).to_have_count(0)

        # 触发记录来自真实本机HTTP与后台任务，不依赖页面创建连接继续存活。
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        response = page.request.post(
            args.url + "/api/v1/schedules",
            headers={"Authorization": "Bearer " + token, "Idempotency-Key": "once-ui"},
            data={
                "title": "一次性证据核对",
                "message": "计算 2+2，保留工具证据",
                "scheduled_at": (datetime.now(UTC) + timedelta(seconds=2)).isoformat(),
            },
        )
        assert response.ok, response.text()
        schedule_id = response.json()["id"]
        page.locator("#schedules-refresh").click()
        once = page.locator(".schedule-row").filter(has_text="一次性证据核对")
        once.get_by_role("button", name="查看触发记录").click()
        expect(page.locator("#schedule-detail")).to_contain_text("打开执行任务", timeout=20000)
        detail = page.request.get(
            args.url + "/api/v1/schedules/" + schedule_id,
            headers={"Authorization": "Bearer " + token},
        ).json()
        assert len(detail["occurrences"]) == 1 and detail["occurrences"][0]["run_id"]
        page.screenshot(path=str(args.output / "02-持久触发记录.png"))

        page.route(
            "**/api/v1/schedules?*",
            lambda route: route.fulfill(status=503, json={"detail": "暂时不可用，请稍后重试"}),
        )
        page.locator("#schedules-refresh").click()
        expect(page.locator("#schedules-status")).to_contain_text("稍后重试")
        page.unroute("**/api/v1/schedules?*")
        page.locator("#schedules-refresh").click()
        expect(page.locator("#schedules-status")).to_have_text("")
        for theme in ("light", "dark"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for width in (390, 768, 1440):
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#schedule-list")).to_be_empty()
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {"页面错误": errors, "响应式场景": 6, "证据": "独立数据库与确定性模型"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
