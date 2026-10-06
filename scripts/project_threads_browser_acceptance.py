"""真实项目多会话、归档、分页与窄屏验收；只连接本机独立测试服务。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/项目会话"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许本机独立测试服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.locator("#username").fill("project-browser-" + str(time.time_ns()))
        page.locator("#password").fill("project-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#project-thread-navigation")).to_be_visible()
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}

        def request(method, path, body=None):
            result = getattr(page.request, method)(
                args.url + "/api/v1" + path,
                headers=headers,
                **({"data": body} if body is not None else {}),
            )
            assert result.ok, result.text()
            return result.json()

        project = request(
            "post",
            "/assets",
            {
                "kind": "project",
                "name": "AegisCode · 边界审查",
                "content": "只分析测试工作区，不访问生产配置。",
                "status": "active",
            },
        )
        injection = "<img src=x onerror=alert(1)>"
        request(
            "post",
            "/assets",
            {"kind": "project", "name": injection, "content": "注入边界测试", "status": "active"},
        )
        page.locator("#threads-refresh").click()
        group = page.locator(".project-thread-group").filter(
            has=page.locator("summary", has_text=project["name"])
        )
        expect(group).to_be_visible()
        group.locator("summary").click()
        group.get_by_role("button", name="新建项目会话").click()
        expect(page.locator("#current-project-label")).to_contain_text(project["name"])
        page.locator("#thread-rename").click()
        page.locator("#thread-title-input").fill("检查参数与权限边界")
        page.locator("#thread-rename-form").get_by_role("button", name="保存").click()
        expect(page.locator("#current-thread-title")).to_have_text("检查参数与权限边界")
        first = request("get", "/threads")[0]
        page.locator("#message").fill("第一会话：计算边界")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        expect(page.locator("#recent-thread-list")).to_contain_text("检查参数与权限边界")
        # 每次刷新会重绘项目树，需要按当前DOM重新展开。
        group.locator("summary").click()
        group.get_by_role("button", name="新建项目会话").click()
        expect(page.locator("#current-thread-title")).to_have_text("新会话")
        page.locator("#thread-rename").click()
        page.locator("#thread-title-input").fill("核对并发与重试")
        page.locator("#thread-rename-form").get_by_role("button", name="保存").click()
        expect(page.locator("#current-thread-title")).to_have_text("核对并发与重试")
        page.locator("#message").fill("第二会话：计算并发")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        page.locator(f"#recent-thread-list [data-thread-id='{first['id']}']").click()
        expect(page.locator("#conversation")).to_contain_text("第一会话：计算边界")
        expect(page.locator("#conversation")).not_to_contain_text("第二会话：计算并发")
        expect(page.locator("#current-thread-title")).to_have_text("检查参数与权限边界")
        page.locator("#thread-archive").click()
        expect(page.locator("#thread-archived-label")).to_be_visible()
        expect(page.locator("#send")).to_be_disabled()
        page.locator("#thread-archive").click()
        expect(page.locator("#send")).to_be_enabled()
        group.locator("summary").click()
        expect(group.locator(".thread-item")).to_have_count(2)
        page.screenshot(path=str(args.output / "01-项目独立会话.png"), full_page=True)
        for index in range(21):
            request("post", "/threads", {"title": f"分页边界{index}", "project_id": project["id"]})
        page.locator("#threads-refresh").click()
        expect(page.locator("#recent-thread-list .thread-item")).to_have_count(20)
        page.locator("#threads-more").click()
        expect(page.locator("#recent-thread-list .thread-item")).to_have_count(23)
        group.locator("summary").click()
        expect(group.locator(".thread-item")).to_have_count(5)
        group.get_by_role("button", name="展开更多会话").click()
        expect(group.locator(".thread-item")).to_have_count(10)
        assert page.locator("#project-thread-list img").count() == 0
        expect(page.locator("#project-thread-list")).to_contain_text(injection)
        page.locator("#theme-toggle").click()
        page.screenshot(path=str(args.output / "02-深色项目树.png"), full_page=True)
        for width in (390, 768):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (
                f"{width}px横向溢出"
            )
        page.screenshot(path=str(args.output / "03-窄屏项目会话.png"), full_page=True)
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#project-thread-navigation")).to_be_hidden()
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {
                "checks": 13,
                "page_errors": errors,
                "model": "确定性测试模型",
                "data": "独立临时数据库",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
