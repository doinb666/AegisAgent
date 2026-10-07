"""工作台布局、易用性与长标题负例验收，仅使用独立本机测试服务。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/工作台新版"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许本机独立测试服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.locator("#username").fill("layout-" + str(time.time_ns()))
        page.locator("#password").fill("layout-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#workspace-theme-slot #theme-toggle")).to_be_visible()
        page.locator("#workspace-summary summary").click()
        expect(page.locator("#memory-preview")).to_contain_text("偏好")
        page.locator("#workspace-summary summary").click()
        expect(page.locator("#workspace-overview")).to_be_hidden()
        page.locator("#mode").select_option("plan")
        expect(page.locator("#mode-hint")).to_contain_text("先拆解步骤")
        page.locator("#mode").select_option("react")
        page.locator(".suggestions button").first.click()
        expect(page.locator("#message")).to_have_value("根据我的知识库回答问题，并列出来源和不足。")
        page.locator("#message").fill("")
        page.screenshot(path=str(args.output / "01-浅色工作空间.png"), full_page=True)

        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        response = page.request.post(
            args.url + "/api/v1/assets",
            headers=headers,
            data={
                "kind": "project",
                "name": "AegisCode · 日常代码审查",
                "content": "先只读分析，修改前提供具体方案与验证步骤。",
                "status": "active",
            },
        )
        assert response.ok, response.text()
        project = response.json()
        page.locator("#sidebar-projects").click()
        expect(page.locator("#run-navigation")).to_be_hidden()
        expect(page.locator("#sidebar-projects")).to_have_attribute("aria-pressed", "true")
        page.locator("#threads-refresh").click()
        group = page.locator(f"[data-project-id='{project['id']}']")
        expect(group).to_be_visible()
        group.locator("summary").click()
        group.get_by_role("button", name="新建项目会话").click()
        expect(page.locator("#current-project-label")).to_contain_text(project["name"])
        expect(page.locator("#threads-status")).to_have_text("")
        page.locator("#thread-rename").click()
        page.locator("#thread-title-input").fill("核对参数校验与权限边界")
        page.locator("#thread-rename-form").get_by_role("button", name="保存标题").click()
        expect(page.locator("#current-thread-title")).to_have_text("核对参数校验与权限边界")
        expect(page.locator("#recent-thread-list .current")).to_have_count(1)
        page.screenshot(path=str(args.output / "02-项目会话与输入区.png"), full_page=True)

        page.locator("#message").fill("写文件并核对预览")
        page.locator("#send").click()
        expect(page.locator("#approval")).to_be_visible(timeout=20000)
        page.locator("#approve").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        page.screenshot(path=str(args.output / "03-任务与审查证据.png"), full_page=True)
        page.locator("#theme-toggle").click()
        page.screenshot(path=str(args.output / "04-深色工作台.png"), full_page=True)
        page.locator("#theme-toggle").click()

        # 输入保持纯文本，极长会话名不能撑宽布局或遮挡发送按钮。
        long_title = "<img src=x onerror=alert(1)>" + "边界" * 85
        page.locator("#thread-rename").click()
        page.locator("#thread-title-input").fill(long_title)
        page.locator("#thread-rename-form").get_by_role("button", name="保存标题").click()
        expect(page.locator("#current-thread-title")).to_have_text(long_title)
        assert page.locator("#thread-context img").count() == 0
        assert page.locator("#recent-thread-list img").count() == 0
        page.keyboard.press("Control+k")
        page.locator("#command-query").fill("搜索任务")
        page.keyboard.press("Enter")
        expect(page.locator("#run-search")).to_be_focused()
        expect(page.locator("#sidebar-history")).to_have_attribute("aria-pressed", "true")

        viewports = [(1440, 900), (1280, 720), (1024, 768), (768, 1024), (390, 844), (844, 390)]
        for theme in ("light", "dark"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for width, height in viewports:
                page.set_viewport_size({"width": width, "height": height})
                expanded = page.locator("#sidebar-toggle").get_attribute("aria-expanded") == "true"
                if width <= 700 and expanded:
                    page.locator("#sidebar-toggle").click()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    theme,
                    width,
                )
                page.locator("#send").scroll_into_view_if_needed()
                assert page.locator("#send").evaluate("""element => {
                    const r = element.getBoundingClientRect();
                    return r.left >= 0 && r.right <= innerWidth
                        && r.top >= 0 && r.bottom <= innerHeight;
                }"""), (theme, width, height)
                assert page.locator("#send").evaluate("""element => {
                    const r = element.getBoundingClientRect();
                    const target = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
                    return element.contains(target);
                }"""), (theme, width, height, "发送区被遮挡")
                page.locator(".task-options summary").click()
                page.locator("#project-mode").scroll_into_view_if_needed()
                expect(page.locator("#project-mode")).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.locator(".task-options summary").click()

        page.set_viewport_size({"width": 390, "height": 844})
        page.emulate_media(reduced_motion="reduce")
        page.locator("#theme-toggle").click()
        page.keyboard.press("Alt+n")
        expect(page.locator("#welcome")).to_be_visible()
        page.locator("#send").scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / "05-窄屏工作台.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 900})
        if page.locator("#sidebar-toggle").get_attribute("aria-expanded") == "false":
            page.locator("#sidebar-toggle").click()
        page.locator("#logout").click()
        expect(page.locator("#auth-theme-slot #theme-toggle")).to_be_visible()
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {
                "viewport_theme_cases": len(viewports) * 2,
                "page_errors": errors,
                "evidence": "独立数据库、真实HTTP与确定性模型，非模型效果评估",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
