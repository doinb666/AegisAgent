"""真实浏览器验收；只连接本地测试服务，不调用外部模型。"""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--tools", action="store_true", help="验收确定性模型测试服务的审批链")
    parser.add_argument("--output", type=Path, default=Path("docs/升级方案/截图"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(arguments.url)
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(arguments.output / "01-登录.png"), full_page=True)
        page.locator("#username").fill("ui-test-" + str(time.time_ns()))
        page.locator("#password").fill("ui-test-password-123")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.screenshot(path=str(arguments.output / "02-任务工作台.png"), full_page=True)
        page.locator('[data-view="memories"]').click()
        page.locator("#add-asset").click()
        page.locator("#asset-name").fill("回复偏好")
        page.locator("#asset-content").fill("回答使用中文，先给结论。")
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-list")).to_contain_text("回复偏好")
        page.get_by_role("button", name="启用", exact=True).click()
        expect(page.locator("#asset-list")).to_contain_text("已启用")
        page.locator('[data-view="projects"]').click()
        page.locator("#add-asset").click()
        page.locator("#asset-name").fill("验证项目")
        page.locator("#asset-content").fill("只处理本测试工作区")
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-list")).to_contain_text("验证项目")
        page.locator('[data-view="documents"]').click()
        page.locator("#document-file").set_input_files(
            {
                "name": "notes.md",
                "mimeType": "text/markdown",
                "buffer": "文档仅允许当前用户召回".encode(),
            }
        )
        page.locator("#upload-form button").click()
        expect(page.locator("#asset-list")).to_contain_text("notes.md", timeout=20000)
        page.locator("#new-task").click()
        page.locator("#message").fill("写文件" if arguments.tools else "未配置模型时请明确失败")
        page.locator("#send").click()
        if arguments.tools:
            expect(page.locator("#approval")).to_be_visible(timeout=20000)
            page.screenshot(path=str(arguments.output / "03-审批.png"), full_page=True)
            page.locator("#approve").click()
            expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
            page.locator("#success").click()
            expect(page.locator("#notice")).to_contain_text("反馈已记录")
            page.locator('[data-view="skills"]').click()
            expect(page.locator("#asset-list")).to_contain_text("任务技能候选")
            page.screenshot(path=str(arguments.output / "04-Skills.png"), full_page=True)
        else:
            expect(page.locator("#run-status")).to_have_text("执行失败", timeout=20000)
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        page.locator("#password").fill("ui-test-password-123")
        page.locator("#login").click()
        expect(page.locator("#memory-preview")).to_contain_text("已加载")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(arguments.output / "05-移动端.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "移动端横向溢出"
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {"浏览器验收": "通过", "页面错误": errors, "工具审批": arguments.tools},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
