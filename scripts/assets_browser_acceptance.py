"""验证内容管理的筛选、分批渲染、就近错误与负例，使用独立本机服务。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/内容管理"))
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
        page.wait_for_load_state("networkidle")
        page.locator("#username").fill("assets-ui-" + str(time.time_ns()))
        page.locator("#password").fill("assets-test-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        for number in range(30):
            response = page.request.post(
                args.url + "/api/v1/assets",
                headers=headers,
                data={
                    "kind": "preference",
                    "name": f"协作偏好 {number:02d}",
                    "content": (
                        "回答先给结论，再给可验证的依据。" * 50
                        if number == 0
                        else "优先使用中文解释；修改代码前说明范围，完成后提供测试结果与未验证项。"
                    ),
                    "status": "active" if number < 10 else "draft",
                    "metadata": {"tags": ["中文"], "description": "日常代码协作"},
                },
            )
            assert response.ok, response.text()
        page.locator('[data-view="memories"]').click()
        expect(page.locator(".asset-row")).to_have_count(24)
        expect(page.locator("#asset-count")).to_contain_text("24 / 30")
        page.locator("#assets-more").click()
        expect(page.locator(".asset-row")).to_have_count(30)
        expect(page.locator("#assets-more")).to_be_hidden()
        page.locator("#asset-search").fill("偏好 07")
        expect(page.locator(".asset-row")).to_have_count(1)
        page.locator("#asset-status-filter").select_option("draft")
        expect(page.locator(".asset-empty")).to_contain_text("没有匹配")
        page.locator("#asset-filters-clear").click()
        expect(page.locator("#asset-search")).to_be_focused()
        expect(page.locator(".asset-row")).to_have_count(24)
        page.locator("#asset-status-filter").select_option("active")
        expect(page.locator(".asset-row")).to_have_count(10)
        page.locator(".asset-content-preview summary").first.click()
        expect(page.locator(".asset-content-preview pre").first).to_contain_text("可验证的依据")
        page.locator(".asset-content-preview summary").first.click()
        page.screenshot(path=str(args.output / "01-记忆搜索与状态筛选.png"), full_page=True)
        page.locator("#asset-filters-clear").click()
        page.locator("#add-asset").click()
        expect(page.locator('#asset-kind option[value="project"]')).to_be_hidden()
        page.locator("#asset-name").fill("   ")
        page.locator("#asset-content").fill("   ")
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-error")).to_contain_text("不能只包含空白")
        expect(page.locator("#asset-error")).to_be_focused()
        page.locator("#asset-name").fill("日常编码偏好")
        page.locator("#asset-content").fill(
            "<script>保持纯文本</script>先列计划，修改后给出验证结果。"
        )
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-form")).to_be_hidden()
        row = page.locator(".asset-row").filter(
            has=page.get_by_role("heading", name="日常编码偏好")
        )
        expect(row).to_be_visible()
        expect(page.locator('#asset-form button[type="submit"]')).to_be_enabled()
        assert row.locator("script").count() == 0
        row.get_by_role("button", name="修订", exact=True).click()
        expect(page.locator("#asset-kind")).to_be_disabled()
        page.route(
            "**/api/v1/assets/*",
            lambda route: (
                route.fulfill(
                    status=409,
                    json={"detail": "内容已被修改，请刷新后核对最新版本"},
                )
                if route.request.method == "PUT"
                else route.continue_()
            ),
        )
        page.locator("#asset-content").fill("待保存的新内容，冲突后应保留。")
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-error")).to_contain_text("最新版本")
        expect(page.locator("#asset-content")).to_have_value("待保存的新内容，冲突后应保留。")
        page.screenshot(path=str(args.output / "02-就近错误与内容保留.png"), full_page=True)
        page.unroute("**/api/v1/assets/*")
        page.locator("#close-editor").click()
        page.locator("#asset-filters-clear").click()

        # 读取失败保留重试入口，恢复后重新获取本人内容。
        page.route(
            "**/api/v1/assets", lambda route: route.fulfill(status=503, json={"detail": "临时繁忙"})
        )
        page.locator("#assets-refresh").click()
        expect(page.locator("#asset-list")).to_contain_text("临时繁忙")
        page.unroute("**/api/v1/assets")
        page.locator("#asset-list").get_by_role("button", name="重新读取").click()
        expect(page.locator(".asset-row")).to_have_count(24)
        page.locator('[data-view="skills"]').click()
        expect(page.locator(".asset-empty")).to_contain_text("SKILL.md")
        page.locator("#add-asset").click()
        page.locator("#asset-name").fill("code-review")
        page.locator("#asset-content").fill("读取代码，核对边界；修改前先审批，完成后运行测试。")
        page.locator('#asset-form button[type="submit"]').click()
        expect(page.locator("#asset-list")).to_contain_text("code-review")
        page.screenshot(path=str(args.output / "03-技能候选管理.png"), full_page=True)
        page.locator('[data-view="documents"]').click()
        page.route(
            "**/api/v1/documents/upload",
            lambda route: route.fulfill(
                status=422,
                json={"detail": "没有抽取到文本，扫描PDF需要先OCR"},
            ),
        )
        page.locator("#document-file").set_input_files(
            {
                "name": "scan.pdf",
                "mimeType": "application/pdf",
                "buffer": b"%PDF-test",
            }
        )
        page.locator("#upload-form button").click()
        expect(page.locator("#upload-error")).to_contain_text("OCR")
        expect(page.locator("#upload-error")).to_be_focused()
        page.unroute("**/api/v1/documents/upload")
        page.locator('[data-view="settings"]').click()
        page.locator("#model-config-builder summary").click()
        expect(page.locator("#provider-copy")).to_be_disabled()
        page.locator("#provider-kind").select_option("ollama")
        page.locator("#provider-model").fill("local-coding-model")
        page.locator("#provider-add").click()
        page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=args.url)
        page.locator("#provider-copy").click()
        expect(page.locator("#provider-copy-status")).to_contain_text("配置已复制")
        page.evaluate("""() => {
            window.originalWrite = navigator.clipboard.writeText.bind(navigator.clipboard);
            navigator.clipboard.writeText = async () => { throw new Error('权限拒绝负例'); };
        }""")
        page.locator("#provider-copy").click()
        expect(page.locator("#provider-copy-status")).to_contain_text("请手动复制")
        assert "AEGIS_MODELS_JSON" in page.evaluate("window.getSelection().toString()")
        page.evaluate("navigator.clipboard.writeText = window.originalWrite")
        page.evaluate("""() => {
            navigator.clipboard.writeText = () => new Promise(resolve => {
                window.finishCopy = resolve;
            });
        }""")
        page.locator("#provider-copy").click()
        page.wait_for_function("typeof window.finishCopy === 'function'")
        page.locator("#provider-clear").click()
        page.evaluate("window.finishCopy()")
        expect(page.locator("#provider-copy-status")).to_be_empty()
        expect(page.locator("#provider-copy")).to_be_disabled()
        page.evaluate("navigator.clipboard.writeText = window.originalWrite")
        page.locator("#provider-add").click()
        page.locator(".settings-intro h3").click()
        page.screenshot(path=str(args.output / "04-模型接入步骤.png"), full_page=True)
        page.locator("#model-config-preview").scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / "06-模型配置与复制.png"), full_page=True)
        for width in (390, 768, 1440):
            page.set_viewport_size({"width": width, "height": 900})
            for view in ("memories", "skills", "documents", "settings"):
                page.locator(f'[data-view="{view}"]').click()
                expect(page.locator("#view-title")).not_to_be_empty()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    width,
                    view,
                )
        page.locator("#theme-toggle").click()
        page.locator('[data-view="memories"]').click()
        page.locator("#asset-search").fill("偏好 07")
        expect(page.locator(".asset-row")).to_have_count(1)
        page.screenshot(path=str(args.output / "05-深色内容管理.png"), full_page=True)
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#asset-search")).to_have_value("")
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {
                "page_errors": errors,
                "negative_cases": ["空白", "XSS", "版本冲突", "503重试", "扫描件422"],
                "responsive_cases": 12,
                "clipboard_cases": ["复制成功", "权限拒绝手动复制", "清空后过期响应失效"],
                "evidence": "独立数据库与故障注入，不代表模型效果",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
