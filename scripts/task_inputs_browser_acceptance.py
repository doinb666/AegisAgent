"""验证模板仅预填、私有资料引用、来源展示和MCP配置导览。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/任务输入"))
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
        page.locator("#username").fill("inputs-ui-" + str(time.time_ns()))
        page.locator("#password").fill("task-inputs-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.get_by_role("button", name="任务模板", exact=True).click()
        expect(page.locator("#template-list")).to_contain_text("计算核对")
        page.screenshot(path=str(args.output / "01-模板与验收要求.png"))
        page.locator(".template-row").filter(has_text="计算核对").get_by_role(
            "button", name="预填任务"
        ).click()
        expect(page.locator("#message")).not_to_be_empty()
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        assert page.request.get(args.url + "/api/v1/runs", headers=headers).json() == []
        page.locator("#message").fill("保留我的原始目标")
        page.get_by_role("button", name="任务模板", exact=True).click()
        row = page.locator(".template-row").filter(has_text="计算核对")
        row.get_by_role("button", name="预填任务").click()
        expect(page.locator("#template-replace")).to_be_visible()
        page.locator("#template-keep").click()
        expect(page.locator("#message")).to_have_value("保留我的原始目标")

        uploaded = page.request.post(
            args.url + "/api/v1/documents/upload",
            headers={**headers, "Idempotency-Key": "ui-private-document"},
            multipart={
                "file": {
                    "name": "私有证据.md",
                    "mimeType": "text/markdown",
                    "buffer": "# 私有资料\n计算核对必须保留来源。".encode(),
                }
            },
        )
        assert uploaded.ok, uploaded.text()
        page.get_by_role("button", name="任务空间", exact=True).click()
        hostile_name = "<img src=x onerror=\"document.title='资料脚本执行'\">"
        page.route(
            "**/api/v1/documents/references?*",
            lambda route: route.fulfill(
                json=[
                    {"id": f"boundary-{index}", "name": name, "version": 1}
                    for index, name in enumerate([hostile_name, "边界二", "边界三", "边界四"])
                ]
            ),
            times=1,
        )
        page.locator("#task-documents").evaluate("element => element.open = true")
        expect(page.locator("#reference-list")).to_contain_text(hostile_name)
        expect(page.locator("#reference-list img")).to_have_count(0)
        choices = page.locator("#reference-list input")
        for index in range(3):
            choices.nth(index).check()
        expect(choices.nth(3)).to_be_disabled()
        page.locator("#reference-selected button").first.click()
        expect(choices.nth(3)).to_be_enabled()
        page.locator("#new-task").click()
        expect(page.locator("#reference-selected")).to_be_empty()
        page.locator("#task-documents").evaluate("element => element.open = true")
        page.locator("#references-refresh").click()
        page.locator("#reference-list").get_by_label("私有证据.md").check()
        expect(page.locator("#reference-selected")).to_contain_text("私有证据.md")
        page.locator("#message").fill("计算 2+3，参考已选择资料并保留来源")
        welcome = page.locator("#welcome p").bounding_box()
        conversation = page.locator("#conversation").bounding_box()
        assert (
            welcome
            and conversation
            and welcome["y"] + welcome["height"] <= conversation["y"] + conversation["height"]
        ), "展开资料后欢迎说明不应被输入区裁切"
        page.screenshot(path=str(args.output / "02-本人资料选择.png"))
        committed = []

        def lost_run_response(route):
            response = route.fetch()
            assert response.ok
            committed.append(response.json()["id"])
            route.fulfill(status=503, json={"detail": "任务创建暂不可用"})

        page.route("**/api/v1/runs", lost_run_response, times=1)
        page.locator("#send").click()
        expect(page.locator("#notice")).to_contain_text("暂不可用")
        expect(page.locator("#reference-selected")).to_contain_text("私有证据.md")
        expect(page.locator("#message")).to_have_value("计算 2+3，参考已选择资料并保留来源")
        page.locator("#send").click()
        expect(page.locator("#run-documents")).to_contain_text("私有证据.md")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        runs = page.request.get(args.url + "/api/v1/runs", headers=headers).json()
        assert len(runs) == 1 and runs[0]["id"] == committed[0]
        page.locator("#new-task").click()
        expect(page.locator("#run-documents")).to_be_hidden()
        expect(page.locator("#reference-selected")).to_be_empty()
        page.get_by_role("button", name="能力与设置", exact=True).click()
        expect(page.locator("#mcp-guide-list")).to_contain_text("暂无获授权")
        page.screenshot(path=str(args.output / "03-MCP授权导览.png"))
        page.route(
            "**/api/v1/mcp/servers",
            lambda route: route.fulfill(status=503, json={"detail": "配置暂不可用，请刷新"}),
            times=1,
        )
        page.locator("#mcp-guide-refresh").click()
        expect(page.locator("#mcp-guide-status")).to_contain_text("暂不可用")
        page.locator("#mcp-guide-refresh").click()
        expect(page.locator("#mcp-guide-status")).to_have_text("")
        for theme in ("light", "dark"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for width in (390, 768, 1440):
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        # 模板打开旧任务时跨越网络等待，退出后旧预填不能污染登录表单或后续账号。
        page.locator(".run-item").filter(has_text="计算 2+3，参考已选择资料并保留来源").click()
        expect(page.locator("#run-documents")).to_contain_text("私有证据.md")
        page.get_by_role("button", name="任务模板", exact=True).click()
        held_run = []
        page.route(
            "**/api/v1/runs/" + committed[0],
            lambda route: held_run.append((route, route.fetch())),
            times=1,
        )
        page.locator(".template-row").filter(has_text="计算核对").get_by_role(
            "button", name="预填任务"
        ).click()
        expect(page.locator("#chat-view")).to_have_attribute("aria-busy", "true")
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        assert len(held_run) == 1
        route, response = held_run.pop()
        route.fulfill(response=response)
        expect(page.locator("#login")).to_be_enabled()
        expect(page.locator("#message")).to_have_value("")
        expect(page.locator("#reference-selected")).to_be_empty()
        expect(page.locator("#mcp-guide-list")).to_be_empty()

        # 旧服务缺能力字段时只降级新增入口，仍可使用原有工作台。
        def legacy_capabilities(route):
            response = route.fetch()
            assert response.ok
            data = response.json()
            data.pop("task_inputs", None)
            route.fulfill(response=response, json=data)

        page.route("**/api/v1/capabilities", legacy_capabilities, times=1)
        page.locator("#password").fill("task-inputs-password")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.get_by_role("button", name="任务模板", exact=True).click()
        expect(page.locator("#templates-unavailable")).to_be_visible()
        expect(page.locator("#templates-refresh")).to_be_hidden()
        expect(page.locator("#template-list")).to_be_empty()
        page.get_by_role("button", name="任务空间", exact=True).click()
        expect(page.locator("#task-documents")).to_be_hidden()
        expect(page.locator("#send")).to_be_enabled()
        page.get_by_role("button", name="能力与设置", exact=True).click()
        expect(page.locator("#mcp-guide")).to_be_hidden()
        assert page.title() != "资料脚本执行"
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {"页面错误": errors, "响应式场景": 6, "资料": "真实私有上传"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
