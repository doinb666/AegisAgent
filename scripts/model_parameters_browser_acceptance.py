"""真实HTTP验证任务模型参数、来源切换、响应丢失恢复与旧服务降级。"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/模型参数"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许独立本机验收服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.locator("#username").fill("parameters-ui-" + str(time.time_ns()))
        page.locator("#password").fill("model-parameters-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#model-parameters")).to_be_visible()
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        capabilities = page.request.get(args.url + "/api/v1/capabilities", headers=headers).json()
        primary = next(
            route for route in capabilities["model_routes"] if route["parameters"]["temperature"]
        )
        backup = next(
            route
            for route in capabilities["model_routes"]
            if not route["parameters"]["temperature"]
        )
        page.locator("#model-parameters").evaluate("element => element.open = true")
        expect(page.locator("#model-temperature")).to_be_disabled()
        expect(page.locator("#model-output-tokens")).to_have_attribute("max", "512")
        expect(page.locator("#model-parameters-hint")).to_contain_text("共同支持")
        page.locator("#model").select_option(primary["id"])
        expect(page.locator("#model-temperature")).to_be_enabled()
        page.locator("#model-temperature").fill("0")
        page.locator("#model-output-tokens").fill("64")
        page.locator("#model-reasoning").select_option("medium")
        expect(page.locator("#model-parameters-summary")).to_have_text("已调整 3 项")
        page.locator("#message").fill("计算 2+3，使用当前模型参数")
        page.screenshot(path=str(args.output / "01-按来源调整参数.png"))
        committed = []

        def lost_response(route):
            response = route.fetch()
            assert response.ok
            committed.append((response.json()["id"], route.request.headers["idempotency-key"]))
            route.fulfill(status=503, json={"detail": "模型任务响应暂不可用"})

        page.route("**/api/v1/runs", lost_response, times=1)
        page.locator("#send").click()
        expect(page.locator("#notice")).to_contain_text("暂不可用")
        expect(page.locator("#model-temperature")).to_have_value("0")
        expect(page.locator("#model-output-tokens")).to_have_value("64")
        expect(page.locator("#model-reasoning")).to_have_value("medium")
        expect(page.locator("#message")).to_have_value("计算 2+3，使用当前模型参数")
        page.locator("#send").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        runs = page.request.get(args.url + "/api/v1/runs", headers=headers).json()
        assert len(runs) == 1 and runs[0]["id"] == committed[0][0]
        expected = {"temperature": 0, "max_output_tokens": 64, "reasoning_effort": "medium"}
        assert runs[0]["model_parameters"] == expected
        event_stream = page.request.get(
            args.url + f"/api/v1/runs/{runs[0]['id']}/events", headers=headers
        ).text()
        model_events = [
            json.loads(line.removeprefix("data: "))
            for block in event_stream.split("\n\n")
            if "event: model\n" in block
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        assert any(event["model_parameters"] == expected for event in model_events)

        page.locator("#new-task").click()
        page.locator("#model-parameters").evaluate("element => element.open = true")
        expect(page.locator("#model-temperature")).to_have_value("")
        expect(page.locator("#model-output-tokens")).to_have_value("")
        expect(page.locator("#model-reasoning")).to_have_value("")
        page.locator("#model-temperature").fill("0.4")
        page.locator("#model-reasoning").select_option("high")
        page.locator("#model").select_option(backup["id"])
        expect(page.locator("#model-temperature")).to_be_disabled()
        expect(page.locator("#model-temperature")).to_have_value("")
        expect(page.locator("#model-reasoning")).to_have_value("")
        expect(page.locator("#model-parameters-status")).to_contain_text("已清除")
        page.locator("#model-output-tokens").fill("513")
        page.locator("#message").fill("超限参数不得提交")
        page.locator("#send").click()
        assert not page.locator("#model-output-tokens").evaluate(
            "element => element.validity.valid"
        )
        assert len(page.request.get(args.url + "/api/v1/runs", headers=headers).json()) == 1
        page.locator("#model-parameters-reset").click()
        page.locator("#model-parameters").evaluate("element => element.open = true")
        expect(page.locator("#model-output-tokens")).to_have_value("")
        for theme in ("light", "dark"):
            if page.locator("html").get_attribute("data-theme") != theme:
                page.locator("#theme-toggle").click()
            for width in (390, 768, 1440):
                page.set_viewport_size({"width": width, "height": 1000})
                page.locator("#model-reasoning").scroll_into_view_if_needed()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#model-output-tokens")).to_have_value("")

        def legacy_capabilities(route):
            response = route.fetch()
            assert response.ok
            data = response.json()
            data.pop("model_parameters", None)
            for item in data.get("model_routes", []):
                item.pop("parameters", None)
            route.fulfill(response=response, json=data)

        page.route("**/api/v1/capabilities", legacy_capabilities, times=1)
        page.locator("#password").fill("model-parameters-password")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#model-parameters")).to_be_hidden()
        expect(page.locator("#send")).to_be_enabled()
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {"页面错误": errors, "响应式场景": 6, "模型参数": "真实HTTP与确定性模型"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
