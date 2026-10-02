"""注册边界与错误提示验收：拦截请求，不写入用户数据库。"""

import argparse
import json
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("仅允许本机服务")
    requests = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def validation_response(route):
            requests.append(route.request.url)
            route.fulfill(status=422, json={"detail": [{
                "loc": ["body", "password"], "type": "string_too_short",
                "msg": "secret-must-not-appear", "input": "secret-must-not-appear",
                "ctx": {"min_length": 8},
            }]})

        page.route("**/api/v1/auth/register", validation_response)
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.locator("#register").click()
        expect(page.locator("#username")).to_be_focused()
        assert not requests, "空表单不应发送注册请求"
        page.locator("#username").fill("registration-boundary-test")
        page.locator("#password").fill("short")
        page.locator("#register").click()
        expect(page.locator("#password")).to_be_focused()
        assert not requests, "短密码不应发送注册请求"
        page.locator("#password").fill("test-password-valid")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_contain_text("密码")
        expect(page.locator("#auth-error")).to_contain_text("8")
        assert "secret-must-not-appear" not in page.locator("#auth-error").inner_text()
        assert len(requests) == 1
        page.unroute("**/api/v1/auth/register", validation_response)
        page.route("**/api/v1/auth/register", lambda route: route.fulfill(
            status=503, content_type="text/html", body="<h1>upstream secret</h1>"))
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_contain_text("503")
        assert "upstream secret" not in page.locator("#auth-error").inner_text()
        malformed = page.evaluate("""() => WorkspaceUI.requestError({detail: [null,
            {loc: {}, type: 'unknown', input: 'hidden'}]}, 422)""")
        assert "格式不正确" in malformed and "hidden" not in malformed
        for location in [{"toString": "invalid"}, "toString", "__proto__"]:
            rendered = page.evaluate("""location => WorkspaceUI.requestError({detail: [
                {loc: ['body', location], type: 'missing'}]}, 422)""", location)
            assert rendered == "提交字段不能为空。"
        # 界面夹具验证审批展开，所有请求仍拦截，不创建真实运行。
        page.evaluate("""() => {
            document.getElementById('shell').hidden = false;
            WorkspaceUI.setInspector(false);
            renderRun({id:'approval-fixture', status:'waiting_approval', step:1,
                approval:{name:'file_write', arguments:{path:'result.txt'}}});
        }""")
        expect(page.locator("#inspector")).to_be_visible()
        expect(page.locator("#approval")).to_be_visible()
        assert not errors, errors
        browser.close()
    result = {"passed": 9, "page_errors": errors, "database_writes": 0}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
