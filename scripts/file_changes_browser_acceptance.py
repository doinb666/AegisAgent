"""本机真实HTTP验证文件差异审批、基线绑定与冲突保留，不代表商业模型效果。"""

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from app.harness import Principal
from app.harness_tools.workspace import workspace_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/文件差异"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许本机独立验收服务")
    data = Path(os.environ.get("AEGIS_UI_TEST_DATA", "")).resolve()
    if data.name != "data" or not data.parent.name.startswith("aegis-browser-"):
        parser.error("必须通过run_ui_acceptance在独立临时数据库运行")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        # 截图使用减少动画偏好，避免捕获深浅主题切换的过渡色。
        page.emulate_media(reduced_motion="reduce")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: (errors.append(dialog.message), dialog.dismiss()))
        page.goto(args.url)
        page.locator("#username").fill("file-change-ui-" + str(time.time_ns()))
        page.locator("#password").fill("file-change-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        token = page.evaluate("sessionStorage.getItem('aegis-token')")
        headers = {"Authorization": "Bearer " + token}
        identity = page.request.get(args.url + "/api/v1/auth/me", headers=headers).json()
        principal = Principal(identity["user_id"], identity["tenant_id"], identity["role"])

        def submit(message="写文件并核对差异与基线", large=False):
            page.locator("#new-task").click()
            page.locator("#message").fill(message)
            page.locator("#send").click()
            expect(page.locator("#approval")).to_be_visible(timeout=20000)
            expect(page.locator("#file-change")).to_be_visible()
            expect(page.locator("#file-change-title")).to_contain_text("新建")
            if large:
                expect(page.locator("#file-change-warning")).to_contain_text("截断")
                expect(page.locator("#file-change-diff img")).to_have_count(0)
            else:
                expect(page.locator("#file-change-diff")).to_contain_text("+已批准")
            run_id = page.evaluate("sessionStorage.getItem('aegis-last-run')")
            run = page.request.get(args.url + "/api/v1/runs/" + run_id, headers=headers).json()
            return run, data / "workspaces" / workspace_key(principal, run_id) / "result.txt"

        run, target = submit()
        assert not target.exists(), "审批前不得写入目标"
        approval = run["approval"]
        bad = page.request.post(
            args.url + "/api/v1/runs/" + run["id"] + "/approval",
            headers=headers,
            data={
                "approved": True,
                "call_id": approval["call_id"],
                "args_hash": approval["hash"],
                "baseline_hash": "0" * 64,
            },
        )
        assert bad.status == 409
        assert not target.exists()
        for theme in ("light", "dark"):
            page.locator("html").evaluate(
                "(element, theme) => element.dataset.theme = theme", theme
            )
            for width in (1440, 768, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("html").evaluate("element => element.dataset.theme = 'light'")
        page.screenshot(path=str(args.output / "01-新建文件差异审批.png"), full_page=True)
        page.locator("#approve").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        assert target.read_bytes() == "已批准".encode()
        expect(page.locator("#file-change")).to_be_hidden()

        run, target = submit()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes("审批后出现的文件，不得覆盖".encode())
        page.locator("#approve").click()
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=20000)
        assert target.read_bytes() == "审批后出现的文件，不得覆盖".encode()
        expect(page.locator("#timeline")).to_contain_text("文件未写入")
        page.locator("#timeline").get_by_text(
            "文件未写入：", exact=False
        ).scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / "02-基线冲突与文件保留.png"), full_page=True)

        submit()
        page.locator("#reject").click()
        expect(page.locator("#run-status")).to_have_text("已停止", timeout=20000)
        expect(page.locator("#file-change")).to_be_hidden()
        submit("验收：差异长文写文件", large=True)
        page.locator("#file-change details summary").click()
        expect(page.locator("#file-change-after")).to_contain_text('<img src=x onerror="alert(1)">')
        expect(page.locator("#file-change img")).to_have_count(0)
        page.screenshot(path=str(args.output / "03-长文截断与安全对照.png"), full_page=True)
        large_id = page.evaluate("sessionStorage.getItem('aegis-last-run')")
        legacy_url = args.url + "/api/v1/runs/" + large_id

        def legacy_approval(route):
            response = route.fetch()
            content = response.json()
            content["approval"].pop("file_write", None)
            route.fulfill(response=response, json=content)

        # 同一页面切换任务后，旧请求的 finally 不得重新启用缺基线审批。
        pending = []
        page.route(legacy_url + "/approval", lambda route: pending.append(route))
        page.locator("#approve").click()
        deadline = time.monotonic() + 5
        while not pending and time.monotonic() < deadline:
            page.wait_for_timeout(25)
        assert pending, "审批故障夹具未触发"
        page.route(legacy_url, legacy_approval)
        page.locator("#run-list button").filter(has_text="验收：差异长文写文件").click()
        expect(page.locator("#approve")).to_be_disabled()
        page.evaluate("""() => {
          window.approvalReenabled = false;
          const button = document.getElementById('approve');
          window.approvalObserver = new MutationObserver(() => {
            if (!button.disabled) window.approvalReenabled = true;
          });
          window.approvalObserver.observe(button, {
            attributes: true, attributeFilter: ['disabled']
          });
        }""")
        with page.expect_response(legacy_url + "/approval"):
            pending.pop().fulfill(status=409, json={"detail": "故障夹具：过期审批"})
        page.wait_for_timeout(250)
        assert not page.evaluate("window.approvalReenabled"), "旧请求重新启用了缺基线批准按钮"
        page.evaluate("window.approvalObserver.disconnect()")
        expect(page.locator("#approve")).to_be_disabled()
        page.unroute(legacy_url + "/approval")
        page.reload()
        expect(page.locator("#file-change-title")).to_have_text("文件版本信息不可用")
        expect(page.locator("#approve")).to_be_disabled()
        page.unroute(legacy_url)
        page.locator("#reject").click()
        expect(page.locator("#run-status")).to_have_text("已停止", timeout=20000)
        page.locator("#new-task").click()
        expect(page.locator("#file-change")).to_be_hidden()
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#file-change-diff")).to_have_text("")
        assert not errors, errors
        browser.close()
    print(
        "文件差异浏览器验收通过：零写入、精确基线、新建、冲突保留、截断HTML、旧审批拒绝及退出；6组响应式"
    )


if __name__ == "__main__":
    main()
