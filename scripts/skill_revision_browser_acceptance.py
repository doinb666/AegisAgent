"""仅本机的 Skill 修订审阅验收：真实 HTTP 闭环与明确标注的浏览器故障注入。"""

import argparse
import copy
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def wait_json(read, predicate, description, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = read()
        assert response.ok, response.text()
        payload = response.json()
        if predicate(payload):
            return payload
        time.sleep(0.1)
    raise AssertionError("等待超时：" + description)


def persisted_events(page, url, headers, run_id):
    response = page.request.get(url + f"/api/v1/runs/{run_id}/events", headers=headers)
    assert response.ok, response.text()
    events = []
    for packet in response.text().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in packet.splitlines() if ": " in line)
        if "data" in fields:
            events.append((fields.get("event"), json.loads(fields["data"])))
    return events


def register(page, url, prefix):
    username = prefix + str(time.time_ns())
    password = "revision-ui-password-123"
    page.locator("#username").fill(username)
    page.locator("#password").fill(password)
    page.locator("#register").click()
    expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
    page.locator("#login").click()
    expect(page.locator("#shell")).to_be_visible()
    token = page.evaluate("sessionStorage.getItem('aegis-token')")
    return {"Authorization": "Bearer " + token}, username, password


def real_fixture(page, url, headers):
    def run(message):
        response = page.request.post(
            url + "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "revision-ui-" + str(time.time_ns())},
            data={"message": message},
        )
        assert response.status == 202, response.text()
        identifier = response.json()["id"]
        wait_json(
            lambda: page.request.get(url + "/api/v1/runs/" + identifier, headers=headers),
            lambda item: item["status"] == "completed",
            "真实工具任务完成",
        )
        return identifier

    source = run("验收：技能修订来源 skill-revision-input 输入检查")
    assets = wait_json(
        lambda: page.request.get(url + "/api/v1/assets", headers=headers),
        lambda items: any(
            item["kind"] == "skill"
            and item["metadata"].get("source_run_id") == source
            and item["metadata"].get("extracted") is True
            for item in items
        ),
        "既有 Evolution 生成来源技能",
    )
    original = next(
        item
        for item in assets
        if item["kind"] == "skill" and item["metadata"].get("extracted") is True
    )
    response = page.request.post(
        url + "/api/v1/assets/" + original["id"] + "/state",
        headers=headers,
        data={"status": "active"},
    )
    assert response.ok, response.text()
    original = response.json()
    target = run("验收：技能修订失败 skill-revision-input 输入检查")
    response = page.request.post(
        url + "/api/v1/runs/" + target + "/feedback",
        headers=headers,
        data={"success": False, "note": "实际输出缺少空输入与异常检查"},
    )
    assert response.ok, response.text()
    assets = wait_json(
        lambda: page.request.get(url + "/api/v1/assets", headers=headers),
        lambda items: any(item["metadata"].get("revision_original_id") for item in items),
        "失败反馈生成真实修订草稿",
    )
    draft = next(item for item in assets if item["metadata"].get("revision_original_id"))
    assert draft["metadata"]["revision_original_id"] == original["id"]
    assert draft["metadata"]["revision_original_version"] == original["version"]
    assert draft["metadata"]["source_run_id"] == target
    assert draft["metadata"]["repair_verified"] is False
    response = page.request.get(url + "/api/v1/assets/" + original["id"], headers=headers)
    assert response.ok and response.json() == original, "修订不得改变原技能"
    return original, draft, target, source


def draft_row(page):
    return page.locator(".asset-row").filter(has=page.locator(".skill-revision-review"))


def open_target(page, message="验收：技能修订失败"):
    page.locator('[data-view="chat"]').click()
    page.locator("#refresh").click()
    page.locator(".run-item").filter(has_text=message).first.click()
    expect(page.locator("#feedback")).to_be_visible()


def injected_assets(page, assets):
    page.route("**/api/v1/assets", lambda route: route.fulfill(json=assets))
    page.locator("#assets-refresh").click()


def defer_fetch(page, path, release_name):
    """只延迟真实 HTTP 的返回，不替换鉴权、数据或服务端状态。"""
    page.evaluate(
        """({path, name}) => {
            const original = window.fetch;
            window.fetch = async (...args) => {
                const response = await original(...args);
                if (String(args[0]) !== path) return response;
                return new Promise(resolve => {
                    window[name] = () => { window.fetch = original; resolve(response); };
                });
            };
        }""",
        {"path": path, "name": release_name},
    )


def wait_deferred(page, release_name):
    page.wait_for_function("name => typeof window[name] === 'function'", arg=release_name)


def release_fetch(page, release_name):
    page.evaluate("name => { window[name](); delete window[name]; }", release_name)


def event_row_contract(page, draft):
    rendered = page.evaluate(
        """draftId => {
            const row = window.WorkspaceUI.eventRow(1, 'skill_revision', {
                state: 'done', draft_assets: [draftId], fingerprint: 'private-event-marker',
                run_config: {secret: 'private-event-marker'}, reason: 'private-event-marker',
            }, {});
            return {text: row.textContent, evidence: row.querySelector('pre').textContent};
        }""",
        draft["id"],
    )
    assert "已生成 1 个未验证修订草稿" in rendered["text"], rendered
    assert "private-event-marker" not in rendered["text"]
    assert json.loads(rendered["evidence"]) == {
        "state": "done",
        "draft_count": 1,
        "automatic_activation": False,
        "repair_verified": False,
    }


def feedback_retention(page, target):
    open_target(page)
    note = "同一任务未提交说明应保留"
    page.locator("#feedback-note").fill(note)
    page.locator(".run-item").filter(has_text="验收：技能修订失败").first.click()
    expect(page.locator("#feedback")).to_be_visible()
    expect(page.locator("#feedback-note")).to_have_value(note)
    page.locator('[data-view="skills"]').click()
    page.locator('[data-view="chat"]').click()
    expect(page.locator("#feedback")).to_be_visible()
    expect(page.locator("#feedback-note")).to_have_value(note)
    path = "**/api/v1/runs/" + target + "/feedback"
    page.route(path, lambda route: route.abort())
    page.locator("#failure").click()
    expect(page.locator("#feedback-error")).to_contain_text("无法连接")
    error = page.locator("#feedback-error").inner_text()
    page.unroute(path)
    page.locator(".run-item").filter(has_text="验收：技能修订失败").first.click()
    expect(page.locator("#feedback")).to_be_visible()
    expect(page.locator("#feedback-note")).to_have_value(note)
    expect(page.locator("#feedback-error")).to_have_text(error)
    page.locator('[data-view="skills"]').click()
    page.locator('[data-view="chat"]').click()
    expect(page.locator("#feedback")).to_be_visible()
    expect(page.locator("#feedback-error")).to_have_text(error)
    page.locator("#feedback-note").fill("旧响应不得清除同任务新说明")
    defer_fetch(page, "/api/v1/runs/" + target + "/feedback", "releaseSameRunFeedback")
    page.locator("#failure").click()
    wait_deferred(page, "releaseSameRunFeedback")
    page.locator(".run-item").filter(has_text="验收：技能修订失败").first.click()
    expect(page.locator("#feedback")).to_be_visible()
    expect(page.locator("#feedback-note")).to_be_enabled()
    page.locator("#feedback-note").fill("同任务新一代未提交说明")
    release_fetch(page, "releaseSameRunFeedback")
    expect(page.locator("#feedback-note")).to_have_value("同任务新一代未提交说明")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/技能修订"))
    parser.add_argument("--red", action="store_true", help="只执行真实 HTTP 生成与缺失入口红例")
    parser.add_argument("--review-red", choices=("event", "feedback"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许独立本机验收服务")
    args.output.mkdir(parents=True, exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        headers, username, password = register(page, args.url, "revision-ui-")
        original, draft, target, source = real_fixture(page, args.url, headers)
        if args.review_red == "event":
            event_row_contract(page, draft)
            raise AssertionError("事件行红例意外通过")
        if args.review_red == "feedback":
            feedback_retention(page, target)
            raise AssertionError("同任务保留红例意外通过")
        event_row_contract(page, draft)
        original_url = args.url + "/api/v1/assets/" + original["id"]
        page.on(
            "request",
            lambda request: requests.append(request.url) if request.url == original_url else None,
        )
        page.locator('[data-view="skills"]').click()
        # 首个断言必须建立在真实服务端生成的草稿上，红例不能用模拟元数据代替。
        expect(page.locator("#asset-list")).to_contain_text("修订草稿", timeout=5000)
        if args.red:
            raise AssertionError("红例意外通过，请检查前端基线")
        row = draft_row(page)
        expect(row).to_have_count(1)
        expect(row).to_contain_text(f"基于原技能v{original['version']}，未验证建议")
        assert not requests, "原技能应惰性读取"
        row.locator(".skill-revision-review summary").click()
        expect(row.locator(".revision-original")).to_contain_text(original["content"])
        expect(row.locator(".revision-suggestion")).to_contain_text(draft["content"])
        summary = row.locator(".skill-revision-review summary")
        summary.focus()
        expect(summary).to_be_focused()
        summary.click()
        summary.click()
        expect(row.locator(".revision-original")).to_be_visible()
        assert len(requests) == 1, "反复展开不能无限请求"
        page.locator("#asset-status-filter").select_option("draft")
        page.locator("#asset-search").fill(original["name"])
        draft_row(page).locator(".skill-revision-review summary").click()
        expect(draft_row(page).locator(".revision-original")).to_be_visible()
        draft_row(page).scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / "01-真实修订草稿对照.png"), full_page=True)
        page.locator("#asset-filters-clear").click()

        # 浏览器故障注入：网络失败、403、404及来源漂移仅测试展示契约。
        for status in (403, 404):
            page.route(
                "**/api/v1/assets/" + original["id"],
                lambda route, request, status=status: route.fulfill(
                    status=status, json={"detail": "禁止泄露的内部详情"}
                ),
            )
            page.locator("#assets-refresh").click()
            draft_row(page).locator(".skill-revision-review summary").click()
            expect(draft_row(page).locator(".revision-status")).to_contain_text("无法读取")
            expect(draft_row(page)).not_to_contain_text("禁止泄露")
            expect(draft_row(page).locator(".revision-original")).to_have_count(0)
            page.unroute("**/api/v1/assets/" + original["id"])
        page.route("**/api/v1/assets/" + original["id"], lambda route: route.abort())
        page.locator("#assets-refresh").click()
        draft_row(page).locator(".skill-revision-review summary").click()
        expect(draft_row(page).get_by_role("button", name="重试读取原技能")).to_be_visible()
        page.unroute("**/api/v1/assets/" + original["id"])
        draft_row(page).get_by_role("button", name="重试读取原技能").click()
        expect(draft_row(page).locator(".revision-original")).to_contain_text(original["content"])
        for field, value in (("version", original["version"] + 1), ("status", "retired")):
            changed = {**original, field: value}
            page.route(
                "**/api/v1/assets/" + original["id"],
                lambda route, request, changed=changed: route.fulfill(json=changed),
            )
            page.locator("#assets-refresh").click()
            draft_row(page).locator(".skill-revision-review summary").click()
            expect(draft_row(page).locator(".revision-status")).to_contain_text("原技能已变化")
            expect(draft_row(page).locator(".revision-original")).to_have_count(0)
            page.unroute("**/api/v1/assets/" + original["id"])
        changed = copy.deepcopy(original)
        changed["metadata"]["source_run_id"] = "another-source"
        page.route("**/api/v1/assets/" + original["id"], lambda route: route.fulfill(json=changed))
        page.locator("#assets-refresh").click()
        draft_row(page).locator(".skill-revision-review summary").click()
        expect(draft_row(page).locator(".revision-status")).to_contain_text("原技能已变化")
        page.screenshot(path=str(args.output / "02-原技能变化提示.png"), full_page=True)
        page.unroute("**/api/v1/assets/" + original["id"])

        malformed = []
        for number, (field, value) in enumerate(
            (
                ("revision_original_id", '<img src=x onerror="alert(1)">'),
                ("revision_original_version", 10**50),
                ("revision_original_version", True),
                ("revision_recall_evidence", None),
                ("revision_feedback_hash", "a" * 100000),
                ("repair_verified", True),
            )
        ):
            item = copy.deepcopy(draft)
            item["id"] = "bad-draft-" + str(number)
            item["metadata"][field] = value
            malformed.append(item)
        ordinary = {**draft, "id": "ordinary-draft", "metadata": {}}
        imported = {**ordinary, "id": "imported-draft", "metadata": {"directory": "导入"}}
        historical = {**draft, "id": "historical-skill", "status": "active"}
        injected_assets(page, [*malformed, ordinary, imported, historical])
        expect(page.locator(".asset-row")).to_have_count(9)
        expect(page.locator(".skill-revision-review")).to_have_count(0)
        assert page.locator("#asset-list img, #asset-list script").count() == 0
        page.unroute("**/api/v1/assets")
        malicious = copy.deepcopy(draft)
        malicious["content"] = (
            '<img src=x onerror="window.revisionXss=true">\n' + "待验证步骤" * 5000
        )
        injected_assets(page, [original, malicious])
        draft_row(page).locator(".skill-revision-review summary").click()
        expect(draft_row(page).locator(".revision-suggestion")).to_contain_text("内容已截断")
        assert page.locator("#asset-list img, #asset-list script").count() == 0
        assert not page.evaluate("window.revisionXss === true")
        page.unroute("**/api/v1/assets")

        page.locator("#assets-refresh").click()
        defer_fetch(page, "/api/v1/assets/" + original["id"], "releaseStableRead")
        draft_row(page).locator(".skill-revision-review summary").click()
        wait_deferred(page, "releaseStableRead")
        page.keyboard.press("Tab")
        summary = draft_row(page).locator(".skill-revision-review summary")
        summary.focus()
        page.evaluate("window.stableRevisionSummary = document.activeElement")
        release_fetch(page, "releaseStableRead")
        expect(draft_row(page).locator(".revision-original")).to_be_visible()
        expect(summary).to_be_focused()
        assert page.evaluate("document.activeElement === window.stableRevisionSummary")
        assert summary.locator("..").get_attribute("open") is not None

        # 延迟真实读取返回后切视图或筛选，不得把旧结果回填新条目。
        for change in ("view", "filter", "refresh"):
            page.locator("#assets-refresh").click()
            defer_fetch(page, "/api/v1/assets/" + original["id"], "releaseRevisionRead")
            draft_row(page).locator(".skill-revision-review summary").click()
            wait_deferred(page, "releaseRevisionRead")
            if change == "view":
                page.locator('[data-view="memories"]').click()
            elif change == "filter":
                page.locator("#asset-search").fill("no-matching-skill")
            else:
                page.locator("#assets-refresh").click()
            release_fetch(page, "releaseRevisionRead")
            expect(page.locator(".revision-original")).to_have_count(0)
            if change == "view":
                page.locator('[data-view="skills"]').click()
            if change == "filter":
                page.locator("#asset-filters-clear").click()

        feedback_retention(page, target)
        open_target(page)
        expect(page.locator("#timeline")).to_contain_text("记录技能修订建议")
        expect(page.locator("#timeline")).not_to_contain_text("修复成功")
        expect(page.locator('label[for="feedback-note"]')).to_have_text("反馈说明（可选）")
        assert page.locator("#feedback-note").get_attribute("maxlength") == "8000"
        page.locator("#feedback-note").fill("🧪" * 4000)
        page.locator("#success").click()
        expect(page.locator("#feedback-note")).to_have_value("")
        expect(page.locator("#feedback-error")).to_be_empty()
        expect(page.locator("#feedback")).to_be_visible()
        assert any(
            kind == "feedback" and data.get("note") == "🧪" * 4000
            for kind, data in persisted_events(page, args.url, headers, target)
        ), "真实 API 应接收 4000 个 Unicode 码点的反馈"
        page.locator("#feedback-note").fill("字" * 4001)
        page.locator("#failure").click()
        expect(page.locator("#feedback-error")).to_contain_text("最多允许 4000")
        expect(page.locator("#feedback-note")).to_have_value("字" * 4001)
        page.locator("#feedback-note").fill("网络失败保留说明 <script>纯文本</script>")
        feedback_path = "**/api/v1/runs/" + target + "/feedback"
        page.route(feedback_path, lambda route: route.abort())
        page.locator("#failure").click()
        expect(page.locator("#feedback-error")).to_contain_text("无法连接")
        expect(page.locator("#feedback-note")).to_have_value(
            "网络失败保留说明 <script>纯文本</script>"
        )
        page.unroute(feedback_path)
        page.locator("#success").click()
        expect(page.locator("#feedback-note")).to_have_value("")
        expect(page.locator("#feedback-error")).to_be_empty()
        page.locator("#failure").click()
        expect(page.locator("#notice")).to_contain_text("反馈已记录")
        page.locator("#feedback-note").fill("切任务应清空")
        page.locator(".run-item").filter(has_text="验收：技能修订来源").first.click()
        expect(page.locator("#feedback-note")).to_have_value("")
        open_target(page)
        page.locator("#feedback-note").fill("原任务反馈不能污染另一个任务")
        defer_fetch(page, "/api/v1/runs/" + target + "/feedback", "releaseFeedback")
        page.locator("#failure").click()
        wait_deferred(page, "releaseFeedback")
        page.locator(".run-item").filter(has_text="验收：技能修订来源").first.click()
        expect(page.locator("#feedback-note")).to_have_value("")
        page.locator("#feedback-note").fill("新任务说明")
        release_fetch(page, "releaseFeedback")
        expect(page.locator("#feedback-note")).to_have_value("新任务说明")
        expect(page.locator("#conversation")).to_contain_text("验收：技能修订来源")
        page.screenshot(path=str(args.output / "03-反馈说明与任务隔离.png"), full_page=True)

        page.locator('[data-view="skills"]').click()
        page.locator("#asset-status-filter").select_option("draft")
        draft_row(page).first.locator(".skill-revision-review summary").click()
        expect(draft_row(page).first.locator(".revision-original")).to_be_visible()
        for theme in ("light", "dark"):
            if page.evaluate("document.documentElement.dataset.theme") != theme:
                page.locator("#theme-toggle").click()
            for width in (390, 768, 1440):
                page.set_viewport_size({"width": width, "height": 1000})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    theme,
                    width,
                )
                summary = draft_row(page).first.locator(".skill-revision-review summary")
                page.keyboard.press("Tab")
                summary.focus()
                expect(summary).to_be_focused()
                assert summary.bounding_box()["height"] >= 44
                assert page.evaluate(
                    """() => {const style=getComputedStyle(document.activeElement);
                        return style.outlineStyle !== 'none'
                            && parseFloat(style.outlineWidth) >= 2;}"""
                )
                page.screenshot(path=str(args.output / f"04-{theme}-{width}.png"), full_page=True)
        page.emulate_media(reduced_motion="reduce")
        assert (
            page.evaluate("getComputedStyle(document.querySelector('button')).transitionDuration")
            == "0s"
        )

        # 真实权限边界：另一账号无法读取本人原技能和草稿。
        page.locator("#assets-refresh").click()
        defer_fetch(page, "/api/v1/assets/" + original["id"], "releaseAccountRead")
        draft_row(page).first.locator(".skill-revision-review summary").click()
        wait_deferred(page, "releaseAccountRead")
        page.locator("#logout").click()
        expect(page.locator("#feedback-note")).to_have_value("")
        other_headers, _, _ = register(page, args.url, "revision-other-")
        release_fetch(page, "releaseAccountRead")
        for identifier in (original["id"], draft["id"]):
            response = page.request.get(
                args.url + "/api/v1/assets/" + identifier, headers=other_headers
            )
            assert response.status in (403, 404), response.text()
            assert original["content"] not in response.text()
        page.locator('[data-view="skills"]').click()
        expect(page.locator(".asset-row")).to_have_count(0)
        assert not errors, errors
        browser.close()
    print(
        json.dumps(
            {
                "真实HTTP": [
                    "工具执行与提炼",
                    "人工启用",
                    "真实召回",
                    "失败反馈生成草稿",
                    "原技能未改变",
                    "跨账号读取拒绝",
                ],
                "浏览器故障注入": [
                    "403",
                    "404",
                    "网络重试",
                    "版本/退役/来源漂移",
                    "畸形元数据",
                    "恶意正文截断",
                    "切视图/筛选/刷新旧响应",
                    "反馈网络保留",
                    "切任务反馈隔离",
                ],
                "视觉": "深浅390/768/1440、44px、键盘焦点、reduced-motion",
                "Unicode边界": "真实API接收4000个表情；4001字提交被前端拦截",
                "页面错误": errors,
                "未验证": "外部真实模型的修订质量、生产数据库、完整无障碍合规",
                "账号": username,
                "来源任务": source,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
