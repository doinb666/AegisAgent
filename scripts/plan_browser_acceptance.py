"""只在独立本机数据库验证持久计划展示；夹具不代表商业模型质量。"""

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/截图/计划执行"))
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
        page.emulate_media(reduced_motion="reduce")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: (errors.append(dialog.message), dialog.dismiss()))
        page.goto(args.url)
        page.locator("#username").fill("plan-ui-" + str(time.time_ns()))
        page.locator("#password").fill("plan-password-123")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        headers = {
            "Authorization": "Bearer " + page.evaluate("sessionStorage.getItem('aegis-token')")
        }

        def submit(message, mode="plan"):
            page.locator("#new-task").click()
            expect(page.locator("#plan-progress")).to_be_hidden()
            page.locator("#mode").select_option(mode)
            page.locator("#message").fill(message)
            page.locator("#send").click()
            expect(page.locator("#run-status")).not_to_have_text("尚未开始任务", timeout=15000)
            page.wait_for_function("() => sessionStorage.getItem('aegis-last-run') !== null")
            return page.evaluate("sessionStorage.getItem('aegis-last-run')")

        def events(run_id):
            response = page.request.get(args.url + f"/api/v1/runs/{run_id}/events", headers=headers)
            assert response.ok
            result = []
            for packet in response.text().split("\n\n"):
                fields = dict(line.split(": ", 1) for line in packet.splitlines() if ": " in line)
                if "data" in fields:
                    result.append((fields.get("event"), json.loads(fields["data"])))
            return result

        def inject(kind, payload):
            page.evaluate(
                "([kind, payload]) => window.addEvent(window.state.cursor + 1, kind, payload)",
                [kind, payload],
            )

        run_id = submit("验收：计划慢任务")
        expect(page.locator("#plan-progress")).to_be_visible(timeout=15000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2", timeout=15000)
        expect(page.locator("#run-status")).to_have_text("正在推进")
        page.reload()
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2")
        assert page.evaluate("sessionStorage.getItem('aegis-last-run')") == run_id
        summary = page.locator('#plan-nodes [data-node-id="n0-2"] summary')
        summary.focus()
        page.keyboard.press("Enter")
        expect(page.locator('#plan-nodes [data-node-id="n0-2"] details')).to_have_attribute(
            "open", ""
        )
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 2/2")
        expect(summary).to_be_focused()
        expect(page.locator('#plan-nodes [data-node-id="n0-2"] details')).to_have_attribute(
            "open", ""
        )
        page.locator("#plan-progress").screenshot(path=str(args.output / "01-两节点与键盘展开.png"))

        evidence_id = submit("验收：计划证据")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(page.locator("#answer")).to_contain_text("结构通过仍需核对结果正确性")
        assert '"risk"' not in page.locator("#answer").inner_text()
        assert '"reason"' not in page.locator("#answer").inner_text()
        page.locator('#plan-nodes [data-node-id="n0-2"] summary').click()
        expect(page.locator("#plan-progress")).to_contain_text("calculator · 来源")
        source_text = page.locator("#plan-progress .plan-source").inner_text()
        source_events = events(evidence_id)
        accepted = next(
            data
            for kind, data in source_events
            if kind == "node_accepted" and data["node_id"] == "n0-2"
        )
        assert accepted["ledger_sources"] and accepted["contract_satisfied"] is True
        assert (
            accepted["correctness_verified"] is False
            and accepted["business_success_verified"] is False
        )
        page.reload()
        expect(page.locator("#plan-summary")).to_contain_text("已通过 2/2")
        page.locator('#plan-nodes [data-node-id="n0-2"] summary').click()
        assert page.locator("#plan-progress .plan-source").inner_text() == source_text
        assert page.evaluate("sessionStorage.getItem('aegis-last-run')") == evidence_id
        # 终态后的经验整理可继续记事件；刷新不得新增模型、工具或节点执行。
        execution = {
            "plan",
            "plan_replanned",
            "node_started",
            "node_accepting",
            "node_accepted",
            "model_request",
            "model_response",
            "tool_started",
            "tool_call",
            "tool_result",
        }
        assert [(kind, data) for kind, data in events(evidence_id) if kind in execution] == [
            (kind, data) for kind, data in source_events if kind in execution
        ]
        for theme in ("light", "dark"):
            page.locator("html").evaluate(
                "(element, theme) => element.dataset.theme = theme", theme
            )
            for width in (1440, 768, 390):
                page.set_viewport_size({"width": width, "height": 1000})
                page.locator("#plan-progress").scroll_into_view_if_needed()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    theme,
                    width,
                )
                assert page.locator("#plan-nodes summary").first.bounding_box()["height"] >= 44
                assert page.locator("#plan-progress").evaluate(
                    "element => element.scrollWidth <= element.clientWidth"
                )
                page.screenshot(path=str(args.output / f"02-工具来源-{theme}-{width}.png"))
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("html").evaluate("element => element.dataset.theme = 'light'")

        repair_id = submit("验收：计划重规划")
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2", timeout=15000)
        prefix_summary = page.locator('#plan-nodes [data-node-id="n0-1"] summary')
        prefix_summary.focus()
        page.keyboard.press("Enter")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(prefix_summary).to_be_focused()
        expect(page.locator('#plan-nodes [data-node-id="n0-1"] details')).to_have_attribute(
            "open", ""
        )
        expect(page.locator("#plan-summary")).to_contain_text("修订 1")
        expect(page.locator('#plan-nodes [data-node-id="n0-1"]')).to_have_count(1)
        expect(page.locator('#plan-nodes [data-node-id="n1-1"]')).to_have_count(1)
        expect(page.locator("#plan-nodes")).to_contain_text("结构通过")
        page.locator("#plan-history summary").click()
        expect(page.locator("#plan-history")).to_contain_text("n0-2")
        expect(page.locator("#plan-history")).to_contain_text("未完整覆盖")
        repair_events = events(repair_id)
        replanned = next(data for kind, data in repair_events if kind == "plan_replanned")
        assert [node["id"] for node in replanned["nodes"]] == ["n0-1", "n1-1"]
        assert replanned["history"][0]["nodes"][0]["id"] == "n0-2"
        page.locator("#plan-progress").screenshot(
            path=str(args.output / "03-重规划保前缀与失败归档.png")
        )
        # 展示级复用通知不能重建归档summary、关闭展开或丢失阅读焦点。
        archive_summary = page.locator("#plan-history summary")
        archive_summary.focus()
        inject(
            "tool_reused",
            {
                "call_id": "history-display-only",
                "source_id": "unmatched-history-source",
                "source_call_id": "old",
                "name": "calculator",
                "reported_known_result": True,
                "business_success_verified": False,
            },
        )
        expect(archive_summary).to_be_focused()
        expect(page.locator("#plan-history details")).to_have_attribute("open", "")

        submit("验收：计划拒绝")
        expect(page.locator("#approval")).to_be_visible(timeout=25000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2")
        page.locator("#reject").click()
        expect(page.locator("#run-status")).to_have_text("已停止", timeout=25000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2")
        expect(page.locator("#plan-nodes")).to_contain_text("已取消")
        page.locator("#plan-progress").screenshot(
            path=str(args.output / "04-拒绝后停止未通过步骤.png")
        )

        budget_id = submit("验收：计划预算")
        expect(page.locator("#run-status")).to_have_text("执行失败", timeout=25000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2")
        expect(page.locator("#plan-nodes")).to_contain_text("执行失败")
        budget_events = events(budget_id)
        assert not any(kind == "tool_started" for kind, _ in budget_events), (
            "预算封口前不得启动工具批次"
        )
        expect(page.locator("#timeline")).to_contain_text("预算不足")
        page.locator("#plan-progress").screenshot(path=str(args.output / "05-预算停止.png"))

        submit("验收：计划失败")
        expect(page.locator("#run-status")).to_have_text("需要人工核对后恢复", timeout=25000)
        expect(page.locator("#plan-summary")).to_contain_text("已通过 1/2")
        expect(page.locator("#plan-nodes")).to_contain_text("待核对")
        page.locator("#plan-progress").screenshot(
            path=str(args.output / "07-真实工具异常中断待核对.png")
        )

        malicious_id = submit("验收：计划恶意")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(page.locator("#plan-progress")).to_contain_text('<img src=x onerror="alert(1)">')
        expect(page.locator("#plan-progress img")).to_have_count(0)
        expect(page.locator("#plan-progress")).to_contain_text("结构契约通过，结果正确性仍需核对")
        page.locator("#plan-progress").screenshot(path=str(args.output / "06-恶意目标安全文本.png"))
        malicious_events = events(malicious_id)
        initial = next(data for kind, data in malicious_events if kind == "plan")
        old_text = page.locator("#plan-progress").inner_text()
        # 以下为展示防御注入，不代表模型效果或真实工具重新执行。
        bad_report = {**accepted, "node_id": "不存在", "plan_revision": 0}
        inject("node_accepted", bad_report)
        inject("node_started", {"node_id": "n0-1", "plan_revision": -1})
        for change in (
            {"revision": None},
            {"revision": float("nan")},
            {"revision": 1e100},
            {"nodes": initial["nodes"] * 5},
            {"revision": 1, "history": [{"revision": 0, "nodes": initial["nodes"]}] * 3},
            {"nodes": [initial["nodes"][0], initial["nodes"][0]]},
            {"revision": 1, "nodes": [{**initial["nodes"][0], "objective": 123}]},
            {"revision": 1, "nodes": [{**initial["nodes"][0], "objective": "超" * 2001}]},
        ):
            inject("plan_replanned", {**initial, **change})
        assert page.locator("#plan-progress").inner_text() == old_text
        inject(
            "tool_reused",
            {
                "call_id": "fake",
                "source_id": "unknown-source",
                "source_call_id": "old",
                "name": "calculator",
                "reported_known_result": True,
                "business_success_verified": False,
            },
        )
        assert "历史结果复用" not in page.locator("#plan-progress").inner_text()
        # 只有匹配真正验收来源，才标识历史复用；复用事件本身不推进节点。
        page.locator("#run-list button").filter(has_text="验收：计划证据").click()
        expect(page.locator("#plan-summary")).to_contain_text("已通过 2/2")
        page.locator('#plan-nodes [data-node-id="n0-2"] summary').click()
        before_summary = page.locator("#plan-summary").inner_text()
        source_id = accepted["ledger_sources"][0]["id"]
        inject(
            "tool_reused",
            {
                "call_id": "display-only",
                "source_id": source_id,
                "source_call_id": "old",
                "name": "calculator",
                "reported_known_result": True,
                "business_success_verified": False,
            },
        )
        expect(page.locator("#plan-progress")).to_contain_text("历史结果复用，非重新执行")
        assert page.locator("#plan-summary").inner_text() == before_summary

        submit("验收：计划回退")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(page.locator("#plan-progress")).to_be_hidden()
        expect(page.locator("#timeline")).to_contain_text("规划失败，记录回退原因：")
        submit("旧式直接执行", "react")
        expect(page.locator("#run-status")).to_have_text("任务完成", timeout=25000)
        expect(page.locator("#plan-progress")).to_be_hidden()
        page.locator("#run-list button").filter(has_text="验收：计划慢任务").click()
        expect(page.locator("#plan-progress")).to_be_visible()
        page.locator('[data-view="settings"]').first.click()
        expect(page.locator("#plan-progress")).to_be_hidden()
        page.locator('[data-view="chat"]').first.click()
        expect(page.locator("#plan-progress")).to_be_visible()
        page.locator("#new-task").click()
        expect(page.locator("#plan-progress")).to_be_hidden()
        expect(page.locator("#plan-nodes")).to_have_text("")
        # 纯展示负例：旧修订、未知节点及伪造正确性标志不能推进步骤。
        private_marker = "PRIVATE-PLAN-INTERNAL-不得出现在任何展示"
        private_fields = {
            "risk": private_marker,
            "reasoning": private_marker,
            "system_prompt": private_marker,
            "model_request_pending": {"body": private_marker},
        }
        projected_nodes = [
            {
                **node,
                **private_fields,
                "inputs": {**node["inputs"], "instruction": private_marker},
                "proposed_calls": [{"model_request": private_marker}],
            }
            for node in initial["nodes"]
        ]
        inject(
            "plan",
            {
                **initial,
                **private_fields,
                "nodes": projected_nodes,
            },
        )
        expect(page.locator("#plan-summary")).to_contain_text("已通过 0/2")
        assert private_marker not in page.locator("#timeline").text_content()
        assert private_marker not in page.locator("body").text_content()
        assert "proposed_calls" not in page.locator("#timeline").text_content()
        untouched = page.locator("#plan-progress").inner_text()
        for payload in (
            {**accepted, "node_id": "n0-2", "plan_revision": -1},
            {**accepted, "node_id": "未知节点", "plan_revision": 0},
            {**accepted, "node_id": "n0-2", "plan_revision": 0, "correctness_verified": True},
            {**accepted, "node_id": "n0-2", "plan_revision": 0, "business_success_verified": True},
        ):
            inject("node_accepted", payload)
        inject("node_started", {"node_id": "n0-2", "plan_revision": float("nan")})
        assert page.locator("#plan-progress").inner_text() == untouched
        assert "结构契约通过" not in page.locator("#timeline").text_content()
        assert "历史结果复用" not in page.locator("#timeline").text_content()
        # 未采用的畸形事件既不转储巨值，也不宣称通过或复用。
        giant = "不应转储" * 20000
        for kind, payload in (
            ("node_accepted", {**accepted, "reason": giant, **private_fields}),
            ("tool_reused", {"source_id": giant, **private_fields}),
            ("plan_fallback", {"plan": None, "reason": giant, **private_fields}),
        ):
            inject(kind, payload)
            last_row = page.locator("#timeline>li").last
            expect(last_row).to_contain_text("计划事件未采用")
            assert len(last_row.text_content()) < 200
            assert last_row.locator("pre").count() == 0
        assert giant not in page.locator("body").text_content()
        # 有效的七类计划事件也只投影公开字段，包含嵌套节点/账本的私有额外键。
        answer_report = {
            "node_id": "n0-1",
            "plan_revision": 0,
            "contract_satisfied": True,
            "correctness_verified": False,
            "business_success_verified": False,
            "reported_call_completed": False,
            "reason": "结构满足，正确性未验证",
            "output": "公开回答",
            "ledger_sources": [],
            **private_fields,
        }
        inject("node_started", {"node_id": "n0-1", "plan_revision": 0, **private_fields})
        inject("node_accepting", {"node_id": "n0-1", "plan_revision": 0, **private_fields})
        inject("node_accepted", answer_report)
        inject(
            "tool_reused",
            {
                "call_id": "public-only",
                "source_id": "unmatched-source",
                "source_call_id": "old",
                "name": "calculator",
                "reported_known_result": True,
                "business_success_verified": False,
                **private_fields,
            },
        )
        revised = {
            **initial,
            "revision": 1,
            "replans_used": 1,
            **private_fields,
            "nodes": projected_nodes,
            "history": [{"revision": 0, "nodes": projected_nodes, **private_fields}],
        }
        inject("plan_replanned", revised)
        inject(
            "plan_fallback",
            {"reason": "公开回退原因", "plan": {**revised, "revision": 2}, **private_fields},
        )
        assert private_marker not in page.locator("#timeline").text_content()
        assert private_marker not in page.locator("body").text_content()
        assert "model_request_pending" not in page.locator("#timeline").text_content()
        assert "proposed_calls" not in page.locator("#timeline").text_content()
        inject("interrupted", {})
        expect(page.locator("#plan-summary")).to_contain_text("已通过 0/2")
        expect(page.locator("#plan-nodes")).to_contain_text("待核对")
        inject("completed", {})
        expect(page.locator("#plan-summary")).to_contain_text("已通过 0/2")
        expect(page.locator("#plan-summary")).to_contain_text("未通过步骤待核对")
        page.locator("#new-task").click()
        expect(page.locator("#plan-progress")).to_be_hidden()
        page.locator("#run-list button").filter(has_text="验收：计划慢任务").click()
        expect(page.locator("#plan-progress")).to_be_visible()
        page.locator("#logout").click()
        expect(page.locator("#auth")).to_be_visible()
        expect(page.locator("#plan-nodes")).to_have_text("")
        page.locator("#username").fill("plan-other-" + str(time.time_ns()))
        page.locator("#password").fill("plan-other-password")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        expect(page.locator("#plan-progress")).to_be_hidden()
        expect(page.locator("#plan-nodes")).to_have_text("")
        assert not errors, errors
        browser.close()
    print(
        "计划浏览器验收通过：真实两节点与来源重放、重规划前缀/归档、拒绝、预算失败、"
        "工具异常中断待核对、XSS与畸形展示负例、计划事件整页公开白名单、"
        "键盘焦点与6组响应式、切任务/退出/账号清空。"
    )


if __name__ == "__main__":
    main()
