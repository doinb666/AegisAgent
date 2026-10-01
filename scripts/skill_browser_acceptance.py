"""真实浏览器验证技能文件导入、资源审核、导出和目录筛选。"""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8772")
    parser.add_argument("--output", type=Path, default=Path("docs/升级方案/截图/Skill文件"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(arguments.url)
        page.wait_for_load_state("networkidle")
        page.locator("#username").fill("skill-ui-" + str(time.time_ns()))
        page.locator("#password").fill("skill-ui-password-123")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.locator('[data-view="skills"]').click()
        page.locator("#import-skill").click()
        page.locator("#skill-file").set_input_files(
            {
                "name": "SKILL.md",
                "mimeType": "text/markdown",
                "buffer": b"---\nname: &a unsafe\ndescription: *a\n---\nunsafe",
            }
        )
        page.locator("#skill-import-form button[type=submit]").click()
        expect(page.locator("#skill-import-error")).not_to_be_empty()
        expect(page.locator("#skill-import-error")).to_be_focused()
        bundle = {
            "document": (
                "---\nname: review-python\ndescription: 审查Python代码的边界\n"
                "tags: [审查]\nallowed-tools: [file_read]\n---\n"
                "先读取代码，再检查边界。参考 references/checklist.md。"
            ),
            "directory": "coding/review",
            "resources": {
                "references/checklist.md": "<script>不得执行</script>\n检查空输入、异常和路径约束。"
            },
        }
        page.locator("#skill-file").set_input_files(
            {
                "name": "review-python.json",
                "mimeType": "application/json",
                "buffer": json.dumps(bundle, ensure_ascii=False).encode(),
            }
        )
        page.locator("#skill-import-form button[type=submit]").click()
        row = page.locator(".asset-row").filter(
            has=page.get_by_role("heading", name="review-python", exact=True)
        )
        expect(row).to_contain_text("候选")
        expect(row).to_contain_text("coding/review")
        row.get_by_role("button", name="查看 references/checklist.md", exact=True).click()
        expect(row.locator(".skill-resource-preview")).to_contain_text("<script>不得执行</script>")
        assert row.locator(".skill-resource-preview script").count() == 0
        with page.expect_download() as info:
            row.get_by_role("button", name="导出文件包", exact=True).click()
        exported = json.loads(info.value.path().read_text(encoding="utf-8"))
        assert exported["resources"] == bundle["resources"]
        assert exported["directory"] == bundle["directory"]
        row.get_by_role("button", name="修订", exact=True).click()
        page.locator("#asset-content").fill("先读取代码，检查输入与异常。")
        page.locator('#asset-form button[type="submit"]').click()
        expect(row).to_contain_text("先读取代码，检查输入与异常。")
        with page.expect_download() as info:
            row.get_by_role("button", name="导出文件包", exact=True).click()
        revised = json.loads(info.value.path().read_text(encoding="utf-8"))
        assert revised["resources"] == bundle["resources"], "修订不能丢失资源"
        assert revised["directory"] == bundle["directory"], "修订不能丢失目录"
        with page.expect_download() as info:
            row.get_by_role("button", name="导出 SKILL.md", exact=True).click()
        assert "review-python" in info.value.path().read_text(encoding="utf-8")
        page.locator("#skill-directory-filter").select_option("coding/review")
        expect(page.locator("#asset-list .asset-row")).to_have_count(1)
        row.get_by_role("button", name="查看 references/checklist.md", exact=True).click()
        page.screenshot(path=str(arguments.output / "01-技能与资源.png"), full_page=True)
        row.get_by_role("button", name="启用", exact=True).click()
        expect(row).to_contain_text("已启用")
        page.locator("#logout").click()
        page.locator("#password").fill("skill-ui-password-123")
        page.locator("#login").click()
        page.locator('[data-view="skills"]').click()
        expect(page.locator("#asset-list")).to_contain_text("review-python")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#import-skill").click()
        page.screenshot(path=str(arguments.output / "02-窄屏导入.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "窄屏横向溢出"
        # 文件读取未完成时切换账号，不应把前一个账号的技能写入新账号。
        page.evaluate("""() => {
          const original = File.prototype.arrayBuffer;
          File.prototype.arrayBuffer = function() {
            if (this.name !== 'race.json') return original.call(this);
            return new Promise(resolve => {
              window.releaseSkillRead = () => resolve(original.call(this).then(bytes => {
                window.skillReadDelivered = true; return bytes;
              }));
            });
          };
        }""")
        race_bundle = {
            "document": bundle["document"].replace("review-python", "cross-account-race")
        }
        page.locator("#skill-file").set_input_files(
            {
                "name": "race.json",
                "mimeType": "application/json",
                "buffer": json.dumps(race_bundle, ensure_ascii=False).encode(),
            }
        )
        page.locator("#skill-import-form button[type=submit]").click()
        page.wait_for_function("() => typeof window.releaseSkillRead === 'function'")
        page.locator("#logout").click()
        page.locator("#username").fill("skill-race-" + str(time.time_ns()))
        page.locator("#password").fill("skill-ui-password-123")
        page.locator("#register").click()
        expect(page.locator("#auth-error")).to_have_text("账号已创建，可以登录。")
        page.locator("#login").click()
        expect(page.locator("#shell")).to_be_visible()
        page.locator('[data-view="skills"]').click()
        page.evaluate("() => window.releaseSkillRead()")
        page.wait_for_function("() => window.skillReadDelivered === true")
        page.wait_for_load_state("networkidle")
        page.wait_for_function(
            "() => !document.querySelector('#skill-import-form button[type=submit]').disabled"
        )
        page.locator('[data-view="skills"]').click()
        expect(page.locator("#asset-list .asset-row")).to_have_count(0)
        assert not errors, errors
        browser.close()
    print(json.dumps({"技能浏览器验收": "通过", "页面错误": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
