"""持久计划纯契约的权限、账本真实性与输出边界。"""

import copy
import importlib
import json

import pytest

from app.harness.errors import HarnessError


def api(name):
    """先验证公开接口存在，使缺失实现呈现为行为断言失败。"""
    try:
        module = importlib.import_module("app.harness.plan_execution")
    except ModuleNotFoundError:
        pytest.fail("持久计划契约模块尚未实现")
    function = getattr(module, name, None)
    assert callable(function), f"缺少公开纯函数：{name}"
    return function


def structured(required=None, output_kind="tool_evidence", dependencies=None):
    return {
        "objective": "读取事实并给出证据",
        "inputs": {"instruction": "读取资料", "from_steps": dependencies or []},
        "acceptance": {
            "output_kind": output_kind,
            "required_tools": ["file_read"] if required is None else required,
        },
    }


def plan_for(required=None, output_kind="tool_evidence"):
    plan = api("normalize_plan")(
        {"steps": [structured(required, output_kind)]}, {"file_read", "calculator"}
    )
    plan["nodes"][0]["tool_call_ids"] = ["call-1"]
    return plan


def record(**changes):
    return {
        "id": "call-1",
        "name": "file_read",
        "status": "done",
        "result": {"content": "真实资料"},
        "tenant_id": "tenant",
        "owner_id": "owner",
        "run_id": "run",
        **changes,
    }


def accept(plan, evidence=None, answer="回答", node_id=None):
    return api("accept_node")(
        plan,
        node_id or plan["nodes"][0]["id"],
        answer,
        [record()] if evidence is None else evidence,
        "tenant",
        "owner",
        "run",
    )


@pytest.mark.parametrize(
    "value",
    [
        {"steps": []},
        {"steps": ["目标"] * 9},
        {"steps": ["目标", structured()]},
        {"steps": [""]},
        {"steps": [" "]},
        {"steps": ["目" * 2001]},
        {"steps": ["目标"], "status": "completed"},
        {"steps": [{**structured(), "status": "completed"}]},
        {"steps": [{**structured(), "id": "伪造"}]},
        {"steps": [{**structured(), "output": "伪造"}]},
        {"steps": [{"objective": "目标"}]},
        {"steps": [structured(dependencies=[0])]},
        {"steps": [structured(dependencies=[True])]},
        {"steps": [structured(required=["execute_shell"])]},
        {"steps": [structured(required=["file_read", "file_read"])]},
        {"steps": [structured(required=[""])]},
        {"steps": [structured(required=[])]},
        {"steps": [structured(output_kind="semantic_success")]},
        {"steps": [{**structured(), "objective": float("nan")}]},
        '{"steps": [NaN]}',
        '{"steps": [Infinity]}',
        '{"steps": [1e999]}',
        '{"steps": ["目标"], "steps": ["注入"]}',
        {"steps": [{**structured(), "inputs": {"instruction": "", "from_steps": []}}]},
        {"steps": [{**structured(), "acceptance": {"required_tools": []}}]},
        {
            "steps": [
                {
                    **structured(),
                    "inputs": {
                        "instruction": "读取",
                        "from_steps": [],
                        "allowed_tools": ["execute_shell"],
                    },
                }
            ]
        },
        {
            "steps": [
                {
                    **structured(),
                    "acceptance": {"output_kind": "answer", "required_tools": [], "verified": True},
                }
            ]
        },
        {"steps": [{**structured(), "inputs": {"instruction": "令" * 4001, "from_steps": []}}]},
    ],
)
def test_normalization_rejects_untrusted_or_invalid_input(value):
    with pytest.raises(HarnessError) as error:
        api("normalize_plan")(value, {"file_read"})
    assert error.value.status_code == 422


@pytest.mark.parametrize("revision", [True, -1, 1.5, 2**31, 10**100])
def test_revision_is_bounded_non_boolean_integer(revision):
    with pytest.raises(HarnessError):
        api("normalize_plan")({"steps": ["目标"]}, set(), revision)


def test_legacy_plan_is_server_owned_and_sequential():
    plan = api("normalize_plan")(json.dumps({"steps": [" 一步 ", "二步"]}), set(), 3)
    assert plan["version"] == 1
    assert plan["revision"] == 3
    assert plan["replans_used"] == 0
    assert plan["budget_policy"] == "shared_actions_v1"
    assert plan["status"] == "active"
    assert plan["current_node_id"] == "n3-1"
    assert plan["history"] == []
    first, second = plan["nodes"]
    assert first["objective"] == "一步"
    assert first["inputs"] == {"instruction": "一步", "from_nodes": []}
    assert second["inputs"]["from_nodes"] == ["n3-1"]
    assert first["acceptance"] == {"output_kind": "answer", "required_tools": []}
    for node in plan["nodes"]:
        assert node["status"] == "pending"
        assert node["proposed_calls"] == node["tool_call_ids"] == []
        assert node["output"] is node["acceptance_result"] is node["blocked_reason"] is None


def test_structured_dependency_references_are_zero_based_and_copied():
    value = {"steps": [structured(), structured(dependencies=[0])]}
    original = copy.deepcopy(value)
    plan = api("normalize_plan")(value, {"file_read"}, 2)
    assert plan["nodes"][1]["inputs"]["from_nodes"] == ["n2-1"]
    plan["nodes"][0]["acceptance"]["required_tools"].append("修改")
    assert value == original


@pytest.mark.parametrize("dependencies", [[-1], [1], [0, 0], [False], ["0"]])
def test_dependencies_cannot_reference_future_duplicate_or_noninteger(dependencies):
    with pytest.raises(HarnessError):
        api("normalize_plan")(
            {"steps": [structured(), structured(dependencies=dependencies)]}, {"file_read"}
        )


def test_entire_canonical_plan_has_character_budget():
    step = structured()
    step["objective"] = "目" * 2000
    step["inputs"]["instruction"] = "令" * 4000
    with pytest.raises(HarnessError):
        api("normalize_plan")({"steps": [step] * 8}, {"file_read"})


@pytest.mark.parametrize(
    "change",
    [
        {"status": "started"},
        {"result": None},
        {"result": {"status": "rejected"}},
        {"result": {"status": "unknown"}},
        {"result": {"status": "error"}},
        {"result": {"status": "cancelled"}},
        {"result": {"status": "interrupted"}},
        {"result": {"success": False}},
        {"result": {"ok": False}},
        {"result": {"error": "执行错误"}},
        {"result": {"errors": ["执行错误"]}},
        {"result": {"children": [{"result": {"status": "failed"}}]}},
        {"tenant_id": "foreign"},
        {"owner_id": "foreign"},
        {"run_id": "foreign"},
        {"id": "previous-node-call"},
        {"name": "calculator"},
    ],
)
def test_acceptance_rejects_ledger_failure_or_wrong_ownership(change):
    result = accept(plan_for(), [record(**change)])
    assert result["contract_satisfied"] is False
    assert result["correctness_verified"] is False
    assert result["reason"]


def test_evidence_must_exactly_cover_unique_node_bound_ids():
    plan = plan_for()
    assert not accept(plan, [])["contract_satisfied"]
    assert not accept(plan, [record(), record()])["contract_satisfied"]
    assert not accept(plan, [record(), record(id="extra")])["contract_satisfied"]
    plan["nodes"][0]["tool_call_ids"] = ["call-1", "call-1"]
    assert not accept(plan)["contract_satisfied"]


@pytest.mark.parametrize(
    "missing", ["id", "name", "status", "result", "tenant_id", "owner_id", "run_id"]
)
def test_all_ledger_fields_are_required(missing):
    evidence = record()
    del evidence[missing]
    assert not accept(plan_for(), [evidence])["contract_satisfied"]


def test_answer_contract_validates_structure_without_claiming_semantic_correctness():
    plan = plan_for([], "answer")
    plan["nodes"][0]["tool_call_ids"] = []
    result = accept(plan, [], "这是未经语义核验的回答")
    assert result["contract_satisfied"] is True
    assert result["correctness_verified"] is False
    assert result["business_success_verified"] is False
    for answer in [None, "", " ", "答" * 131073]:
        assert not accept(plan, [], answer)["contract_satisfied"]


def test_unknown_business_success_allows_known_completed_tool_evidence_without_answer():
    plan = plan_for()
    original = copy.deepcopy(plan)
    evidence = [record()]
    saved_evidence = copy.deepcopy(evidence)
    result = accept(plan, evidence, None)
    assert result["contract_satisfied"] is True
    assert result["reported_call_completed"] is True
    assert result["business_success_verified"] is False
    assert result["correctness_verified"] is False
    assert result["output"]
    assert result["ledger_sources"][0]["id"] == "call-1"
    assert plan == original
    assert evidence == saved_evidence


def test_required_tools_apply_to_answer_contract_too():
    plan = plan_for(["file_read"], "answer")
    assert not accept(plan, [record(name="calculator")])["contract_satisfied"]


def test_dependency_must_be_completed_with_nonempty_output_and_known_node():
    plan = api("normalize_plan")({"steps": ["第一步", "第二步"]}, set())
    second_id = plan["nodes"][1]["id"]
    assert not accept(plan, [], node_id=second_id)["contract_satisfied"]
    plan["nodes"][0]["status"] = "completed"
    assert not accept(plan, [], node_id=second_id)["contract_satisfied"]
    plan["nodes"][0]["output"] = "前序输出"
    assert accept(plan, [], node_id=second_id)["contract_satisfied"]
    assert not accept(plan, [], node_id="invented")["contract_satisfied"]


def test_tool_call_cannot_be_reused_from_previous_node():
    plan = api("normalize_plan")({"steps": [structured(), structured()]}, {"file_read"})
    for node in plan["nodes"]:
        node["tool_call_ids"] = ["call-1"]
    assert not accept(plan, node_id=plan["nodes"][1]["id"])["contract_satisfied"]


def test_evidence_summaries_and_count_are_bounded():
    plan = plan_for()
    result = accept(plan, [record(result={"content": "资" * 500000})], None)
    assert result["contract_satisfied"] is True
    assert len(result["ledger_sources"][0]["summary"]) <= 2000
    assert len(result["output"]) <= 131072
    plan["nodes"][0]["tool_call_ids"] = [f"call-{i}" for i in range(17)]
    evidence = [record(id=identifier) for identifier in plan["nodes"][0]["tool_call_ids"]]
    assert not accept(plan, evidence)["contract_satisfied"]


def test_nested_results_and_nonfinite_numbers_fail_closed():
    nested = {"content": "资料"}
    for _ in range(40):
        nested = {"children": [nested]}
    assert not accept(plan_for(), [record(result=nested)])["contract_satisfied"]
    assert not accept(plan_for(), [record(result={"value": float("nan")})])["contract_satisfied"]


def test_wide_results_fail_closed_without_unbounded_report():
    result = accept(plan_for(), [record(result={"items": [1] * 2049})])
    assert result["contract_satisfied"] is False
    assert len(json.dumps(result, ensure_ascii=False)) < 5000


def test_unrelated_node_call_bindings_must_remain_bounded():
    plan = api("normalize_plan")({"steps": ["第一步", "第二步"]}, set())
    plan["nodes"][0]["status"] = "completed"
    plan["nodes"][0]["output"] = "输出"
    plan["nodes"][0]["tool_call_ids"] = [f"old-{index}" for index in range(17)]
    assert not accept(plan, [], node_id=plan["nodes"][1]["id"])["contract_satisfied"]


def test_tool_evidence_rejects_oversized_answer_even_when_ledger_is_complete():
    assert not accept(plan_for(), answer="答" * 131073)["contract_satisfied"]


def test_generic_ledger_result_never_proves_business_semantics():
    result = accept(plan_for(), [record(result={"success": True, "value": 2})])
    assert result["contract_satisfied"] is True
    assert result["business_success_verified"] is False
    assert result["correctness_verified"] is False


def test_raw_json_character_budget_applies_before_parsing():
    raw = json.dumps({"steps": ["目标"]}, ensure_ascii=False)
    boundary = raw + " " * (32768 - len(raw))
    assert api("normalize_plan")(boundary, set())["nodes"][0]["objective"] == "目标"
    for oversized in [boundary + " ", " " * 32769]:
        with pytest.raises(HarnessError) as error:
            api("normalize_plan")(oversized, set())
        assert error.value.status_code == 422
        assert "原始" in error.value.detail
    with pytest.raises(HarnessError) as error:
        api("normalize_plan")(" " * 32768, set())
    assert error.value.status_code == 422


def test_normalization_and_acceptance_share_tool_name_length_boundary():
    name = "工" * 256
    plan = api("normalize_plan")({"steps": [structured([name])]}, {name})
    plan["nodes"][0]["tool_call_ids"] = ["call-1"]
    result = accept(plan, [record(name=name)])
    assert result["contract_satisfied"] is True
    too_long = name + "工"
    with pytest.raises(HarnessError) as error:
        api("normalize_plan")({"steps": [structured([too_long])]}, {too_long})
    assert error.value.status_code == 422


@pytest.mark.parametrize("field", ["exit_code", "returncode"])
@pytest.mark.parametrize("code", [1, -1, 137, True, False, "0", 0.0, None])
def test_process_exit_fields_reject_failure_or_noninteger_boolean_values(field, code):
    result = accept(plan_for(), [record(result={field: code, "output": "程序输出"})])
    assert result["contract_satisfied"] is False
    assert result["correctness_verified"] is False


@pytest.mark.parametrize("field", ["exit_code", "returncode"])
def test_zero_process_exit_code_is_structure_evidence_only(field):
    result = accept(plan_for(), [record(result={field: 0, "output": "程序输出"})])
    assert result["contract_satisfied"] is True
    assert result["business_success_verified"] is False


@pytest.mark.parametrize("field", ["isError", "is_error"])
def test_mcp_error_flags_are_rejected_even_inside_child_results(field):
    result = accept(plan_for(), [record(result={"children": [{field: True}]})])
    assert result["contract_satisfied"] is False
    clear = accept(plan_for(), [record(result={field: False, "content": []})])
    assert clear["contract_satisfied"] is True
    assert clear["business_success_verified"] is False


@pytest.mark.parametrize("status", ["pending", "blocked", "rejected", "unknown", False, None])
def test_actual_collaboration_acceptance_requires_verified_status(status):
    result = accept(
        plan_for(),
        [record(result={"children": [{"status": "completed", "acceptance": {"status": status}}]})],
    )
    assert result["contract_satisfied"] is False


def test_verified_collaboration_acceptance_does_not_claim_business_correctness():
    result = accept(plan_for(), [record(result={"acceptance": {"status": "verified"}})])
    assert result["contract_satisfied"] is True
    assert result["business_success_verified"] is False


def test_unknown_business_fields_are_not_generalized_into_failure():
    result = accept(
        plan_for(),
        [record(result={"score": 0, "verdict": "negative", "acceptance": {"threshold": 0}})],
    )
    assert result["contract_satisfied"] is True
    assert result["business_success_verified"] is False


@pytest.mark.parametrize(
    "status",
    [" " * 257, " " * 1000000 + "failed"],
    ids=["oversized-whitespace", "oversized-failure"],
)
def test_oversized_status_fails_closed_before_interpretation(status):
    result = accept(plan_for(), [record(result={"status": status})])
    assert result["contract_satisfied"] is False
    assert "状态文本超限" in result["reason"]


def test_status_character_boundary_still_recognizes_failure():
    failed = accept(plan_for(), [record(result={"status": " " * 250 + "failed"})])
    assert failed["contract_satisfied"] is False
    assert "失败" in failed["reason"]
    known = accept(plan_for(), [record(result={"status": " " * 252 + "done"})])
    assert known["contract_satisfied"] is True
    assert known["business_success_verified"] is False


def test_extremely_long_ledger_ids_are_rejected_and_256_boundary_is_accepted():
    plan = plan_for()
    assert not accept(plan, [record(id=" " * 1000000 + "call-1")])["contract_satisfied"]
    identifier = "账" * 256
    plan["nodes"][0]["tool_call_ids"] = [identifier]
    assert accept(plan, [record(id=identifier)])["contract_satisfied"] is True
