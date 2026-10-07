// 只消费本人持久事件的公开契约；结构通过不证明结果正确性。
type Source = {id: string; name: string; status: string; summary: string};
type Report = {contract_satisfied: boolean; reason: string; ledger_sources: Source[]};
type PlanNode = {id: string; objective: string; inputs: {from_nodes: string[]};
  acceptance: {output_kind: string; required_tools: string[]}; status: string;
  acceptance_result: Report | null; blocked_reason: string | null};
type Plan = {revision: number; nodes: PlanNode[]; history: {revision: number; nodes: PlanNode[]}[]};
type NodeView = {row: HTMLLIElement; summary: HTMLElement; title: HTMLElement;
  status: HTMLElement; body: HTMLElement; signature: string};
type HistoryView = {details: HTMLDetailsElement; nodes: Map<string, {row: HTMLElement; signature: string}>};

const nodeStates = ["pending", "running", "accepting", "completed", "blocked"];
const labels: Record<string, string> = {pending: "等待前序", running: "执行中", accepting: "验收中",
  completed: "结构通过", blocked: "阻断"};
const terminalLabels: Record<string, string> = {completed: "运行已结束，未通过步骤待核对",
  failed: "已停止 · 执行失败", cancelled: "已停止 · 已取消", interrupted: "待核对 · 执行中断"};
const integer = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= 2147483647;
const record = (value: any): boolean => !!value && typeof value === "object" && !Array.isArray(value);
function text(value: unknown, limit: number, empty = true): value is string {
  return typeof value === "string" && value.length <= limit * 2 && [...value].length <= limit
    && (empty || !!value.trim());
}
const identifier = (value: unknown): value is string => text(value, 256, false);
function strings(value: any, limit = 16): value is string[] {
  return Array.isArray(value) && value.length <= limit && value.every(identifier) && new Set(value).size === value.length;
}
function report(value: any): Report | null {
  if (!record(value) || typeof value.contract_satisfied !== "boolean"
    || value.correctness_verified !== false || value.business_success_verified !== false
    || typeof value.reported_call_completed !== "boolean" || !text(value.reason, 2000)
    || !(value.output === null || text(value.output, 131072))
    || (value.contract_satisfied && !text(value.output, 131072, false))
    || !Array.isArray(value.ledger_sources) || value.ledger_sources.length > 16) return null;
  const sources: Source[] = [];
  for (const source of value.ledger_sources) {
    if (!record(source) || ![source.id, source.name, source.tenant_id, source.owner_id, source.run_id, source.status].every(identifier)
      || !text(source.summary, 2000) || typeof source.reported_call_completed !== "boolean"
      || (value.contract_satisfied && (source.status !== "done" || source.reported_call_completed !== true))
      || source.business_success_verified !== false || sources.some(item => item.id === source.id)) return null;
    sources.push({id: source.id, name: source.name, status: source.status, summary: source.summary});
  }
  return {contract_satisfied: value.contract_satisfied, reason: value.reason, ledger_sources: sources};
}
function nodes(value: any, archived = false): PlanNode[] | null {
  if (!Array.isArray(value) || !value.length || value.length > 8) return null;
  const result: PlanNode[] = [];
  for (const item of value) {
    if (!record(item) || !identifier(item.id) || result.some(node => node.id === item.id)
      || !text(item.objective, 2000, false) || !record(item.inputs) || !text(item.inputs.instruction, 4000, false)
      || !strings(item.inputs.from_nodes, 8) || (!archived && item.inputs.from_nodes.some(id => !result.some(node => node.id === id)))
      || !record(item.acceptance) || !["answer", "tool_evidence"].includes(item.acceptance.output_kind)
      || !strings(item.acceptance.required_tools) || (item.acceptance.output_kind === "tool_evidence" && !item.acceptance.required_tools.length)
      || !nodeStates.includes(item.status) || !Array.isArray(item.proposed_calls) || item.proposed_calls.length > 16
      || !strings(item.tool_call_ids) || !(item.output === null || text(item.output, 131072))
      || !(item.blocked_reason === null || text(item.blocked_reason, 2000))) return null;
    const acceptance = item.acceptance_result === null ? null : report(item.acceptance_result);
    if ((item.acceptance_result !== null && !acceptance) || (item.status === "completed" && !acceptance?.contract_satisfied)
      || (item.status === "blocked" && acceptance?.contract_satisfied)) return null;
    result.push({id: item.id, objective: item.objective, inputs: {from_nodes: [...item.inputs.from_nodes]},
      acceptance: {output_kind: item.acceptance.output_kind, required_tools: [...item.acceptance.required_tools]},
      status: item.status, acceptance_result: acceptance, blocked_reason: item.blocked_reason});
  }
  return result;
}
function snapshot(value: any): Plan | null {
  if (!record(value) || value.version !== 1 || !integer(value.revision) || ![0, 1].includes(value.replans_used)
    || !["active", "completed"].includes(value.status) || !Array.isArray(value.history) || value.history.length > 2) return null;
  const current = nodes(value.nodes);
  if (!current || !(value.current_node_id === null || current.some(node => node.id === value.current_node_id))) return null;
  const history: Plan["history"] = [];
  for (const archived of value.history) {
    if (!record(archived) || !integer(archived.revision) || archived.revision >= value.revision
      || history.some(item => item.revision === archived.revision)) return null;
    const archivedNodes = nodes(archived.nodes, true);
    if (!archivedNodes) return null;
    history.push({revision: archived.revision, nodes: archivedNodes});
  }
  return {revision: value.revision, nodes: current, history};
}
export function isPlanEvent(type: string): boolean {
  return ["plan", "plan_replanned", "plan_fallback", "node_started", "node_accepting", "node_accepted", "tool_reused"].includes(type);
}
export function publicPlanEvent(type: string, data: any): Record<string, unknown> | null {
  if (!record(data)) return null;
  if (["plan", "plan_replanned"].includes(type)) return snapshot(data);
  if (type === "plan_fallback") {
    if (!text(data.reason, 2000, false)) return null;
    const plan = data.plan === null ? null : snapshot(data.plan);
    return data.plan !== null && !plan ? null : {reason: data.reason, plan};
  }
  if (type === "tool_reused") {
    if (![data.source_id, data.call_id, data.source_call_id, data.name].every(identifier)
      || data.reported_known_result !== true || data.business_success_verified !== false) return null;
    return {call_id: data.call_id, source_id: data.source_id, source_call_id: data.source_call_id,
      name: data.name, reported_known_result: true, business_success_verified: false};
  }
  if (!["node_started", "node_accepting", "node_accepted"].includes(type)
    || !integer(data.plan_revision) || !identifier(data.node_id)) return null;
  const identity = {node_id: data.node_id, plan_revision: data.plan_revision};
  if (type !== "node_accepted") return identity;
  const accepted = report(data);
  return accepted ? {...identity, ...accepted, correctness_verified: false, business_success_verified: false} : null;
}
function element(tag: string, content = "", className = ""): HTMLElement {
  const result = document.createElement(tag);
  result.textContent = content; result.className = className;
  return result;
}

export function initializePlanProgress() {
  const section = document.getElementById("plan-progress")!;
  const summary = document.getElementById("plan-summary")!;
  const list = document.getElementById("plan-nodes")!;
  const archive = document.getElementById("plan-history")!;
  const live = document.getElementById("plan-live")!;
  let plan: Plan | null = null, terminal = "";
  const views = new Map<string, NodeView>(), reusedSources = new Set<string>();
  const histories = new Map<number, HistoryView>();

  function clear(): void {
    plan = null; terminal = ""; views.clear(); histories.clear(); reusedSources.clear();
    section.hidden = true; summary.textContent = ""; live.textContent = "";
    list.replaceChildren(); archive.replaceChildren(); archive.hidden = true;
  }
  function detail(node: PlanNode): HTMLElement {
    const body = element("div", "", "plan-detail");
    body.append(element("p", `必需工具：${node.acceptance.required_tools.join("、") || "无（非空回答契约）"}`),
      element("p", `前序依赖：${node.inputs.from_nodes.join("、") || "无"}`));
    const reason = node.acceptance_result?.reason || node.blocked_reason;
    if (reason) body.append(element("p", `验收说明：${reason}`));
    for (const source of node.acceptance_result?.ledger_sources || []) {
      const evidence = element("div", "", "plan-source");
      evidence.append(element("p", `${source.name} · 来源 ${source.id} · 账本 ${source.status}`));
      if (reusedSources.has(source.id)) evidence.append(element("p", "历史结果复用，非重新执行", "plan-reused"));
      evidence.append(element("p", source.summary)); body.append(evidence);
    }
    return body;
  }
  function render(): void {
    if (!plan) return;
    section.hidden = false;
    const passed = plan.nodes.filter(node => node.status === "completed").length;
    const active = plan.nodes.find(node => ["running", "accepting", "blocked"].includes(node.status));
    let state = "等待执行";
    if (active) state = labels[active.status];
    if (terminal) state = terminal === "completed" && passed === plan.nodes.length ? "结构通过" : terminalLabels[terminal];
    const caption = `已通过 ${passed}/${plan.nodes.length} · ${state} · 修订 ${plan.revision}`;
    if (summary.textContent !== caption) { summary.textContent = caption; live.textContent = caption; }
    const ids = new Set(plan.nodes.map(node => node.id));
    for (const [id, view] of views) if (!ids.has(id)) { view.row.remove(); views.delete(id); }
    plan.nodes.forEach((node, index) => {
      let view = views.get(node.id);
      if (!view) {
        const row = document.createElement("li"), details = document.createElement("details");
        row.dataset.nodeId = node.id;
        const title = element("span", "", "plan-objective"), status = element("span", "", "plan-state");
        const heading = element("summary"); heading.append(title, status);
        const body = element("div"); details.append(heading, body); row.append(details);
        view = {row, summary: heading, title, status, body, signature: ""};
        views.set(node.id, view); list.append(row);
      }
      const signature = JSON.stringify([node, terminal, [...reusedSources]]);
      if (signature === view.signature) return;
      view.signature = signature;
      view.title.textContent = `${index + 1}. ${node.objective}`;
      view.status.textContent = terminal && !["completed", "blocked"].includes(node.status) ? terminalLabels[terminal] : labels[node.status];
      view.row.dataset.status = node.status;
      // 保留summary节点、焦点和原生展开状态，只更新无交互的证据正文。
      view.body.replaceChildren(detail(node));
    });
    archive.hidden = !plan.history.length;
    for (const [revision, view] of histories) {
      if (!plan.history.some(history => history.revision === revision)) { view.details.remove(); histories.delete(revision); }
    }
    for (const history of plan.history) {
      let view = histories.get(history.revision);
      if (!view) {
        const details = document.createElement("details");
        details.append(element("summary", `归档修订 ${history.revision} · ${history.nodes.length} 个未完成步骤`));
        view = {details, nodes: new Map()}; histories.set(history.revision, view); archive.append(details);
      }
      for (const [id, saved] of view.nodes) {
        if (!history.nodes.some(node => node.id === id)) { saved.row.remove(); view.nodes.delete(id); }
      }
      for (const node of history.nodes) {
        let saved = view.nodes.get(node.id);
        if (!saved) {
          const row = element("div", "", "plan-archived-node");
          saved = {row, signature: ""}; view.nodes.set(node.id, saved); view.details.append(row);
        }
        const signature = JSON.stringify([node, [...reusedSources]]);
        if (saved.signature === signature) continue;
        saved.signature = signature;
        saved.row.replaceChildren(element("p", `${node.id} · ${node.objective} · ${labels[node.status]}`), detail(node));
      }
    }
  }
  function settle(status: string): void {
    if (!Object.hasOwn(terminalLabels, status)) return;
    terminal = status; render();
  }
  function accept(type: string, data: any): boolean {
    if (!isPlanEvent(type)) return false;
    const projected = publicPlanEvent(type, data);
    if (!projected) return false;
    if (["plan", "plan_replanned", "plan_fallback"].includes(type)) {
      if (type === "plan_fallback" && projected.plan === null) return !plan;
      const next = (type === "plan_fallback" ? projected.plan : projected) as Plan;
      if (!next || (plan && next.revision <= plan.revision)) return false;
      plan = next; render(); return true;
    }
    if (!plan) return false;
    if (type === "tool_reused") {
      if (reusedSources.size >= 32) return false;
      reusedSources.add(projected.source_id as string); render(); return true;
    }
    if (projected.plan_revision !== plan.revision) return false;
    const node = plan.nodes.find(item => item.id === projected.node_id);
    if (!node || ["completed", "blocked"].includes(node.status)) return false;
    if (type === "node_accepted") {
      const accepted = projected as unknown as Report;
      node.acceptance_result = accepted;
      node.status = accepted.contract_satisfied ? "completed" : "blocked";
      node.blocked_reason = accepted.contract_satisfied ? null : accepted.reason;
    } else if (type === "node_accepting") node.status = "accepting";
    else if (node.status === "pending") node.status = "running";
    render();
    return true;
  }
  clear();
  return {accept, settle, clear};
}
