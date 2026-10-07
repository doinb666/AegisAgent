"use strict";

import { copyText } from "./clipboard";

// 工作区选项与模型配置示例均使用安全文本；不保存或读取服务端密钥。
const WorkspaceOptions = (() => {
  type ModelEntry = {
    provider: string;
    label: string;
    model: string;
    priority: number;
    base_url?: string;
    api_key_env?: string;
    api_version?: string;
  };

  const byId = (id: string): any => document.getElementById(id);
  const node = (tag: string, text = "", className = ""): any => {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  let nodes: any[] = [];
  const acceptances = new Map<string, any>();
  let entries: ModelEntry[] = [];
  let configGeneration = 0;
  const states = {pending:"等待前驱",running:"正在执行",completed:"执行完成",failed:"执行失败",blocked:"已阻断",cancelled:"已停止",interrupted:"需核对"};

  function configureCollaboration(cap, writable) {
    const modes = cap.collaboration?.project_modes || [];
    byId("collaboration-mode").disabled = !writable;
    byId("project-mode").disabled = !writable;
    for (const option of byId("project-mode").options) {
      option.disabled = Boolean(option.value) && !modes.includes(option.value);
    }
    if (!modes.includes(byId("project-mode").value)) byId("project-mode").value = "";
    byId("collaboration-hint").textContent = modes.length
      ? "准备代码目录需批准，子任务只读父目录。协作默认方式可由主控按任务合法选择调整，最多两个子任务。"
      : "管理员尚未绑定仓库，代码准备入口已停用。仍可委派只读分析，最多两个子任务。";
  }

  function clearCollaboration() {
    nodes = []; acceptances.clear();
    byId("collaboration-nodes").replaceChildren();
    byId("collaboration-review").hidden = true;
  }

  function renderCollaboration(type, data, openRun) {
    if (type === "collaboration_graph" && Array.isArray(data.nodes)) nodes = data.nodes;
    if (type === "collaboration_acceptance") acceptances.set(data.run_id, data);
    if (!nodes.length) return;
    byId("collaboration-review").hidden = false;
    byId("collaboration-nodes").replaceChildren(...nodes.map(item => {
      const row = node("li"), acceptance = acceptances.get(item.run_id);
      row.append(node("strong", item.id), node("span", states[item.status] || item.status, "status-tag"));
      row.append(node("p", item.depends_on?.length ? `前驱：${item.depends_on.join("、")}` : "独立节点", "muted"));
      if (acceptance) {
        const verified = acceptance.status === "verified";
        row.append(node("p", verified ? "账本验收通过" : "账本验收未通过", verified ? "muted" : "error"));
        for (const reason of acceptance.reasons || []) row.append(node("p", reason, "muted"));
        const names = (acceptance.evidence || []).map(call => call.name);
        if (names.length) row.append(node("p", `完成工具：${names.join("、")}`, "muted"));
      }
      if (item.run_id) {
        const button = node("button", "查看子任务来源"); button.type = "button";
        button.onclick = () => openRun(item.run_id); row.append(button);
      }
      return row;
    }));
  }

  function updateProvider() {
    const kind = byId("provider-kind").value;
    const local = kind === "ollama";
    byId("provider-key-env").disabled = local;
    byId("provider-key-env").required = !local;
    byId("provider-base").required = kind === "custom" || kind === "azure";
    byId("provider-version-row").hidden = kind !== "azure";
    byId("provider-version").required = kind === "azure";
    byId("provider-error").textContent = "";
  }
  byId("provider-kind").onchange = updateProvider;
  updateProvider();
  byId("provider-form").onsubmit = event => {
    event.preventDefault();
    const base = byId("provider-base").value.trim();
    if (base) {
      let endpoint: URL;
      try {
        endpoint = new URL(base);
      } catch {
        byId("provider-error").textContent = "基础地址不是有效 URL。";
        return;
      }
      if (!["http:", "https:"].includes(endpoint.protocol) || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
        byId("provider-error").textContent = "基础地址须为不含凭证、查询和片段的 HTTP(S) 地址。";
        return;
      }
    }
    if (entries.length >= 32) { byId("provider-error").textContent = "最多配置 32 路来源。"; return; }
    const provider = byId("provider-kind").value;
    const entry: ModelEntry = {provider, label:byId("provider-label").value.trim(), model:byId("provider-model").value.trim(), priority:Number(byId("provider-priority").value)};
    if (!entry.model) { byId("provider-error").textContent = "模型名称不能为空。"; return; }
    if (base) entry.base_url = base;
    if (provider !== "ollama") entry.api_key_env = byId("provider-key-env").value;
    if (provider === "azure") entry.api_version = byId("provider-version").value;
    if (entries.some(previous => previous.provider === provider && previous.model === entry.model && previous.base_url === entry.base_url)) {
      byId("provider-error").textContent = "同一来源与模型已在示例中。"; return;
    }
    entries.push(entry);
    configGeneration++;
    byId("provider-error").textContent = "";
    // 这是 dotenv 文本，单引号仅按 dotenv 规则转义，不能作为 shell 命令执行。
    byId("model-config-preview").textContent = "AEGIS_MODELS_JSON='" + JSON.stringify(entries).replaceAll("'", "\\'") + "'";
    byId("provider-copy").disabled = false;
    byId("provider-copy-status").textContent = "";
  };
  byId("provider-copy").onclick = async () => {
    const button = byId("provider-copy");button.disabled = true;
    const generation = configGeneration;
    const current = () => generation === configGeneration;
    await copyText(byId("model-config-preview").textContent, byId("model-config-preview"),
      byId("provider-copy-status"), "配置", current);
    if (current()) button.disabled = entries.length === 0;
  };
  byId("provider-clear").onclick = () => {
    entries = []; configGeneration++;byId("provider-error").textContent = "";
    byId("model-config-preview").textContent = "尚未生成配置。已有服务配置不会被这里修改。";
    byId("provider-copy").disabled = true;byId("provider-copy-status").textContent = "";
  };
  return Object.freeze({configureCollaboration, clearCollaboration, renderCollaboration});
})();

window.WorkspaceOptions = WorkspaceOptions;

export default WorkspaceOptions;
