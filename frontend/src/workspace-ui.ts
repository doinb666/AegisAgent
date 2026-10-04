"use strict";

import WorkspaceOptions from "./workspace-options";

// 展示层只创建安全文本节点，不解释模型或文件返回的 HTML。
const WorkspaceUI = (() => {
  const byId = (id: string): any => document.getElementById(id);
  const node = (tag: string, text = "", className = ""): any => {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  let fileGeneration = 0;
  let previewGeneration = 0;
  const themeKey = "aegis-theme";

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const label = theme === "dark" ? "浅色主题" : "深色主题";
    byId("theme-label").textContent = label;
    byId("theme-toggle").setAttribute("aria-label", `切换为${label}`);
    try { localStorage.setItem(themeKey, theme); } catch { /* 浏览器禁止存储时保留当前主题。 */ }
  }
  let savedTheme;
  try { savedTheme = localStorage.getItem(themeKey); } catch { savedTheme = null; }
  setTheme(savedTheme === "dark" ? "dark" : "light");
  byId("theme-toggle").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

  function setInspector(expanded) {
    byId("inspector").hidden = !expanded;
    byId("shell").classList.toggle("inspector-collapsed", !expanded);
    byId("inspector-toggle").setAttribute("aria-expanded", String(expanded));
    byId("inspector-toggle").setAttribute("aria-label", expanded ? "折叠审查区" : "展开审查区");
  }
  byId("inspector-toggle").onclick = () => setInspector(byId("inspector").hidden);
  setInspector(false);

  function requestError(data, status) {
    if(typeof data?.detail === "string") return data.detail.slice(0,2000);
    if(Array.isArray(data?.detail)) {
      const fields={username:"账号",password:"密码",message:"任务内容",name:"名称",content:"内容",model:"模型",mode:"执行方式",session_id:"会话",directory:"技能目录",document:"技能文件",expected_version:"版本",query:"搜索内容"};
      const messages=data.detail.filter(error=>error && typeof error==="object").slice(0,4).map(error=>{
        const location=Array.isArray(error.loc) ? error.loc.at(-1) : null;
        const field=typeof location==="string" && Object.hasOwn(fields,location) ? fields[location] : "提交字段";
        const limit=error.ctx?.min_length ?? error.ctx?.max_length;
        const bounded=typeof limit==="number" && Number.isFinite(limit) && limit>=0 && limit<=1000000;
        if(error.type==="missing") return `${field}不能为空`;
        if(error.type==="string_too_short") return bounded ? `${field}至少需要 ${limit} 个字符` : `${field}过短`;
        if(error.type==="string_too_long") return bounded ? `${field}最多允许 ${limit} 个字符` : `${field}过长`;
        if(error.type==="literal_error") return `${field}选项无效，请重新选择`;
        return `${field}格式不正确，请检查后重试`;
      });
      if(messages.length) return messages.join("；")+"。";
    }
    return `请求失败 (${status})，请稍后重试；若持续失败，请检查服务状态。`;
  }

  function renderCapabilities(cap) {
    const models = cap.models || [], tools = cap.tools || [];
    const roles = {admin: "管理员", operator: "操作员", viewer: "只读成员"};
    const rows = [
      ["模型来源", (cap.model_routes || []).map(route=>`${route.label} · ${route.model} (${route.provider})`).join("；") || models.join("、") || "未配置。可在下方生成配置示例，再由管理员写入环境变量并重启服务。"],
      ["工具", tools.join("、") || "未配置工具"],
      ["执行沙箱", cap.sandbox ? "已配置，代码执行仍需审批；连通状态未验证。" : "未配置，代码执行将被拒绝。"],
      ["当前角色", roles[cap.role] || cap.role || "未提供"],
      ["步数预算", `每个任务最多 ${cap.max_steps} 步`]
    ];
    byId("capability-list").replaceChildren(...rows.map(([title, value]) => {
      const row = node("div"); row.append(node("dt", title), node("dd", value)); return row;
    }));
    // 仅保留展示契约中的配置证据，避免未知字段包含密钥。
    byId("capabilities").textContent = JSON.stringify({models, tools, sandbox: cap.sandbox, role: cap.role, max_steps: cap.max_steps}, null, 2);
  }
  function renderOverview(overview, statuses) {
    const running = overview.runs.by_status.running || 0;
    const waiting = overview.runs.by_status.waiting_approval || 0;
    byId("workspace-overview").textContent = `本人任务 ${overview.runs.total} · 正在推进 ${running} · 待审批 ${waiting} · 资产 ${overview.assets.total}`;
    byId("workspace-overview").title = Object.entries(overview.runs.by_status).map(([key, count]) => `${statuses[key] || key} ${count}`).join("；");
  }
  function clearFiles() {
    fileGeneration++; previewGeneration++;
    byId("file-list").textContent = "打开任务后可查看文件。";
    byId("file-preview").textContent = "";
    byId("file-preview").hidden = true;
    byId("files-refresh").disabled = false;
  }
  async function loadFiles(id, api, current) {
    const generation = ++fileGeneration;
    previewGeneration++;
    const valid = () => current() && generation === fileGeneration;
    byId("file-list").textContent = "正在读取任务文件…";
    byId("file-preview").textContent = ""; byId("file-preview").hidden = true;
    byId("files-refresh").disabled = true;
    try {
      const result = await api(`/runs/${id}/files`);
      if (!valid()) return;
      const rows = result.files.map(file => {
        const button = node("button"); button.type = "button";
        button.append(node("span", file.path), node("small", `${file.bytes} B`));
        button.onclick = async () => {
          const preview = ++previewGeneration;
          const previewValid = () => valid() && preview === previewGeneration;
          byId("file-preview").hidden = false; byId("file-preview").textContent = "正在读取文件…";
          try {
            const content = await api(`/runs/${id}/file?path=${encodeURIComponent(file.path)}`);
            if (!previewValid()) return;
            byId("file-preview").textContent = `${content.path}\n${content.truncated ? "内容已截断，仅显示前 64 KiB。\n" : ""}\n${content.content}`;
          } catch (error) { if (previewValid()) byId("file-preview").textContent = `无法预览：${error.message}`; }
        };
        return button;
      });
      byId("file-list").replaceChildren(...rows);
      if (!rows.length) byId("file-list").textContent = "此任务尚无可预览文件。";
      if (result.truncated) byId("file-list").append(node("p", "文件列表已截断，请缩小任务文件范围。", "muted"));
    } catch (error) { if (valid()) byId("file-list").textContent = `文件列表读取失败：${error.message}`; }
    finally { if (valid()) byId("files-refresh").disabled = false; }
  }
  function eventRow(id, type, data, statuses) {
    const names = {queued: "任务已排队", running: "任务开始推进", assets_recalled: "召回本人记忆与能力", bootstrap: "加载任务上下文", model: "收到模型响应", model_request: "请求模型", model_response: "收到模型响应", tool_started: "开始工具调用", tool_call: "准备工具调用", tool_result: "收到工具结果", risk_review: "记录独立风险审查", reflection: "记录回答复核", waiting_approval: "请求审批", approval: "记录审批决定", feedback: "记录任务反馈", plan: "生成执行计划", plan_fallback: "规划失败，记录回退原因", checkpoint: "保存执行进度"};
    const caption = names[type] || statuses[type] || "保存任务事件";
    let description = caption;
    if (data.name) description += `：${data.name}`;
    if (type === "approval") description += data.approved ? "，已批准" : "，已拒绝";
    if (type === "feedback") description += data.success ? "，目标已达成" : "，仍需改进";
    if (data.error) description += `：${String(data.error).slice(0,120)}`;
    const row = node("li"); row.append(node("span", `#${id} · ${caption}`), node("div", description));
    const details = node("details"), evidence = node("pre", JSON.stringify(data, null, 2));
    details.append(node("summary", "查看完整事件证据"), evidence); row.append(details);
    return row;
  }
  const configureCollaboration = (...args: any[]) => (
    WorkspaceOptions.configureCollaboration as (...values: any[]) => any
  )(...args);
  const clearCollaboration = () => WorkspaceOptions.clearCollaboration();
  const renderCollaboration = (...args: any[]) => (
    WorkspaceOptions.renderCollaboration as (...values: any[]) => any
  )(...args);
  return Object.freeze({node, setInspector, requestError, renderCapabilities, renderOverview, clearFiles, loadFiles, eventRow, configureCollaboration, clearCollaboration, renderCollaboration});
})();

window.WorkspaceUI = WorkspaceUI;

export default WorkspaceUI;
