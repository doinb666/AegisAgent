interface Template {
  id: string; title: string; description: string; message: string;
  required_tools: string[]; expected_output: string; available: boolean;
}
interface Reference { id: string; name: string; version: number; content_hash?: string; truncated?: boolean }
interface Server { name: string; tools: string[]; credential_ready: boolean; configuration_only: boolean }
interface Options {
  api: (path: string, options?: object) => Promise<any>;
  identity: () => string | null;
  message: () => string;
  prefill: (message: string) => Promise<void>;
}
const element = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
function node(tag: string, text: string, className = ""): HTMLElement {
  const result = document.createElement(tag); result.textContent = text; result.className = className; return result;
}
function failure(error: unknown): string { return error instanceof Error ? error.message : "服务暂不可用，请刷新。"; }
const toolNames: Record<string, string> = {knowledge_search: "本人资料检索", calculator: "计算器", skill_read: "技能读取"};

export function initializeTaskInputs(options: Options) {
  let enabled = false, epoch = 0, view = "chat", templateTicket = 0, referenceTicket = 0, serverTicket = 0;
  let references: Reference[] = [], more = false, loading = false, replacement: Template | null = null;
  const selected = new Map<string, Reference>();
  const current = () => {
    const identity = options.identity(), generation = epoch;
    return () => !!identity && identity === options.identity() && generation === epoch;
  };
  async function useTemplate(template: Template): Promise<void> {
    if (!enabled || !template.available) return;
    if (options.message().trim()) {
      replacement = template;
      element("template-replace-panel").hidden = false;
      element("template-replace-description").textContent = `已有任务内容。是否用「${template.title}」替换？保留原内容不会执行任务。`;
      element("template-keep").focus(); return;
    }
    await prefill(template);
  }
  async function prefill(template: Template): Promise<void> {
    const valid = current();
    try { await options.prefill(template.message); }
    catch (error) { if (valid()) element("template-status").textContent = failure(error); }
  }
  async function templates(): Promise<void> {
    if (!enabled || view !== "templates") return;
    const context = current(), ticket = ++templateTicket;
    const valid = () => context() && view === "templates" && ticket === templateTicket;
    element("template-status").textContent = "正在读取模板…";
    try {
      const rows: Template[] = await options.api("/task-templates"); if (!valid()) return;
      const list = element("template-list"); list.replaceChildren();
      for (const template of rows) {
        const item = node("article", "", "template-row");
        item.append(node("h4", template.title), node("p", template.description),
          node("p", `所需能力：${template.required_tools.map(name => toolNames[name] || name).join("、") || "基于提供内容进行分析"}`, "muted"),
          node("p", `验收要求：${template.expected_output}`, "template-evidence"));
        const button = document.createElement("button"); button.type = "button";
        button.textContent = template.available ? "预填任务" : "缺少所需工具";
        button.disabled = !template.available; button.onclick = () => { void useTemplate(template); };
        item.append(button); list.append(item);
      }
      if (!rows.length) list.append(node("p", "暂无任务模板，可直接描述你的目标。", "empty"));
      element("template-status").textContent = "";
    } catch (error) { if (valid()) element("template-status").textContent = `读取失败：${failure(error)}`; }
  }
  function renderReferences(): void {
    const list = element("reference-list"); list.replaceChildren();
    for (const reference of references) {
      const row = document.createElement("label"); row.className = "reference-option";
      const checkbox = document.createElement("input"); checkbox.type = "checkbox";
      checkbox.checked = selected.has(reference.id); checkbox.disabled = !checkbox.checked && selected.size >= 3;
      const title = node("span", reference.name); title.append(node("small", `版本 ${reference.version}`, "muted"));
      checkbox.onchange = () => {
        if (checkbox.checked && selected.size < 3) selected.set(reference.id, reference);
        else selected.delete(reference.id);
        renderReferences(); renderSelected();
      };
      row.append(checkbox, title); list.append(row);
    }
    if (!references.length) list.append(node("p", "暂无可引用资料。先在知识库上传文本资料，再刷新。", "muted"));
    element("references-more").hidden = !more;
  }
  function renderSelected(): void {
    const list = element("reference-selected"); list.replaceChildren();
    for (const reference of selected.values()) {
      const button = document.createElement("button"); button.type = "button";
      button.textContent = `${reference.name} ×`; button.setAttribute("aria-label", `移除资料 ${reference.name}`);
      button.onclick = () => { selected.delete(reference.id); renderReferences(); renderSelected(); };
      list.append(button);
    }
    element("references-count").textContent = `已选 ${selected.size}/3 · 创建任务时核验最新版本`;
  }
  async function documents(append = false): Promise<void> {
    if (!enabled || (append && (loading || !more))) return;
    const context = current(), ticket = ++referenceTicket, valid = () => context() && ticket === referenceTicket;
    loading = true; element<HTMLButtonElement>("references-more").disabled = true;
    element("references-status").textContent = "正在读取本人资料…";
    try {
      const query = new URLSearchParams({limit: "20"});
      if (append && references.length) query.set("before", references.at(-1)!.id);
      const page: Reference[] = await options.api(`/documents/references?${query}`); if (!valid()) return;
      references = append ? [...new Map([...references, ...page].map(row => [row.id, row])).values()] : page;
      more = page.length === 20; renderReferences(); element("references-status").textContent = "";
    } catch (error) { if (valid()) element("references-status").textContent = `读取失败：${failure(error)}`; }
    finally { if (valid()) { loading = false; element<HTMLButtonElement>("references-more").disabled = false; } }
  }
  async function servers(): Promise<void> {
    if (!enabled || view !== "settings") return;
    const context = current(), ticket = ++serverTicket;
    const valid = () => context() && ticket === serverTicket && view === "settings";
    element("mcp-guide-status").textContent = "正在读取授权配置…";
    try {
      const rows: Server[] = await options.api("/mcp/servers"); if (!valid()) return;
      const list = element("mcp-guide-list"); list.replaceChildren();
      for (const server of rows) {
        const item = node("article", "", "mcp-guide-row");
        item.append(node("h4", server.name), node("p", server.credential_ready ? "配置就绪 · 未连接验证" : "凭据缺失 · 请联系管理员"),
          node("p", `允许的工具：${server.tools.join("、") || "未配置工具白名单"}`, "muted"));
        list.append(item);
      }
      if (!rows.length) list.append(node("p", "暂无获授权的 MCP 服务。管理员需配置服务、工具白名单及用户授权；只读成员不开放外部工具。", "muted"));
      element("mcp-guide-status").textContent = "";
    } catch (error) { if (valid()) element("mcp-guide-status").textContent = `读取失败：${failure(error)}`; }
  }
  element("templates-refresh").onclick = () => { void templates(); };
  element("references-refresh").onclick = () => { void documents(); };
  element("references-more").onclick = () => { void documents(true); };
  element<HTMLDetailsElement>("task-documents").ontoggle = () => {
    if (element<HTMLDetailsElement>("task-documents").open && !references.length) void documents();
  };
  element("mcp-guide-refresh").onclick = () => { void servers(); };
  element("template-keep").onclick = () => { replacement = null; element("template-replace-panel").hidden = true; };
  element("template-replace").onclick = () => {
    const template = replacement; replacement = null; element("template-replace-panel").hidden = true;
    if (template) void prefill(template);
  };
  function resetSelection(): void { selected.clear(); renderSelected(); renderReferences(); }
  function showRun(run: {document_references?: Reference[]} | null): void {
    const panel = element("run-documents"); panel.replaceChildren();
    const sources = run?.document_references || []; panel.hidden = sources.length === 0;
    if (sources.length) panel.append(node("span", "本次任务引用：", "muted"));
    for (const reference of sources) panel.append(node("span", `${reference.name} · 版本 ${reference.version}${reference.truncated ? " · 预览已截断" : ""}`));
  }
  function clear(): void {
    epoch++; templateTicket++; referenceTicket++; serverTicket++; enabled = false;
    references = []; loading = false; more = false; replacement = null; resetSelection(); showRun(null);
    element("template-list").replaceChildren(); element("mcp-guide-list").replaceChildren();
    element("reference-list").replaceChildren(); element("task-documents").removeAttribute("open");
    element("template-replace-panel").hidden = true; element("references-more").hidden = true;
    for (const id of ["template-status", "references-status", "mcp-guide-status"]) element(id).textContent = "";
  }
  return {
    clear, resetSelection, showRun,
    documentIds: (): string[] => [...selected.keys()].sort(),
    configure(capabilities: any): void {
      clear(); enabled = capabilities.task_inputs?.enabled === true;
      element("task-documents").hidden = !enabled; element("mcp-guide").hidden = !enabled;
      element("templates-unavailable").hidden = enabled; element("templates-refresh").hidden = !enabled;
      renderSelected();
    },
    show(value: string): void {
      view = value; templateTicket++; serverTicket++;
      if (view === "templates") void templates();
      if (view === "settings") void servers();
    },
  };
}
