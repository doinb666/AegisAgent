interface Command {
  title: string;
  description: string;
  run: () => void;
}

function element<T extends HTMLElement>(id: string): T {
  const result = document.getElementById(id);
  if (!result) throw new Error(`缺少工作台元素：${id}`);
  return result as T;
}

export function initializeNavigation(
  isLoggedIn: () => boolean,
  newTask: () => void,
  showView: (view: string) => void,
): { close: () => void } {
  const shell = element("shell");
  const sidebar = element("workspace-sidebar");
  const toggle = element<HTMLButtonElement>("sidebar-toggle");
  const panel = element("quick-actions");
  const open = element<HTMLButtonElement>("command-toggle");
  const query = element<HTMLInputElement>("command-query");
  const results = element("command-results");
  const count = element("command-count");
  let previousFocus: HTMLElement | null = null;

  function showSidebarSection(projects: boolean): void {
    element("sidebar-project-panel").hidden = !projects;
    element("run-navigation").hidden = projects;
    element("sidebar-projects").setAttribute("aria-pressed", String(projects));
    element("sidebar-history").setAttribute("aria-pressed", String(!projects));
  }
  element("sidebar-projects").onclick = () => showSidebarSection(true);
  element("sidebar-history").onclick = () => showSidebarSection(false);

  function setSidebar(expanded: boolean): void {
    sidebar.hidden = !expanded;
    shell.classList.toggle("sidebar-collapsed", !expanded);
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.setAttribute("aria-label", expanded ? "折叠导航" : "展开导航");
  }
  toggle.onclick = () => setSidebar(Boolean(sidebar.hidden));
  setSidebar(!window.matchMedia("(max-width: 700px)").matches);

  function close(): void {
    if (panel.hidden) return;
    panel.hidden = true;
    open.setAttribute("aria-expanded", "false");
    if (previousFocus?.isConnected && previousFocus.getClientRects().length) previousFocus.focus();
    else open.focus();
    query.value = "";
    results.replaceChildren();
    count.textContent = "";
  }

  const commands: Command[] = [
    {title: "新任务", description: "描述新的目标，Alt N", run: newTask},
    ...[
      ["chat", "任务空间", "回到当前任务与执行证据"],
      ["projects", "项目", "管理项目目标与边界"],
      ["memories", "长期记忆", "审阅个人偏好、约束与经验"],
      ["skills", "技能库", "Skills：审阅、修订和复用技能"],
      ["documents", "知识库", "上传与检索私有资料"],
      ["schedules", "定时任务", "安排只读任务、暂停计划与查看触发记录"],
      ["settings", "能力与设置", "查看模型接入和企业成员"],
    ].map(([view, title, description]) => ({title, description, run: () => showView(view)})),
    {title: "搜索任务", description: "按内容和状态查找历史任务", run: () => {
      setSidebar(true); showSidebarSection(false); element("run-search").focus();
    }},
    {title: "项目会话", description: "展开项目树、最近会话与归档", run: () => {
      setSidebar(true); showSidebarSection(true); element("sidebar-projects").focus();
    }},
    {title: "切换任务审查", description: "展开或折叠审批与工具证据", run: () => element("inspector-toggle").click()},
  ];

  function render(): void {
    const term = query.value.trim().toLocaleLowerCase();
    const matches = commands.filter(command => `${command.title} ${command.description}`.toLocaleLowerCase().includes(term));
    results.replaceChildren(...matches.map(command => {
      const row = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      const title = document.createElement("span");
      title.textContent = command.title;
      const description = document.createElement("small");
      description.textContent = command.description;
      button.append(title, description);
      button.onclick = () => { close(); command.run(); };
      row.append(button);
      return row;
    }));
    count.textContent = matches.length ? `${matches.length} 个入口，方向键选择，Enter 打开。` : "没有匹配的入口，请换个关键词。";
  }
  function toggleCommands(): void {
    if (!isLoggedIn()) return;
    if (!panel.hidden) { close(); return; }
    previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    panel.hidden = false;
    open.setAttribute("aria-expanded", "true");
    render();
    query.focus();
  }
  open.onclick = toggleCommands;
  element("command-close").onclick = close;
  query.oninput = render;
  panel.addEventListener("keydown", event => {
    if (event.key === "Enter" && event.target === query) {
      event.preventDefault(); results.querySelector<HTMLButtonElement>("button")?.click();
    } else if (["ArrowDown", "ArrowUp"].includes(event.key)) {
      event.preventDefault();
      const buttons = Array.from(results.querySelectorAll<HTMLButtonElement>("button"));
      const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
      const next = event.key === "ArrowDown" ? index + 1 : index - 1;
      if (next < 0) query.focus();
      else buttons[Math.min(next, buttons.length - 1)]?.focus();
    }
  });
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k" && isLoggedIn()) {
      event.preventDefault(); toggleCommands();
    } else if (event.key === "Escape" && !panel.hidden) {
      event.preventDefault(); close();
    }
  });
  return {close};
}
