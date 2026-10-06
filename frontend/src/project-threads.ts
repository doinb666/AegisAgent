interface Project { id: string; name: string; status: string; }
export interface Thread {
  id: string; title: string; session_id: string; project_id: string | null;
  project_name: string | null; archived: boolean; latest_run_id: string | null;
}
interface Options {
  api: (path: string, options?: object) => Promise<any>;
  identity: () => string | null;
  canWrite: () => boolean;
  select: (thread: Thread) => Promise<void>;
  notice: (message: string) => void;
  changed: (thread: Thread | null) => void;
}
const element = (id: string): HTMLElement => document.getElementById(id)!;
const text = (tag: string, content: string, className = ""): HTMLElement => {
  const node = document.createElement(tag); node.textContent = content; node.className = className;
  return node;
};
const mergeById = <T extends { id: string }>(previous: T[], page: T[]): T[] =>
  [...new Map([...previous, ...page].map(item => [item.id, item])).values()];

export function initializeProjectThreads(options: Options) {
  let enabled = false, generation = 0, archived = false;
  let refreshTicket = 0, refreshing = false;
  let projects: Project[] = [], threads: Thread[] = [], selected: Thread | null = null;
  const pending = new Map<string, string>();
  const current = () => {
    const token = options.identity(), epoch = generation;
    return () => !!token && token === options.identity() && epoch === generation;
  };
  const perform = async (action: () => Promise<void>, button?: HTMLButtonElement): Promise<void> => {
    const valid = current();
    if (button) button.disabled = true;
    try { await action(); }
    catch (error) { if (valid()) options.notice(error instanceof Error ? error.message : "会话操作失败"); }
    finally { if (button && valid()) button.disabled = false; }
  };

  function show(thread: Thread | null): void {
    selected = thread;
    options.changed(thread);
    element("thread-context").hidden = !thread;
    element("thread-rename-form").hidden = true;
    if (!thread) return;
    element("current-project-label").textContent = thread.project_name ? `项目 · ${thread.project_name}` : "独立会话";
    element("current-thread-title").textContent = thread.title;
    element("thread-archived-label").hidden = !thread.archived;
    element("thread-actions").hidden = !options.canWrite();
    element("thread-archive").textContent = thread.archived ? "恢复会话" : "归档会话";
  }

  function threadButton(thread: Thread): HTMLButtonElement {
    const button = document.createElement("button");
    button.type = "button"; button.className = "thread-item";
    button.textContent = thread.title; button.title = thread.title;
    button.dataset.threadId = thread.id;
    button.onclick = () => { void perform(() => options.select(thread), button); };
    return button;
  }

  async function create(project?: Project): Promise<void> {
    if (!enabled || !options.canWrite()) return;
    const valid = current();
    const body = JSON.stringify({ title: "新会话", project_id: project?.id || null });
    if (!pending.has(body)) pending.set(body, crypto.randomUUID());
    const thread: Thread = await options.api("/threads", {
      method: "POST", headers: { "Idempotency-Key": pending.get(body)! }, body,
    });
    if (!valid()) return;
    pending.delete(body);
    if (project && !thread.project_name) thread.project_name = project.name;
    await options.select(thread);
    await refresh();
  }

  function renderProjects(): void {
    const list = element("project-thread-list"); list.replaceChildren();
    if (!projects.length) list.append(text("p", "启用项目后可在这里创建独立会话。", "muted"));
    for (const project of projects) {
      const group = document.createElement("details"); group.className = "project-thread-group";
      const heading = document.createElement("summary"); heading.textContent = project.name;
      const items = document.createElement("div"); items.className = "project-thread-items";
      const start = document.createElement("button"); start.type = "button";
      start.textContent = "新建项目会话"; start.disabled = !options.canWrite();
      start.onclick = () => { void perform(() => create(project), start); };
      const more = document.createElement("button"); more.type = "button"; more.textContent = "展开更多会话";
      let loaded: Thread[] = [], loading = false, initialized = false;
      const load = async (append: boolean): Promise<void> => {
        if (loading) return;
        loading = true; more.disabled = true;
        const valid = current();
        const params = new URLSearchParams({project_id: project.id, limit: "5", archived: String(archived)});
        if (append && loaded.length) params.set("before", loaded.at(-1)!.id);
        try {
          const page: Thread[] = await options.api(`/threads?${params}`);
          if (!valid() || !group.isConnected) return;
          loaded = append ? mergeById(loaded, page) : page;
          items.replaceChildren(...loaded.map(threadButton));
          if (!loaded.length) items.append(text("p", "暂无会话", "muted"));
          more.hidden = page.length < 5; more.textContent = "展开更多会话"; initialized = true;
        } catch (error) {
          if (valid() && group.isConnected) {
            items.replaceChildren(...loaded.map(threadButton), text("p", "读取失败，可点击下方重试。", "muted"));
            more.hidden = false; more.textContent = "重试读取会话";
          }
        } finally { loading = false; if (valid()) more.disabled = false; }
      };
      group.append(heading, start, items, more); list.append(group);
      more.hidden = true;
      more.onclick = () => { void load(initialized); };
      group.ontoggle = () => { if (group.open && !initialized) void load(false); };
    }
  }

  async function refresh(moreThreads = false, moreProjects = false): Promise<void> {
    if (!enabled || !options.identity()) return;
    if (refreshing && (moreThreads || moreProjects)) return;
    const context = current(), ticket = ++refreshTicket;
    const valid = () => context() && ticket === refreshTicket;
    refreshing = true;
    for (const id of ["threads-more", "projects-more"]) (element(id) as HTMLButtonElement).disabled = true;
    element("threads-status").textContent = "正在读取会话…";
    try {
      if (!moreThreads) {
        const params = new URLSearchParams({limit: "20"});
        if (moreProjects && projects.length) params.set("before", projects.at(-1)!.id);
        const page: Project[] = await options.api(`/workspace/projects?${params}`);
        if (!valid()) return;
        projects = moreProjects ? mergeById(projects, page) : page;
        element("projects-more").hidden = page.length < 20;
        renderProjects();
      }
      const params = new URLSearchParams({limit: "20", archived: String(archived)});
      if (moreThreads && threads.length) params.set("before", threads.at(-1)!.id);
      const page: Thread[] = await options.api(`/threads?${params}`);
      if (!valid()) return;
      threads = moreThreads ? mergeById(threads, page) : page;
      const list = element("recent-thread-list");
      list.replaceChildren(...threads.map(threadButton));
      if (!threads.length) list.append(text("p", archived ? "暂无归档会话。" : "新任务会自动建立独立会话。", "muted"));
      element("threads-more").hidden = page.length < 20;
      element("threads-status").textContent = "";
    } catch (error) {
      if (valid()) element("threads-status").textContent = `读取失败，可刷新重试：${error instanceof Error ? error.message : "服务不可用"}`;
    } finally {
      if (ticket === refreshTicket) {
        refreshing = false;
        for (const id of ["threads-more", "projects-more"]) (element(id) as HTMLButtonElement).disabled = false;
      }
    }
  }

  element("threads-refresh").onclick = () => { generation++; void refresh(); };
  element("threads-more").onclick = event => { void perform(() => refresh(true), event.currentTarget as HTMLButtonElement); };
  element("projects-more").onclick = event => { void perform(() => refresh(false, true), event.currentTarget as HTMLButtonElement); };
  element("threads-archive-toggle").onclick = () => {
    archived = !archived; generation++;
    element("threads-caption").textContent = archived ? "归档会话" : "最近会话";
    element("threads-archive-toggle").textContent = archived ? "返回最近" : "查看归档";
    void refresh();
  };
  element("thread-rename").onclick = () => {
    if (!selected) return;
    element("thread-rename-form").hidden = false;
    const input = element("thread-title-input") as HTMLInputElement; input.value = selected.title; input.focus();
  };
  element("thread-rename-cancel").onclick = () => { element("thread-rename-form").hidden = true; };
  const modify = async (body: object): Promise<void> => {
    if (!selected || !options.canWrite()) return;
    const id = selected.id, valid = current();
    const thread: Thread = await options.api(`/threads/${encodeURIComponent(id)}`, {method: "PATCH", body: JSON.stringify(body)});
    if (!valid()) return;
    if (selected?.id === id) show(thread);
    await refresh();
  };
  element("thread-rename-form").onsubmit = event => {
    event.preventDefault();
    const title = (element("thread-title-input") as HTMLInputElement).value;
    void perform(() => modify({title}), (event as SubmitEvent).submitter as HTMLButtonElement);
  };
  element("thread-archive").onclick = event => {
    if (selected) void perform(() => modify({archived: !selected!.archived}), event.currentTarget as HTMLButtonElement);
  };

  return {
    enabled: () => enabled,
    configure: (capabilities: any): void => {
      generation++; enabled = capabilities.threads?.enabled === true;
      element("project-thread-navigation").hidden = !enabled;
      if (enabled) void refresh();
    },
    clear: (): void => {
      generation++; enabled = false; projects = []; threads = []; archived = false;
      refreshTicket++; refreshing = false;
      element("threads-caption").textContent = "最近会话";
      element("threads-archive-toggle").textContent = "查看归档";
      pending.clear(); show(null);
      element("project-thread-list").replaceChildren(); element("recent-thread-list").replaceChildren();
      element("project-thread-navigation").hidden = true;
    },
    refresh, show, create,
    isArchived: () => selected?.archived === true,
    forRun: async (threadId: string | null, valid: () => boolean): Promise<void> => {
      if (!enabled || !threadId) { show(null); return; }
      const thread: Thread = await options.api(`/threads/${encodeURIComponent(threadId)}`);
      if (valid()) show(thread);
    },
  };
}
