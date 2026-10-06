interface Notification {
  id: number; run_id: string; type: string; title: string; created: number; read_at: number | null;
}
interface Options {
  api: (path: string, options?: object) => Promise<any>;
  identity: () => string | null;
  openRun: (id: string) => Promise<unknown>;
}
const element = (id: string): HTMLElement => document.getElementById(id)!;
const labels: Record<string, string> = {
  completed: "任务完成", failed: "任务失败", waiting_approval: "等待审批", interrupted: "需要人工核对",
};

export function initializeNotifications(options: Options) {
  const toggle = element("notifications-toggle"), panel = element("notifications-panel");
  let enabled = false, generation = 0, ticket = 0, unreadTicket = 0, loading = false;
  let rows: Notification[] = [], timer: ReturnType<typeof setInterval> | undefined;
  const current = () => {
    const identity = options.identity(), epoch = generation;
    return () => !!identity && identity === options.identity() && epoch === generation;
  };
  const status = (message: string) => { element("notifications-status").textContent = message; };

  async function unread(): Promise<void> {
    if (!enabled || !options.identity()) return;
    const context = current(), serial = ++unreadTicket;
    const valid = () => context() && serial === unreadTicket;
    try {
      const result = await options.api("/notifications/unread");
      if (!valid()) return;
      const count = Math.max(0, Number(result.count) || 0);
      element("notifications-count").textContent = count > 99 ? "99+" : String(count);
      element("notifications-count").hidden = count === 0;
      toggle.setAttribute("aria-label", `站内通知，${count}条未读`);
      toggle.title = "查看完成、失败和审批提醒";
    } catch {
      if (valid()) toggle.title = "通知暂不可用，可打开后重试";
    }
  }

  function close(focus = true): void {
    panel.hidden = true; toggle.setAttribute("aria-expanded", "false");
    if (focus) toggle.focus();
  }

  function render(): void {
    const list = element("notifications-list"); list.replaceChildren();
    for (const notification of rows) {
      const item = document.createElement("li"); item.className = "notification-item";
      item.dataset.read = String(notification.read_at !== null);
      const button = document.createElement("button"); button.type = "button";
      const label = document.createElement("span"), title = document.createElement("strong");
      label.textContent = `${labels[notification.type] || notification.type}${notification.read_at === null ? " · 未读" : ""}`;
      title.textContent = notification.title; button.append(label, title);
      button.onclick = async () => {
        const valid = current(); button.disabled = true;
        try {
          await options.api(`/notifications/${notification.id}/read`, {method: "POST"});
          if (!valid()) return;
          notification.read_at ??= Date.now() / 1000;
          render(); await unread();
          if (!valid()) return;
          // 成功读到任务才收起；读取失败时保留可点击重试的通知。
          const opened = await options.openRun(notification.run_id);
          if (valid() && opened === true) close();
        } catch (error) {
          if (valid()) status(`读取失败，请重试：${error instanceof Error ? error.message : "服务不可用"}`);
        } finally { if (valid() && button.isConnected) button.disabled = false; }
      };
      item.append(button); list.append(item);
    }
    if (!rows.length) {
      const empty = document.createElement("li"); empty.className = "muted";
      empty.textContent = "完成、失败或等待审批的任务会出现在这里。"; list.append(empty);
    }
  }

  async function refresh(more = false): Promise<void> {
    if (!enabled || !options.identity() || (more && loading)) return;
    const context = current(), serial = ++ticket, valid = () => context() && serial === ticket;
    loading = true; (element("notifications-more") as HTMLButtonElement).disabled = true;
    status("正在读取通知…");
    try {
      const params = new URLSearchParams({limit: "20"});
      if (more && rows.length) params.set("before", String(rows.at(-1)!.id));
      const page: Notification[] = await options.api(`/notifications?${params}`);
      if (!valid()) return;
      rows = more ? [...new Map([...rows, ...page].map(row => [row.id, row])).values()] : page;
      render(); element("notifications-more").hidden = page.length < 20;
      status(""); await unread();
    } catch (error) {
      if (valid()) status(`读取失败，可刷新重试：${error instanceof Error ? error.message : "服务不可用"}`);
    } finally {
      if (serial === ticket) {
        loading = false; (element("notifications-more") as HTMLButtonElement).disabled = false;
      }
    }
  }

  toggle.onclick = () => {
    if (!panel.hidden) { close(); return; }
    panel.hidden = false; toggle.setAttribute("aria-expanded", "true");
    element("notifications-close").focus(); void refresh();
  };
  element("notifications-close").onclick = () => close();
  element("notifications-refresh").onclick = () => { void refresh(); };
  element("notifications-more").onclick = () => { void refresh(true); };
  panel.onkeydown = event => { if (event.key === "Escape") { event.preventDefault(); close(); } };

  function clear(): void {
    generation++; ticket++; unreadTicket++; enabled = false; loading = false; rows = [];
    clearInterval(timer); timer = undefined; close(false); toggle.hidden = true;
    element("notifications-list").replaceChildren(); element("notifications-count").hidden = true;
    toggle.setAttribute("aria-label", "站内通知"); status("");
  }
  return {
    clear,
    configure: (capabilities: any): void => {
      clear(); enabled = capabilities.notifications?.enabled === true;
      toggle.hidden = !enabled;
      if (enabled) { void unread(); timer = setInterval(() => { void unread(); }, 30000); }
    },
    refresh: unread,
  };
}
