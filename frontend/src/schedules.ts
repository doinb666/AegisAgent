interface Occurrence { id: string; status: string; run_id: string | null; scheduled_at: string; reason: string | null }
interface Schedule {
  id: string; title: string; message: string; status: string; version: number;
  next_at: string | null; interval_seconds: number | null; occurrences?: Occurrence[];
}
interface Options {
  api: (path: string, options?: object) => Promise<any>;
  identity: () => string | null;
  canWrite: () => boolean;
  openRun: (id: string) => Promise<unknown>;
}
const element = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const labels: Record<string, string> = {active: "已启用", paused: "已暂停", cancelled: "已取消", completed: "已结束触发"};
const occurrenceLabels: Record<string, string> = {submitted: "已创建任务", pending: "等待触发", leased: "正在触发", retry: "等待重试", skipped: "已跳过", failed: "触发失败"};
const reasons: Record<string, string> = {
  user_run_quota: "运行数量达到配额，稍后重试", identity_permission_revoked: "执行权限已撤销",
  database_temporarily_unavailable: "数据库暂不可用",
  schedule_inactive: "计划已暂停或取消", previous_run_active: "上一轮任务仍在执行",
  superseded_by_latest: "错过的旧触发已由最近一次替代", occurrence_lease_lost: "触发租约已过期，等待恢复",
  schedule_run_idempotency_collision: "触发标识发生冲突，请联系管理员核对",
};
const date = (value: string | null): string => value ? new Date(value).toLocaleString() : "无后续触发";
function node(tag: string, text: string, className = ""): HTMLElement {
  const result = document.createElement(tag); result.textContent = text; result.className = className; return result;
}
function errorText(error: unknown): string { return error instanceof Error ? error.message : "服务暂不可用，请重试。"; }

export function initializeSchedules(options: Options) {
  let enabled = false, visible = false, epoch = 0, listTicket = 0, detailTicket = 0;
  let rows: Schedule[] = [], selected: string | null = null, more = false, loading = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: {body: string; key: string} | null = null;
  const form = element<HTMLFormElement>("schedule-form");
  const status = (text: string) => { element("schedules-status").textContent = text; };
  const current = () => {
    const identity = options.identity(), generation = epoch;
    return () => !!identity && identity === options.identity() && generation === epoch;
  };
  function button(text: string, action: () => void, danger = false): HTMLButtonElement {
    const result = document.createElement("button"); result.type = "button"; result.textContent = text;
    if (danger) result.className = "danger"; result.onclick = action; return result;
  }
  function render(): void {
    const list = element("schedule-list"); list.replaceChildren();
    if (!rows.length) list.append(node("p", "还没有定时计划。可以安排资料检索或计算核对。", "empty"));
    for (const row of rows) {
      const item = node("article", "", "schedule-row"), heading = node("div", "", "schedule-heading");
      heading.append(node("h4", row.title), node("span", labels[row.status] || row.status, "schedule-state"));
      item.append(heading, node("p", row.message, "schedule-message"),
        node("p", `${row.interval_seconds ? `每 ${row.interval_seconds} 秒` : "一次性"} · ${date(row.next_at)}`, "muted"));
      const actions = node("div", "", "schedule-actions");
      actions.append(button("查看触发记录", () => { void detail(row.id); }));
      if (options.canWrite() && ["active", "paused"].includes(row.status)) {
        const action = row.status === "active" ? "pause" : "resume";
        actions.append(button(action === "pause" ? "暂停" : "恢复", () => { void change(row, action, item); }),
          button("取消计划", () => { void change(row, "cancel", item); }, true));
      }
      item.append(actions); list.append(item);
    }
    element("schedules-more").hidden = !more;
  }
  async function refresh(append = false): Promise<void> {
    if (!enabled || !visible || (append && (loading || !more))) return;
    const context = current(), ticket = ++listTicket, valid = () => context() && ticket === listTicket;
    loading = true; element<HTMLButtonElement>("schedules-more").disabled = true;
    status("正在读取计划…");
    try {
      const query = new URLSearchParams({limit: "20"});
      if (append && rows.length) query.set("before", rows.at(-1)!.id);
      const page: Schedule[] = await options.api(`/schedules?${query}`);
      if (!valid()) return;
      const known = new Map(rows.map(row => [row.id, row]));
      const latest = page.map(row => {
        const previous = known.get(row.id);
        return previous && previous.version > row.version ? previous : row;
      });
      rows = append ? [...new Map([...rows, ...latest].map(row => [row.id, row])).values()] : latest;
      more = page.length === 20; render(); status("");
    } catch (error) { if (valid()) status(`读取失败：${errorText(error)}`); }
    finally { if (valid()) { loading = false; element<HTMLButtonElement>("schedules-more").disabled = false; } }
  }
  async function change(row: Schedule, action: string, item: HTMLElement): Promise<void> {
    const valid = current(); item.querySelectorAll<HTMLButtonElement>("button").forEach(control => { control.disabled = true; });
    status("正在更新计划…");
    try {
      await options.api(`/schedules/${row.id}`, {method: "PATCH", body: JSON.stringify({action, expected_version: row.version})});
      if (!valid()) return;
      await refresh(); if (selected === row.id) await detail(row.id);
    } catch (error) { if (valid()) status(`更新失败：${errorText(error)}。可刷新后重试。`); }
    finally { if (valid()) item.querySelectorAll<HTMLButtonElement>("button").forEach(control => { control.disabled = false; }); }
  }
  function queueDetail(): void {
    clearTimeout(timer);
    if (visible && selected && !document.hidden) timer = setTimeout(() => { if (selected) void detail(selected, true); }, 3000);
  }
  async function detail(id: string, background = false): Promise<void> {
    if (!enabled || !visible) return;
    selected = id;
    const context = current(), ticket = ++detailTicket;
    const valid = () => context() && visible && selected === id && ticket === detailTicket;
    const panel = element("schedule-detail"); panel.hidden = false;
    if (!background) panel.replaceChildren(node("p", "正在读取触发记录…", "muted"));
    try {
      const row: Schedule = await options.api(`/schedules/${id}`);
      if (!valid()) return;
      const cached = rows.findIndex(item => item.id === row.id);
      if (cached >= 0 && rows[cached].version <= row.version) { rows[cached] = row; render(); }
      panel.replaceChildren(node("h3", `${row.title} · 触发记录`),
        node("p", "这里只记录触发情况；执行结果请打开对应任务查看。取消计划不会停止已创建的任务。", "muted"));
      const controls = node("div", "", "schedule-actions");
      controls.append(button("刷新记录", () => { void detail(id); }), button("收起记录", () => {
        selected = null; detailTicket++; clearTimeout(timer); panel.hidden = true;
      })); panel.append(controls);
      if (!row.occurrences?.length) panel.append(node("p", "暂无触发记录", "empty"));
      for (const occurrence of row.occurrences || []) {
        const entry = node("div", "", "schedule-occurrence");
        entry.append(node("p", `${date(occurrence.scheduled_at)} · ${occurrenceLabels[occurrence.status] || occurrence.status}`));
        const reason = occurrence.reason;
        if (reason) entry.append(node("p", reasons[reason] || reason, "muted"));
        if (occurrence.run_id) entry.append(button("打开执行任务", () => { void openOccurrence(occurrence.run_id!); }));
        panel.append(entry);
      }
    } catch (error) { if (valid()) status(`记录读取失败：${errorText(error)}`); }
    finally { if (valid()) queueDetail(); }
  }
  async function openOccurrence(id: string): Promise<void> {
    const valid = current();
    try { await options.openRun(id); } catch (error) { if (valid()) status(errorText(error)); }
  }
  form.onsubmit = async event => {
    event.preventDefault(); if (!enabled || !options.canWrite()) return;
    const submit = form.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    if (submit.disabled) return;
    const error = element("schedule-error"); error.textContent = "";
    const title = element<HTMLInputElement>("schedule-title").value.trim();
    const message = element<HTMLTextAreaElement>("schedule-message").value.trim();
    const timestamp = new Date(element<HTMLInputElement>("schedule-time").value);
    const interval = element<HTMLSelectElement>("schedule-frequency").value === "interval" ? Number(element<HTMLInputElement>("schedule-interval").value) : null;
    if (!title || !message) { error.textContent = "标题与任务内容不能只包含空白。"; error.focus(); return; }
    if (!Number.isFinite(timestamp.getTime())) { error.textContent = "请选择有效的本地时间。"; error.focus(); return; }
    if (interval !== null && (!Number.isInteger(interval) || interval < 60 || interval > 2678400)) { error.textContent = "间隔须为 60 到 2678400 秒的整数。"; error.focus(); return; }
    const body = JSON.stringify({title, message, scheduled_at: timestamp.toISOString(), interval_seconds: interval});
    // 已提交但响应丢失的同一请求允许原键恢复；不能因为触发时间已过而改键重建。
    if (timestamp.getTime() <= Date.now() && pending?.body !== body) { error.textContent = "请选择未来的本地时间。"; error.focus(); return; }
    if (!pending || pending.body !== body) pending = {body, key: crypto.randomUUID()};
    const request = pending, valid = current(); submit.disabled = true;
    try {
      await options.api("/schedules", {method: "POST", headers: {"Idempotency-Key": request.key}, body});
      if (!valid()) return;
      pending = null; form.reset(); form.hidden = true; updateFrequency(); await refresh();
    } catch (failure) { if (valid()) { error.textContent = errorText(failure); error.focus(); } }
    finally { if (valid()) submit.disabled = false; }
  };
  function updateFrequency(): void {
    const interval = element<HTMLInputElement>("schedule-interval");
    interval.disabled = element<HTMLSelectElement>("schedule-frequency").value !== "interval";
    element("schedule-interval-field").hidden = interval.disabled;
  }
  element("schedule-frequency").onchange = updateFrequency;
  element("schedule-add").onclick = () => { form.hidden = !form.hidden; if (!form.hidden) element("schedule-title").focus(); };
  element("schedule-form-close").onclick = () => { form.hidden = true; element("schedule-add").focus(); };
  element("schedules-refresh").onclick = () => { void refresh(); };
  element("schedules-more").onclick = () => { void refresh(true); };
  document.addEventListener("visibilitychange", queueDetail);
  function clear(): void {
    epoch++; listTicket++; detailTicket++; clearTimeout(timer);
    enabled = false; visible = false; selected = null; rows = []; more = false; loading = false; pending = null;
    form.reset(); form.hidden = true; updateFrequency();
    form.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled = false;
    element("schedule-error").textContent = ""; status("");
    element("schedule-list").replaceChildren(); element("schedule-detail").replaceChildren();
    element("schedule-detail").hidden = true; element("schedules-more").hidden = true;
  }
  return {
    clear,
    configure(capabilities: any): void {
      clear(); enabled = capabilities.schedules?.enabled === true;
      element("schedule-unavailable").hidden = enabled;
      element("schedule-add").hidden = !enabled || !options.canWrite();
      element("schedules-refresh").hidden = !enabled;
    },
    show(value: boolean): void {
      visible = value; clearTimeout(timer); detailTicket++;
      if (value) { void refresh(); if (selected) void detail(selected); }
    },
  };
}
