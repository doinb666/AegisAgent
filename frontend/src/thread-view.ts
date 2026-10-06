import { createMessage } from "./reply-view";

interface ThreadItem {
  id: string;
  message: string;
  answer: string | null;
  error: string | null;
  status: string;
}

interface ThreadPage {
  items: ThreadItem[];
  has_more: boolean;
  next_before: string | null;
}

export function createThreadHistory(
  runId: string,
  api: (path: string) => Promise<ThreadPage>,
  valid: () => boolean,
  statuses: Record<string, string>,
): HTMLElement {
  const history = document.createElement("section");
  history.setAttribute("aria-label", "之前的会话问答");
  const more = document.createElement("button");
  more.type = "button";
  more.textContent = "正在加载会话…";
  const status = document.createElement("p");
  status.className = "empty";
  status.setAttribute("role", "status");
  const turns = document.createElement("div");
  history.append(more, status, turns);
  let before: string | null = null;
  let loading = false;
  const seen = new Set<string>();

  const load = async (): Promise<void> => {
    if (loading || !valid()) return;
    loading = true;
    more.disabled = true;
    more.textContent = "正在加载会话…";
    status.textContent = "";
    try {
      const params = new URLSearchParams({ limit: "20" });
      if (before) params.set("before", before);
      const page = await api(`/runs/${encodeURIComponent(runId)}/thread?${params}`);
      if (!valid()) return;
      const viewport = history.parentElement;
      const previousHeight = viewport?.scrollHeight || 0;
      const previousTop = viewport?.scrollTop || 0;
      const firstPage = before === null;
      const fragment = document.createDocumentFragment();
      for (const item of page.items) {
        if (item.id === runId || seen.has(item.id)) continue;
        seen.add(item.id);
        fragment.append(createMessage("user", item.message));
        fragment.append(createMessage(
          "assistant", item.answer || item.error || statuses[item.status] || item.status,
        ));
      }
      turns.prepend(fragment);
      before = page.next_before;
      more.hidden = !page.has_more;
      more.textContent = "加载更早的问答";
      status.textContent = seen.size ? `已加载 ${seen.size} 轮历史问答` : "";
      if (viewport) {
        viewport.scrollTop = firstPage
          ? viewport.scrollHeight
          : previousTop + viewport.scrollHeight - previousHeight;
      }
    } catch (error) {
      if (!valid()) return;
      more.hidden = false;
      more.textContent = "重试加载会话";
      status.textContent = `历史加载失败：${error instanceof Error ? error.message : "请稍后重试"}`;
    } finally {
      loading = false;
      if (valid()) more.disabled = false;
    }
  };
  more.onclick = () => { void load(); };
  void load();
  return history;
}
