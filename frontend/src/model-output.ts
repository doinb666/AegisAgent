// 公开模型文本是临时预览；最终答复以Run.answer为准，不解释片段中的HTML。
export function initializeModelOutput(container: HTMLElement) {
  let callId = "", sequence = 0, characters = 0, withdrawn = false;
  let article: HTMLElement | null = null, label: HTMLElement | null = null, body: HTMLElement | null = null;
  let content: Text | null = null;

  function removePreview(): void {
    article?.remove(); article = null; label = null; body = null;
    content = null; characters = 0;
  }
  function clear(): void {
    removePreview(); callId = ""; sequence = 0; withdrawn = false;
  }
  function show(): void {
    if (article) return;
    article = document.createElement("article");
    article.id = "model-output"; article.className = "message assistant model-output";
    label = document.createElement("span"); label.id = "model-output-status";
    label.className = "message-label"; label.setAttribute("role", "status");
    label.textContent = "模型输出中，尚未验收";
    body = document.createElement("div"); body.id = "model-output-body";
    body.className = "model-output-body";
    content = document.createTextNode(""); body.append(content);
    article.append(label, body); container.append(article);
  }
  function accept(type: string, data: Record<string, unknown>, active: boolean): void {
    if (!active || !data || typeof data.call_id !== "string" || !data.call_id || data.call_id.length > 128
      || !Number.isSafeInteger(data.seq) || Number(data.seq) < 0) return;
    if (type === "model_output_started") {
      if (data.seq !== 0 || data.call_id === callId) return;
      clear(); callId = data.call_id; return;
    }
    if (data.call_id !== callId || Number(data.seq) <= sequence) return;
    sequence = Number(data.seq);
    if (type === "model_output_retracted") {
      removePreview(); withdrawn = true;
      return;
    }
    if (type === "model_output_finished") {
      if (data.reason === "tools") removePreview();
      if (label && !withdrawn) label.textContent = data.preview_truncated
        ? "预览已截断，等待完整任务结果" : "输出已收齐，正在完成任务核对";
      withdrawn = true;
      return;
    }
    if (type !== "model_output_delta" || withdrawn) return;
    // Python偏移按Unicode码点计数，中文与非BMP字符必须使用相同口径。
    const length = typeof data.text === "string" && data.text.length <= 8192 ? [...data.text].length : -1;
    if (length < 0 || length > 4096 || data.offset !== characters || characters + length > 131072) {
      withdrawn = true; removePreview(); return;
    }
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
    characters += length; show(); content!.appendData(data.text as string);
    if (nearBottom) container.scrollTop = container.scrollHeight;
  }
  return { clear, accept, settle(status: string): void { if (!["queued", "running"].includes(status)) clear(); } };
}
