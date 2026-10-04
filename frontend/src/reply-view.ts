// 所有模型内容均以文本节点渲染，仅支持有界的段落、标题、行内代码和代码围栏。
function appendInline(parent: HTMLElement, text: string): void {
  const pieces = text.split(/(`[^`\n]+`)/g);
  for (const piece of pieces) {
    if (piece.startsWith("`") && piece.endsWith("`") && piece.length > 2) {
      const code = document.createElement("code");
      code.textContent = piece.slice(1, -1);
      parent.append(code);
    } else {
      parent.append(document.createTextNode(piece));
    }
  }
}

function codeBlock(language: string, content: string): HTMLElement {
  const block = document.createElement("section");
  block.className = "code-block";
  const header = document.createElement("div");
  header.className = "code-heading";
  const label = document.createElement("span");
  label.textContent = language || "代码";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "复制代码";
  copy.setAttribute("aria-label", "复制代码");
  const pre = document.createElement("pre");
  pre.tabIndex = 0;
  const code = document.createElement("code");
  code.textContent = content;
  pre.append(code);
  const status = document.createElement("span");
  status.className = "copy-status";
  status.setAttribute("role", "status");
  copy.onclick = async () => {
    copy.disabled = true;
    try {
      await navigator.clipboard.writeText(content);
      copy.textContent = "已复制";
      status.textContent = "代码已复制到剪贴板。";
    } catch {
      copy.textContent = "复制代码";
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
      pre.focus();
      status.textContent = "剪贴板不可用，已选中代码，请手动复制。";
    } finally {
      copy.disabled = false;
    }
  };
  header.append(label, copy);
  block.append(header, pre, status);
  return block;
}

export function renderReply(text: string): HTMLElement {
  const body = document.createElement("div");
  body.className = "reply-body";
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let paragraph: string[] = [];
  const flush = () => {
    if (!paragraph.length) return;
    const element = document.createElement("p");
    appendInline(element, paragraph.join("\n"));
    body.append(element);
    paragraph = [];
  };
  for (let index = 0; index < lines.length; index++) {
    const fence = /^ {0,3}(`{3,}|~{3,})([^\s`]*)\s*$/.exec(lines[index]);
    if (fence) {
      flush();
      const content: string[] = [];
      const close = new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
      while (++index < lines.length && !close.test(lines[index])) content.push(lines[index]);
      body.append(codeBlock(fence[2].slice(0, 40), content.join("\n")));
    } else if (!lines[index].trim()) {
      flush();
    } else {
      const heading = /^#{1,4}\s+(.+)$/.exec(lines[index]);
      if (heading) {
        flush();
        const element = document.createElement("h3");
        appendInline(element, heading[1]);
        body.append(element);
      } else {
        paragraph.push(lines[index]);
      }
    }
  }
  flush();
  return body;
}
