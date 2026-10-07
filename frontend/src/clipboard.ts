// 剪贴板失败时选中安全文本，保留键盘手动复制路径。
export async function copyText(
  text: string, target: HTMLElement, status: HTMLElement, label: string,
  current: () => boolean = () => true,
): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    if (!current()) return false;
    status.textContent = `${label}已复制到剪贴板。`;
    return true;
  } catch {
    if (!current()) return false;
    const range = document.createRange(); range.selectNodeContents(target);
    const selection = window.getSelection();
    selection?.removeAllRanges(); selection?.addRange(range);
    target.focus();
    status.textContent = `剪贴板不可用，已选中${label}，请手动复制。`;
    return false;
  }
}
