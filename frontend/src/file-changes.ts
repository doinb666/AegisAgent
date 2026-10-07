// 差异只作为审批预览；基线由服务端冻结，模型和页面均不能改写它。
type FileApproval = {
  name?: string;
  arguments?: {path?: string};
  file_write?: {
    version: number;
    path: string;
    baseline_hash: string;
    baseline: {exists: boolean; bytes: number; sha256?: string | null};
    preview: {before: string; after: string; diff: string; truncated: boolean};
  };
};

export function initializeFileChanges() {
  const section = document.getElementById("file-change")!;
  const title = document.getElementById("file-change-title")!;
  const summary = document.getElementById("file-change-summary")!;
  const difference = document.getElementById("file-change-diff")!;
  const before = document.getElementById("file-change-before")!;
  const after = document.getElementById("file-change-after")!;
  const warning = document.getElementById("file-change-warning")!;

  function clear(): void {
    section.hidden = true;
    for (const element of [title, summary, difference, before, after, warning]) element.textContent = "";
    section.querySelectorAll("details").forEach(element => element.open = false);
  }

  function boundedText(text: unknown, limit: number): boolean {
    return typeof text === "string" && text.length <= limit * 2 && [...text].length <= limit;
  }

  function valid(approval: FileApproval): boolean {
    const change = approval.file_write;
    if (!change || change.version !== 1 || change.path !== approval.arguments?.path
      || typeof change.path !== "string" || change.path.length > 512
      || typeof change.baseline_hash !== "string" || !/^[a-f0-9]{64}$/.test(change.baseline_hash)) return false;
    const baseline = change.baseline, preview = change.preview;
    return !!baseline && typeof baseline.exists === "boolean"
      && Number.isSafeInteger(baseline.bytes) && baseline.bytes >= 0 && baseline.bytes <= 1000000
      && (!baseline.exists || /^[a-f0-9]{64}$/.test(baseline.sha256 || ""))
      && !!preview && typeof preview.truncated === "boolean"
      && [preview.before, preview.after].every(text => boundedText(text, 8192))
      && boundedText(preview.diff, 16384);
  }

  function baselineHash(approval: FileApproval): string | null {
    if (approval.name !== "file_write") return null;
    if (!valid(approval)) throw new Error("文件审批缺少有效版本与差异，请重新提出修改任务。");
    return approval.file_write!.baseline_hash;
  }

  function canApprove(approval: FileApproval | null): boolean {
    return !!approval && (approval.name !== "file_write" || valid(approval));
  }

  function render(approval: FileApproval | null): boolean {
    clear();
    if (!approval || approval.name !== "file_write") return true;
    section.hidden = false;
    if (!valid(approval)) {
      title.textContent = "文件版本信息不可用";
      warning.textContent = "无法安全批准这次写入。请拒绝旧操作，再重新提出修改任务。";
      return false;
    }
    const change = approval.file_write!, preview = change.preview;
    title.textContent = `${change.baseline.exists ? "修改" : "新建"} · ${change.path}`;
    summary.textContent = change.baseline.exists
      ? `原文件 ${change.baseline.bytes} B，版本 ${change.baseline.sha256!.slice(0, 12)}。保存前会再次核对版本。`
      : "文件尚不存在。保存时若已有同名文件，将拒绝覆盖。";
    const lines = preview.diff.split("\n"), fragment = document.createDocumentFragment();
    for (const text of lines.slice(0, 400)) {
      const line = document.createElement("div");
      line.textContent = text || " ";
      if (text.startsWith("+") && !text.startsWith("+++")) line.className = "diff-added";
      if (text.startsWith("-") && !text.startsWith("---")) line.className = "diff-removed";
      fragment.append(line);
    }
    difference.replaceChildren(fragment);
    if (!preview.diff) difference.textContent = "预览范围内没有文本差异，请核对完整参数。";
    before.textContent = preview.before || "（原文件为空或尚不存在）";
    after.textContent = preview.after || "（拟写内容为空）";
    warning.textContent = preview.truncated || lines.length > 400
      ? "差异已截断，前后内容也有长度限制。请在完整参数中核对全部拟写内容，不确定时拒绝。"
      : "+ 表示新增，− 表示删除。批准仅对当前文件版本和本次拟写内容有效。";
    return true;
  }
  return {clear, render, baselineHash, canApprove};
}
