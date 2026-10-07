type RecordValue = Record<string, unknown>;
interface RevisionSource {
  originalId: string;
  version: number;
  sourceRunId: string;
}
type Api = (path: string, options?: unknown) => Promise<unknown>;

const maxVersion = 2147483647;
const maxPreview = 8000;

function record(value: unknown): value is RecordValue {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function isAssetReference(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9_-]{1,36}$/.test(value);
}

function version(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 1 && Number(value) <= maxVersion;
}

// 校验服务端公开来源契约；前端展示校验不代表鉴权或修复验收。
export function revisionSource(asset: unknown): RevisionSource | null {
  if (!record(asset) || asset.kind !== "skill" || asset.status !== "draft"
    || !isAssetReference(asset.id) || !version(asset.version) || typeof asset.content !== "string"
    || !record(asset.metadata)) return null;
  const metadata = asset.metadata;
  const evidence = metadata.revision_recall_evidence;
  const boundary = metadata.revision_permission_boundary;
  if (!isAssetReference(metadata.revision_original_id)
    || metadata.revision_original_id === asset.id
    || !version(metadata.revision_original_version)
    || !isAssetReference(metadata.source_run_id)
    || !isAssetReference(metadata.revision_original_source_run_id)
    || metadata.extracted !== true || metadata.repair_verified !== false
    || metadata.manual_verified !== false || metadata.user_verified !== false
    || ![metadata.revision_feedback_hash, metadata.revision_fingerprint].every(
      value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value),
    )
    || !record(evidence) || !record(boundary)) return null;
  if (evidence.asset_id !== metadata.revision_original_id
    || evidence.version !== metadata.revision_original_version
    || evidence.source_run_id !== metadata.revision_original_source_run_id
    || !Number.isSafeInteger(evidence.event_id) || Number(evidence.event_id) < 1) return null;
  const tools = boundary.allowed_tools;
  if (tools !== null && (!Array.isArray(tools) || tools.length > 128
    || !tools.every(tool => typeof tool === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(tool)))) return null;
  return {
    originalId: metadata.revision_original_id,
    version: metadata.revision_original_version,
    sourceRunId: metadata.revision_original_source_run_id,
  };
}

export function boundedSkillText(value: unknown, limit = maxPreview): string {
  if (typeof value !== "string") return "正文格式无效，无法展示。";
  const text = value.slice(0, limit);
  return value.length > limit ? `${text}\n\n内容已截断，仅展示前 ${limit} 个字符；请通过原有正文入口核对。` : text;
}

function textNode<K extends keyof HTMLElementTagNameMap>(tag: K, text: string, className = ""): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  return element;
}

export function appendSkillRevisionReview(
  row: HTMLElement, asset: unknown, api: Api, current: () => boolean,
): void {
  const source = revisionSource(asset);
  if (!source || !record(asset)) return;
  row.append(textNode("p", `修订草稿 · 基于原技能v${source.version}，未验证建议`, "revision-origin"));
  const details = document.createElement("details");
  details.className = "skill-revision-review";
  const summary = textNode("summary", "对照原技能");
  const status = textNode("p", "展开后核对原技能版本与来源。", "revision-status muted");
  status.setAttribute("role", "status");
  const comparison = document.createElement("div");
  comparison.className = "revision-comparison";
  const suggestion = document.createElement("section");
  suggestion.append(textNode("h4", "未验证建议"), textNode("pre", boundedSkillText(asset.content), "revision-suggestion"));
  details.append(summary, status, comparison);
  row.append(details);
  let requestState: "idle" | "loading" | "settled" = "idle";
  const valid = () => current() && details.isConnected;

  async function load(): Promise<void> {
    if (requestState !== "idle" || !valid()) return;
    requestState = "loading";
    status.textContent = "正在核对原技能…";
    comparison.replaceChildren(suggestion);
    try {
      const original = await api(`/assets/${encodeURIComponent(source.originalId)}`);
      if (!valid()) return;
      requestState = "settled";
      if (!record(original) || original.id !== source.originalId || original.kind !== "skill"
        || original.status !== "active" || original.version !== source.version
        || !record(original.metadata) || original.metadata.source_run_id !== source.sourceRunId
        || typeof original.content !== "string") {
        status.textContent = "原技能已变化，当前版本、状态或来源不再匹配。无法展示生成时原文，请核对版本记录。";
        return;
      }
      status.textContent = `原技能v${source.version}与来源匹配；召回只证明进入上下文，不证明使用或失败因果。启用前仍需验证。`;
      const originalSection = document.createElement("section");
      originalSection.append(textNode("h4", `原技能v${source.version}（仍活动）`),
        textNode("pre", boundedSkillText(original.content), "revision-original"));
      comparison.prepend(originalSection);
    } catch (failure) {
      if (!valid()) return;
      requestState = "settled";
      const code = record(failure) ? failure.status : undefined;
      if (code === 403 || code === 404) {
        status.textContent = "无法读取原技能，内容不存在或当前账号已无权限。";
        return;
      }
      status.textContent = "原技能读取失败，请确认网络与服务后手动重试；建议正文已保留。";
      const retry = textNode("button", "重试读取原技能");
      retry.type = "button";
      retry.onclick = () => { retry.remove(); requestState = "idle"; void load(); };
      comparison.append(retry);
    }
  }
  // 更新内容区，保留同一个原生 summary、展开状态及键盘焦点。
  details.ontoggle = () => { if (details.open) void load(); };
}

export function publicSkillRevisionEvent(value: unknown): RecordValue {
  const data = record(value) ? value : {};
  const states = ["queued", "started", "done", "skipped"];
  const state = typeof data.state === "string" && states.includes(data.state) ? data.state : "unknown";
  const drafts = Array.isArray(data.draft_assets) && data.draft_assets.length <= 2
    && data.draft_assets.every(isAssetReference) ? data.draft_assets.length : 0;
  return {state, draft_count: drafts, automatic_activation: false, repair_verified: false};
}

export function skillRevisionHint(value: unknown): string {
  const data = publicSkillRevisionEvent(value);
  if (data.state === "queued") return "修订建议已排队；原技能继续保持，草稿不会自动启用。";
  if (data.state === "started") return "正在生成未验证修订建议，不重放原任务工具。";
  if (data.state === "done" && Number(data.draft_count) > 0) return `已生成 ${data.draft_count} 个未验证修订草稿，请在技能库核对来源与正文。`;
  if (data.state === "skipped") return "本次修订已跳过，没有可保存的可信修订建议。";
  return "修订状态待核对，未确认修复或启用。";
}
