interface AssetSummary {
  id: string;
  kind: string;
  name: string;
  content: string;
  version: number;
  status: string;
  metadata: { directory?: string; description?: string; tags?: string[]; source_run_id?: string };
}

// 筛选仅作用于服务端已返回的本人内容，不额外发请求或保存用户数据。
export function initializeAssetBrowser(changed: () => void) {
  const query = document.getElementById("asset-search") as HTMLInputElement;
  const status = document.getElementById("asset-status-filter") as HTMLSelectElement;
  const more = document.getElementById("assets-more") as HTMLButtonElement;
  const clear = document.getElementById("asset-filters-clear") as HTMLButtonElement;
  const count = document.getElementById("asset-count")!;
  let limit = 24;

  function update(): void { limit = 24; changed(); }
  query.oninput = update;
  status.onchange = update;
  more.onclick = () => { limit += 24; changed(); };
  clear.onclick = () => { reset(); changed(); query.focus(); };

  function reset(): void {
    query.value = ""; status.value = ""; limit = 24;
    count.textContent = ""; more.hidden = true; clear.hidden = true;
  }

  function select<T extends AssetSummary>(assets: T[]): T[] {
    const term = query.value.trim().toLocaleLowerCase();
    const matches = assets.filter(asset => {
      if (status.value && asset.status !== status.value) return false;
      const metadata = asset.metadata || {};
      const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
      const searchable = [asset.name, metadata.description, metadata.directory,
        ...tags].join(" ").toLocaleLowerCase();
      return !term || searchable.includes(term);
    });
    const visible = matches.slice(0, limit);
    count.textContent = `显示 ${visible.length} / ${matches.length} 条 · 已读取 ${assets.length} 条`;
    more.hidden = visible.length >= matches.length;
    clear.hidden = !term && !status.value;
    return visible;
  }
  function reveal(name: string): void {
    reset(); query.value = name;
  }
  return {reset, select, reveal};
}
