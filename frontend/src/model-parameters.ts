interface Capabilities {
  temperature?: boolean;
  output_tokens?: boolean;
  output_token_parameter?: string | null;
  max_output_tokens?: number;
  reasoning_efforts?: string[];
}
interface Route { id: string; model: string; parameters?: Capabilities }
interface Parameters { temperature?: number; max_output_tokens?: number; reasoning_effort?: string }

const element = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const effortNames: Record<string, string> = {
  none: "不额外推理", minimal: "精简", low: "轻量", medium: "标准", high: "深入", xhigh: "更深入",
};

export function initializeModelParameters(model: () => string) {
  const panel = element<HTMLDetailsElement>("model-parameters");
  const temperature = element<HTMLInputElement>("model-temperature");
  const output = element<HTMLInputElement>("model-output-tokens");
  const effort = element<HTMLSelectElement>("model-reasoning");
  const status = element("model-parameters-status");
  let automatic: Capabilities | null = null, routes: Route[] = [], active: Capabilities = {};

  function summary(): void {
    const count = [temperature.value, output.value, effort.value].filter(Boolean).length;
    element("model-parameters-summary").textContent = count ? `已调整 ${count} 项` : "沿用模型配置";
  }
  function render(): void {
    const selected = model();
    active = selected ? routes.find(route => route.id === selected || route.model === selected)?.parameters || {} : automatic || {};
    panel.hidden = automatic === null && !routes.some(route => route.parameters);
    const previousEffort = effort.value;
    const outputEnabled = active.output_tokens === true || !!active.output_token_parameter;
    let removed = false;
    if (!active.temperature && temperature.value) { temperature.value = ""; removed = true; }
    if (!outputEnabled && output.value) { output.value = ""; removed = true; }
    temperature.disabled = active.temperature !== true;
    output.disabled = !outputEnabled;
    output.max = String(active.max_output_tokens || 4096);
    effort.replaceChildren(new Option("沿用模型配置", ""));
    for (const value of active.reasoning_efforts || []) effort.add(new Option(effortNames[value] || value, value));
    effort.disabled = !active.reasoning_efforts?.length;
    if (active.reasoning_efforts?.includes(previousEffort)) effort.value = previousEffort;
    else if (previousEffort) removed = true;
    const available = active.temperature || outputEnabled || !effort.disabled;
    element("model-parameters-hint").textContent = available
      ? `${selected ? "按当前来源声明的能力调整" : "自动选择仅提供所有候选共同支持的参数"}。${outputEnabled ? `输出上限 ${output.max} Token；` : ""}留空沿用服务端配置，Token上限不等于费用硬配额。`
      : "当前来源未声明可调整参数，沿用服务端配置。管理员需先核对接口能力再开放。";
    status.textContent = removed ? "已清除当前模型不支持的参数，请核对其余设置。" : "";
    summary();
  }
  function reset(): void {
    temperature.value = ""; output.value = ""; effort.value = "";
    panel.open = false; status.textContent = ""; summary();
  }
  for (const input of [temperature, output, effort]) input.addEventListener("input", summary);
  element("model-parameters-reset").onclick = reset;
  element("model").addEventListener("change", render);

  return {
    reset,
    clear(): void { automatic = null; routes = []; reset(); render(); },
    configure(capabilities: {model_parameters?: Capabilities; model_routes?: Route[]}): void {
      automatic = capabilities.model_parameters || null; routes = capabilities.model_routes || [];
      reset(); render();
    },
    values(): Parameters {
      const result: Parameters = {};
      if (!temperature.disabled && temperature.value) {
        const value = Number(temperature.value);
        if (!Number.isFinite(value) || value < 0 || value > 2) throw new Error("采样温度须为0至2之间的数值。");
        result.temperature = value;
      }
      if (!output.disabled && output.value) {
        const value = Number(output.value);
        if (!Number.isInteger(value) || value < 1 || value > Number(output.max)) throw new Error(`输出上限须为1至${output.max}之间的整数。`);
        result.max_output_tokens = value;
      }
      if (!effort.disabled && effort.value) {
        if (!active.reasoning_efforts?.includes(effort.value)) throw new Error("当前来源不支持所选推理强度，请重新选择。");
        result.reasoning_effort = effort.value;
      }
      return result;
    },
  };
}
