"use strict";

import WorkspaceUI from "./workspace-ui";
import { initializeNavigation } from "./navigation";
import { createMessage } from "./reply-view";
import { createThreadHistory } from "./thread-view";

const $ = (id: string): any => document.getElementById(id);
const apiRoot = "/api/v1";
const welcomeTemplate = $("welcome").cloneNode(true);
const state: any = {token: sessionStorage.getItem("aegis-token"), user: null, run: null, session: null,
  view: "chat", cursor: 0, stream: null, assets: [], editingAsset: null, generation: 0, pendingRequest: null,
  viewGeneration: 0, assetsGeneration: 0, runsGeneration: 0, overviewGeneration: 0, runs: [], modelRoutes: [], runsLoading: false, runLoading: false, hasMore: false};
const statuses = {queued:"等待执行",running:"正在推进",waiting_approval:"等待你的审批",completed:"任务完成",failed:"执行失败",cancelled:"已停止",interrupted:"需要人工核对后恢复"};
const kinds = {profile:"偏好",preference:"偏好",constraint:"约束",memory:"记忆",episodic:"经验",skill:"Skill",procedure:"步骤",project:"项目",document:"文档"};
const assetStates = {draft:"候选",active:"已启用",retired:"已退役"};
function notice(text) { $("notice").textContent = text; }
const canWrite = () => state.user && state.user.role !== "viewer";
function contextCurrent() {
  const token = state.token, generation = state.generation, viewGeneration = state.viewGeneration;
  return () => token === state.token && generation === state.generation && viewGeneration === state.viewGeneration;
}
function runCurrent(id, generation, token) {
  return token === state.token && generation === state.generation && state.view === "chat" && state.run?.id === id;
}
function setRunLoading(loading) {
  state.runLoading=loading;
  $("send").disabled=loading;
  $("chat-view").setAttribute("aria-busy",String(loading));
  $("send").title=loading ? "正在读取任务，完成后可继续" : "";
}
async function api(path: string, options: any = {}) {
  const requestToken=state.token;
  const headers = new Headers(options.headers || {});
  if (requestToken) headers.set("Authorization", `Bearer ${requestToken}`);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(apiRoot + path, {...options, headers});
  let data;
  try { data = await response.json(); }
  catch { if(response.ok) throw new Error("服务返回了无法读取的数据，请刷新后重试。"); }
  if(requestToken!==state.token) throw new Error("账号已切换，请在当前空间重新操作。");
  if (!response.ok) {
    if (response.status === 401 && state.user) resetLogin();
    throw new Error(WorkspaceUI.requestError(data, response.status));
  }
  return data;
}
async function guard(action, button=null) {
  const token=state.token;
  const valid=contextCurrent();
  if (button) button.disabled = true;
  try { await action(); } catch(error) { if(valid()) notice(error.message); }
  finally { if (button && token===state.token) button.disabled = button.id==="send" && state.runLoading; }
}
function resetLogin() {
  navigation.close();
  clearTimeout(searchTimer);
  state.generation++; state.stream?.abort(); state.user=null; state.token=null;
  state.pendingRequest=null; state.run=null; state.session=null;
  state.editingAsset=null;$("skill-import-form").reset();closeSkillImport();
  $("skill-import-form").querySelector('button[type="submit"]').disabled=false;
  state.assets=[];$("asset-list").replaceChildren();
  $("run-list").replaceChildren();$("asset-form").reset();$("upload-form").reset();$("member-form").reset();
  newTask(); state.viewGeneration++; state.assetsGeneration++; state.runsGeneration++; state.overviewGeneration++;
  state.runs=[];state.runsLoading=false;state.hasMore=false;
  $("run-search").value="";$("run-filter").value="";$("runs-more").hidden=true;
  $("workspace-overview").textContent="";$("workspace-overview").removeAttribute("title");
  $("capability-list").replaceChildren();$("capabilities").textContent="";$("memory-preview").textContent="";$("approval-data").textContent="";$("notice").textContent="";
  $("model").replaceChildren(new Option("自动选择模型", ""));
  state.modelRoutes=[];
  WorkspaceUI.configureCollaboration({},false);
  $("provider-clear").click();
  $("provider-form").reset();$("provider-kind").dispatchEvent(new Event("change"));
  $("member-form").closest("details").hidden=true;
  $("account").textContent="";
  sessionStorage.removeItem("aegis-pending-request"); sessionStorage.removeItem("aegis-last-run");
  sessionStorage.removeItem("aegis-token"); $("shell").hidden=true; $("auth").hidden=false;
}
async function enter(user) {
  state.user=user; $("auth").hidden=true; $("shell").hidden=false;
  const token=state.token;
  const valid=contextCurrent();
  $("account").textContent=`${user.username || "我的账号"} · ${{admin:"管理员",operator:"操作员",viewer:"只读成员"}[user.role] || user.role}`;
  $("import-skill").disabled=!canWrite();
  $("member-form").closest("details").hidden=user.role!=="admin";
  await refreshRuns(); if(token!==state.token)return;
  const cap=await api("/capabilities");
  if(token!==state.token)return;
  $("model").replaceChildren(new Option("自动选择模型", ""));
  state.modelRoutes=cap.model_routes || [];
  if(state.modelRoutes.length) state.modelRoutes.forEach(route=>$("model").add(new Option(`${route.label} · ${route.model}`,route.id)));
  else cap.models.forEach(m=>$("model").add(new Option(m,m)));
  WorkspaceUI.renderCapabilities(cap);
  WorkspaceUI.configureCollaboration(cap,canWrite());
  $("member-form").closest("details").hidden=cap.role!=="admin";
  const bootstrap=await api("/auth/me");
  if(token!==state.token)return;
  $("memory-preview").textContent=bootstrap.bootstrap.memories.length ? `已加载 ${bootstrap.bootstrap.memories.length} 条长期记忆与约束` : "可以在长期记忆中建立你的协作偏好。";
  if(valid())notice("已进入你的私有工作空间");
  await refreshOverview();
}
$("auth-form").addEventListener("submit", async event=>{
  event.preventDefault(); $("auth-error").textContent=""; $("login").disabled=true;
  try {
    const result=await api("/auth/login",{method:"POST",body:JSON.stringify({username:$("username").value,password:$("password").value})});
    state.token=result.token; sessionStorage.setItem("aegis-token",result.token); $("password").value="";
    await enter(result.user);
  } catch(error) { $("auth-error").textContent=error.message; }
  finally { $("login").disabled=false; }
});
$("register").addEventListener("click", async()=>{
  $("auth-error").textContent="";
  const password=$("password");
  password.setCustomValidity(password.value.length>0 && password.value.length<8 ? "密码至少需要 8 个字符。" : "");
  if(!$("auth-form").reportValidity()) {
    $("auth-error").textContent="请检查账号和密码，密码至少需要 8 个字符。";
    return;
  }
  $("register").disabled=true;
  try { await api("/auth/register", {method:"POST",body:JSON.stringify({username:$("username").value,password:$("password").value})}); $("auth-error").textContent="账号已创建，可以登录。"; }
  catch(error) { $("auth-error").textContent=error.message; }
  finally { $("register").disabled=false; }
});
$("password").addEventListener("input",()=>$("password").setCustomValidity(""));
$("logout").onclick=()=>guard(async()=>{ await api("/auth/logout",{method:"POST"}); resetLogin(); });
async function refreshOverview() {
  const token=state.token, generation=++state.overviewGeneration;
  try {
    const overview=await api("/workspace/overview");
    if(token===state.token && generation===state.overviewGeneration) WorkspaceUI.renderOverview(overview,statuses);
  } catch(error) { if(token===state.token && generation===state.overviewGeneration) $("workspace-overview").textContent=`统计暂不可用：${error.message}`; }
}
function renderRuns() {
  $("run-list").replaceChildren();
  if(!state.runs.length) $("run-list").append(WorkspaceUI.node("p", "未找到任务。可以调整筛选或开始新任务。", "empty"));
  for(const run of state.runs) {
    const button=document.createElement("button"); button.className="run-item"+(state.run?.id===run.id?" current":"");
    button.dataset.status=run.status;
    button.append(WorkspaceUI.node("span",run.message,"run-title"));button.title=run.message;
    const small=document.createElement("small"); small.textContent=statuses[run.status] || run.status; button.append(small);
    button.onclick=()=>guard(()=>openRun(run.id)); $("run-list").append(button);
  }
  $("runs-more").hidden=!state.hasMore;
  $("runs-more").disabled=state.runsLoading;
}
async function refreshRuns(more=false) {
  if(more && (state.runsLoading || !state.hasMore))return;
  const token=state.token, generation=++state.runsGeneration;
  const valid=()=>token===state.token && generation===state.runsGeneration;
  const params=new URLSearchParams({limit:"20"});
  if($("run-search").value) params.set("query",$("run-search").value);
  if($("run-filter").value) params.set("status",$("run-filter").value);
  if(more && state.runs.length)params.set("before",state.runs.at(-1).id);
  state.runsLoading=true;$("runs-more").disabled=true;
  if(!more) $("run-list").replaceChildren(WorkspaceUI.node("p","正在读取任务…","empty"));
  try {
    const runs=await api(`/runs?${params}`);if(!valid())return;
    state.runs=more ? [...state.runs,...runs] : runs;
    state.hasMore=runs.length===20;renderRuns();
  } catch(error) {
    if(valid()) { if(!more) $("run-list").replaceChildren(WorkspaceUI.node("p",`任务读取失败：${error.message}`,"empty"));else notice(error.message); }
  } finally { if(valid()){state.runsLoading=false;$("runs-more").disabled=false;} }
}
function message(role,text) {
  $("conversation").append(createMessage(role,text));
}
function renderRun(run) {
  state.run=run; state.session=run.session_id;
  $("run-status").textContent=statuses[run.status] || run.status;
  $("run-status").dataset.status=run.status;
  const route=state.modelRoutes.find(item=>item.id===run.model);
  $("run-meta").textContent=`步骤 ${run.step || 0}\nTrace ${run.trace_id}\n模型 ${route?.label || run.model || "自动选择"}\n协作默认 ${run.collaboration_mode || "team"}\n代码目录 ${run.project_mode || "未选择"}`;
  $("cancel").hidden=!canWrite() || !['queued','running','waiting_approval'].includes(run.status);
  $("feedback").hidden=!canWrite() || !['completed','failed'].includes(run.status);
  $("approval").hidden=run.status!=="waiting_approval";
  if(run.status==="waiting_approval") WorkspaceUI.setInspector(true);
  if(run.approval) $("approval-data").textContent=JSON.stringify({工具:run.approval.name,参数:run.approval.arguments},null,2);
  $("approve").hidden=!canWrite();$("reject").hidden=!canWrite();
  const previous=$("answer"); previous?.remove();
  if(run.answer || run.error) { message("assistant",run.answer || run.error); $("conversation").lastChild.id="answer"; }
  $("conversation").scrollTop=$("conversation").scrollHeight;
}
async function openRun(id) {
  state.generation++; state.stream?.abort(); state.cursor=0; $("timeline").replaceChildren();
  WorkspaceUI.clearFiles(); state.run=null;state.session=null;
  WorkspaceUI.clearCollaboration();
  showView("chat");$("conversation").replaceChildren();
  setRunLoading(true);$("connection").textContent="正在读取任务";
  $("run-status").textContent="正在读取任务…";$("run-meta").textContent="";$("approval-data").textContent="";
  $("run-status").removeAttribute("data-status");
  for(const control of ['approval','cancel','feedback'])$(control).hidden=true;
  const generation=state.generation,token=state.token,valid=contextCurrent();
  try {
    const run=await api(`/runs/${id}`);if(!valid())return;
    message("user",run.message); renderRun(run);WorkspaceUI.setInspector(true);
    $("conversation").prepend(createThreadHistory(
      id, api, () => runCurrent(id,generation,token), statuses,
    ));
    sessionStorage.setItem("aegis-last-run",id);renderRuns();
    watch(id,generation,token);loadRunFiles();
    return true;
  } catch(error) {
    if(valid()) {
      $("run-status").textContent="任务读取失败";$("connection").textContent="未连接任务";
      notice(`无法读取任务：${error.message}。请重新选择任务，或开始新任务。`);
    }
  } finally { if(valid())setRunLoading(false); }
}
function addEvent(id,type,data) {
  if(id<=state.cursor) return; state.cursor=id;
  $("timeline").append(WorkspaceUI.eventRow(id,type,data,statuses));
  WorkspaceUI.renderCollaboration(type,data,childId=>guard(()=>openRun(childId)));
}
async function watch(id,generation,token) {
  const valid=()=>runCurrent(id,generation,token);
  if(!valid())return;
  const controller=new AbortController(); state.stream=controller;
  try {
    const response=await fetch(`${apiRoot}/runs/${id}/events?after=${state.cursor}`,{headers:{Authorization:`Bearer ${token}`},signal:controller.signal});
    if(!valid())return;
    if(!response.ok) throw new Error(`事件连接失败 (${response.status})`);
    $("connection").textContent="已连接";
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer="";
    while(true) {
      const {done,value}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
      let boundary;
      while((boundary=buffer.indexOf("\n\n"))>=0) {
        const packet=buffer.slice(0,boundary); buffer=buffer.slice(boundary+2);
        const lines=packet.split("\n"), values: Record<string, string>={};
        for(const line of lines) { const colon=line.indexOf(":"); if(colon>0) values[line.slice(0,colon)]=line.slice(colon+1).trim(); }
        if(!valid())return;
        if(values.event==="auth_expired") { resetLogin(); return; }
        if(values.id && values.data) addEvent(Number(values.id),values.event,JSON.parse(values.data));
      }
    }
    if(!valid()) return;
    const run=await api(`/runs/${id}`);if(!valid())return;
    const changed=state.run.status!==run.status;
    renderRun(run);
    if(changed && !['queued','running'].includes(run.status))loadRunFiles();
    await refreshRuns();if(!valid())return;
    await refreshOverview();if(!valid())return;
    if(['queued','running'].includes(run.status)) setTimeout(()=>watch(id,generation,token),500);
  } catch(error) {
    if(error.name==="AbortError" || !valid()) return;
    $("connection").textContent="正在重连"; notice("连接中断，任务仍在后台运行，正在恢复事件。");
    setTimeout(()=>{if(valid()) watch(id,generation,token);},1500);
  }
}
function newTask() {
  state.generation++; state.stream?.abort(); state.run=null; state.session=null; state.cursor=0;
  setRunLoading(false);
  WorkspaceUI.clearFiles();sessionStorage.removeItem("aegis-last-run");$("message").value="";$("approval-data").textContent="";
  WorkspaceUI.clearCollaboration();
  $("conversation").replaceChildren(welcomeTemplate.cloneNode(true));
  WorkspaceUI.setInspector(false);
  $("timeline").replaceChildren(); $("run-status").textContent="尚未开始任务";$("run-status").removeAttribute("data-status"); $("run-meta").textContent="";$("connection").textContent="尚未连接任务";
  for(const id of ['approval','cancel','feedback']) $(id).hidden=true;
  showView("chat"); $("message").focus();
}
$("new-task").onclick=newTask; $("refresh").onclick=()=>guard(()=>refreshRuns());
$("runs-more").onclick=()=>guard(()=>refreshRuns(true));
let searchTimer;
function invalidateRunSearch() {
  clearTimeout(searchTimer);state.runsGeneration++;state.runsLoading=false;state.hasMore=false;
  $("runs-more").hidden=true;$("run-list").replaceChildren(WorkspaceUI.node("p","正在筛选任务…","empty"));
}
$("run-search").oninput=()=>{invalidateRunSearch();searchTimer=setTimeout(()=>guard(()=>refreshRuns()),250);};
$("run-filter").onchange=()=>{invalidateRunSearch();guard(()=>refreshRuns());};
function loadRunFiles() {
  if(!state.run || state.view!=="chat")return;
  const id=state.run.id,generation=state.generation,token=state.token;
  return WorkspaceUI.loadFiles(id,api,()=>runCurrent(id,generation,token));
}
$("files-refresh").onclick=()=>{if(state.run)loadRunFiles();else WorkspaceUI.clearFiles();};
$("conversation").addEventListener("click",event=>{
  const button=event.target.closest("[data-prompt]");
  if(button) { $("message").value=button.dataset.prompt;$("message").focus(); }
});
$("composer").addEventListener("submit", event=>{
  event.preventDefault();if(state.runLoading)return;
  guard(async()=>{
    const valid=contextCurrent();
    const body=JSON.stringify({message:$("message").value,session_id:state.session,mode:$("mode").value,model:$("model").value || null,collaboration_mode:canWrite()?$("collaboration-mode").value:null,project_mode:canWrite()?($("project-mode").value || null):null});
    if(!state.pendingRequest || state.pendingRequest.body!==body) state.pendingRequest={body,key:crypto.randomUUID()};
    sessionStorage.setItem("aegis-pending-request",JSON.stringify(state.pendingRequest));
    const run=await api("/runs",{method:"POST",headers:{"Idempotency-Key":state.pendingRequest.key},body});
    if(!valid())return;
    state.pendingRequest=null;sessionStorage.removeItem("aegis-pending-request");
    $("message").value=""; await openRun(run.id);
  },$("send"));
});
$("message").addEventListener("keydown",event=>{if(event.ctrlKey && event.key==="Enter") {event.preventDefault();$("composer").requestSubmit();}});
document.addEventListener("keydown",event=>{if(event.altKey && event.key.toLowerCase()==="n" && state.user) {event.preventDefault();newTask();}});
const runActions: Array<[string, string, Record<string, boolean>]> = [
  ["approve","approval",{approved:true}],
  ["reject","approval",{approved:false}],
  ["cancel","cancel",{}],
  ["success","feedback",{success:true}],
  ["failure","feedback",{success:false}],
];
for(const [id,path,body] of runActions) {
  $(id).onclick=()=>guard(async()=>{ const runId=state.run.id;
    if(!canWrite())return;
    const valid=contextCurrent();
    const payload=path==="approval" ? {...body,call_id:state.run.approval.call_id,args_hash:state.run.approval.hash} : body;
    await api(`/runs/${runId}/${path}`,{method:"POST",body:JSON.stringify(payload)});if(!valid())return;
    const opened=await openRun(runId);
    if(opened)notice(path==="feedback"?"反馈已记录，经验可在 Skills 中管理。":"任务状态已更新"); },$(id));
}
async function showView(view) {
  if(view==="chat" && state.view==="chat")return;
  if(view==="chat" && state.view!=="chat" && state.run)return openRun(state.run.id);
  state.view=view;state.viewGeneration++;state.assetsGeneration++;
  if(view!=="chat"){state.generation++;state.stream?.abort();WorkspaceUI.clearFiles();setRunLoading(false);}
  const titles: Record<string, string>={chat:"任务空间",projects:"项目",memories:"长期记忆",skills:"Skills",documents:"知识库",settings:"能力与设置"};
  $("view-title").textContent=titles[view]; $("location").textContent=titles[view];
  $("chat-view").hidden=view!=="chat"; $("settings-view").hidden=view!=="settings";
  $("assets-view").hidden=['chat','settings'].includes(view);
  document.querySelectorAll<HTMLElement>("[data-view]").forEach(b=>b.classList.toggle("selected",b.dataset.view===view));
  if(!['chat','settings'].includes(view)) {
    $("asset-form").hidden=true; $("upload-form").hidden=view!=="documents" || !canWrite(); $("add-asset").hidden=view==="documents" || !canWrite();
    $("skill-controls").hidden=view!=="skills";
    closeSkillImport();
    $("import-skill").disabled=!canWrite();
    $("assets-description").textContent={memories:"偏好与约束跨会话保留。候选需要你确认，退役内容不再进入任务。",skills:"从执行经验中提炼的能力。检查适用边界与来源，再启用或修订。",documents:"上传资料供本账号检索。支持文本型 PDF、Markdown 与 TXT。",projects:"保存项目目标与边界。选择项目后，新任务在同一线程连续推进。"}[view];
    await loadAssets();
  }
}
document.querySelectorAll<HTMLElement>("[data-view]").forEach(b=>b.onclick=()=>guard(()=>showView(b.dataset.view)));
const navigation = initializeNavigation(() => Boolean(state.user), newTask, view => guard(() => showView(view)));
async function loadAssets() {
  const token=state.token,view=state.view,generation=++state.assetsGeneration;
  const valid=()=>token===state.token && view===state.view && generation===state.assetsGeneration;
  $("asset-list").replaceChildren(WorkspaceUI.node("p","正在读取资产…","empty"));
  let all;
  try { all=await api("/assets"); }
  catch(error) { if(valid()) $("asset-list").replaceChildren(WorkspaceUI.node("p",`资产读取失败：${error.message}`,"empty"));return; }
  if(!valid())return;
  const filter={skills:['skill','procedure'],memories:['profile','preference','constraint','memory','episodic'],projects:['project'],documents:['document']}[view];
  state.assets=all.filter(a=>filter.includes(a.kind));
  if(state.view==="skills") refreshSkillDirectories();
  renderAssets();
  await refreshOverview();
}
async function mutateAsset(path, payload) {
  if(!canWrite())return;
  const valid=contextCurrent();
  await api(path,{method:"POST",body:JSON.stringify(payload)});
  if(valid())await loadAssets();
}
function refreshSkillDirectories() {
  const select=$("skill-directory-filter"), previous=select.value;
  select.replaceChildren(new Option("全部目录", ""));
  const directories: string[]=[...new Set<string>(state.assets.map(a=>a.metadata.directory || "未分类"))].sort();
  directories.forEach(directory=>select.add(new Option(directory,directory)));
  if(directories.includes(previous)) select.value=previous;
}
function renderAssets() {
  const directory=state.view==="skills" ? $("skill-directory-filter").value : "";
  const assets=state.assets.filter(a=>!directory || (a.metadata.directory || "未分类")===directory);
  $("asset-list").replaceChildren();
  if(!assets.length) { const empty=document.createElement("p");empty.className="empty";empty.textContent="这里还没有内容。添加资料或完成任务后，经验会逐步积累。";$("asset-list").append(empty); }
  for(const asset of assets) {
    const row=document.createElement("article"); row.className="asset-row";
    const h=document.createElement("h3");h.textContent=asset.name;
    const meta=document.createElement("div");meta.className="asset-meta";meta.textContent=`${kinds[asset.kind] || asset.kind} · ${assetStates[asset.status] || asset.status} · v${asset.version}`;
    const content=document.createElement("div"); content.className="asset-detail";content.textContent=asset.content.slice(0,2400);
    row.append(h,meta,content);
    if(asset.kind==="skill") appendSkillControls(row,asset);
    if(asset.metadata.source_run_id) {const source=document.createElement("button");source.textContent="查看来源任务";source.onclick=()=>guard(()=>openRun(asset.metadata.source_run_id));row.append(source);}
    if(asset.kind==="project") {
      const use=document.createElement("button");use.textContent="在此项目开始任务";use.onclick=()=>{newTask();state.session=asset.id;$("message").value=`项目：${asset.name}\n约束：${asset.content}\n任务：`;};row.append(use);
    }
    for(const [label,next] of [["启用","active"],["退役","retired"]]) {
      if(!canWrite() || asset.status===next || asset.kind==="document") continue;
      const button=document.createElement("button");button.textContent=label;button.onclick=()=>guard(()=>mutateAsset(`/assets/${asset.id}/state`,{status:next}),button);row.append(button);
    }
    const edit=document.createElement("button");edit.textContent="修订";edit.onclick=()=>{
      $("asset-form").hidden=false;state.editingAsset=asset;
      $("asset-name").value=asset.name;$("asset-content").value=asset.content;$("asset-kind").value=asset.kind;
    };if(canWrite() && asset.kind!=="document")row.append(edit);
    const history=document.createElement("button");history.textContent="版本记录";history.onclick=()=>guard(async()=>{
      row.querySelector(".version-history")?.remove(); const versions=document.createElement("div");versions.className="version-history";
      const valid=contextCurrent();
      const entries=await api(`/assets/${asset.id}/history`);if(!valid() || !row.isConnected)return;
      let pre=document.createElement("pre");pre.textContent=JSON.stringify(entries,null,2);versions.append(pre);
      for(const entry of entries) {if(!canWrite() || entry.version===asset.version)continue;const restore=document.createElement("button");restore.textContent=`恢复 v${entry.version}`;
        restore.onclick=()=>guard(()=>mutateAsset(`/assets/${asset.id}/restore`,{version:entry.version}),restore);versions.append(restore);}
      row.append(versions);
    });row.append(history);
    $("asset-list").append(row);
  }
}
function downloadText(filename,content,type) {
  const url=URL.createObjectURL(new Blob([content],{type}));
  const link=document.createElement("a"); link.href=url;link.download=filename;
  document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function appendSkillControls(row,asset) {
  const descriptor=document.createElement("p");descriptor.className="asset-meta";
  const metadata=asset.metadata;
  descriptor.textContent=`目录：${metadata.directory || "未分类"} · ${metadata.description || "请检查适用条件后启用"}`;
  row.append(descriptor);
  const exportActions: Array<[string, boolean]> = [["导出 SKILL.md",false],["导出文件包",true]];
  for(const [label,bundle] of exportActions) {
    const button=document.createElement("button");button.textContent=label;
    button.onclick=()=>guard(async()=>{
      const valid=contextCurrent();
      const result=await api(`/skills/${asset.id}/export`);
      if(!valid() || !row.isConnected)return;
      downloadText(bundle?`skill-${asset.id}.json`:"SKILL.md",bundle?JSON.stringify(result,null,2):result.document,bundle?"application/json":"text/markdown");
    },button);row.append(button);
  }
  for(const path of Object.keys(metadata.resources || {})) {
    const button=document.createElement("button");button.textContent=`查看 ${path}`;
    button.onclick=()=>guard(async()=>{
      const valid=contextCurrent();
      const resource=await api(`/skills/${asset.id}/resources?path=${encodeURIComponent(path)}`);
      if(!valid() || !row.isConnected)return;
      let preview=row.querySelector(".skill-resource-preview");
      if(!preview) {preview=document.createElement("pre");preview.className="skill-resource-preview";row.append(preview);}
      preview.textContent=`${resource.path}\n\n${resource.content}`;
    },button);row.append(button);
  }
}
function closeSkillImport() {
  $("skill-import-form").hidden=true;$("import-skill").setAttribute("aria-expanded","false");
}
$("skill-directory-filter").onchange=renderAssets;
$("import-skill").onclick=()=>{
  $("skill-import-form").hidden=false;$("skill-import-error").textContent="";
  $("import-skill").setAttribute("aria-expanded","true");$("skill-file").focus();
};
$("close-skill-import").onclick=()=>{closeSkillImport();$("import-skill").focus();};
$("skill-import-form").addEventListener("submit",async event=>{
  event.preventDefault();const button=event.submitter;button.disabled=true;
  if(!canWrite()){button.disabled=false;return;}
  const importToken=state.token, importGeneration=state.generation,importView=state.viewGeneration;
  const valid=()=>importToken===state.token && importGeneration===state.generation && importView===state.viewGeneration;
  $("skill-import-error").textContent="";
  try {
    const file=$("skill-file").files[0];
    if(!file) throw new Error("请先选择技能文件。");
    const isBundle=file.name.toLowerCase().endsWith(".json");
    if(!isBundle && !file.name.toLowerCase().endsWith(".md")) throw new Error("请选择 Markdown 或 JSON 文件。");
    if(file.size>(isBundle?524288:32768)) throw new Error("技能文件超出大小限制。");
    const bytes=await file.arrayBuffer();let text;
    if(!valid()) return;
    try {text=new TextDecoder("utf-8",{fatal:true}).decode(bytes);}
    catch {throw new Error("技能文件必须使用有效的 UTF-8 编码。");}
    let payload: any={document:text};
    if(isBundle) {
      try {payload=JSON.parse(text);}
      catch {throw new Error("JSON 文件包格式无效，请检查文件内容。");}
    }
    if(!payload || typeof payload!=="object" || Array.isArray(payload) || typeof payload.document!=="string") throw new Error("文件包必须包含 document 技能文本。");
    const directory=$("skill-directory").value.trim() || payload.directory || "";
    await api("/skills/import",{method:"POST",body:JSON.stringify({document:payload.document,directory,resources:payload.resources || {}})});
    if(!valid())return;
    $("skill-import-form").reset();closeSkillImport();
    if(state.view==="skills") await loadAssets();
    if(!valid())return;
    notice("已导入技能候选。审阅正文和资源后再启用。");$("import-skill").focus();
  } catch(error) {
    if(valid()) {
      $("skill-import-error").textContent=error.message;$("skill-import-error").focus();
    }
  } finally {if(importToken===state.token) button.disabled=false;}
});
$("add-asset").onclick=()=>{$("asset-form").hidden=false;state.editingAsset=null;$("asset-name").value="";$("asset-content").value="";$("asset-kind").value={skills:"skill",projects:"project",memories:"profile"}[state.view];$("asset-name").focus();};
$("close-editor").onclick=()=>{$("asset-form").hidden=true;};
$("asset-form").addEventListener("submit",event=>{
  event.preventDefault();guard(async()=>{
    if(!canWrite())return;
    const valid=contextCurrent();
    const existing=state.editingAsset;
    const payload: any={kind:$("asset-kind").value,name:$("asset-name").value,content:$("asset-content").value,status:"draft"};
    if(existing) {payload.metadata=existing.metadata;payload.expected_version=existing.version;}
    await api(existing?`/assets/${existing.id}`:"/assets",{method:existing?"PUT":"POST",body:JSON.stringify(payload)});
    if(!valid())return;
    $("asset-form").hidden=true;state.editingAsset=null;await loadAssets();if(valid())notice("已保存候选，可审阅后启用");
  },event.submitter);
});
$("upload-form").addEventListener("submit",event=>{
  event.preventDefault();
  guard(async()=>{
    if(!canWrite())return;
    const valid=contextCurrent();
    const file=$("document-file").files[0];
    if(!file) throw new Error("请先选择文档");
    const form=new FormData();form.append("file",file);
    const uploaded=await api("/documents/upload",{method:"POST",body:form});
    if(!valid())return;
    await loadAssets();
    if(!valid())return;
    const warnings=uploaded.indexing?.warnings || [];
    let status=uploaded.reused ? "已复用原有文档" : "文档已入库";
    if(warnings.length) status=`文档已保存。${warnings.join("；")}`;
    notice(status);
  },event.submitter);
});
$("member-form").addEventListener("submit",event=>{event.preventDefault();guard(async()=>{
  if(state.user?.role!=="admin")return;
  const valid=contextCurrent();
  await api("/auth/members",{method:"POST",body:JSON.stringify({username:$("member-name").value,password:$("member-password").value,role:$("member-role").value})});
  if(valid()){$("member-form").reset();notice("企业成员已创建");}
},event.submitter);});
if(state.token) guard(async()=>{
  const valid=contextCurrent();
  const user=await api("/auth/me");if(!valid())return;
  await enter(user);if(!valid())return;
  const id=sessionStorage.getItem("aegis-last-run");if(id)await openRun(id);
});
if(state.token && sessionStorage.getItem("aegis-pending-request")) {
  try {state.pendingRequest=JSON.parse(sessionStorage.getItem("aegis-pending-request"));} catch {sessionStorage.removeItem("aegis-pending-request");}
}

// 保留现有浏览器验收与只读诊断入口；业务能力仍受后端鉴权约束。
window.state = state;
window.enter = enter;
window.addEvent = addEvent;
window.renderRun = renderRun;
