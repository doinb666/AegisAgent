"use strict";
const $ = (id) => document.getElementById(id);
const apiRoot = "/api/v1";
const state = {token: sessionStorage.getItem("aegis-token"), user: null, run: null, session: null,
  view: "chat", cursor: 0, stream: null, assets: [], editingAsset: null, generation: 0, pendingRequest: null};
const statuses = {queued:"等待执行",running:"正在推进",waiting_approval:"等待你的审批",completed:"任务完成",failed:"执行失败",cancelled:"已停止",interrupted:"需要人工核对后恢复"};
const kinds = {profile:"偏好",preference:"偏好",constraint:"约束",memory:"记忆",episodic:"经验",skill:"Skill",procedure:"步骤",project:"项目",document:"文档"};
const assetStates = {draft:"候选",active:"已启用",retired:"已退役"};
function notice(text) { $("notice").textContent = text; }
async function api(path, options={}) {
  const requestToken=state.token;
  const headers = new Headers(options.headers || {});
  if (requestToken) headers.set("Authorization", `Bearer ${requestToken}`);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(apiRoot + path, {...options, headers});
  const data = await response.json();
  if(requestToken!==state.token) throw new Error("账号已切换，请在当前空间重新操作。");
  if (!response.ok) {
    if (response.status === 401 && state.user) resetLogin();
    throw new Error(typeof data.detail === "string" ? data.detail : `请求失败 (${response.status})`);
  }
  return data;
}
async function guard(action, button=null) {
  if (button) button.disabled = true;
  try { await action(); } catch(error) { notice(error.message); }
  finally { if (button) button.disabled = false; }
}
function resetLogin() {
  state.generation++; state.stream?.abort(); state.user=null; state.token=null;
  state.pendingRequest=null; state.run=null; state.session=null;
  state.editingAsset=null;$("skill-import-form").reset();closeSkillImport();
  $("skill-import-form").querySelector('button[type="submit"]').disabled=false;
  state.assets=[];$("asset-list").replaceChildren();
  $("run-list").replaceChildren();$("asset-form").reset();$("upload-form").reset();$("member-form").reset();
  newTask();
  sessionStorage.removeItem("aegis-pending-request"); sessionStorage.removeItem("aegis-last-run");
  sessionStorage.removeItem("aegis-token"); $("shell").hidden=true; $("auth").hidden=false;
}
async function enter(user) {
  state.user=user; $("auth").hidden=true; $("shell").hidden=false;
  $("account").textContent=`${user.username || "我的账号"} · ${user.role}`;
  await refreshRuns();
  const cap=await api("/capabilities");
  $("model").replaceChildren(new Option("自动选择模型", ""));
  cap.models.forEach(m=>$("model").add(new Option(m,m)));
  $("capabilities").textContent=`可用模型：${cap.models.join("、") || "未配置，请设置服务端模型环境变量"}\n\n工具：${cap.tools.join("、")}\n\n沙箱：${cap.sandbox ? "已配置，执行需审批" : "未配置，代码执行将被拒绝"}\n角色：${cap.role}\n单任务步数预算：${cap.max_steps}`;
  $("member-form").closest("details").hidden=cap.role!=="admin";
  const bootstrap=await api("/auth/me");
  $("memory-preview").textContent=bootstrap.bootstrap.memories.length ? `已加载 ${bootstrap.bootstrap.memories.length} 条长期记忆与约束` : "可以在长期记忆中建立你的协作偏好。";
  notice("已进入你的私有工作空间");
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
  $("auth-error").textContent=""; $("register").disabled=true;
  try { await api("/auth/register", {method:"POST",body:JSON.stringify({username:$("username").value,password:$("password").value})}); $("auth-error").textContent="账号已创建，可以登录。"; }
  catch(error) { $("auth-error").textContent=error.message; }
  finally { $("register").disabled=false; }
});
$("logout").onclick=()=>guard(async()=>{ await api("/auth/logout",{method:"POST"}); resetLogin(); });
async function refreshRuns() {
  const runs=await api("/runs"); $("run-list").replaceChildren();
  for(const run of runs) {
    const button=document.createElement("button"); button.className="run-item"+(state.run?.id===run.id?" current":"");
    button.append(document.createTextNode(run.message.slice(0,30)));
    const small=document.createElement("small"); small.textContent=statuses[run.status] || run.status; button.append(small);
    button.onclick=()=>guard(()=>openRun(run.id)); $("run-list").append(button);
  }
}
function message(role,text) {
  const article=document.createElement("article"); article.className=`message ${role}`;
  const label=document.createElement("span"); label.className="message-label"; label.textContent=role==="user"?"你":"AegisCode";
  article.append(label,document.createTextNode(text)); $("conversation").append(article);
}
function renderRun(run) {
  state.run=run; state.session=run.session_id;
  $("run-status").textContent=statuses[run.status] || run.status;
  $("run-meta").textContent=`步骤 ${run.step || 0}\nTrace ${run.trace_id}\n模型 ${run.model || "自动选择"}`;
  $("cancel").hidden=!['queued','running','waiting_approval'].includes(run.status);
  $("feedback").hidden=!['completed','failed'].includes(run.status);
  $("approval").hidden=run.status!=="waiting_approval";
  if(run.approval) $("approval-data").textContent=JSON.stringify({工具:run.approval.name,参数:run.approval.arguments},null,2);
  const previous=$("answer"); previous?.remove();
  if(run.answer || run.error) { message("assistant",run.answer || run.error); $("conversation").lastChild.id="answer"; }
  $("conversation").scrollTop=$("conversation").scrollHeight;
}
async function openRun(id) {
  state.generation++; state.stream?.abort(); state.cursor=0; $("timeline").replaceChildren();
  await showView("chat"); $("conversation").replaceChildren();
  const run=await api(`/runs/${id}`); message("user",run.message); renderRun(run);
  sessionStorage.setItem("aegis-last-run",id); await refreshRuns();
  watch(id,state.generation);
}
function addEvent(id,type,data) {
  if(id<=state.cursor) return; state.cursor=id;
  const row=document.createElement("li"); const caption=document.createElement("span"); caption.textContent=`#${id} · ${type}`;
  row.append(caption,document.createTextNode(JSON.stringify(data))); $("timeline").append(row);
}
async function watch(id,generation) {
  const controller=new AbortController(); state.stream=controller;
  try {
    const response=await fetch(`${apiRoot}/runs/${id}/events?after=${state.cursor}`,{headers:{Authorization:`Bearer ${state.token}`},signal:controller.signal});
    if(!response.ok) throw new Error(`事件连接失败 (${response.status})`);
    $("connection").textContent="已连接";
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer="";
    while(true) {
      const {done,value}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
      let boundary;
      while((boundary=buffer.indexOf("\n\n"))>=0) {
        const packet=buffer.slice(0,boundary); buffer=buffer.slice(boundary+2);
        const lines=packet.split("\n"), values={};
        for(const line of lines) { const colon=line.indexOf(":"); if(colon>0) values[line.slice(0,colon)]=line.slice(colon+1).trim(); }
        if(values.event==="auth_expired") { resetLogin(); return; }
        if(values.id && values.data && generation===state.generation) addEvent(Number(values.id),values.event,JSON.parse(values.data));
      }
    }
    if(generation!==state.generation) return;
    const run=await api(`/runs/${id}`); renderRun(run); await refreshRuns();
    if(['queued','running'].includes(run.status)) setTimeout(()=>watch(id,generation),500);
  } catch(error) {
    if(error.name==="AbortError" || generation!==state.generation) return;
    $("connection").textContent="正在重连"; notice("连接中断，任务仍在后台运行，正在恢复事件。");
    setTimeout(()=>{if(state.token && generation===state.generation) watch(id,generation);},1500);
  }
}
function newTask() {
  state.generation++; state.stream?.abort(); state.run=null; state.session=null; state.cursor=0;
  $("conversation").replaceChildren(); message("assistant","描述你的新目标。可以先选择项目、上传资料，或设置长期约束。");
  $("timeline").replaceChildren(); $("run-status").textContent="尚未开始任务"; $("run-meta").textContent="";
  for(const id of ['approval','cancel','feedback']) $(id).hidden=true;
  showView("chat"); $("message").focus();
}
$("new-task").onclick=newTask; $("refresh").onclick=()=>guard(refreshRuns);
document.querySelectorAll("[data-prompt]").forEach(button=>button.onclick=()=>{$("message").value=button.dataset.prompt;$("message").focus();});
$("composer").addEventListener("submit", event=>{
  event.preventDefault(); guard(async()=>{
    const body=JSON.stringify({message:$("message").value,session_id:state.session,mode:$("mode").value,model:$("model").value || null});
    if(!state.pendingRequest || state.pendingRequest.body!==body) state.pendingRequest={body,key:crypto.randomUUID()};
    sessionStorage.setItem("aegis-pending-request",JSON.stringify(state.pendingRequest));
    const run=await api("/runs",{method:"POST",headers:{"Idempotency-Key":state.pendingRequest.key},body});
    state.pendingRequest=null;sessionStorage.removeItem("aegis-pending-request");
    $("message").value=""; await openRun(run.id);
  },$("send"));
});
$("message").addEventListener("keydown",event=>{if(event.ctrlKey && event.key==="Enter") {event.preventDefault();$("composer").requestSubmit();}});
document.addEventListener("keydown",event=>{if(event.altKey && event.key.toLowerCase()==="n" && state.user) {event.preventDefault();newTask();}});
for(const [id,path,body] of [["approve","approval",{approved:true}],["reject","approval",{approved:false}],["cancel","cancel",{}],["success","feedback",{success:true}],["failure","feedback",{success:false}]]) {
  $(id).onclick=()=>guard(async()=>{ const runId=state.run.id;
    const payload=path==="approval" ? {...body,call_id:state.run.approval.call_id,args_hash:state.run.approval.hash} : body;
    await api(`/runs/${runId}/${path}`,{method:"POST",body:JSON.stringify(payload)}); await openRun(runId); notice(path==="feedback"?"反馈已记录，经验可在 Skills 中管理。":"任务状态已更新"); },$(id));
}
async function showView(view) {
  state.view=view;
  const titles={chat:"任务空间",projects:"项目",memories:"长期记忆",skills:"Skills",documents:"知识库",settings:"能力与设置"};
  $("view-title").textContent=titles[view]; $("location").textContent=titles[view];
  $("chat-view").hidden=view!=="chat"; $("settings-view").hidden=view!=="settings";
  $("assets-view").hidden=['chat','settings'].includes(view);
  document.querySelectorAll("[data-view]").forEach(b=>b.classList.toggle("selected",b.dataset.view===view));
  if(!['chat','settings'].includes(view)) {
    $("asset-form").hidden=true; $("upload-form").hidden=view!=="documents"; $("add-asset").hidden=view==="documents";
    $("skill-controls").hidden=view!=="skills";
    closeSkillImport();
    $("import-skill").disabled=state.user.role==="viewer";
    $("assets-description").textContent={memories:"偏好与约束跨会话保留。候选需要你确认，退役内容不再进入任务。",skills:"从执行经验中提炼的能力。检查适用边界与来源，再启用或修订。",documents:"上传资料供本账号检索。支持文本型 PDF、Markdown 与 TXT。",projects:"保存项目目标与边界。选择项目后，新任务在同一线程连续推进。"}[view];
    await loadAssets();
  }
}
document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>guard(()=>showView(b.dataset.view)));
async function loadAssets() {
  const all=await api("/assets");
  const filter={skills:['skill','procedure'],memories:['profile','preference','constraint','memory','episodic'],projects:['project'],documents:['document']}[state.view];
  state.assets=all.filter(a=>filter.includes(a.kind));
  if(state.view==="skills") refreshSkillDirectories();
  renderAssets();
}
function refreshSkillDirectories() {
  const select=$("skill-directory-filter"), previous=select.value;
  select.replaceChildren(new Option("全部目录", ""));
  const directories=[...new Set(state.assets.map(a=>a.metadata.directory || "未分类"))].sort();
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
      if(asset.status===next || asset.kind==="document") continue;
      const button=document.createElement("button");button.textContent=label;button.onclick=()=>guard(async()=>{await api(`/assets/${asset.id}/state`,{method:"POST",body:JSON.stringify({status:next})});await loadAssets();},button);row.append(button);
    }
    const edit=document.createElement("button");edit.textContent="修订";edit.onclick=()=>{
      $("asset-form").hidden=false;state.editingAsset=asset;
      $("asset-name").value=asset.name;$("asset-content").value=asset.content;$("asset-kind").value=asset.kind;
    };if(asset.kind!=="document")row.append(edit);
    const history=document.createElement("button");history.textContent="版本记录";history.onclick=()=>guard(async()=>{
      row.querySelector(".version-history")?.remove(); const versions=document.createElement("div");versions.className="version-history";
      const entries=await api(`/assets/${asset.id}/history`);let pre=document.createElement("pre");pre.textContent=JSON.stringify(entries,null,2);versions.append(pre);
      for(const entry of entries) {if(entry.version===asset.version)continue;const restore=document.createElement("button");restore.textContent=`恢复 v${entry.version}`;
        restore.onclick=()=>guard(async()=>{await api(`/assets/${asset.id}/restore`,{method:"POST",body:JSON.stringify({version:entry.version})});await loadAssets();},restore);versions.append(restore);}
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
  for(const [label,bundle] of [["导出 SKILL.md",false],["导出文件包",true]]) {
    const button=document.createElement("button");button.textContent=label;
    button.onclick=()=>guard(async()=>{
      const result=await api(`/skills/${asset.id}/export`);
      downloadText(bundle?`skill-${asset.id}.json`:"SKILL.md",bundle?JSON.stringify(result,null,2):result.document,bundle?"application/json":"text/markdown");
    },button);row.append(button);
  }
  for(const path of Object.keys(metadata.resources || {})) {
    const button=document.createElement("button");button.textContent=`查看 ${path}`;
    button.onclick=()=>guard(async()=>{
      const resource=await api(`/skills/${asset.id}/resources?path=${encodeURIComponent(path)}`);
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
  const importToken=state.token, importGeneration=state.generation;
  $("skill-import-error").textContent="";
  try {
    const file=$("skill-file").files[0];
    if(!file) throw new Error("请先选择技能文件。");
    const isBundle=file.name.toLowerCase().endsWith(".json");
    if(!isBundle && !file.name.toLowerCase().endsWith(".md")) throw new Error("请选择 Markdown 或 JSON 文件。");
    if(file.size>(isBundle?524288:32768)) throw new Error("技能文件超出大小限制。");
    const bytes=await file.arrayBuffer();let text;
    if(importToken!==state.token || importGeneration!==state.generation) return;
    try {text=new TextDecoder("utf-8",{fatal:true}).decode(bytes);}
    catch {throw new Error("技能文件必须使用有效的 UTF-8 编码。");}
    let payload={document:text};
    if(isBundle) {
      try {payload=JSON.parse(text);}
      catch {throw new Error("JSON 文件包格式无效，请检查文件内容。");}
    }
    if(!payload || typeof payload!=="object" || Array.isArray(payload) || typeof payload.document!=="string") throw new Error("文件包必须包含 document 技能文本。");
    const directory=$("skill-directory").value.trim() || payload.directory || "";
    await api("/skills/import",{method:"POST",body:JSON.stringify({document:payload.document,directory,resources:payload.resources || {}})});
    $("skill-import-form").reset();closeSkillImport();
    if(state.view==="skills") await loadAssets();
    notice("已导入技能候选。审阅正文和资源后再启用。");$("import-skill").focus();
  } catch(error) {
    if(importToken===state.token && importGeneration===state.generation) {
      $("skill-import-error").textContent=error.message;$("skill-import-error").focus();
    }
  } finally {if(importToken===state.token) button.disabled=false;}
});
$("add-asset").onclick=()=>{$("asset-form").hidden=false;state.editingAsset=null;$("asset-name").value="";$("asset-content").value="";$("asset-kind").value={skills:"skill",projects:"project",memories:"profile"}[state.view];$("asset-name").focus();};
$("close-editor").onclick=()=>{$("asset-form").hidden=true;};
$("asset-form").addEventListener("submit",event=>{
  event.preventDefault();guard(async()=>{
    const existing=state.editingAsset;
    const payload={kind:$("asset-kind").value,name:$("asset-name").value,content:$("asset-content").value,status:"draft"};
    if(existing) {payload.metadata=existing.metadata;payload.expected_version=existing.version;}
    await api(existing?`/assets/${existing.id}`:"/assets",{method:existing?"PUT":"POST",body:JSON.stringify(payload)});
    $("asset-form").hidden=true;state.editingAsset=null;await loadAssets();notice("已保存候选，可审阅后启用");
  },event.submitter);
});
$("upload-form").addEventListener("submit",event=>{
  event.preventDefault();
  guard(async()=>{
    const file=$("document-file").files[0];
    if(!file) throw new Error("请先选择文档");
    const form=new FormData();form.append("file",file);
    const uploaded=await api("/documents/upload",{method:"POST",body:form});
    await loadAssets();
    const warnings=uploaded.indexing?.warnings || [];
    let status=uploaded.reused ? "已复用原有文档" : "文档已入库";
    if(warnings.length) status=`文档已保存。${warnings.join("；")}`;
    notice(status);
  },event.submitter);
});
$("member-form").addEventListener("submit",event=>{event.preventDefault();guard(async()=>{await api("/auth/members",{method:"POST",body:JSON.stringify({username:$("member-name").value,password:$("member-password").value,role:$("member-role").value})});$("member-form").reset();notice("企业成员已创建");},event.submitter);});
if(state.token) guard(async()=>{const user=await api("/auth/me");await enter(user);const id=sessionStorage.getItem("aegis-last-run");if(id)await openRun(id);});
if(state.token && sessionStorage.getItem("aegis-pending-request")) {
  try {state.pendingRequest=JSON.parse(sessionStorage.getItem("aegis-pending-request"));} catch {sessionStorage.removeItem("aegis-pending-request");}
}
