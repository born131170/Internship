# bootstrap.py — создаёт недостающие файлы проекта. Запуск: python bootstrap.py
from pathlib import Path
ROOT=Path(__file__).resolve().parent

REQ="""fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
httpx>=0.27
numpy>=1.24
opencv-python>=4.9
mediapipe>=0.10.14
"""

SNAP=r'''
"""Цифровые слепки жестов/мимики + поиск идентичных эпизодов (z-norm + DTW)."""
from __future__ import annotations
import math, time, uuid
import numpy as np

DEFAULT_CHANNELS=["smile","blink","jawO","browUp","yc","pc","hand_speed","aperture"]
N=64

def _resample(vals, n=N):
    vals=np.asarray(vals,dtype=float)
    idx=np.where(~np.isnan(vals))[0]
    if len(idx)==0: return np.zeros(n)
    if len(idx)==1: return np.full(n, vals[idx[0]])
    return np.interp(np.linspace(0,len(vals)-1,n), idx, vals[idx])

def _znorm(a):
    a=np.asarray(a,dtype=float); s=a.std()
    return (a-a.mean())/(s if s>1e-9 else 1.0)

def dtw_dist(a, b, band=12):
    n,m=len(a),len(b); INF=float("inf")
    D=np.full((n+1,m+1),INF); D[0,0]=0.0
    for i in range(1,n+1):
        lo=max(1,i-band); hi=min(m,i+band)
        ai=a[i-1]
        for j in range(lo,hi+1):
            c=(ai-b[j-1])**2
            D[i,j]=c+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return math.sqrt(D[n,m]/max(n,m))

def create_snapshot(records, t0, t1, channels, name):
    win=[r for r in records if t0<=r["t"]<=t1]
    if len(win)<4: raise ValueError("окно слишком короткое")
    series={}; stats={}
    for ch in channels:
        raw=np.array([r.get(ch,float("nan")) for r in win],dtype=float)
        series[ch]=_znorm(_resample(raw)).tolist()
        stats[ch]={"mean":float(np.nanmean(raw)) if not np.all(np.isnan(raw)) else None,
                   "std":float(np.nanstd(raw)) if not np.all(np.isnan(raw)) else None}
    return {"id":uuid.uuid4().hex[:10],"name":name,"created":int(time.time()),
            "channels":list(channels),"duration":round(t1-t0,2),"t0":t0,"t1":t1,"n":N,
            "series":series,"stats":stats}

def search(records, snap, hop=0.25, top_k=10, refine=25):
    if not records: return []
    chs=[c for c in snap["channels"] if c in snap["series"]]
    if not chs: return []
    T=np.array([r["t"] for r in records])
    M={c:np.array([r.get(c,float("nan")) for r in records],dtype=float) for c in chs}
    S={c:np.asarray(snap["series"][c],dtype=float) for c in chs}
    dur=float(snap["duration"]); t1lim=T[-1]
    cands=[]; t=T[0]
    while t+dur<=t1lim+1e-6:
        a=np.searchsorted(T,t); b=np.searchsorted(T,t+dur,side="right")
        if b-a>=4:
            d=0.0
            for c in chs: d+=np.sqrt(np.mean((_znorm(_resample(M[c][a:b]))-S[c])**2))
            cands.append((t,t+dur,d/len(chs)))
        t+=hop
    cands.sort(key=lambda x:x[2])
    out=[]
    for a,b,d in cands[:refine]:
        ia=np.searchsorted(T,a); ib=np.searchsorted(T,b,side="right")
        dd=0.0
        for c in chs: dd+=dtw_dist(_znorm(_resample(M[c][ia:ib])), S[c])/len(chs)
        out.append({"t0":round(float(a),2),"t1":round(float(b),2),"distance":round(dd,4)})
    out.sort(key=lambda x:x["distance"])
    return out[:top_k]
'''

HTML=r'''
<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PersonaScope — Мультимодальный анализ личности</title>
<style>
:root{--bg:#0b0e14;--panel:#131824;--panel2:#1a2030;--acc:#8b5cf6;--cy:#22d3ee;--txt:#e5e7eb;--mut:#94a3b8;--ok:#34d399;--bad:#f87171}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt);display:flex;min-height:100vh}
aside{width:230px;background:#0e1220;border-right:1px solid #1f2937;padding:16px 12px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:6px}
.logo{font-weight:700;color:#fff}.logo small{display:block;color:var(--mut);font-weight:400}
nav button{display:flex;gap:8px;align-items:center;width:100%;text-align:left;background:none;border:0;color:var(--mut);padding:9px 10px;border-radius:8px;cursor:pointer;font-size:13px}
nav button:hover{background:#161c2c;color:var(--txt)}nav button.active{background:linear-gradient(90deg,#3b2a63,#232a3d);color:#fff;border:1px solid #4c3a86}
main{flex:1;padding:22px 26px;max-width:1200px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:18px 0 8px}.sub{color:var(--mut);margin-bottom:16px}
.card{background:var(--panel);border:1px solid #1f2937;border-radius:12px;padding:16px;margin-bottom:16px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button.btn{background:var(--acc);border:0;color:#fff;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:13px}
button.btn.sec{background:#243049}button.btn:disabled{opacity:.5;cursor:not-allowed}
input,select,textarea{background:#0f1523;border:1px solid #2b3650;color:var(--txt);border-radius:8px;padding:8px 10px;font-size:13px}
textarea{width:100%;min-height:150px;font-family:ui-monospace,Consolas,monospace}
video{width:100%;border-radius:10px;background:#000}
.drop{border:2px dashed #2b3650;transition:.15s;text-align:center}
.drop.over{border-color:var(--cy);background:#12203a}
#jobbox{margin-top:auto;background:#10162a;border:1px solid #223;border-radius:10px;padding:10px;font-size:12px;color:var(--mut)}
.bar{height:6px;background:#22304a;border-radius:4px;margin-top:6px;overflow:hidden}.bar>i{display:block;height:100%;width:0;background:var(--cy)}
.chip{display:inline-block;border-radius:6px;padding:1px 7px;font-size:11px;background:#22304a;color:var(--mut)}
.chip.ok{background:#0e3b2e;color:var(--ok)}.chip.bad{background:#3b1414;color:var(--bad)}
.dash{display:grid;grid-template-columns:1fr 250px;gap:12px}
.inds{background:var(--panel2);border-radius:10px;padding:10px}
.inds h3{margin:2px 0 8px;font-size:13px;text-align:center}
.ind{display:flex;align-items:center;gap:8px;padding:3px 2px;font-size:12px}
.led{width:12px;height:12px;border-radius:3px;background:#5a6472;flex:none}
.led.on{background:var(--cy);box-shadow:0 0 6px var(--cy)}
.ind .ct{margin-left:auto;color:var(--cy);font-variant-numeric:tabular-nums}
canvas.chart{width:100%;height:74px;background:#10141f;border-radius:8px;margin-top:6px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.ep{background:var(--panel2);border:1px solid #263149;border-radius:10px;overflow:hidden;font-size:12px}
.ep img{width:100%;height:120px;object-fit:cover;display:block}
.ep .b{padding:8px}.ep .tc{color:var(--cy)}
.syscard{background:var(--panel);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:14px}
.syscard h3{margin:0 0 8px;color:var(--cy);font-size:15px}
.sb{display:grid;grid-template-columns:150px 1fr 34px;gap:8px;align-items:center;font-size:12px;margin:3px 0}
.sb .t{height:8px;background:#22304a;border-radius:4px;overflow:hidden}.sb .t>i{display:block;height:100%;background:linear-gradient(90deg,#7c3aed,#22d3ee)}
.gauge{width:120px;height:120px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:700;
 background:conic-gradient(var(--ok) calc(var(--v)*1%),#22304a 0)}
pre.out{background:#0d1220;border:1px solid #223;border-radius:10px;padding:12px;max-height:420px;overflow:auto;font-size:12px}
.tag{display:inline-block;background:#243049;border-radius:6px;padding:2px 8px;font-size:11px;color:var(--cy);margin-right:6px}
.lbl{color:var(--mut);font-size:12px}
.warn{background:#3b2f10;border:1px solid #6b5618;color:#fbbf24;border-radius:8px;padding:8px 10px;font-size:12px;margin:8px 0}
.notice{background:#10162a;border:1px dashed #2b3650;border-radius:12px;padding:18px;color:var(--mut)}
</style></head><body>
<aside>
  <div class="logo">PersonaScope<small>v2.3 Pro · мультимодальный анализ</small></div>
  <nav>
    <button data-p="upload" class="active">⬆ Загрузка видео</button>
    <button data-p="analysis">📊 Анализ</button>
    <button data-p="truth">🛡 Правдивость</button>
    <button data-p="snapshot">🖐 Цифровой слепок</button>
    <button data-p="search">🔍 Поиск эпизодов</button>
    <button data-p="llm">⚙ Настройки LLM</button>
    <button data-p="prompt">✏ Промпт</button>
  </nav>
  <div id="jobbox">
    Движок (MediaPipe, локально): <span class="chip" id="mpchip">—</span><br>
    LLM (внешний API): <span class="chip" id="llmchip">не подключён</span>
    <div id="jstate" style="margin-top:6px"></div><div class="bar"><i id="jbar"></i></div><div id="jmsg"></div>
  </div>
</aside>
<main>
<section id="p-upload">
  <h1>Научный анализ личности</h1>
  <div class="sub">Стадия 1 — измерения MediaPipe по кадрам видео (без LLM). Стадия 2 — трактовка LLM, только после явного запуска и только при подключённом API.</div>
  <div class="card drop" id="drop">
    <div id="dropHint">Перетащите видеофайл сюда<br><span class="lbl">или выберите через диалог (mp4 / mov / webm)</span></div>
    <div class="row" style="justify-content:center;margin-top:10px">
      <input type="file" id="file" accept="video/*">
      <button class="btn" id="btnUp">Загрузить и анализировать</button>
      <label class="lbl">stride <input id="stride" type="number" value="2" style="width:60px"></label>
    </div>
  </div>
  <div class="card"><video id="srcVideo" controls></video></div>
  <div class="card" id="dashCard" hidden>
    <div class="row" style="justify-content:space-between"><h2 style="margin:0">Дашборд видео-анализа (MediaPipe)</h2><span class="tag" id="dashTag"></span></div>
    <div class="dash" style="margin-top:10px">
      <div><video id="dashVideo" controls></video>
        <canvas class="chart" id="cVAD"></canvas><canvas class="chart" id="cEMO"></canvas>
        <canvas class="chart" id="cDYN"></canvas><canvas class="chart" id="cHEAD"></canvas><canvas class="chart" id="cHANDS"></canvas>
      </div>
      <div class="inds"><h3>Индикаторы</h3><div id="inds"></div>
        <div class="ind" style="margin-top:8px;border-top:1px solid #2b3650;padding-top:6px"><b>Всего:</b><span class="ct" id="tot"></span></div></div>
    </div>
  </div>
</section>
<section id="p-analysis" hidden><h1>Оценка типа личности</h1><div class="sub">Big Five · MBTI · Эннеаграмма · Temperament · HEXACO · PID-5 (DSM-5)</div>
  <div class="row"><button class="btn" id="btnGoLLM">Запустить LLM-анализ</button><span class="lbl" id="anState"></span></div>
  <div id="sysCards"></div></section>
<section id="p-truth" hidden><h1>Оценка правдивости</h1><div class="sub">Эвристика MediaPipe (без LLM): моргание, избегание взгляда, руки у лица, напряжение.</div>
  <div class="card row"><div class="gauge" id="gauge" style="--v:0">–</div><div id="cueList"></div></div>
  <div class="card" id="llmTruth" hidden><h3>Вердикт LLM</h3><div id="llmTruthBody"></div></div></section>
<section id="p-snapshot" hidden><h1>Цифровой слепок жеста / эмоции</h1><div class="sub">Выделите временное окно на плеере и сохраните мультимодальный шаблон.</div>
  <div class="card row">
    <button class="btn sec" id="m0">⟦ Начало</button><input id="t0" type="number" step="0.1" value="0" style="width:80px">
    <button class="btn sec" id="m1">Конец ⟧</button><input id="t1" type="number" step="0.1" value="3" style="width:80px">
    <input id="sname" placeholder="название слепка" style="width:180px">
    <button class="btn" id="btnSnap">Создать слепок</button></div>
  <div class="card"><span class="lbl">Каналы:</span><div id="chans" class="row"></div></div>
  <div class="card"><h2>Сохранённые слепки</h2><div id="snapList" class="grid"></div>
    <div class="row" style="margin-top:10px"><input type="file" id="snapFile" accept=".json"><button class="btn sec" id="btnSnapUp">Загрузить слепок</button></div></div></section>
<section id="p-search" hidden><h1>Поиск идентичных эпизодов</h1><div class="sub">Скользящее окно + z-norm + DTW по каналам слепка.</div>
  <div class="card row"><select id="snapSel"></select><input id="topk" type="number" value="8" style="width:60px">
    <button class="btn" id="btnSearch">Найти</button></div>
  <div class="card"><div id="resList" class="grid"></div></div></section>
<section id="p-llm" hidden><h1>Настройки LLM (API)</h1>
  <div class="card"><div class="row" style="margin-bottom:8px"><span class="lbl" style="width:110px">Base URL</span><input id="cBase" style="width:340px"></div>
  <div class="row" style="margin-bottom:8px"><span class="lbl" style="width:110px">API key</span><input id="cKey" type="password" style="width:340px"></div>
  <div class="row" style="margin-bottom:8px"><span class="lbl" style="width:110px">Модель</span><input id="cModel" style="width:240px"></div>
  <div class="row"><span class="lbl" style="width:110px">Temperature</span><input id="cTemp" type="number" step="0.1" value="0.3" style="width:80px">
  <button class="btn" id="btnSaveCfg">Сохранить</button><button class="btn sec" id="btnTestCfg">Проверить</button><span id="cfgMsg" class="lbl"></span></div></div></section>
<section id="p-prompt" hidden><h1>Окно промпта для LLM</h1>
  <div class="card"><textarea id="prompt"></textarea>
  <div class="row" style="margin-top:10px"><button class="btn" id="btnRun">Выполнить анализ</button><span class="lbl" id="runState"></span></div></div>
  <div class="card"><h2>RAW-ответ</h2><pre class="out" id="rawOut">—</pre></div></section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const api=async(p,o={})=>{const r=await fetch(p,o); if(!r.ok) throw new Error((await r.text()).slice(0,500)); return r.json();};
const errText=e=>{try{return JSON.parse(e.message).error||e.message}catch(_){return e.message}};
let VID=null,SUM=null,MET=null,RESULT=null,POLL=null;
const CH_LIST=["smile","blink","jawO","browUp","browDown","eyeWide","yc","pc","hand_speed","aperture","hfd","menergy"];
const fmt=t=>{t=Math.max(0,t|0);return String(t/60|0).padStart(2,"0")+":"+String(t%60).padStart(2,"0")};
$$("nav button").forEach(b=>b.onclick=()=>{$$("nav button").forEach(x=>x.classList.remove("active"));b.classList.add("active");
  $$("main section").forEach(s=>s.hidden=true);$("#p-"+b.dataset.p).hidden=false;});
$("#chans").innerHTML=CH_LIST.map(c=>`<label class="tag"><input type="checkbox" value="${c}" ${["smile","blink","jawO","browUp","yc","pc","hand_speed","aperture"].includes(c)?"checked":""}> ${c}</label>`).join("");
$("#prompt").value="Проведи углублённую оценку типа личности по системам Big Five, MBTI, Эннеаграмма, Temperament Theory, HEXACO и PID-5 (DSM-5), а также оценку правдивости человека на видео. Язык отчёта: русский. Для каждой системы приведи 2-4 эпизода-доказательства СТРОГО из входного списка эпизодов.";
const drop=$("#drop");
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("over");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("over");}));
drop.addEventListener("drop",e=>{const f=e.dataTransfer.files&&e.dataTransfer.files[0]; if(f) startUpload(f);});
$("#btnUp").onclick=()=>{const f=$("#file").files[0]; if(f) startUpload(f); else alert("Выберите файл или перетащите видео в зону");};
async function startUpload(f){
  if(!f.type.startsWith("video/")) return alert("Нужен видеофайл (тип: "+f.type+")");
  $("#jstate").textContent="загрузка…"; $("#mpchip").textContent="загрузка"; $("#mpchip").className="chip";
  const fd=new FormData(); fd.append("file",f);
  const v=await api("/api/videos",{method:"POST",body:fd}); VID=v.id;
  $("#srcVideo").src=`/api/videos/${VID}/file?path=source.mp4`;
  await api(`/api/videos/${VID}/analyze?stride=${+$("#stride").value||2}`,{method:"POST"}); poll();}
function poll(){clearInterval(POLL); POLL=setInterval(async()=>{const j=await api(`/api/jobs/${VID}`);
  $("#jstate").textContent="Движок: "+j.state; $("#jmsg").textContent=j.msg||""; $("#jbar").style.width=(j.progress*100)+"%";
  if(j.state==="done"){clearInterval(POLL); $("#mpchip").textContent="done"; $("#mpchip").className="chip ok"; await loadDash();}
  if(j.state==="error"){clearInterval(POLL); $("#mpchip").textContent="ошибка"; $("#mpchip").className="chip bad"; alert("Ошибка MediaPipe: "+j.msg);}},900);}
async function loadDash(){SUM=await api(`/api/videos/${VID}/summary`); MET=await api(`/api/videos/${VID}/metrics`);
  $("#dashCard").hidden=false; $("#dashTag").textContent=`${SUM.duration_sec}s · ${SUM.n_frames} кадров · без LLM`;
  $("#dashVideo").src=`/api/videos/${VID}/file?path=dashboard.mp4`;
  $("#inds").innerHTML=Object.keys(SUM.pattern_names).map(p=>`<div class="ind"><span class="led ${SUM.counters[p]?"on":""}"></span><span>${p}: ${SUM.pattern_names[p]}</span><span class="ct">${SUM.counters[p]}</span></div>`).join("");
  $("#tot").textContent=Object.values(SUM.counters).reduce((a,b)=>a+b,0);
  const nan=v=>v==null?NaN:v, col=k=>MET.map(r=>nan(r[k]));
  const vad1=MET.map(r=>nan(r.smile)-(nan(r.frown)+nan(r.browDown)+nan(r.noseW)));
  const vad2=MET.map(r=>nan(r.jawO)+nan(r.eyeWide));
  const emo=MET.map(r=>({joy:0,surprise:1,anger:2,sadness:3,fear:4,disgust:5})[r.emotion]??NaN);
  draw("#cVAD",[{v:vad1,c:"#34d399"},{v:vad2,c:"#ef4444"}],-1,1,"VAD (valence / arousal)");
  draw("#cEMO",[{v:emo,c:"#fbbf24"}],0,5,"EMO (emotion)");
  draw("#cDYN",[{v:col("hand_speed"),c:"#3b82f6"}],0,.5,"DYN (hand speed)");
  draw("#cHEAD",[{v:col("yc"),c:"#f59e0b"},{v:col("pc"),c:"#fde047"}],-.4,.4,"HEAD (yaw / pitch)");
  draw("#cHANDS",[{v:col("aperture"),c:"#38bdf8"},{v:col("hfd"),c:"#4ade80"}],0,4,"HANDS (aperture / hand-face)");
  renderTruth(await api(`/api/videos/${VID}/truth`));
  RESULT=await api(`/api/videos/${VID}/llm/result`); renderAnalysis(); refreshSnaps();}
function draw(sel,series,y0,y1,label){const cv=$(sel),dpr=devicePixelRatio||1,W=cv.clientWidth,H=74;
  cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext("2d");g.scale(dpr,dpr);g.clearRect(0,0,W,H);
  g.fillStyle="#94a3b8";g.font="10px sans";g.fillText(label,4,11);
  const n=series[0].v.length,st=Math.max(1,Math.ceil(n/W));
  for(const s of series){g.strokeStyle=s.c;g.beginPath();let pen=false;
    for(let i=0;i<n;i+=st){const v=s.v[i]; if(v==null||isNaN(v)){pen=false;continue;}
      const x=i/n*W,y=H-((Math.min(Math.max(v,y0),y1)-y0)/(y1-y0))*(H-16)-4;
      pen?g.lineTo(x,y):(g.moveTo(x,y),pen=true);} g.stroke();}}
function renderTruth(tr){ if(tr){ $("#gauge").style.setProperty("--v",tr.heuristic); $("#gauge").textContent=tr.heuristic;
  $("#cueList").innerHTML=`<div class="lbl">Эвристика MediaPipe (без LLM): ${tr.heuristic}/100</div><ul>`+
  Object.entries(tr.cues).map(([k,v])=>`<li><b>${k}</b>: ${v}</li>`).join("")+"</ul>"; }
  const t=RESULT&&RESULT.parsed&&RESULT.parsed.truthfulness;
  $("#llmTruth").hidden=!t;
  if(t) $("#llmTruthBody").innerHTML=`<b>Score: ${t.score}/100</b> — ${t.verdict||""}<ul>`+
    (t.cues||[]).map(c=>`<li>${c.cue} <span class="lbl">(${c.direction||""})</span></li>`).join("")+"</ul>";}
const SYS=[["big_five","Big Five (OCEAN)",p=>[["Openness",p.openness],["Conscientiousness",p.conscientiousness],["Extraversion",p.extraversion],["Agreeableness",p.agreeableness],["Neuroticism",p.neuroticism]]],
 ["mbti","MBTI",p=>[["E / I",p.axes?.E_I],["S / N",p.axes?.S_N],["T / F",p.axes?.T_F],["J / P",p.axes?.J_P]],p=>"Тип: "+p.type],
 ["enneagram","Эннеаграмма",p=>[["Тип",p.type*10]],p=>`Тип ${p.type}${p.wing?" ("+p.wing+")":""}`],
 ["temperament","Temperament Theory",p=>[["Сангвиник",p.sanguine],["Холерик",p.choleric],["Меланхолик",p.melancholic],["Флегматик",p.phlegmatic]]],
 ["hexaco","HEXACO",p=>[["H",p.H],["E",p.E],["X",p.X],["A",p.A],["C",p.C],["O",p.O]]],
 ["pid5","PID-5 (DSM-5)",p=>[["Neg. affect",p.negative_affect],["Detachment",p.detachment],["Antagonism",p.antagonism],["Disinhibition",p.disinhibition],["Psychoticism",p.psychoticism]]]];
const epById=id=>SUM&&SUM.episodes.find(e=>e.id===id);
function renderAnalysis(){const p=RESULT&&RESULT.parsed;
  if(!p){ const msg=RESULT
    ? "Ответ модели не распарсен как JSON — результаты НЕ применены. Смотрите RAW в разделе «Промпт»."
    : "LLM-анализ НЕ выполнялся. Оценок по системам нет и быть не может без реального ответа API. Подключите LLM в «Настройки LLM» (кнопка «Проверить» должна дать OK) и нажмите «Запустить LLM-анализ». Всё, что видно на «Загрузка видео» и в «Правдивость» — прямые измерения MediaPipe по кадрам, без генерации.";
    $("#sysCards").innerHTML=`<div class="notice">${msg}</div>`; return;}
  const w=(RESULT.warnings||[]);
  $("#sysCards").innerHTML=(w.length?`<div class="warn">Предупреждения валидации (анти-галлюцинации):<ul>${w.map(x=>`<li>${x}</li>`).join("")}</ul></div>`:"")+
  `<div class="syscard"><h3>Заключение</h3>${p.summary||""} <span class="tag">confidence ${(p.confidence??0).toFixed(2)}</span></div>`+
  SYS.map(([k,t,bars,head])=>{const d=p[k]; if(!d)return"";
    const ev=(p.evidence||{})[k]||[];
    return `<div class="syscard"><h3>${t} ${head?`<span class="tag">${head(d)}</span>`:""}${d._reliable===false?'<span class="tag">ненадёжно</span>':""}</h3>`+
    bars(d).filter(b=>b[1]!=null).map(([n,v])=>`<div class="sb"><span>${n}</span><span class="t"><i style="width:${Math.min(100,v)}%"></i></span><b>${Math.round(v)}</b></div>`).join("")+
    (d.notes?`<div class="lbl" style="margin:6px 0">${d.notes}</div>`:"")+
    `<div class="grid" style="margin-top:8px">${ev.map(e=>{const ep=epById(e.episode_id);
      return `<div class="ep">${ep?`<img src="/api/videos/${VID}/file?path=${ep.thumb}">`:""}<div class="b"><span class="tc">${e.timecode||""}</span> ${ep?"· "+ep.name:""}<br>${e.rationale||""}</div></div>`;}).join("")||'<span class="lbl">доказательства не указаны</span>'}</div></div>`;}).join("");
  renderTruth();}
async function runLLM(){ if(!VID) return alert("Сначала загрузите видео");
  let st; try{ st=await api("/api/llm/status"); }catch(e){ return alert(errText(e)); }
  if(!st.configured){ $("#anState").textContent="LLM не подключён — заполните настройки"; setLLMChip(false);
    return alert("LLM не настроен: укажите Base URL, API key и модель, затем «Проверить»."); }
  $("#anState").textContent=`LLM думает… (реальный запрос: ${st.model} @ ${st.base_url})`;
  try{
    const r=await api(`/api/videos/${VID}/llm/analyze`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:$("#prompt").value})});
    RESULT=r; $("#rawOut").textContent=r.raw; renderAnalysis(); setLLMChip(true,st.model);
    $("#anState").textContent="готово"+((r.warnings||[]).length?`, предупреждений: ${r.warnings.length}`:"");
  }catch(e){ RESULT=null; renderAnalysis(); $("#anState").textContent="ошибка: "+errText(e); setLLMChip(false); }}
$("#btnGoLLM").onclick=runLLM; $("#btnRun").onclick=runLLM;
$("#m0").onclick=()=>$("#t0").value=$("#srcVideo").currentTime.toFixed(1);
$("#m1").onclick=()=>$("#t1").value=$("#srcVideo").currentTime.toFixed(1);
$("#btnSnap").onclick=async()=>{const chans=$$("#chans input:checked").map(i=>i.value);
  await api(`/api/videos/${VID}/snapshots`,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({t0:+$("#t0").value,t1:+$("#t1").value,name:$("#sname").value||"слепок",channels:chans})}); refreshSnaps();};
$("#btnSnapUp").onclick=async()=>{const f=$("#snapFile").files[0]; if(!f)return;
  const fd=new FormData(); fd.append("file",f); await api("/api/snapshots/upload",{method:"POST",body:fd}); refreshSnaps();};
async function refreshSnaps(){const s=await api("/api/snapshots");
  $("#snapList").innerHTML=s.map(x=>`<div class="ep"><div class="b"><b>${x.name}</b><br><span class="tc">${x.duration}s</span> · ${x.channels.length} кан. · id ${x.id}</div></div>`).join("")||"<span class='lbl'>пусто</span>";
  $("#snapSel").innerHTML=s.map(x=>`<option value="${x.id}">${x.name} (${x.duration}s)</option>`).join("");}
$("#btnSearch").onclick=async()=>{const sid=$("#snapSel").value; if(!sid||!VID)return;
  const r=await api(`/api/videos/${VID}/search`,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({snapshot_id:sid,top_k:+$("#topk").value})});
  $("#resList").innerHTML=r.results.map((x,i)=>`<div class="ep"><div class="b">#${i+1} <span class="tc">${fmt(x.t0)}–${fmt(x.t1)}</span><br>distance: <b>${x.distance}</b><br>
    <button class="btn sec" onclick="jump(${x.t0})">▶ показать</button></div></div>`).join("")||"<span class='lbl'>ничего не найдено</span>";};
window.jump=t=>{document.querySelector('nav button[data-p=upload]').click(); const v=$("#srcVideo"); v.currentTime=t; v.play();};
function setLLMChip(ok,model){const c=$("#llmchip"); c.className="chip "+(ok?"ok":"bad"); c.textContent=ok?("подключён: "+(model||"")):"не подключён";}
$("#btnSaveCfg").onclick=async()=>{await api("/api/llm/config",{method:"PUT",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({base_url:$("#cBase").value,api_key:$("#cKey").value,model:$("#cModel").value,temperature:+$("#cTemp").value})});
  $("#cfgMsg").textContent="сохранено"; refreshStatus();};
$("#btnTestCfg").onclick=async()=>{ $("#cfgMsg").textContent="проверка…";
  try{ const r=await api("/api/llm/test",{method:"POST"}); $("#cfgMsg").textContent="OK: "+r.reply; setLLMChip(true,$("#cModel").value); }
  catch(e){ $("#cfgMsg").textContent="ошибка: "+errText(e); setLLMChip(false); } };
async function refreshStatus(){ try{ const s=await api("/api/llm/status"); setLLMChip(s.configured, s.configured?s.model:null); }catch(_){} }
(async()=>{const c=await api("/api/llm/config"); $("#cBase").value=c.base_url; $("#cKey").value=c.api_key||""; $("#cModel").value=c.model; $("#cTemp").value=c.temperature; refreshStatus(); renderAnalysis();})();
</script></body></html>
'''

(ROOT/"static").mkdir(exist_ok=True)
(ROOT/"requirements.txt").write_text(REQ, encoding="utf-8")
(ROOT/"snapshots.py").write_text(SNAP.lstrip("\n"), encoding="utf-8")
(ROOT/"static"/"index.html").write_text(HTML.lstrip("\n"), encoding="utf-8")
print("[OK] созданы: requirements.txt, snapshots.py, static/index.html")