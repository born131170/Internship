"""LLM v4.1: детерминированный скоринг копируется из scores_block, отказ невозможен."""
from __future__ import annotations
import json, os, re
from pathlib import Path
import httpx

SCHEMA="""{
 "summary": "3-5 предложений нарративного заключения",
 "big_five": {"openness":0,"conscientiousness":0,"extraversion":0,"agreeableness":0,"neuroticism":0,"notes":"..."},
 "mbti": {"type":"ENFJ","axes":{"E_I":55,"S_N":65,"T_F":40,"J_P":60},"notes":"..."},
 "enneagram": {"type":3,"wing":"3w2","notes":"..."},
 "temperament": {"sanguine":0,"choleric":0,"melancholic":0,"phlegmatic":0,"notes":"..."},
 "hexaco": {"H":0,"E":0,"X":0,"A":0,"C":0,"O":0,"notes":"..."},
 "pid5": {"negative_affect":0,"detachment":0,"antagonism":0,"disinhibition":0,"psychoticism":0,"notes":"..."},
 "truthfulness": {"score":0,"verdict":"...","cues":[{"cue":"...","direction":"повышает/понижает доверие"}]},
 "evidence": {"big_five":[{"episode_id":"P10_00","timecode":"00:12-00:16","rationale":"..."}],
              "mbti":[],"enneagram":[],"temperament":[],"hexaco":[],"pid5":[],"truthfulness":[]},
 "confidence": 0.0
}"""

SYSTEM_PROMPT=(
 "Ты — эксперт по поведенческой диагностике PersonaScope. Вход: JSON измерений MediaPipe (мимика, поза, руки, взгляд, "
 "речь, голос), счётчики паттернов P01-P12, список эпизодов с таймкодами, детерминированные скоринги scores_block и формула truth_heuristic.\n"
 "КОНТРАКТ:\n"
 "1) Числовые скоринги систем и truthfulness.score вычислены детерминированным модулем (scores_block): скопируй их в поля ответа без изменений.\n"
 "2) summary и notes = СОДЕРЖАТЕЛЬНАЯ интерпретационная заключительная оценка: какие маркеры (паттерны, каналы, эпизоды) "
 "поддерживают каждый скор; поведенческие гипотезы причинно-следствия; динамика по видео. Пиши конкретно: счётчики, окна, каналы.\n"
 "3) evidence: для КАЖДОЙ системы процитируй >=2 эпизода из входного списка (episode_id + timecode + rationale <=20 слов); "
 "при слабых маркерах цитируй эпизоды, вошедшие в композит. Пустой evidence = нарушение контракта.\n"
 "4) В summary/notes/verdict ЗАПРЕЩЕНЫ: рассуждения о лицензиях, профессионализме, «алгоритм не обосновывает», "
 "«не является диагнозом», воспроизведение дисклеймеров и методологических оговорок — такие тексты в отчёт не входят.\n"
 "5) truthfulness: score из scores_block; verdict и cues — текстом по маркерам (моргание, избегание, руки у лица, напряжение, темп речи, паузы, f0).\n"
 "6) MBTI: >50 по оси = полюс; Enneagram 1..9 с крылом; Temperament — проценты, сумма ~100.\n"
 "7) Вывод: СТРОГО один валидный JSON без markdown по схеме; rationale <=20 слов."
)

DEFAULT_USER_PROMPT=(
    "Проведи оценку по системам Big Five, MBTI, Эннеаграмма, Temperament Theory, HEXACO, PID-5 (DSM-5) "
    "и оценку правдивости по предоставленным невербальным метрикам. Язык: русский. Соблюдай КОНТРАКТ системного промпта: "
    "числовые гипотезы обязательны, тотальный null запрещён, доказательства — только из входного списка эпизодов."
)

HEUR_DEF=(
    "truth_heuristic (legacy-формула конвейера) = clamp(5..95, 100 - |blink_per_min-20|*1 - avoidance_ratio*60 - "
    "tension_ratio*40 - hand_face_ratio*30 - min(40,|head_yaw_std-0.1|*120) - "
    "max(0,|speech_rate_syll_per_sec-4|*3) - min(20,pause_per_min*1.5)). "
    "ВАЖНО: итоговый score правдивости берётся не отсюда, а из psychometrics.truthfulness.score "
    "(валидированные маркеры нагрузки/арузала, поле psychometrics в evidence) — числа для ответа копируются из scores_block. "
    "Это невалидированный невербальный приор утечки/нагрузки, НЕ вероятность лжи; использовать только как один из cues."
)

_ENV_KEYS={"base_url":"LLM_BASE_URL","api_key":"LLM_API_KEY","model":"LLM_MODEL",
           "temperature":"LLM_TEMPERATURE","timeout":"LLM_TIMEOUT"}

def _dotenv(path: Path) -> dict:
    """Минимальный парсер .env-файла (без внешних зависимостей): KEY=VALUE, # — комментарий."""
    out={}
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k,v=line.split("=",1)
            k=k.strip(); v=v.strip().strip('"').strip("'")
            if k and v: out[k]=v
    except Exception:
        pass
    return out

def load_config(data_dir: Path):
    """Приоритет источников: переменные окружения > .env.local > data/llm_config.json.

    data/llm_config.json пишет интерфейс; .env.local (в корне проекта или в рабочем каталоге)
    и переменные окружения позволяют держать ключ вне рабочей папки — .env.local в .gitignore.
    """
    cfg={"base_url":"https://sharelim.net/v1","api_key":"","model":"gpt-6-astra","temperature":0.3}
    p=Path(data_dir)/"llm_config.json"
    if p.exists():
        try:
            loaded=json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded,dict): cfg.update(loaded)
        except Exception: pass
    env={}
    for root in (Path(__file__).resolve().parent, Path.cwd()):
        f=root/".env.local"
        if f.exists():
            env=_dotenv(f); break
    for key,envname in _ENV_KEYS.items():
        val=os.environ.get(envname) or env.get(envname)
        if val in (None,""): continue
        if key in ("temperature","timeout"):
            try: cfg[key]=float(val)
            except (TypeError,ValueError): continue
        else:
            cfg[key]=val
    return cfg

def compute_confidence(parsed, warnings) -> float:
    """Честный индикатор полноты LLM-ответа 0..1 (не уверенность модели!)."""
    if not isinstance(parsed, dict): return 0.0
    blocks={"big_five":["openness","conscientiousness","extraversion","agreeableness","neuroticism"],
            "hexaco":["H","E","X","A","C","O"],
            "pid5":["negative_affect","detachment","antagonism","disinhibition","psychoticism"],
            "temperament":["sanguine","choleric","melancholic","phlegmatic"]}
    filled=total=0
    for blk,keys in blocks.items():
        d=parsed.get(blk)
        if isinstance(d,dict):
            total+=len(keys)
            filled+=sum(1 for k in keys if isinstance(d.get(k),(int,float)) and not isinstance(d.get(k),bool))
    mb=parsed.get("mbti")
    if isinstance(mb,dict):
        ax=mb.get("axes") or {}
        ks=["E_I","S_N","T_F","J_P"]; total+=4
        filled+=sum(1 for k in ks if isinstance(ax.get(k),(int,float)))
    en=parsed.get("enneagram")
    if isinstance(en,dict):
        total+=1; filled+=1 if isinstance(en.get("type"),(int,float)) else 0
    t=parsed.get("truthfulness")
    if isinstance(t,dict):
        total+=1; filled+=1 if isinstance(t.get("score"),(int,float)) else 0
    ev=parsed.get("evidence") or {}
    syskeys=["big_five","mbti","enneagram","temperament","hexaco","pid5","truthfulness"]
    total+=len(syskeys)
    filled+=sum(1 for k in syskeys if isinstance(ev.get(k),list) and len(ev.get(k))>0)
    if isinstance(parsed.get("summary"),str) and parsed["summary"].strip():
        total+=1; filled+=1
    base=filled/max(1,total)
    pen=min(0.3, 0.05*len(set(warnings or [])))
    return round(max(0.0,min(1.0,base-pen)),2)

def save_config(data_dir: Path, cfg: dict):
    (Path(data_dir)/"llm_config.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=1),encoding="utf-8")

def describe_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        body=(e.response.text or "").strip()
        return f"API вернул HTTP {e.response.status_code}: {body[:400]}"
    if isinstance(e, httpx.ConnectError):
        return f"Не удалось подключиться к хосту: {e.__cause__ or e}"
    if isinstance(e, httpx.TimeoutException):
        return "Таймаут запроса к API (90 c)"
    return f"{type(e).__name__}: {e}"

def is_retriable(e: Exception) -> bool:
    code=getattr(getattr(e,"response",None),"status_code",None)
    if code in (524,502,503,504,429): return True
    return isinstance(e,(httpx.TimeoutException,httpx.ConnectError))

def _as_text(c) -> str:
    if isinstance(c,str): return c
    if c is None: return ""
    if isinstance(c,list):
        out=[]
        for b in c:
            if isinstance(b,dict): out.append(str(b.get("text") or ""))
            else: out.append(str(b))
        return "".join(out)
    return json.dumps(c,ensure_ascii=False)

DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))  # настраивается через env или cfg["timeout"]

def chat(cfg: dict, messages: list, timeout=None):
    url=cfg["base_url"].rstrip("/")+"/chat/completions"
    headers={"Authorization":f"Bearer {cfg.get('api_key','')}"}
    payload={"model":cfg.get("model"),"temperature":cfg.get("temperature",0.3),"messages":messages}
    to=float(timeout or cfg.get("timeout") or DEFAULT_TIMEOUT)
    try:
        r=httpx.post(url,json=payload,headers=headers,timeout=to)
    except httpx.TimeoutException as e:
        raise type(e)(f"Таймаут запроса к API ({int(to)} c)") from e
    r.raise_for_status()
    data=r.json()
    msg=None; fr=""
    try: msg=data["choices"][0]["message"]; fr=data["choices"][0].get("finish_reason") or ""
    except Exception: msg=data.get("message") or data
    return _as_text(msg.get("content") if isinstance(msg,dict) else msg), (fr if isinstance(fr,str) else "")

def ping(cfg: dict): return chat(cfg,[{"role":"user","content":"Ответь одним словом: ok"}])[0]

def repair_json_response(cfg: dict, broken: str):
    msgs=[{"role":"system","content":"Ты — восстановитель JSON. Верни ТОЛЬКО один валидный JSON-объект без markdown и пояснений."},
          {"role":"user","content":"Следующий ответ оборван/испорчен. Восстанови его до валидного JSON по исходной схеме:\n\n"+broken[:16000]}]
    return chat(cfg,msgs)[0]

def _repair_json(s: str):
    for cut in range(len(s), max(0,len(s)-600), -1):
        frag=s[:cut]; stack=[]; instr=False; esc=False; ok=True
        for ch in frag:
            if instr:
                if esc: esc=False
                elif ch=="\\": esc=True
                elif ch=='"': instr=False
                continue
            if ch=='"': instr=True
            elif ch in "{[": stack.append(ch)
            elif ch in "}]":
                if not stack: ok=False; break
                stack.pop()
        if not ok or instr: continue
        f2=frag.rstrip()
        while f2 and f2[-1] in ",:": f2=f2[:-1]
        cand=f2+"".join("}" if c=="{" else "]" for c in reversed(stack))
        try: return json.loads(cand)
        except Exception: continue
    return None

def parse_json(text):
    if not isinstance(text,str): text=_as_text(text)
    t=text.strip()
    t=re.sub(r"^```[a-zA-Z]*\s*","",t); t=re.sub(r"\s*```$","",t)
    try: return json.loads(t)
    except Exception: pass
    m=re.search(r"\{.*\}",t,re.S)
    if m:
        cand=m.group(0)
        try: return json.loads(cand)
        except Exception:
            fixed=_repair_json(cand)
            if fixed is not None: return fixed
    return None

def _cue_list(cues) -> list:
    """Нормализует поле cues к списку.

    Модель иногда возвращает здесь СТРОКУ вместо массива — тогда прежний код
    (`for c in (t.get("cues") or [])`) разбирал её по символам, и в интерфейсе
    вердикт печатался по букве на строку. Список из одиночных символов (след
    того же бага, уже сохранённый в llm_result.json) склеивается обратно —
    поэтому старые анализы чинятся без повторного запроса к модели.
    """
    if isinstance(cues, str):
        return [cues] if cues.strip() else []
    if not isinstance(cues, (list, tuple)):
        return []
    items = list(cues)
    if len(items) >= 3 and all(isinstance(c, str) and len(c) <= 1 for c in items):
        joined = "".join(items).strip()
        return [joined] if joined else []
    return items


def validate_result(parsed: dict, summary: dict):
    warn=[]
    if not isinstance(parsed,dict): return None,["Ответ модели не является JSON-объектом"]
    ids={e["id"] for e in summary.get("episodes",[])}
    ev=parsed.get("evidence")
    if isinstance(ev,dict):
        for key,items in ev.items():
            if not isinstance(items,list): continue
            clean=[]
            for it in items:
                if not isinstance(it,dict): continue
                eid=it.get("episode_id")
                if eid and eid not in ids:
                    warn.append(f"{key}: ссылка на несуществующий эпизод '{eid}' удалена")
                    continue
                clean.append(it)
            ev[key]=clean
    def clamp(d,keys,blk):
        for k in keys:
            v=d.get(k)
            if isinstance(v,(int,float)) and not isinstance(v,bool):
                if v<0 or v>100:
                    warn.append(f"{blk}.{k}={v} вне 0..100 — ограничено")
                    d[k]=float(min(100,max(0,v)))
    for blk,keys in [("big_five",["openness","conscientiousness","extraversion","agreeableness","neuroticism"]),
                     ("hexaco",["H","E","X","A","C","O"]),
                     ("pid5",["negative_affect","detachment","antagonism","disinhibition","psychoticism"])]:
        if isinstance(parsed.get(blk),dict): clamp(parsed[blk],keys,blk)
    t=parsed.get("temperament")
    if isinstance(t,dict):
        vals=[t[k] for k in ("sanguine","choleric","melancholic","phlegmatic") if isinstance(t.get(k),(int,float))]
        if vals and abs(sum(vals)-100)>5:
            warn.append(f"temperament: сумма процентов {round(sum(vals),1)} != 100 — значения помечены ненадёжными")
            t["_reliable"]=False
    tt=parsed.get("truthfulness")
    if isinstance(tt,dict):
        # нормализация cues ДО детерминированной подстановки: модель может вернуть
        # строки вместо объектов {"cue","direction"} — фронт печатал «undefined ()»
        nc=[]
        for c in _cue_list(tt.get("cues")):
            if isinstance(c,dict):
                nm=str(c.get("cue") or c.get("name") or c.get("marker") or "").strip()
                dr=str(c.get("direction") or c.get("note") or "").strip()
                if nm: nc.append({"cue":nm,"direction":dr})
            elif isinstance(c,str) and c.strip():
                nc.append({"cue":c.strip(),"direction":""})
        tt["cues"]=nc
        # жёсткая гарантия: score берётся ТОЛЬКО из детерминированного скоринга движка;
        # модель не может привнести своё число (раньше "Score: 50/100" приходил прямо из LLM)
        try:
            import scoring as _sc
            det=_sc.compute_scores(summary) if isinstance(summary,dict) else None
            dt=(det or {}).get("truthfulness") or {}
            dscore=dt.get("score")
            if isinstance(dscore,(int,float)):
                if isinstance(tt.get("score"),(int,float)) and abs(float(tt["score"])-float(dscore))>2:
                    warn.append(f"truthfulness.score={tt['score']} заменён детерминированным значением движка {dscore} "
                                f"(источник: {dt.get('source','уточняется')}; LLM не меняет числа)")
                tt["score"]=dscore
                if not (tt.get("verdict") or "").strip():
                    tt["verdict"]=dt.get("verdict","")
        except Exception as e:
            warn.append(f"truthfulness: детерминированная подстановка не выполнена ({e})")
    fillmap=[("big_five",["openness","conscientiousness","extraversion","agreeableness","neuroticism"]),
             ("hexaco",["H","E","X","A","C","O"]),
             ("pid5",["negative_affect","detachment","antagonism","disinhibition","psychoticism"])]
    for blk,keys in fillmap:
        d=parsed.get(blk)
        if isinstance(d,dict):
            miss=[k for k in keys if not isinstance(d.get(k),(int,float))]
            if miss:
                for k in miss: d[k]=50.0
                warn.append(f"{blk}: модель вернула блок без скорингов — заполнено популяционной нормой 50 (shrinkage), confidence не завышается")
    mb=parsed.get("mbti")
    if isinstance(mb,dict):
        ax=mb.get("axes")
        if not isinstance(ax,dict) or not any(isinstance(ax.get(k),(int,float)) for k in ("E_I","S_N","T_F","J_P")):
            mb["axes"]={"E_I":50,"S_N":50,"T_F":50,"J_P":50}
            warn.append("mbti: оси отсутствуют — заполнено 50/50 (shrinkage)")
    tp=parsed.get("temperament")
    if isinstance(tp,dict):
        ks=("sanguine","choleric","melancholic","phlegmatic")
        if not any(isinstance(tp.get(k),(int,float)) for k in ks):
            for k in ks: tp[k]=25.0
            warn.append("temperament: проценты отсутствуют — заполнено 25/25/25/25 (shrinkage)")
    return parsed,warn

def build_evidence(summary: dict, max_episodes=40):
    eps=summary.get("episodes",[])
    counts={}
    for e in eps: counts[e["pattern"]]=counts.get(e["pattern"],0)+1
    evidence={"video":{"duration_sec":summary["duration_sec"],"fps":summary["fps"],"frames":summary["n_frames"]},
            "pattern_counters":summary["counters"],"pattern_events":summary["events"],
            "pattern_names":summary["pattern_names"],"stats":summary["stats"],
            "emotion_distribution":summary["emotions"],"truth_cues":summary["truth_cues"],
            "truth_heuristic":summary["truth_heuristic"],
            "truth_heuristic_definition":HEUR_DEF,
            "method":summary.get("method"),
            "episode_counts_by_pattern":counts,
            "episodes":[{"id":e["id"],"pattern":e["pattern"],"name":e["name"],"t0":e["t0"],"t1":e["t1"]} for e in eps[:max_episodes]],
            "episodes_truncated":len(eps)>max_episodes}
    # детерминированный психометрический слой (Big Five LR + кадры-доказательства) — если есть
    ps=summary.get("personality")
    if isinstance(ps,dict):
        bf=ps.get("big_five") or {}
        det={t:{k:v for k,v in (tr or {}).items() if k in ("score","ci95","n_indicators","status")}
             for t,tr in (bf.get("_detail") or {}).items() if isinstance(tr,dict)}
        ev={}
        for t,tr in (bf.get("_detail") or {}).items():
            if isinstance(tr,dict):
                ev[t]=[{kk:x.get(kk) for kk in ("var","value","lr","tier","direction","note","episode_id","pattern","t0","t1")}
                       for x in tr.get("evidence_frames",[])]
        evidence["psychometrics"]={"version":ps.get("version"),
            "big_five_scores":{k:v for k,v in bf.items() if k!="_detail"},
            "big_five_detail":det,"evidence_frames":ev,
            "behavior":ps.get("behavior"),"derived":ps.get("derived"),
            "truthfulness":ps.get("truthfulness"),
            "model_validity":ps.get("model_validity"),
            "instruction":"Используй psychometrics как приоритетный количественный базис для big_five; эпизоды из evidence_frames — реальные кадры-доказательства (ссылайся на их episode_id)."}
    return evidence

def _force_format(cite_cap: int) -> str:
    base=("\n\nФОРМАТ: верни РОВНО ОДИН валидный JSON по схеме. Без markdown, без текста вне JSON. "
     "rationale <=20 слов. Тотальный null запрещён контрактом. Блок без числовых полей (только notes) — нарушение контракта: числа обязательны всегда.")
    if cite_cap>0:
        base+=f" В evidence каждой системы <= {cite_cap} эпизодов."
    else:
        base+=" В evidence всех систем верни пустые списки: только scores, summary, truthfulness."
    return base

def run_analysis(cfg: dict, user_prompt: str, evidence: dict, cite_cap: int=6, retries: int=2):
    """Запрос к LLM с внутренними автоповторами на сетевые ошибки/таймауты.
    Системный промпт берётся из cfg["system_prompt"], если он задан и не совпадает
    со значением по умолчанию (тогда используется встроенный SYSTEM_PROMPT)."""
    import time as _t
    sysp=(cfg.get("system_prompt") or "").strip() or SYSTEM_PROMPT
    msgs=[{"role":"system","content":sysp},
          {"role":"user","content":(user_prompt or DEFAULT_USER_PROMPT)+_force_format(cite_cap)+"\n\nEVIDENCE JSON:\n"+json.dumps(evidence,ensure_ascii=False)}]
    last=None
    for attempt in range(retries+1):
        try:
            return chat(cfg,msgs)
        except Exception as e:
            last=e
            if attempt<retries and is_retriable(e):
                _t.sleep(1.5*(attempt+1)); continue
            raise
    raise last
