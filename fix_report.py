# fix_report.py — ТОЛЬКО: убрать дисклеймер из отчёта/контракта + гарантированные скрины кадров в доказательствах.
from pathlib import Path
import py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parent

def rep(text,old,new,label,required=True):
    if old not in text:
        if required: raise SystemExit(f"[!] якорь не найден: {label}")
        print("[=] пропускаю:",label); return text
    return text.replace(old,new,1)

# --- 1) scoring.py: auto_evidence (скрины кадров гарантированно) ---
sc=(ROOT/"scoring.py").read_text(encoding="utf-8")
sc=rep(sc,'    return {"big_five":bf,"mbti":mbti,"enneagram":enn,"temperament":temp,"hexaco":hexa,"pid5":pid5,',
        '    ae=auto_evidence(s)\n    return {"big_five":bf,"mbti":mbti,"enneagram":enn,"temperament":temp,"hexaco":hexa,"pid5":pid5,"auto_evidence":ae,','scoring return ae')
sc+='''
SYS_PATTERNS={"big_five":["P10","P11","P04","P03","P12"],"mbti":["P03","P10","P07","P09"],
 "enneagram":["P02","P11"],"temperament":["P04","P09","P11","P10"],"hexaco":["P10","P12","P06"],
 "pid5":["P11","P12","P07"],"truthfulness":["P06","P12","P11","P01"]}

def _tc(t): return f"{int(t//60):02d}:{int(t%60):02d}"

def auto_evidence(s, max_items=4):
    """Детерминированные скрин-кадры доказательств по релевантным паттернам каждой системы."""
    eps=s.get("episodes",[]); out={}
    for key,pats in SYS_PATTERNS.items():
        items=[]
        for pid in pats:
            for e in eps:
                if e["pattern"]!=pid: continue
                items.append({"episode_id":e["id"],"timecode":_tc(e["t0"])+"-"+_tc(e["t1"]),
                              "rationale":f"[auto] маркер {pid} ({e['name']}), окно {e['t0']}-{e['t1']} c, {e['frames']} кадров — вход композита системы"})
                if len(items)>=max_items: break
            if len(items)>=max_items: break
        out[key]=items
    return out

def enrich_parsed(parsed, s, min_items=2, max_items=6):
    """Если модель дала мало цитат — добираем детерминированные скрин-кадры."""
    if not isinstance(parsed,dict): return parsed
    ev=parsed.get("evidence")
    if not isinstance(ev,dict): ev={}; parsed["evidence"]=ev
    ae=auto_evidence(s,max_items)
    for key,auto in ae.items():
        cur=[x for x in ev.get(key,[]) if isinstance(x,dict)]
        have={x.get("episode_id") for x in cur}
        for a in auto:
            if len(cur)>=min_items: break
            if a["episode_id"] in have: continue
            cur.append(a); have.add(a["episode_id"])
        ev[key]=cur[:max_items]
    return parsed
'''
(ROOT/"scoring.py").write_text(sc,encoding="utf-8")
print("[OK] scoring: auto_evidence + enrich_parsed")

# --- 2) llm.py: контракт без дисклеймера, с обязательными цитатами ---
l=(ROOT/"llm.py").read_text(encoding="utf-8")
l=rep(l,' "disclaimer": "гипотезы по невербальным маркерам; не клинический диагноз и не установленная правдивость",\n','','schema disclaimer')
i=l.index('SYSTEM_PROMPT=('); j=l.index('DEFAULT_USER_PROMPT=')
NEW_PROMPT='''SYSTEM_PROMPT=(
 "Ты — эксперт по поведенческой диагностике PersonaScope. Вход: JSON измерений MediaPipe (мимика, поза, руки, взгляд, "
 "речь, голос), счётчики паттернов P01-P12, список эпизодов с таймкодами, детерминированные скоринги scores_block и формула truth_heuristic.\\n"
 "КОНТРАКТ:\\n"
 "1) Числовые скоринги систем и truthfulness.score вычислены детерминированным модулем (scores_block): скопируй их в поля ответа без изменений.\\n"
 "2) summary и notes = СОДЕРЖАТЕЛЬНАЯ интерпретационная заключительная оценка: какие маркеры (паттерны, каналы, эпизоды) "
 "поддерживают каждый скор; поведенческие гипотезы причинно-следствия; динамика по видео. Пиши конкретно: счётчики, окна, каналы.\\n"
 "3) evidence: для КАЖДОЙ системы процитируй >=2 эпизода из входного списка (episode_id + timecode + rationale <=20 слов); "
 "при слабых маркерах цитируй эпизоды, вошедшие в композит. Пустой evidence = нарушение контракта.\\n"
 "4) В summary/notes/verdict ЗАПРЕЩЕНЫ: рассуждения о лицензиях, профессионализме, «алгоритм не обосновывает», "
 "«не является диагнозом», воспроизведение дисклеймеров и методологических оговорок — такие тексты в отчёт не входят.\\n"
 "5) truthfulness: score из scores_block; verdict и cues — текстом по маркерам (моргание, избегание, руки у лица, напряжение, темп речи, паузы, f0).\\n"
 "6) MBTI: >50 по оси = полюс; Enneagram 1..9 с крылом; Temperament — проценты, сумма ~100.\\n"
 "7) Вывод: СТРОГО один валидный JSON без markdown по схеме; rationale <=20 слов."
)

'''
l=l[:i]+NEW_PROMPT+l[j:]
l=rep(l,'base+=" В evidence всех систем верни пустые списки: только scores, summary, truthfulness, disclaimer."',
        'base+=" В evidence всех систем верни пустые списки: только scores, summary, truthfulness."','force disclaimer')
(ROOT/"llm.py").write_text(l,encoding="utf-8")
print("[OK] llm: контракт = анализ и цитаты, без дисклеймеров")

# --- 3) app.py: enrich после валидации ---
a=(ROOT/"app.py").read_text(encoding="utf-8")
a=rep(a,'            parsed,w=llm.validate_result(parsed,s); warnings+=w',
        '            parsed,w=llm.validate_result(parsed,s); warnings+=w\n            parsed=scoring.enrich_parsed(parsed,s)','app enrich')
(ROOT/"app.py").write_text(a,encoding="utf-8")
print("[OK] app: доказательства добираются детерминированно")

# --- 4) index.html: убрать дисклеймер из UI/PDF + fallback на auto_evidence ---
h=(ROOT/"static"/"index.html").read_text(encoding="utf-8")
h=rep(h,'${p.disclaimer?`<div class="lbl" style="margin-top:6px">${p.disclaimer}</div>`:""}','','ui disclaimer')
h=rep(h,'${p.disclaimer?`<p><i>${p.disclaimer}</i></p>`:""}','','pdf disclaimer')
h=rep(h,'    const ev=(p&&p.evidence)?(p.evidence[k]||[]):[];',
        '    const ev=((p&&p.evidence&&(p.evidence[k]||[]).length)?p.evidence[k]:((sc&&sc.auto_evidence)?(sc.auto_evidence[k]||[]):[]));','ev fallback auto')
(ROOT/"static"/"index.html").write_text(h,encoding="utf-8")
print("[OK] ui: дисклеймер убран, скрины кадров = evidence модели ИЛИ auto-evidence")

for f in ("scoring.py","llm.py","app.py"):
    py_compile.compile(str(ROOT/f),doraise=True); print(f"[OK] {f} компилируется")
r=subprocess.run([sys.executable,"-c","import app; print('[OK] import app')"],cwd=str(ROOT),capture_output=True,text=True)
print(r.stdout.strip() or r.stderr.strip()[-600:])
sys.exit(0 if r.returncode==0 else 1)