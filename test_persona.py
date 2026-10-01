# test_persona.py — регрессионные проверки PersonaScope по разметке и инвариантам.
# Запуск при работающем сервере:
#   python test_persona.py                       (видео = последнее, разметка по умолчанию)
#   python test_persona.py <id_видео> 0:2 26:28 74.5:76.5
import json, sys, urllib.request
BASE="http://127.0.0.1:8000"
def get(p):
    with urllib.request.urlopen(BASE+p, timeout=900) as r: return json.loads(r.read().decode("utf-8"))
def post(p, body):
    req=urllib.request.Request(BASE+p, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r: return json.loads(r.read().decode("utf-8"))
def raw(p):
    with urllib.request.urlopen(BASE+p, timeout=900) as r: return r.read().decode("utf-8")

ok=True
def check(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ")+name+((" | "+detail) if detail else ""))
    ok = ok and bool(cond)

vid=sys.argv[1] if len(sys.argv)>1 else get("/api/videos/latest")["id"]
marks=[tuple(map(float,w.split(":"))) for w in sys.argv[2:]] or [(0,2),(26,28),(74.5,76.5)]
print("video:",vid," markup:",marks)

s=get(f"/api/videos/{vid}/summary"); met=get(f"/api/videos/{vid}/metrics")
check("summary+metrics загружаются", bool(s and met), f"{s['duration_sec']}s / {s['n_frames']} кадров")
check("metrics парсятся браузером (нет токена NaN)", "NaN" not in raw(f"/api/videos/{vid}/metrics"))
check("метрики свежие (pipeline_version=4.3-hybrid)", s.get("pipeline_version")=="4.3-hybrid",
      f"версия метрик: {s.get('pipeline_version')} — выполните: python fix_final2.py rerun {vid}")
check("method P06 зафиксирован", bool(s.get("method",{}).get("P06")), str(s.get("method",{}).get("P06",""))[:80])

eps=[e for e in s["episodes"] if e["pattern"]=="P06"]
print("P06 эпизоды summary:", [(e["t0"],e["t1"]) for e in eps])
for a,b in marks:
    hit=[e for e in eps if abs(e["t0"]-a)<=1.5 and abs(e["t1"]-b)<=1.5]
    check(f"разметка {a}-{b}s закрыта ровно 1 эпизодом P06", len(hit)==1, str([(e['t0'],e['t1']) for e in hit]))
check("P06: лишних эпизодов нет", len(eps)==len(marks), f"summary={len(eps)} markup={len(marks)}")

t0,t1=marks[0]
snap=post(f"/api/videos/{vid}/snapshots",{"t0":t0,"t1":t1,"name":"gt",
    "channels":["hfd","hfd2","tif","hand_speed","yc","pc","smile","blink"]})
check("слепок создан", bool(snap.get("id")), f"id={snap.get('id')} active={snap.get('pattern_active')}")
res=post(f"/api/videos/{vid}/search",{"snapshot_id":snap["id"],"top_k":100,"pattern":"P06"})
r=res.get("results",[]); m=res.get("meta",{})
check("search: целевой паттерн P06", m.get("pattern")=="P06", str(m.get("pattern")))
check("search: количество == счётчику P06", m.get("total")==len(eps), f"search={m.get('total')} summary={len(eps)}")
check("search: окна = реальные эпизоды (<=15 c)", all(x["t1"]-x["t0"]<=15 for x in r), str([(x['t0'],x['t1']) for x in r][:5]))
for a,b in marks:
    hit=[x for x in r if abs(x["t0"]-a)<=1.5 and abs(x["t1"]-b)<=1.5]
    check(f"search закрыл разметку {a}-{b}s", len(hit)==1)

tr=get(f"/api/videos/{vid}/truth")
check("правдивость: cues содержат аудио-метрики", "speech_ratio" in tr.get("cues",{}), str(list(tr.get("cues",{}).keys()))[:90])
print("ИТОГ:", "ALL PASS" if ok else "ЕСТЬ FAIL")
sys.exit(0 if ok else 1)
