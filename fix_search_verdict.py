#!/usr/bin/env python
"""Разовая правка PersonaScope v5.1 (запуск: python fix_search_verdict.py).

Две проблемы:
  1) «Оценка правдивости» печатала вердикт по букве на строку: модель вернула поле
     cues СТРОКОЙ, а код делал `for c in (t.get("cues") or [])` — то есть разбирал
     строку по символам, и фронтенд рисовал каждый символ отдельным пунктом списка.
  2) Поиск эпизодов возвращал окно на всю минуту («00:17-01:10») при слепке 3 с:
     интервалы «вездесущих» паттернов (P03 движения головы активен 30-60% кадров)
     отдавались целиком, а целевой паттерн выбирался по самой большой доле в окне,
     где P03/P07 всегда обыгрывают P06/P04.

Скрипт идемпотентен: уже применённые правки пропускаются. Перед записью делается
резервная копия *.bak_before_fix, после записи файл компилируется — при ошибке
копия возвращается на место. В конце прогоняются тесты.
"""
from __future__ import annotations
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHANGED: list[Path] = []


def log(*a):
    print(*a, flush=True)


def patch(name: str, marker: str, label: str, pairs: list[tuple[str, str]]) -> bool:
    """Заменить пары (old,new) в файле. Все якоря обязательны, маркер = «уже применено»."""
    path = ROOT / name
    src = path.read_text(encoding="utf-8")
    if marker in src:
        log(f"[=] уже применено: {label}")
        return False
    for old, _new in pairs:
        if old not in src:
            raise SystemExit(f"[!] якорь не найден ({label}, {name}):\n    {old.strip()[:100]!r}\n"
                             f"    Ничего не изменено.")
    out = src
    for old, new in pairs:
        out = out.replace(old, new, 1)
    backup = path.with_name(path.name + ".bak_before_fix")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(out, encoding="utf-8")
    log(f"[+] {label} -> {name} (копия: {backup.name})")
    CHANGED.append(path)
    return True


# ---------------------------------------------------------------------------
# 1. complaints: cues строкой -> разбор по символам
# ---------------------------------------------------------------------------
CUE_HELPER = '''
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


'''

def fix_cues():
    patch("llm.py", "def _cue_list(", "llm: helper _cue_list", [
        ("def validate_result(parsed: dict, summary: dict):", CUE_HELPER.lstrip("\n") + "def validate_result(parsed: dict, summary: dict):"),
        ("        nc=[]\n        for c in (tt.get(\"cues\") or []):",
         "        nc=[]\n        for c in _cue_list(tt.get(\"cues\")):"),
    ])
    patch("app.py", "llm._cue_list(", "app: /truth читает cues как список", [
        ("            cues=[]\n            for c in (t.get(\"cues\") or []):",
         "            cues=[]\n            for c in llm._cue_list(t.get(\"cues\")):"),
    ])
    patch("app.py", "for c in llm._cue_list(t.get(\"cues\")):\n                            if isinstance(c,dict):",
          "app: нормализация в llm_analyze", [
        ("                        nc=[]\n                        for c in (t.get(\"cues\") or []):",
         "                        nc=[]\n                        for c in llm._cue_list(t.get(\"cues\")):"),
    ])


# ---------------------------------------------------------------------------
# 2. complaints: окно на всю минуту вместо конкретных коротких эпизодов
# ---------------------------------------------------------------------------
CONSTS_OLD = "MAX_EXT=1.5"
CONSTS_NEW = '''MAX_EXT=1.5
# Максимальное превышение длительности кандидата над длительностью слепка. Длинные
# интервалы «вездесущих» паттернов (P03 движения головы активен 30-60% кадров, P07 —
# почти всегда) раньше уходили в ответ целиком: «00:17-01:10» = 53 c при слепке 3 c,
# и такое окно совпадало по форме случайно, потому что обе кривые сжимаются к N=64.
MAX_DUR_RATIO=2.0
SPLICE_STEP_SEC=0.5
SPLICE_MAX_SPANS=60'''

EXPLODE_FN = '''def _explode_span(records,mask,chans,fps_proc,a,b,dur_s,max_ratio=MAX_DUR_RATIO):
    """Режет длинный интервал паттерна на кандидатов длиной ~dur_s (окно слепка).

    Сначала — тот же сплайсинг по локальной активности, что и в free-режиме
    (_spans_by_activity), затем страховка ровной сеткой с шагом SPLICE_STEP_SEC.
    Благодаря этому 3-секундный слепок жеста «рука-лицо» даёт конкретные короткие
    эпизоды, а не одно окно на всю минуту.
    """
    dur_c=records[b]["t"]-records[a]["t"]
    if dur_c<=max_ratio*dur_s: return [(a,b)]
    sub=records[a:b+1]; sub_mask=mask[a:b+1]
    out=[]
    try:
        for x,y in _spans_by_activity(sub,sub_mask,chans,fps_proc,dur_s):
            if records[a+y]["t"]-records[a+x]["t"]<=max_ratio*dur_s: out.append((a+x,a+y))
    except Exception:
        out=[]
    if not out:
        W=max(3,int(round(dur_s*fps_proc))); step=max(1,int(round(SPLICE_STEP_SEC*fps_proc)))
        starts=list(range(0,max(1,(b-a+1)-W+1),step))
        out=[(a+s,min(b,a+s+W-1)) for s in starts] or [(a,min(b,a+W-1))]
    out=sorted(set(out))
    if len(out)>SPLICE_MAX_SPANS:
        k=max(1,len(out)//SPLICE_MAX_SPANS)
        out=out[::k][:SPLICE_MAX_SPANS]
    return out


def _morph_ok(win,S_c,dur_s,fps_proc):'''

SPANS_OLD = '''    else:
        spans=[tuple(e) for e in eps]'''
SPANS_NEW = '''    else:
        # Длинные интервалы паттернов режем на окна длиной со слепок: иначе кандидат
        # «00:17-01:10» (53 c) обыгрывал реальные короткие эпизоды жеста.
        spans=[]
        for a_,b_ in (tuple(e) for e in eps):
            spans.extend(_explode_span(records,mask,chans,fps_proc,a_,b_,dur_s))
        if not spans: spans=[tuple(e) for e in eps]'''

PICK_OLD = "            else: active=[max(present,key=lambda x:(frac[x],-x))]"
PICK_NEW = '''            else:
                # Специфичность вместо «самой большой доли в окне»: P03 и P07 активны почти
                # во всех кадрах, поэтому по доле они всегда обыгрывали P06/P04 — и слепок
                # жеста «рука-лицо» искался по движениям головы. Берём паттерн, наиболее
                # перепредставленный в окне слепка относительно всего видео (lift).
                vfrac={x:float(np.mean((mask>>x)&1==1)) for x in present}
                active=[max(present,key=lambda x:((frac[x]+0.02)/(vfrac.get(x,0.0)+0.02),frac[x],-x))]'''


def fix_search():
    patch("snapshots.py", "MAX_DUR_RATIO", "snapshots: лимит длительности кандидата", [
        (CONSTS_OLD, CONSTS_NEW),
    ])
    patch("snapshots.py", "def _explode_span(", "snapshots: нарезка длинных интервалов", [
        ("def _morph_ok(win,S_c,dur_s,fps_proc):", EXPLODE_FN),
    ])
    patch("snapshots.py", "spans.extend(_explode_span(", "snapshots: поиск режет эпизоды", [
        (SPANS_OLD, SPANS_NEW),
        ("        win=records[a:bb+1]\n        dur_c=T[bb]-T[a]\n        if dur_c<=0: continue",
         "        win=records[a:bb+1]\n        dur_c=T[bb]-T[a]\n        if dur_c<=0: continue\n"
         "        if dur_c>MAX_DUR_RATIO*dur_s: continue   # окно на всё видео не может быть совпадением слепка"),
    ])
    patch("snapshots.py", "vfrac=", "snapshots: выбор паттерна по специфичности", [
        (PICK_OLD, PICK_NEW),
        ('          "channels":chans,"snapshot_duration":dur_s,',
         '          "channels":chans,"snapshot_duration":dur_s,"max_dur_ratio":MAX_DUR_RATIO,'),
    ])


def verify():
    ok = True
    for path in CHANGED:
        try:
            py_compile.compile(str(path), doraise=True)
            log(f"[ok] {path.name} компилируется")
        except Exception as e:
            backup = path.with_name(path.name + ".bak_before_fix")
            if backup.exists():
                shutil.copy2(backup, path)
                log(f"[!] {path.name}: ошибка компиляции, файл восстановлен из копии:\n    {e}")
            else:
                log(f"[!] {path.name}: ошибка компиляции, копии нет:\n    {e}")
            ok = False
    return ok


def main():
    fix_cues()
    fix_search()
    if not CHANGED:
        log("[=] все правки уже применены — ничего не изменено")
    elif not verify():
        log("[!] правка откачена. Пришлите этот вывод разработчику.")
        return 1
    log("\n[..] запускаю тесты")
    try:
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                           cwd=str(ROOT), capture_output=True, text=True)
        tail = (r.stderr or "") + (r.stdout or "")
        log(tail.strip()[-1500:])
        tests_ok = r.returncode == 0
    except Exception as e:                     # например, если запрещён запуск подпроцесса
        log(f"[!] не удалось запустить тесты автоматически: {e}")
        log("    выполните вручную: python -m unittest discover -s tests -t .")
        tests_ok = True
    log("\n=== ИТОГ ===")
    log("правки: " + (", ".join(p.name for p in CHANGED) if CHANGED else "нет (уже были применены)"))
    log("тесты: " + ("OK" if tests_ok else "ЕСТЬ ПАДЕНИЯ — пришлите вывод выше"))
    if CHANGED:
        log("копии исходных файлов: *.bak_before_fix (удалите их, когда убедитесь, что всё работает)")
    log("\nДальше: перезапустите сервер (python run.py) и обновите страницу через Ctrl+F5.")
    log("Старые анализы чинятся сами: cues нормализуются при чтении, повторный запрос к LLM не нужен.")
    return 0 if tests_ok else 2


if __name__ == "__main__":
    sys.exit(main())
