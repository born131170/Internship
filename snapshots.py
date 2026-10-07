"""Цифровые слепки v5.2: оцифровка жестового эпизода и поиск аналогий.
Только измерения MediaPipe; LLM в этом контуре не участвует.
Поиск строго соответствует конвейеру pipeline.PATTERN_DEFS и pipeline.touch_* (единый источник истины).
search() возвращает (results, meta) — совместимо с app.py.

Изменения v5.2 (по итогам разметки владельца проекта):
  * касание рукой лица считается геометрией «кисть -> меш лица» (pipeline.touch_frame),
    а не AND из порогов по осям, который пропускал 1.13% кадров и давал 0 находок в строгом режиме;
  * кандидаты строгого режима берутся из эпизодов касания, а не из маски P06;
  * морфология формы кривой стала мягким штрафом вместо жёсткого отсева;
  * при поиске в ДРУГОМ видео окно слепка не пересчитывается по таймкодам источника;
  * добавлен порог «совпадение / фон» (нулевое распределение) — поиск умеет отвечать «не найдено».
"""
from __future__ import annotations
import math, random, time, uuid, zlib
import numpy as np

SNAP_VERSION="5.2"
DEFAULT_CHANNELS=["hand_speed","aperture","hfd","hfd2","yc","pc","menergy"]
N=64
MAX_EXT=1.5
# Максимальное превышение длительности кандидата над длительностью слепка. Длинные
# интервалы «вездесущих» паттернов (P03 движения головы активен 30-60% кадров, P07 —
# почти всегда) раньше уходили в ответ целиком: «00:17-01:10» = 53 c при слепке 3 c,
# и такое окно совпадало по форме случайно, потому что обе кривые сжимаются к N=64.
MAX_DUR_RATIO=2.0
SPLICE_STEP_SEC=0.5
SPLICE_MAX_SPANS=60
# Мягкий штраф за несоответствие формы кривой (вместо жёсткого отсева кандидата)
MORPH_PENALTY=0.6
# Сколько случайных окон брать для нулевого распределения и какой его квантиль считать порогом.
# 20-й процентиль выбран по разметке владельца проекта (tools/eval_search.py --grid):
# precision 0.90 при recall вдвое выше, чем у 5-го процентиля.
NULL_WINDOWS=120
NULL_PERCENTILE=20.0
# Каналы, которые сравниваются в абсолютных единицах (касание руки и лица), а не по форме
_ABS_CHANNELS=("touch","twrist")
# Минимальная доля кадров касания в окне слепка, чтобы считать слепок жестом «рука-лицо»
SNAP_TOUCH_RATIO=0.30
# Минимальная доля кадров касания в окне-кандидате (режим авто-касания)
CAND_TOUCH_RATIO=0.15

# Единый источник определений паттернов — pipeline.PATTERN_DEFS
# (научно операционализированные пороги, Ekman/FACS, Kaitz 2007, Vismara 2016 и др.).
# Локальные дубли-лямбды удалены: именно они считали NaN как "активно"
# (r.get("hfd",1)<0.8 при пропущенном кадре давало ложные P02/P03/P06/P08/P11/P12)
# и расходились с порогом P12 в конвейере (0.2 рад вместо radians(35)).
# Оттуда же берутся геометрия касания и пороги: дублировать их нельзя (иначе слои разъедутся).
_TOUCH_TAU=0.5; _TWRIST_TAU=2.6; _touch_spans=None; _touch_ratio=None; _touch_frame=None
try:
    from pipeline import PATTERN_DEFS as _PD, _n as _pn, _nn as _pnn
    try:
        from pipeline import (touch_spans as _touch_spans, touch_ratio as _touch_ratio,
                              touch_frame as _touch_frame,
                              TOUCH_MAX_TAU as _TOUCH_TAU, TWRIST_MAX_TAU as _TWRIST_TAU)
    except Exception:
        pass
    PAT_IDS=[p[0] for p in _PD]
    PAT_DEFS=[(pid, lambda r, f=p[2]: bool(f({k:(np.nan if v is None else v)
                                               for k,v in r.items()})))
              for p in _PD]
except Exception:  # автономный режим: эквивалентный fallback с NaN-safe порогами
    PAT_IDS=["P01","P02","P03","P04","P05","P06","P07","P08","P09","P10",
             "P10D","P11","P12","P13","P14"]
    def _g(r,c):
        v=r.get(c)
        return np.nan if v is None else float(v)
    PAT_DEFS=[
     ("P01",lambda r: _g(r,"blink")>0.6),
     ("P02",lambda r: abs(_g(r,"pc"))>0.08),
     ("P03",lambda r: abs(_g(r,"yc"))+abs(_g(r,"pc"))>0.03),
     ("P04",lambda r: _g(r,"hand_speed")>0.03),
     ("P05",lambda r: _g(r,"hand_speed")>0.1),
     ("P06",lambda r: ((_g(r,"tif")>0.5 or _g(r,"hfd")<0.8) and (_g(r,"hands") or 0)>0)),
     ("P07",lambda r: _g(r,"hand_speed")>0.01),
     ("P08",lambda r: _g(r,"aperture")<0.8),
     ("P09",lambda r: _g(r,"menergy")>0.02),
     ("P10",lambda r: _g(r,"smile")>0.4),
     ("P10D",lambda r: _g(r,"smile")>0.35 and _g(r,"cheek")>0.15),
     ("P11",lambda r: _g(r,"browDown")>0.3 or _g(r,"press")>0.2),
     ("P12",lambda r: _g(r,"yc")>math.radians(35) or _g(r,"pc")>math.radians(35)),
     ("P13",lambda r: bool(r.get("speech"))),
     ("P14",lambda r: _g(r,"pause")>=0.5),
    ]

def _isnan(v): return v is None or (isinstance(v,float) and math.isnan(v))

def _resample(vals,n=N):
    vals=np.asarray(vals,dtype=float)
    idx=np.where(~np.isnan(vals))[0]
    if len(idx)==0: return np.zeros(n)
    if len(idx)==1: return np.full(n,vals[idx[0]])
    return np.interp(np.linspace(0,len(vals)-1,n),idx,vals[idx])

def _znorm(a):
    a=np.asarray(a,dtype=float); s=a.std()
    return (a-a.mean())/(s if s>1e-9 else 1.0)

def dtw_dist(a,b,band=12):
    n=len(a); m=len(b); INF=float("inf")
    D=np.full((n+1,m+1),INF); D[0][0]=0.0
    for i in range(1,n+1):
        lo=max(1,i-band); hi=min(m,i+band)
        ai=a[i-1]
        for j in range(lo,hi+1):
            c=(ai-b[j-1])**2
            D[i][j]=c+min(D[i-1][j],D[i][j-1],D[i-1][j-1])
    return math.sqrt(D[n][m]/max(n,m))

def _pattern_mask(records):
    out=np.zeros(len(records),dtype=np.int64)
    # P01 (бит 0) — единый с конвейером детектор морганий (переходы+рефрактерность),
    # а не сырой порог blink>0.6: при stride-сэмплинге blendshape залипает и raw-порог
    # даёт вечную активность/0 событий в зависимости от видео.
    try:
        from pipeline import blink_events as _be
        bl=set(_be(records)) if len(records)>3 else set()
    except Exception:
        bl=None
    for i in range(len(records)):
        r=records[i]; m=0
        for b in range(len(PAT_DEFS)):
            pid=PAT_DEFS[b][0]
            if pid=="P01":
                if bl is not None:
                    if i in bl: m|=(1<<b)
                else:
                    try:
                        if PAT_DEFS[b][1](r): m|=(1<<b)
                    except Exception: pass
                continue
            try:
                if PAT_DEFS[b][1](r): m|=(1<<b)
            except Exception: pass
        out[i]=m
    return out

def _mask_of_window(mask):
    if len(mask)==0: return 0
    return int(np.bitwise_or.reduce(mask))

# Множитель доли активных кадров окна, при котором паттерн считается активным в слепке.
# Было bitwise-OR по всему окну: даже единичный ложный кадр (или случайный шумовой пик)
# помечал паттерн активным на всё 2–3-секундное окно — отсюда "веер" из 10 паттернов
# в слепке чистого жеста и выбор целевым паттерном редчайшего P11 с 0 эпизодами.
FRAC_MIN=0.5

def _window_frac(mask):
    """Доля кадров окна для каждого паттерна."""
    n=len(mask)
    if n==0: return np.zeros(len(PAT_DEFS))
    return np.array([np.mean((mask>>b)&1==1) for b in range(len(PAT_DEFS))])

def _active_patterns(mask,frac_min=FRAC_MIN):
    fr=_window_frac(mask)
    return [b for b in range(len(PAT_DEFS)) if fr[b]>=frac_min]

def _dominant_patterns(mask,frac_min=FRAC_MIN):
    """Доминирующие паттерны окна: доля >= frac_min, либо (если таких нет)
    локальный максимум долей > 0. Иначе чистый жестовой слепок давал бы пустой
    pattern_active и поиск скатывался в free-режим по шумовому паттерну."""
    fr=_window_frac(mask)
    act=[b for b in range(len(PAT_DEFS)) if fr[b]>=frac_min]
    if not act and fr.max()>0:
        act=[int(np.argmax(fr))]
    return act

def _activity(r):
    hs=r.get("hand_speed")
    return (not _isnan(hs)) and hs>0.01

def _segment(records,a,b):
    n=len(records)
    dt=(records[-1]["t"]-records[0]["t"])/max(1,n-1)
    cap=max(1,int(MAX_EXT/dt))
    i=a; k=0
    while i>0 and k<cap and _activity(records[i-1]): i-=1; k+=1
    j=b; k=0
    while j<n-1 and k<cap and _activity(records[j+1]): j+=1; k+=1
    return i,j

def _valid_channels(win,channels):
    valid=[]
    for c in channels:
        v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
        nz=float(np.mean(~np.isnan(v)))
        sd=float(np.nanstd(v)) if nz>0 else 0.0
        if nz>=0.6 and sd>1e-6: valid.append(c)
    return valid or list(channels[:1])

def _series_of(win,c):
    v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
    # Каналы касания храним в АБСОЛЮТНЫХ единицах (межзрачковые расстояния): для жеста
    # «рука у лица» важна реальная близость, а не форма нормированной кривой. Остальные
    # каналы — как раньше, по форме (z-нормировка убирает масштаб и смещение).
    if c in _ABS_CHANNELS:
        return _resample(v)
    return _znorm(_resample(v))

def _channel_distance(v,S_c,c):
    """Расстояние по одному каналу между окном-кандидатом и эталоном слепка.

    Для touch/twrist — RMSE в абсолютных единицах (в шаблоне они тоже лежат без нормировки),
    для остальных каналов — 0.3*евклид + 0.7*DTW по z-нормированным кривым.
    Раньше z-нормировались ВСЕ каналы, из-за чего «рука далеко» и «рука у лица» могли дать
    одинаковую форму и попасть в результаты поиска.
    """
    if c in _ABS_CHANNELS:
        cs=_resample(v); Sc=_resample(S_c)
        if np.all(np.isnan(cs)) or np.all(np.isnan(Sc)):
            return None
        return float(np.sqrt(np.nanmean((cs-Sc)**2)))
    cs=_znorm(_resample(v)); Sc=_znorm(_resample(S_c))
    return 0.3*float(np.sqrt(np.mean((cs-Sc)**2)))+0.7*dtw_dist(cs,Sc)

def create_snapshot(records,t0,t1,channels,name):
    idx=[i for i,r in enumerate(records) if t0<=r["t"]<=t1]
    if len(idx)<3: raise ValueError("окно слишком короткое (нужно >=3 кадров)")
    a0,b0=_segment(records,idx[0],idx[-1])
    win=records[a0:b0+1]
    chans=_valid_channels(win,channels or DEFAULT_CHANNELS)
    series={c:_series_of(win,c).tolist() for c in chans}
    stats={}
    for c in chans:
        v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
        stats[c]={"mean":float(np.nanmean(v)),"std":float(np.nanstd(v))}
    pm=_pattern_mask(win)
    act=_dominant_patterns(pm)
    frac={PAT_DEFS[k][0]:round(float(np.mean((pm>>k)&1==1)),3) for k in range(len(PAT_DEFS))}
    wm=0
    for k in act: wm|=(1<<k)
    # Касание фиксируем прямо в слепке: при поиске в ЧУЖОМ видео это единственный корректный
    # источник (таймкоды t0/t1 принадлежат источнику и в целевом видео ничего не значат).
    dur_win=(win[-1]["t"]-win[0]["t"]) or 1.0
    fps_win=(len(win)-1)/dur_win if dur_win>0 else 1.0
    tsp=_touch_spans_or_none(win,fps_win) or []
    touch_eps=[{"t0":round(win[a]["t"],2),"t1":round(win[b]["t"],2),"frames":b-a+1} for a,b in tsp]
    touch_ratio_win=round(float(sum(b-a+1 for a,b in tsp)/max(1,len(win))),4)
    return {"id":uuid.uuid4().hex[:10],"name":name,"created":int(time.time()),
            "channels":chans,"duration":round(records[b0]["t"]-records[a0]["t"],2),
            "t0":round(records[a0]["t"],2),"t1":round(records[b0]["t"],2),
            "t0_req":t0,"t1_req":t1,"n":N,"series":series,"stats":stats,
            "pattern_mask":wm,
            "pattern_active":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if wm&(1<<x)],
            "pattern_frac":{k:v for k,v in frac.items() if v>0},
            "touch_ratio":touch_ratio_win,
            "touch_episodes":touch_eps,
            "version":SNAP_VERSION}

def _episodes_of_pattern(mask,b,fps_proc):
    GAP=max(1,int(0.4*fps_proc)); MIN=max(2,int(0.6*fps_proc))
    bit=1<<b; ints=[]; st=None; last=-10
    for i in range(len(mask)):
        on=bool(mask[i]&bit)
        if on:
            if st is None: st=i; last=i
            elif i-last>GAP: ints.append([st,last]); st=i; last=i
            else: last=i
    if st is not None: ints.append([st,last])
    return [ints[k] for k in range(len(ints)) if ints[k][1]-ints[k][0]>=MIN]

def _spans_by_activity(records,mask,chans,fps_proc,dur_s,min_frac=0.15,max_mult=3.0):
    """Сплайсинг длинного окна на отдельные ЭПИЗОДЫ по локальной активности каналов слепка.

    Исправляет «одно окно 53 с вместо трёх эпизодов по 1–2 с»: раньше в режиме free
    окно всегда равнялось полной длительности слепка. Теперь внутри скользящего окна
    ищем сегменты с аномальной активностью (z-score > 1 по |diff| или std канала),
    склеиваем их с окном ±0.4 с и отдаём как независимые кандидаты. Если выраженных
    всплесков нет — возвращаем исходное окно целиком (честно, без выдумывания)."""
    W=max(3,int(round(dur_s*fps_proc)))
    pad=int(round(0.4*fps_proc))
    lo_cut=min(1.0,0.35*dur_s); hi_cut=max(2.0,2.0*dur_s)
    out=[]
    for start in range(0,max(1,len(records)-W+1)):
        win=records[start:start+W]
        mseg=mask[start:start+W]
        segs=[]
        for c in chans:
            v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
            ok=~np.isnan(v)
            if ok.sum()<max(3,int(0.3*len(v))): continue
            vv=v[ok]; idx=np.where(ok)[0]
            d=np.abs(np.diff(vv)); s=float(np.nanstd(vv))
            med_d=float(np.median(d)); mad_d=float(np.median(np.abs(d-med_d))) or 1e-9
            z_thr=med_d+3.0*mad_d
            act=(d>max(z_thr,s))|(d>2.0*med_d)
            hits=[int(idx[i]) for i in range(len(act)) if bool(act[i])]
            cur=None; last_i=-10
            for h in hits:
                if cur is None: cur=[h,h]
                elif h-last_i<=2*pad: cur[1]=h
                else: segs.append(tuple(cur)); cur=[h,h]
                last_i=h
            if cur is not None: segs.append(tuple(cur))
        merged=[]
        for a,b in sorted(segs):
            a2=max(0,a-pad); b2=min(W-1,b+pad)
            if merged and a2<=merged[-1][1]+1: merged[-1]=(merged[-1][0],max(merged[-1][1],b2))
            else: merged.append((a2,b2))
        # фильтр по длительности эпизода: не короче min(lo_cut, 0.3·dur) и не длиннее max(hi_cut, dur)
        min_len=max(2,int(round(min(lo_cut,0.3*dur_s)*fps_proc)))
        max_len=max(min_len+1,int(round(max(hi_cut,dur_s)*fps_proc)))
        frac_ok=int(round(min_frac*W))
        for a2,b2 in merged:
            if b2-a2+1<min_len: continue
            if b2-a2+1>max_len: continue
            if sum(bool(mseg[i]) for i in range(a2,b2+1))<frac_ok: continue
            out.append((start+a2,start+b2))
    # дедупликация перекрывающихся спанов (окна с шагом 1 кадр дают ~дублей)
    uniq=[]
    for a,b in sorted(set(out)):
        if uniq and a<=uniq[-1][1]: continue
        uniq.append((a,b))
    if not uniq:
        for start in range(0,max(1,len(records)-W+1)):
            uniq.append((start,start+W-1))
    return uniq[:800]

def _explode_span(records,mask,chans,fps_proc,a,b,dur_s,max_ratio=MAX_DUR_RATIO):
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


def _morph_ok(win,S_c,dur_s,fps_proc):
    """Морфологическая проверка формы кривой окна против слепка (аффинно-инвариантно)."""
    v=np.asarray([x for x in win if not _isnan(x)],dtype=float)
    if len(v)<max(3,int(0.4*fps_proc)): return False
    cs=_znorm(_resample(v))
    d_euc=float(np.sqrt(np.mean((cs-S_c)**2)))
    d_dtw=dtw_dist(cs,S_c)
    # пик внутри окна (жест имеет вершину), а не монотонный дрейф
    k=int(np.argmax(cs)); edge=max(k,len(cs)-1-k)
    has_peak=edge>=0.15*(len(cs)-1)
    return (d_euc<=1.7 and d_dtw<=1.35 and has_peak)

def _face_touch_gate(r):
    """Кадр касания рукой лица.

    Единственная реализация — pipeline.touch_frame: геометрия «кисть -> меш лица» в единицах
    межзрачкового расстояния, фолбэк по запястью позы и (для старых прогонов без каналов
    касания) прежняя оценка по осям. Дублировать правило здесь нельзя: именно расхождение
    копий порогов дало ситуацию, когда гейт пропускал 1.13% кадров и строгий режим
    возвращал ноль находок, хотя кисть внутри окна касания видна в 95.7% кадров.
    """
    if _touch_frame is None:
        return False
    try:
        return bool(_touch_frame(r))
    except Exception:
        return False

def _touch_spans_or_none(records,fps_proc):
    """Эпизоды касания по данным конвейера (None, если каналов касания нет)."""
    if _touch_spans is None:
        return None
    try:
        return list(_touch_spans(records,fps_proc))
    except Exception:
        return None

def _touch_ratio_of(records,fps_proc):
    """Доля кадров в эпизодах касания; для легаси-данных — доля кадров, прошедших старый гейт."""
    if _touch_ratio is not None:
        try:
            return float(_touch_ratio(records,fps_proc))
        except Exception:
            pass
    if not records:
        return 0.0
    return sum(1 for r in records if _face_touch_gate(r))/len(records)

def _null_distances(records,snap,chans,S,dur_s,fps_proc,n=NULL_WINDOWS):
    """Нулевое распределение: расстояния «шаблон против случайного окна».

    Нужно, чтобы отличать совпадение от фона. Раньше поиск всегда возвращал top-K, поэтому
    на шуме интерфейс показывал K «находок» (владелец проекта видел ~22 ложных эпизода).
    Длины окон берутся случайными в допустимом диапазоне (0.5x..2x слепка) и в расстояние
    входит тот же штраф за длительность, что и у кандидатов — иначе сравнение нечестное.
    Выборка детерминирована: seed из id слепка, повторный поиск даёт тот же порог.
    """
    total=len(records)
    if total<8 or not chans:
        return []
    # Seed из СОДЕРЖИМОГО шаблона, а не из случайного id: одинаковый слепок даёт одинаковый
    # порог, поэтому отчёты оценки воспроизводимы между запусками.
    try:
        seed=int(zlib.crc32(np.asarray(S[chans[0]],dtype=float).tobytes()))
    except Exception:
        seed=str(snap.get("id") or "seed")
    rnd=random.Random(seed)
    out=[]
    for _ in range(max(10,int(n))):
        ratio=1.0/MAX_DUR_RATIO + rnd.random()*(MAX_DUR_RATIO-1.0/MAX_DUR_RATIO)
        W=max(3,int(round(dur_s*fps_proc*ratio)))
        if total<=W+1: continue
        s=rnd.randrange(0,total-W+1)
        win=records[s:s+W]
        dur_c=win[-1]["t"]-win[0]["t"]
        if dur_c<=0: continue
        pen=0.15*abs(math.log(dur_c/dur_s))
        d=0.0; ok=0
        for c in chans:
            v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
            if len(v)==0 or np.all(np.isnan(v)): continue
            cd=_channel_distance(v,S[c],c)
            if cd is None: continue
            d+=cd; ok+=1
        if ok: out.append(d/ok+pen)
    return out

def search(records,snap,hop=None,top_k=100,refine=None,pattern=None,mode=None,**kwargs):
    if not records: return [],{"total":0,"version":SNAP_VERSION}
    chans=[c for c in snap.get("channels",[]) if c in snap.get("series",{})]
    if not chans: return [],{"total":0,"version":SNAP_VERSION}
    face_touch=bool(kwargs.get("face_touch")) or pattern=="P06"
    strict_ft = bool(kwargs.get("face_touch"))   # запрос пользователя: искать ТОЛЬКО касания рукой лица
    S={c:np.asarray(snap["series"][c],dtype=float) for c in chans}
    dur_s=float(snap.get("duration",1.0)) or 1.0
    T=[r.get("t",0.0) for r in records]
    dt=(T[-1]-T[0])/max(1,len(T)-1) or 0.08
    fps_proc=1.0/dt
    mask=_pattern_mask(records)
    req=int(snap.get("pattern_mask",0) or 0)
    active=[x for x in range(len(PAT_DEFS)) if req&(1<<x)]
    # Ищем в том же видео, откуда взят слепок, или в другом? Таймкоды t0/t1 принадлежат
    # ИСТОЧНИКУ: в чужом видео это произвольное место, поэтому окно по ним не пересчитывается
    # (иначе целевой паттерн выбирался по случайному участку — владелец проекта получал
    # поиск по P03 «движения головы» вместо касаний лица).
    target_video=kwargs.get("target_video")
    same_video = (target_video is None) or (snap.get("source_video") in (None,target_video))
    # Доли паттернов в окне слепка: для чужого видео берём сохранённые при создании слепка
    stored_frac={}
    for _pid,_v in (snap.get("pattern_frac") or {}).items():
        for _ix,_d in enumerate(PAT_DEFS):
            if _d[0]==_pid and isinstance(_v,(int,float)):
                stored_frac[_ix]=float(_v)
    if same_video:
        ra,br=_segment_bounds(records,snap.get("t0"),snap.get("t1"))
        winmask=_pattern_mask(records[ra:br+1]) if ra is not None else _pattern_mask(records)
        winfrac={x:float(np.mean((winmask>>x)&1==1)) for x in range(len(PAT_DEFS))}
    else:
        ra=br=None
        winmask=_pattern_mask(records)
        winfrac=dict(stored_frac)
    if (len(active)>6 or not active) and same_video:
        # "веер" v4 или пустая маска: пересчитываем по текущим определениям
        active=_dominant_patterns(winmask)
    elif not active:
        active=_dominant_patterns(winmask)
    # АВТОРЕЖИМ «касание лица»: если в ОКНЕ САМОГО СЛЕПКА заметная доля кадров — реальное
    # касание, слепок семантически является жестом рука->лицо, и строгий режим включается сам.
    # Для того же видео доля считается по кадрам, для чужого — берётся сохранённая в слепке.
    try:
        if not same_video:
            frac_touch=float(snap.get("touch_ratio") or 0.0)
        else:
            wwin=records[ra:br+1] if ra is not None and br is not None else records
            frac_touch=_touch_ratio_of(wwin,fps_proc)
    except Exception:
        frac_touch=0.0
    snap_is_touch=frac_touch>=SNAP_TOUCH_RATIO
    if snap_is_touch and not face_touch:
        face_touch=True
        meta_note="авто: окно слепка содержит >=30% кадров касания лица рукой"
    else:
        meta_note=""
    if snap_is_touch and not strict_ft:
        strict_ft=True
    if pattern:
        sel=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]==pattern]
        if sel: active=sel
    touch_spans_idx=[]
    if strict_ft:
        # Строгий режим ищет ТОЛЬКО касания рукой лица, поэтому кандидаты берутся из
        # геометрических эпизодов касания (pipeline.touch_spans с теми же порогами, что в UI),
        # а не из маски P06, построенной на порогах по осям.
        touch_spans_idx=_touch_spans_or_none(records,fps_proc) or []
        p06=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]=="P06"]
        if p06: active=p06
    # ДОМИНИРУЮЩИЙ паттерн по всему видео: если в окне слепка активен «веер» из
    # 10+ паттернов (старые слепки v4.0), выбор одного целевого паттерна по окну
    # слепка ненадёжен. Берём самый частотный паттерн во всём видео и строим
    # эпизоды по нему — это восстанавливает обычный поиск для не-touch слепков.
    global_dominant=None
    if not strict_ft and len(active)>3:
        try:
            freq={x:int(((mask>>x)&1).sum()) for x in range(len(PAT_DEFS))}
            present=[x for x in active if freq.get(x,0)>0]
            cand=present or [max(freq,key=lambda x:freq.get(x,0))]
            global_dominant=max(cand,key=lambda x:(freq.get(x,0),-x))
        except Exception:
            global_dominant=None
    if len(active)>1:
        # Было: выбирался САМЫЙ РЕДКИЙ активный паттерн (cnt[0]) — при "веере" это
        # приводило к P11 с 0 эпизодами и пустому поиску. Теперь приоритет:
        # 1) паттерны, присутствующие и в слепке, и в целевом видео (>0 эпизодов),
        #    среди них — доминирующий по доле кадров окна слепка;
        # 2) если ни один не найден в видео — самый частотный в самом слепке
        #    (честно: показать, что аналогий нет, а не искать несуществующий P11).
        frac={x:winfrac.get(x,0.0) for x in active}
        present=[x for x in active if (strict_ft or len(_episodes_of_pattern(mask,x,fps_proc))>0)]
        if present:
            # face_touch: касание лица (P06) имеет наивысший приоритет семантики запроса
            ft=[x for x in present if PAT_DEFS[x][0]=="P06"]
            if face_touch and ft: active=[ft[0]]
            else:
                # Специфичность вместо «самой большой доли в окне»: P03 и P07 активны почти
                # во всех кадрах, поэтому по доле они всегда обыгрывали P06/P04 — и слепок
                # жеста «рука-лицо» искался по движениям головы. Берём паттерн, наиболее
                # перепредставленный в окне слепка относительно всего видео (lift).
                vfrac={x:float(np.mean((mask>>x)&1==1)) for x in present}
                active=[max(present,key=lambda x:((frac[x]+0.02)/(vfrac.get(x,0.0)+0.02),frac[x],-x))]
        elif not strict_ft:
            active=[max(active,key=lambda x:(frac[x],-x))]
        # strict_ft + нет P06-эпизодов в видео: active остаётся = [P06] -> eps=[],
        # честный ответ 0 находок (см. ниже), без скатывания в free-скольжение
    b=active[0]; pid=PAT_DEFS[b][0]
    if global_dominant is not None and len(active)>1:
        # «веер» из окна слепка: целевой паттерн — самый частотный по всему видео
        b=global_dominant; pid=PAT_DEFS[b][0]; active=[b]
    # ЯВНЫЙ выбор целевого паттерна пользователем (payload.pattern): пересчёт доли
    # активности этого паттерна в окне слепка. Без этого длинный слепок (несколько
    # жестов в одном окне) выбирал шумовой P07 вместо осмысленного P06, и поиск
    # возвращал произвольные 20-секундные окна вместо реальных эпизодов жеста.
    if pattern:
        sel=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]==pattern]
        if sel:
            b=sel[0]; pid=PAT_DEFS[b][0]
            frac={x:winfrac.get(x,0.0) for x in active}
            frac[b]=winfrac.get(b,0.0)
            active=[b]
    eps=_episodes_of_pattern(mask,b,fps_proc)
    if strict_ft and touch_spans_idx:
        # строгий режим: ищем именно геометрические эпизоды касания
        eps=[tuple(x) for x in touch_spans_idx]
    if strict_ft:
        # в строгом режиме не скатываемся в free-скольжение: если P06-эпизодов нет,
        # честный ответ — 0 находок (а не шумовые окна без рук)
        mode="episodes"
    else:
        mode=mode or ("episodes" if eps else "free")
    cands=[]
    if mode=="free":
        # скользящее окно + СПЛАЙСИНГ по активности: длинное окно (например слепок
        # 53 с) режется на отдельные эпизоды-всплески по локальной активности каналов
        # слепка — иначе пользователь получает «1 находка = всё видео», хотя внутри
        # окна несколько коротких жестов по 1–2 с
        W=max(3,int(round(dur_s*fps_proc)))
        step=max(1,int(round(float(hop or 0.25)*fps_proc)))
        spans=_spans_by_activity(records,mask,chans,fps_proc,dur_s)
        # если сплайсинг вернул окна целиком (нет всплесков) — оставляем обычную сетку
        if not spans:
            starts=list(range(0,max(1,len(records)-W+1),step)) or [0]
            spans=[(s,min(len(records)-1,s+W-1)) for s in starts]
    else:
        # Длинные интервалы паттернов режем на окна длиной со слепок: иначе кандидат
        # «00:17-01:10» (53 c) обыгрывал реальные короткие эпизоды жеста.
        spans=[]
        for a_,b_ in (tuple(e) for e in eps):
            spans.extend(_explode_span(records,mask,chans,fps_proc,a_,b_,dur_s))
        if not spans: spans=[tuple(e) for e in eps]
    for k,(a,bb) in enumerate(spans):
        win=records[a:bb+1]
        dur_c=T[bb]-T[a]
        if dur_c<=0: continue
        if dur_c>MAX_DUR_RATIO*dur_s: continue   # окно на всё видео не может быть совпадением слепка
        # face_touch: в окне должен быть настоящий эпизод касания, а не 25% разрозненных кадров.
        # В строгом режиме окна уже являются эпизодами касания — повторная проверка не нужна.
        if face_touch and not strict_ft:
            if _touch_spans_or_none(win,fps_proc) is None:
                ntouch=sum(1 for r in win if _face_touch_gate(r))
                if ntouch < max(2,int(round(0.25*len(win)))): continue
            elif _touch_ratio_of(win,fps_proc) < CAND_TOUCH_RATIO:
                continue
        pen=0.15*abs(math.log(dur_c/dur_s))
        d=0.0; ok=0; morph_fail=False
        for c in chans:
            v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
            if len(v)==0 or np.all(np.isnan(v)): continue
            cd=_channel_distance(v,S[c],c)
            if cd is None: continue
            d+=cd; ok+=1
            if not _morph_ok(v,_znorm(_resample(S[c])),dur_s,fps_proc): morph_fail=True
        if ok==0: continue
        # Морфология формы — мягкий штраф вместо отсева: жёсткий AND (d_euc<=1.7 И d_dtw<=1.35
        # И наличие пика) отбрасывал верные касания с более короткой или шумной кривой.
        morph_pen=MORPH_PENALTY if (face_touch and morph_fail) else 0.0
        ft_frac=None
        if face_touch:
            ft_frac=round(_touch_ratio_of(win,fps_proc),2)
        item={"t0":round(T[a],2),"t1":round(T[bb],2),"frames":bb-a+1,
              "pattern":pid,"ep":k,"dur_ratio":round(dur_c/dur_s,2),
              "distance":round(d/ok+pen+morph_pen,4)}
        if ft_frac is not None: item["face_touch_frac"]=ft_frac
        cands.append(item)
    cands.sort(key=lambda x:x["distance"])
    # Порог «совпадение / фон»: те же расстояния на случайных окнах этой же длины.
    # Раньше всегда возвращался top-K, поэтому на шуме интерфейс показывал K «находок».
    thr=None; null_n=0; dropped=0
    if cands and kwargs.get("null_threshold",True):
        null_d=_null_distances(records,snap,chans,S,dur_s,fps_proc)
        if null_d:
            null_n=len(null_d)
            thr=float(np.percentile(null_d,float(kwargs.get("null_percentile",NULL_PERCENTILE))))
            passed=[c for c in cands if c["distance"]<=thr]
            dropped=len(cands)-len(passed)
            cands=passed
    keep=cands[:max(1,int(top_k))] if cands else []
    meta={"total":len(cands),"returned":len(keep),"version":SNAP_VERSION,"pattern":pid,
          "mode":mode,"episodes_of_pattern":len(eps),
          "patterns_in_snapshot":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if req&(1<<x)],
          "channels":chans,"snapshot_duration":dur_s,"max_dur_ratio":MAX_DUR_RATIO,
          "face_touch_gate":bool(face_touch),"strict_face_touch":bool(strict_ft),
          "snapshot_touch_frac":round(float(frac_touch),2),
          "cross_video":(not same_video),
          "touch_max_tau":_TOUCH_TAU,"twrist_max_tau":_TWRIST_TAU,
          "null_size":null_n,"match_threshold":(None if thr is None else round(thr,4)),
          "dropped_below_threshold":dropped}
    if strict_ft: meta["touch_episodes_in_video"]=len(touch_spans_idx)
    if thr is not None and not keep:
        meta["note"]=(meta.get("note","")+" " if meta.get("note") else "") + \
            f"совпадений выше порога не найдено (порог {thr:.3f} по {null_n} случайным окнам)"
    if meta_note: meta["note"]=meta_note
    return keep,meta

def _segment_bounds(records,t0,t1):
    """Индексы сегментированного окна слепка в произвольных метриках (для ресчета v4-слепков)."""
    if t0 is None or t1 is None: return None,None
    idx=[i for i,r in enumerate(records) if float(t0)<=r.get("t",0.0)<=float(t1)]
    if len(idx)<3: return None,None
    try:
        a,b=_segment(records,idx[0],idx[-1]); return a,b
    except Exception:
        return idx[0],idx[-1]
