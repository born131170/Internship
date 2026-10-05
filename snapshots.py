"""Цифровые слепки v5.0: корректная оцифровка жестового эпизода и поиск аналогий.
Только измерения MediaPipe; LLM в этом контуре не участвует.
Поиск строго соответствует конвейеру pipeline.PATTERN_DEFS (единый источник истины).
search() возвращает (results, meta) — совместимо с app.py."""
from __future__ import annotations
import math, time, uuid
import numpy as np

SNAP_VERSION="5.0"
DEFAULT_CHANNELS=["hand_speed","aperture","hfd","hfd2","yc","pc","menergy"]
N=64
MAX_EXT=1.5

# Единый источник определений паттернов — pipeline.PATTERN_DEFS
# (научно операционализированные пороги, Ekman/FACS, Kaitz 2007, Vismara 2016 и др.).
# Локальные дубли-лямбды удалены: именно они считали NaN как "активно"
# (r.get("hfd",1)<0.8 при пропущенном кадре давало ложные P02/P03/P06/P08/P11/P12)
# и расходились с порогом P12 в конвейере (0.2 рад вместо radians(35)).
try:
    from pipeline import PATTERN_DEFS as _PD, _n as _pn, _nn as _pnn
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
    return _znorm(_resample(v))

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
    return {"id":uuid.uuid4().hex[:10],"name":name,"created":int(time.time()),
            "channels":chans,"duration":round(records[b0]["t"]-records[a0]["t"],2),
            "t0":round(records[a0]["t"],2),"t1":round(records[b0]["t"],2),
            "t0_req":t0,"t1_req":t1,"n":N,"series":series,"stats":stats,
            "pattern_mask":wm,
            "pattern_active":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if wm&(1<<x)],
            "pattern_frac":{k:v for k,v in frac.items() if v>0},
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
    """Строгий кадр касания рукой лица (Kaitz 2007; Vismara 2016; Ekman FACS AU для рук).
    Требует РЕАЛЬНО детектированную кисть И подтверждение близости к лицу НЕ МЕНЕЕ ДВУХ
    независимых каналов одновременно: раньше достаточно было одного (например, ложного
    tif=1 при проецировании запястья в бокс лица поверх одежды — «руки нет, а тач есть»).
    Теперь: hands>0 + face_conf>=0.3 + минимум два из {tif, hfd<0.8, hfd2<0.55}."""
    try:
        hands=float(r.get("hands") or 0)
    except Exception:
        return False
    if hands<=0: return False
    fc=r.get("face_conf")
    if fc is not None and not _isnan(float(fc)) and float(fc)<0.3: return False
    hfd=r.get("hfd"); hfd2=r.get("hfd2"); tif=r.get("tif")
    signals=0
    if tif is not None and not _isnan(float(tif)) and float(tif)>0.5: signals+=1
    if hfd is not None and not _isnan(float(hfd)) and float(hfd)<0.8: signals+=1
    if hfd2 is not None and not _isnan(float(hfd2)) and float(hfd2)<0.55: signals+=1
    return signals>=2

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
    # Совместимость со слепками v4.0 (веер из-за bitwise-OR и NaN-дефолтов):
    # пересчитываем активные паттерны по текущим определениям pipeline.
    ra,br=_segment_bounds(records,snap.get("t0"),snap.get("t1"))
    winmask=_pattern_mask(records[ra:br+1]) if ra is not None else _pattern_mask(records)
    if len(active)>6 or not active:
        # "веер" v4 или пустая маска: пересчитываем по текущим определениям
        active=_dominant_patterns(winmask)
    # АВТОРЕЖИМ «касание лица» (исправление «находит эпизоды без рук/лица»):
    # если в ОКНЕ САМОГО СЛЕПКА доля кадров с реальным touch-gate >= 30%, слепок
    # семантически является жестом рука->лицо, и строгий P06-гейт включается
    # автоматически — даже когда пользователь НЕ отметил чекбокс строгого режима.
    try:
        wlen=max(1,(br-ra+1)) if ra is not None and br is not None else max(1,len(winmask))
        wwin=records[ra:br+1] if ra is not None and br is not None else records
        frac_touch=sum(1 for r in wwin if _face_touch_gate(r))/wlen
    except Exception:
        frac_touch=0.0
    snap_is_touch=frac_touch>=0.30
    if snap_is_touch and not face_touch:
        face_touch=True
        meta_note="авто: окно слепка содержит >=30% кадров касания лица рукой"
    else:
        meta_note=""
    if snap_is_touch and not strict_ft:
        # для корректной семантики жеста рука->лицо P06 обязателен и в слепке,
        # иначе целевые окна без лица проходят по одному лишь морфологическому сходству кривой hfd
        strict_ft=True
    if pattern:
        sel=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]==pattern]
        if sel: active=sel
    if strict_ft:
        # семантика запроса пользователя: искать ТОЛЬКО касания рукой лица (P06),
        # независимо от маски слепка (в т.ч. старых v4.0 с "веером")
        p06=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]=="P06"]
        if p06: active=p06
    if len(active)>1:
        # Было: выбирался САМЫЙ РЕДКИЙ активный паттерн (cnt[0]) — при "веере" это
        # приводило к P11 с 0 эпизодами и пустому поиску. Теперь приоритет:
        # 1) паттерны, присутствующие и в слепке, и в целевом видео (>0 эпизодов),
        #    среди них — доминирующий по доле кадров окна слепка;
        # 2) если ни один не найден в видео — самый частотный в самом слепке
        #    (честно: показать, что аналогий нет, а не искать несуществующий P11).
        frac={x:float(np.mean((winmask>>x)&1==1)) for x in active}
        present=[x for x in active if len(_episodes_of_pattern(mask,x,fps_proc))>0]
        if present:
            # face_touch: касание лица (P06) имеет наивысший приоритет семантики запроса
            ft=[x for x in present if PAT_DEFS[x][0]=="P06"]
            if face_touch and ft: active=[ft[0]]
            else: active=[max(present,key=lambda x:(frac[x],-x))]
        elif not strict_ft:
            active=[max(active,key=lambda x:(frac[x],-x))]
        # strict_ft + нет P06-эпизодов в видео: active остаётся = [P06] -> eps=[],
        # честный ответ 0 находок (см. ниже), без скатывания в free-скольжение
    b=active[0]; pid=PAT_DEFS[b][0]
    eps=_episodes_of_pattern(mask,b,fps_proc)
    if strict_ft:
        # в строгом режиме не скатываемся в free-скольжение: если P06-эпизодов нет,
        # честный ответ — 0 находок (а не шумовые окна без рук)
        mode="episodes"
    else:
        mode=mode or ("episodes" if eps else "free")
    cands=[]
    if mode=="free":
        # скользящее окно без привязки к паттерну: работает всегда, даже когда
        # пороговый паттерн в целевом видео не сегментируется на эпизоды
        W=max(3,int(round(dur_s*fps_proc)))
        step=max(1,int(round(float(hop or 0.25)*fps_proc)))
        starts=list(range(0,max(1,len(records)-W+1),step)) or [0]
        spans=[(s,min(len(records)-1,s+W-1)) for s in starts]
    else:
        spans=[tuple(e) for e in eps]
    for k,(a,bb) in enumerate(spans):
        win=records[a:bb+1]
        dur_c=T[bb]-T[a]
        if dur_c<=0: continue
        # face_touch: хотя бы в 25% кадров окна реально детектирована кисть у лица
        if face_touch:
            ntouch=sum(1 for r in win if _face_touch_gate(r))
            if ntouch < max(2,int(round(0.25*len(win)))): continue
        pen=0.15*abs(math.log(dur_c/dur_s))
        d=0.0; ok=0; morph_fail=False
        for c in chans:
            v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
            if len(v)==0 or np.all(np.isnan(v)): continue
            cs=_znorm(_resample(v))
            de=float(np.sqrt(np.mean((cs-S[c])**2)))
            dd=dtw_dist(cs,S[c])
            d+=0.3*de+0.7*dd; ok+=1
            if not _morph_ok(v,S[c],dur_s,fps_proc): morph_fail=True
        if ok==0: continue
        if face_touch and morph_fail: continue   # строгий режим: форма кривой обязательна
        ft_frac=None
        if face_touch:
            ft_frac=round(sum(1 for r in win if _face_touch_gate(r))/len(win),2)
        item={"t0":round(T[a],2),"t1":round(T[bb],2),"frames":bb-a+1,
              "pattern":pid,"ep":k,"dur_ratio":round(dur_c/dur_s,2),
              "distance":round(d/ok+pen,4)}
        if ft_frac is not None: item["face_touch_frac"]=ft_frac
        cands.append(item)
    cands.sort(key=lambda x:x["distance"])
    keep=cands[:max(1,int(top_k))]
    meta={"total":len(cands),"returned":len(keep),"version":SNAP_VERSION,"pattern":pid,
          "mode":mode,"episodes_of_pattern":len(eps),
          "patterns_in_snapshot":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if req&(1<<x)],
          "channels":chans,"snapshot_duration":dur_s,
          "face_touch_gate":bool(face_touch),"strict_face_touch":bool(strict_ft),
          "snapshot_touch_frac":round(float(frac_touch),2)}
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
