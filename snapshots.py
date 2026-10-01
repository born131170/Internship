"""Цифровые слепки v4.0: корректная оцифровка жестового эпизода и поиск аналогий.
Только измерения MediaPipe; LLM в этом контуре не участвует.
search() возвращает (results, meta) — совместимо с app.py."""
from __future__ import annotations
import math, time, uuid
import numpy as np

SNAP_VERSION="4.0"
DEFAULT_CHANNELS=["hand_speed","aperture","hfd","hfd2","yc","pc","menergy"]
N=64
MAX_EXT=1.5

PAT_DEFS=[
 ("P01",lambda r: r.get("blink",0)>0.6),
 ("P02",lambda r: abs(r.get("pc",0))>0.08),
 ("P03",lambda r: abs(r.get("yc",0))+abs(r.get("pc",0))>0.03),
 ("P04",lambda r: r.get("hand_speed",0)>0.03),
 ("P05",lambda r: r.get("hand_speed",0)>0.1),
 ("P06",lambda r: ((r.get("tif",0)>0.5 or r.get("hfd",1)<0.8) and r.get("hands",0)>0)),
 ("P07",lambda r: r.get("hand_speed",0)>0.01),
 ("P08",lambda r: r.get("aperture",1)<0.8),
 ("P09",lambda r: r.get("menergy",0)>0.02),
 ("P10",lambda r: r.get("smile",0)>0.4),
 ("P11",lambda r: r.get("browDown",0)>0.3 or r.get("press",0)>0.2),
 ("P12",lambda r: abs(r.get("yc",0))>0.2 or r.get("pc",0)>0.2),
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
    out=np.zeros(len(records),dtype=np.int32)
    for i in range(len(records)):
        r=records[i]; m=0
        for b in range(len(PAT_DEFS)):
            try:
                if PAT_DEFS[b][1](r): m|=(1<<b)
            except Exception: pass
        out[i]=m
    return out

def _mask_of_window(mask):
    if len(mask)==0: return 0
    return int(np.bitwise_or.reduce(mask))

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
    a,b=_segment(records,idx[0],idx[-1])
    win=records[a:b+1]
    chans=_valid_channels(win,channels or DEFAULT_CHANNELS)
    series={c:_series_of(win,c).tolist() for c in chans}
    stats={}
    for c in chans:
        v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
        stats[c]={"mean":float(np.nanmean(v)),"std":float(np.nanstd(v))}
    pm=_pattern_mask(win); wm=_mask_of_window(pm)
    return {"id":uuid.uuid4().hex[:10],"name":name,"created":int(time.time()),
            "channels":chans,"duration":round(records[b]["t"]-records[a]["t"],2),
            "t0":round(records[a]["t"],2),"t1":round(records[b]["t"],2),
            "t0_req":t0,"t1_req":t1,"n":N,"series":series,"stats":stats,
            "pattern_mask":wm,
            "pattern_active":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if wm&(1<<x)],
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

def search(records,snap,hop=None,top_k=100,refine=None,pattern=None,**kwargs):
    if not records: return [],{"total":0,"version":SNAP_VERSION}
    chans=[c for c in snap.get("channels",[]) if c in snap.get("series",{})]
    if not chans: return [],{"total":0,"version":SNAP_VERSION}
    S={c:np.asarray(snap["series"][c],dtype=float) for c in chans}
    dur_s=float(snap.get("duration",1.0)) or 1.0
    T=[r.get("t",0.0) for r in records]
    dt=(T[-1]-T[0])/max(1,len(T)-1) or 0.08
    fps_proc=1.0/dt
    mask=_pattern_mask(records)
    req=int(snap.get("pattern_mask",0) or 0)
    active=[x for x in range(len(PAT_DEFS)) if req&(1<<x)] or list(range(len(PAT_DEFS)))
    if pattern:
        sel=[x for x in range(len(PAT_DEFS)) if PAT_DEFS[x][0]==pattern]
        if sel: active=sel
    if len(active)>1:
        cnt=sorted((len(_episodes_of_pattern(mask,x,fps_proc)),x) for x in active)
        active=[cnt[0][1]]
    b=active[0]; pid=PAT_DEFS[b][0]
    eps=_episodes_of_pattern(mask,b,fps_proc)
    cands=[]
    for k in range(len(eps)):
        a,bb=eps[k]
        win=records[a:bb+1]
        dur_c=T[bb]-T[a]
        if dur_c<=0: continue
        pen=0.15*abs(math.log(dur_c/dur_s))
        d=0.0; ok=0
        for c in chans:
            v=np.array([r.get(c,float("nan")) for r in win],dtype=float)
            if len(v)==0 or np.all(np.isnan(v)): continue
            cs=_znorm(_resample(v))
            de=float(np.sqrt(np.mean((cs-S[c])**2)))
            dd=dtw_dist(cs,S[c])
            d+=0.3*de+0.7*dd; ok+=1
        if ok==0: continue
        cands.append({"t0":round(T[a],2),"t1":round(T[bb],2),"frames":bb-a+1,
                      "pattern":pid,"ep":k,"dur_ratio":round(dur_c/dur_s,2),
                      "distance":round(d/ok+pen,4)})
    cands.sort(key=lambda x:x["distance"])
    keep=cands[:max(1,int(top_k))]
    meta={"total":len(cands),"returned":len(keep),"version":SNAP_VERSION,"pattern":pid,
          "patterns_in_snapshot":[PAT_DEFS[x][0] for x in range(len(PAT_DEFS)) if req&(1<<x)],
          "channels":chans,"snapshot_duration":dur_s}
    return keep,meta
