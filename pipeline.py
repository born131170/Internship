"""PersonaScope pipeline v4.2 (финальная сборка): MediaPipe + аудио + геометрический P06 + приглушённый неон."""
from __future__ import annotations
import json, math, subprocess, wave
from collections import deque
from pathlib import Path
import cv2
import numpy as np

POSE_LINKS=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
HAND_LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
EMO_CODES={"joy":0,"surprise":1,"anger":2,"sadness":3,"fear":4,"disgust":5}

def _n(v):
    """NaN-safe: None/NaN -> False-нейтральное значение 0 для порогов."""
    if v is None: return 0.0
    if isinstance(v, float) and math.isnan(v): return 0.0
    return float(v)

def _nn(v):
    """True если значение измеримо (не None/NaN)."""
    if v is None: return False
    return not (isinstance(v, float) and math.isnan(v))

# Научно операционализированные паттерны (пороги — из литературы, см. psychometrics.NORM_SOURCES):
# P01 blink-эпизоды; P02 кивки (ритмический pitch-осцилляторный паттерн); P03 мобильность головы;
# P04/P05 иллюстративные жесты; P06 самоуспокаивающие касания лица (adaptors, Kaitz 2007 / Vismara 2016);
# P07 мелкая моторная активность (fidgeting); P08 закрытая позиция кисти; P09 смены позы;
# P10 улыбка; P10D Duchenne-улыбка (mouthSmile + orbicularis oculi — Ekman & Friesen 1978);
# P11 напряжение (browDown/mouthPress/lipPress); P12 отведение взгляда (>35° от медианы головы);
# P13 вокализация (активная речь); P14 пауза >=0.5 c (когнитивная нагрузка, Vrij 2008).
PATTERN_DEFS=[
    ("P01","Моргание",            lambda m: _n(m["blink"])>0.6),
    ("P02","Кивок при слушании",   lambda m: _n(m["pc"])>0.08),
    ("P03","Движения головы",      lambda m: (_n(m["yc"])+_n(m["pc"]))>0.03),
    ("P04","Жестовый эпизод",      lambda m: _n(m["hand_speed"])>0.03),
    ("P05","Амплитуда/скорость жестов", lambda m: _n(m["hand_speed"])>0.1),
    ("P06","Рука около лица",      lambda m: ((_n(m["tif"])>0.5 or (_nn(m["hfd"]) and m["hfd"]<0.8)) and m["hands"]>0)),
    ("P07","Повторные мелкие движения", lambda m: _n(m["hand_speed"])>0.01),
    ("P08","Положение рук",        lambda m: _nn(m["aperture"]) and m["aperture"]<0.8),
    ("P09","Смена позы",           lambda m: _n(m["menergy"])>0.02),
    ("P10","Подъём уголков губ",   lambda m: _n(m["smile"])>0.4),
    ("P10D","Duchenne-улыбка",     lambda m: _n(m["smile"])>0.35 and _n(m["cheek"])>0.15),
    ("P11","Брови / напряжение губ", lambda m: _n(m["browDown"])>0.3 or _n(m["press"])>0.2),
    ("P12","Отведение взгляда",    lambda m: _n(m["yc"])>math.radians(35) or _n(m["pc"])>math.radians(35)),
    ("P13","Вокализация",          lambda m: bool(m.get("speech"))),
    ("P14","Пауза в речи ≥0.5 c",  lambda m: _n(m["pause"])>=0.5),
]
PIPELINE_VERSION="5.0-psychometric"
STATS_CH=["smile","frown","browUp","browDown","eyeWide","blink","jawO","noseW","press","cheek",
          "yaw","pitch","yc","pc","hand_speed","aperture","hfd","hfd2","tif","menergy","face_conf","pose_conf",
          "rms","f0","srate","pause"]

MUTED_BG=(46,13,21); MUTED_BAND=(58,24,33); MUTED_GRID=(70,58,96); MUTED_TXT=(150,138,175)
PAL_MUTED=[(160,74,230),(246,92,139),(191,212,45),(248,189,56),(182,114,244)]

def _ffmpeg_exe():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil; return shutil.which("ffmpeg")

def _extract_audio(video: Path, wav: Path):
    exe=_ffmpeg_exe()
    if not exe: return False
    try:
        subprocess.run([exe,"-y","-i",str(video),"-ac","1","-ar","16000","-f","wav",str(wav)],
                       check=True,capture_output=True,timeout=600)
        return wav.exists()
    except Exception:
        return False

def _read_wav(wav: Path):
    with wave.open(str(wav),"rb") as w:
        sr=w.getframerate(); data=w.readframes(w.getnframes())
    return sr, np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0

def _audio_features(sr, x, times, win_sec):
    n=len(times); rms=np.zeros(n); f0=np.full(n,np.nan)
    w=int(win_sec*sr)
    for i,t in enumerate(times):
        a=int(t*sr); b=min(len(x),a+w)
        if a>=len(x) or b-a<256: continue
        seg=x[a:b]; rms[i]=float(np.sqrt(np.mean(seg**2)))
        seg=seg-np.mean(seg)
        corr=np.correlate(seg,seg,"full")[len(seg)-1:]
        lo=int(sr/400); hi=min(len(corr)-1,int(sr/70))
        if hi>lo and corr[0]>0:
            c=corr[lo:hi]
            if c.size:
                k=lo+int(np.argmax(c)); v=c[np.argmax(c)]/corr[0]
                if v>0.3: f0[i]=sr/k
    speech_thr=max(0.008,float(np.percentile(rms,30))*1.6) if n else 0.0
    speech=rms>speech_thr
    peaks=[]
    for i in range(1,n-1):
        if speech[i] and rms[i]>=rms[i-1] and rms[i]>rms[i+1] and rms[i]>speech_thr*1.3:
            if not peaks or times[i]-peaks[-1]>=0.12: peaks.append(times[i])
    srate=np.zeros(n); pause=np.zeros(n); sil_start=None; pi=0
    for i in range(n):
        t=times[i]
        while pi<len(peaks) and peaks[pi]<t-2.0: pi+=1
        srate[i]=sum(1 for p in peaks[pi:] if p<=t)/2.0
        if speech[i]: sil_start=None; pause[i]=0.0
        else:
            if sil_start is None: sil_start=t
            pause[i]=t-sil_start
    voicemo=np.full(n,np.nan)
    idx=np.where(speech & ~np.isnan(f0))[0]
    if len(idx)>10:
        mE=rms[idx].mean(); sE=rms[idx].std()+1e-9
        mF=np.nanmean(f0[idx]); sF=np.nanstd(f0[idx])+1e-9
        for i in idx:
            zE=(rms[i]-mE)/sE; zF=(f0[i]-mF)/sF
            if zE>0.7 and zF>0.7: voicemo[i]=1
            elif zE>0.7 and zF<-0.2: voicemo[i]=2
            elif zE<-0.6 and zF<-0.6: voicemo[i]=3
            elif zF>1.2: voicemo[i]=4
            else: voicemo[i]=0
    return rms,f0,speech,srate,pause,voicemo,speech_thr

def _strip_bg(canvas,x0,y0,w,h):
    cv2.rectangle(canvas,(x0,y0),(x0+w-1,y0+h-1),MUTED_BG,-1)
    NB=7; gap=4; bw=max(8,(w-gap*(NB+1))//NB)
    for b in range(NB):
        x=x0+gap+b*(bw+gap)
        if x+bw<x0+w: cv2.rectangle(canvas,(x,y0+3),(x+bw,y0+h-4),MUTED_BAND,-1)
    for i in range(1,4):
        y=y0+int(i*h/4); cv2.line(canvas,(x0+2,y),(x0+w-2,y),MUTED_GRID,1,cv2.LINE_AA)

def _draw_series(canvas,x0,y0,w,h,hist,ymin,ymax,color,fill=True,step=False):
    if len(hist)<2: return
    last=list(hist)[-w:]
    pad=3; ih=h-2*pad; base=y0+h-pad
    def Y(v): return y0+pad+int((1.0-float(np.clip((v-ymin)/(ymax-ymin),0,1)))*ih)
    if step:
        segs=[]; xa=None; yy=None
        for i,v in enumerate(last):
            bad=v is None or (isinstance(v,float) and math.isnan(v))
            xx=x0+int(i*w/len(last))
            if bad:
                if xa is not None: segs.append((xa,xx,yy)); xa=None; yy=None
                continue
            ny=Y(v)
            if xa is None: xa=xx; yy=ny
            elif ny!=yy:
                segs.append((xa,xx,yy)); xa=xx; yy=ny
        if xa is not None: segs.append((xa,x0+w-1,yy))
        for x1,x2,yv in segs:
            cv2.line(canvas,(x1,yv),(x2,yv),color,2,cv2.LINE_AA)
        return
    pts=[]
    for i,v in enumerate(last):
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        pts.append([x0+int(i*w/len(last)),Y(v)])
    if len(pts)<2: return
    P=np.array(pts,np.int32)
    if fill:
        poly=np.vstack([P,[[pts[-1][0],base]],[[pts[0][0],base]]]).astype(np.int32)
        sub=canvas[y0:y0+h, x0:x0+w]
        tmp=sub.copy()
        cv2.fillPoly(tmp,[poly-[x0,y0]],color)
        cv2.addWeighted(tmp,0.16,sub,0.84,0,sub)
    cv2.polylines(canvas,[P],False,color,1,cv2.LINE_AA)

def _extract_episodes(records, fps_proc, fps_src, video, ep_dir, make_clips, width=480, per_pat=0, models_dir=Path("models")):
    eps=[]; n=len(records)
    pnames={pid:nm for pid,nm,_ in PATTERN_DEFS}
    if n<4: return eps
    GAP=max(1,int(0.4*fps_proc)); MIN=max(2,int(0.6*fps_proc))
    for pid,pname,fn in PATTERN_DEFS:
        act=[bool(fn(r)) for r in records]; ints=[]; start=None; last=-10
        for i,a in enumerate(act):
            if a:
                if start is None: start=i; last=i
                elif i-last>GAP: ints.append((start,last)); start=i; last=i
                else: last=i
        if start is not None: ints.append((start,last))
        ints=[(a,b) for a,b in ints if (b-a)>=MIN]
        ints.sort(key=lambda ab: ab[1]-ab[0], reverse=True)
        for k,(a,b) in enumerate(ints if per_pat<=0 else ints[:per_pat]):
            mid=(a+b)//2
            # измеримые метрики эпизода — «доказательство» в числовом виде
            def _mval(ch):
                vs=[records[i].get(ch) for i in range(a,b+1)]
                vs=[float(v) for v in vs if v is not None and not (isinstance(v,float) and math.isnan(v))]
                return round(float(np.mean(vs)),4) if vs else None
            metrics={ch:_mval(ch) for ch in ("smile","cheek","blink","hand_speed","hfd","tif","menergy","yc","pc","f0","rms","pause") }
            metrics["hands_max"]=int(max(records[i].get("hands",0) or 0 for i in range(a,b+1)))
            eps.append({"id":f"{pid}_{k:02d}","pattern":pid,"name":pname,
                        "t0":round(records[a]["t"],2),"t1":round(records[b]["t"],2),
                        "mid":round(records[mid]["t"],2),
                        "frames":b-a+1,"thumb":f"episodes/{pid}_{k:02d}.jpg","clip":f"episodes/{pid}_{k:02d}.mp4",
                        "metrics":metrics})
    eps.sort(key=lambda e:e["t0"])
    if not eps: return eps
    cap=cv2.VideoCapture(str(video))
    fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640; fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    cw=width; chh=max(2,int(fh*width/fw))
    import mediapipe as _mp
    from mediapipe.tasks import python as _mppy
    from mediapipe.tasks.python import vision as _mpv
    _snap_f=_snap_p=_snap_h=None
    def _open_snap_models():
        nonlocal _snap_f,_snap_p,_snap_h
        md=Path(models_dir)
        _snap_f=_mpv.FaceLandmarker.create_from_options(_mpv.FaceLandmarkerOptions(
            base_options=_mppy.BaseOptions(model_asset_path=str(md/"face_landmarker.task")),
            running_mode=_mpv.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True))
        _snap_p=_mpv.PoseLandmarker.create_from_options(_mpv.PoseLandmarkerOptions(
            base_options=_mppy.BaseOptions(model_asset_path=str(md/"pose_landmarker.task")),
            running_mode=_mpv.RunningMode.VIDEO,num_poses=1))
        _snap_h=_mpv.HandLandmarker.create_from_options(_mpv.HandLandmarkerOptions(
            base_options=_mppy.BaseOptions(model_asset_path=str(md/"hand_landmarker.task")),
            running_mode=_mpv.RunningMode.VIDEO,num_hands=2))
    try:
        _open_snap_models()
    except Exception:
        _snap_f=_snap_p=_snap_h=None
    def _close_snap_models():
        for m in (_snap_f,_snap_p,_snap_h):
            try:
                if m is not None: m.close()
            except Exception: pass
    # доказательные скрин-кадры: исходный кадр + наложение контуров MediaPipe и подписи метрик эпизода
    def _evidence_frame(fr, ep):
        h0,w0=fr.shape[:2]; sc=cw/w0
        img=cv2.resize(fr,(cw,chh)); hh,ww=img.shape[:2]
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        im=_mp.Image(image_format=_mp.ImageFormat.SRGB,data=rgb)
        ts=int(ep["mid"]*1000)
        try:
            if _snap_f is not None:
                frr=_snap_f.detect_for_video(im,ts)
                if frr.face_landmarks:
                    for pt in frr.face_landmarks[0]: cv2.circle(img,(int(pt.x*ww),int(pt.y*hh)),1,(0,255,0),-1)
            if _snap_p is not None:
                prr=_snap_p.detect_for_video(im,ts)
                if prr.pose_landmarks:
                    pts=np.array([[p.x*ww,p.y*hh] for p in prr.pose_landmarks[0]])
                    for a,b in POSE_LINKS: cv2.line(img,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),(0,255,255),2)
            if _snap_h is not None:
                hrr=_snap_h.detect_for_video(im,ts)
                for hd in (hrr.hand_landmarks or []):
                    hp=np.array([[p.x*ww,p.y*hh] for p in hd])
                    for a,b in HAND_LINKS: cv2.line(img,tuple(hp[a].astype(int)),tuple(hp[b].astype(int)),(255,0,255),1)
        except Exception:
            pass
        nm=pnames.get(ep["pattern"],ep["name"])
        lines=[f"{ep['pattern']} {nm}  t={ep['t0']:.1f}-{ep['t1']:.1f}s"]
        mm=ep.get("metrics") or {}
        kv=[f"{k}={mm[k]}" for k in ("smile","cheek","blink","hand_speed","hfd","tif","menergy","yc","pc","f0","pause") if mm.get(k) is not None][:6]
        if kv: lines.append(", ".join(kv))
        pad=6; lh=17; bw=min(ww-10,max(180,max(len(l) for l in lines)*9+12)); bh=len(lines)*lh+10
        ov=img.copy(); cv2.rectangle(ov,(4,hh-bh-4),(4+bw,hh-4),(0,0,0),-1)
        cv2.addWeighted(ov,0.62,img,0.38,0,img)
        for i,ln in enumerate(lines):
            cv2.putText(img,ln,(10,hh-bh+lh*(i+1)-2),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,255),1,cv2.LINE_AA)
        return img
    rec_by_t=None
    for ep in eps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ep["mid"]*1000)
        ok,fr=cap.read()
        img=_evidence_frame(fr,ep) if ok else np.zeros((chh,cw,3),np.uint8)
        cv2.imwrite(str(ep_dir/Path(ep["thumb"]).name), img)
        if not make_clips: continue
        cap.set(cv2.CAP_PROP_POS_MSEC, ep["t0"]*1000)
        wr=cv2.VideoWriter(str(ep_dir/Path(ep["clip"]).name), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (cw,chh))
        while True:
            ok,fr=cap.read()
            if not ok: break
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if t>ep["t1"]: break
            if t>=ep["t0"]-0.01: wr.write(cv2.resize(fr,(cw,chh)))
        wr.release()
    cap.release()
    _close_snap_models()
    return eps

def analyze_video(video: Path, out_dir: Path, stride=2, width=640, max_frames=0,
                  make_dashboard=True, make_clips=True, progress_cb=None, models_dir=Path("models")):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ep_dir=out_dir/"episodes"; ep_dir.mkdir(exist_ok=True)
    fo=mp_vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(Path(models_dir)/"face_landmarker.task")),
        running_mode=mp_vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True)
    po=mp_vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(Path(models_dir)/"pose_landmarker.task")),
        running_mode=mp_vision.RunningMode.VIDEO,num_poses=1)
    ho=mp_vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(Path(models_dir)/"hand_landmarker.task")),
        running_mode=mp_vision.RunningMode.VIDEO,num_hands=2)
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); out_fps=fps/stride; fps_proc=out_fps
    audio=None
    wav=out_dir/"audio.wav"
    n_est=int(n_total/stride)+2 if n_total else 0
    if n_est and _extract_audio(video,wav):
        try:
            sr,x=_read_wav(wav)
            times=[i*stride/fps for i in range(n_est)]
            audio=_audio_features(sr,x,times,win_sec=max(0.05,2*stride/fps))
        except Exception:
            audio=None
    W=1400
    H_vad=[deque(maxlen=W) for _ in range(3)]; H_emo=deque(maxlen=W)
    H_dyn=[deque(maxlen=W) for _ in range(2)]; H_head=[deque(maxlen=W) for _ in range(2)]; H_hands=[deque(maxlen=W) for _ in range(2)]
    H_rms=deque(maxlen=W); H_f0=deque(maxlen=W); H_vemo=deque(maxlen=W)
    yaw_hist=[]; pitch_hist=[]; writer=None; real=0; idx=0; records=[]; prev_pose=None; prev_wr=None
    with mp_vision.FaceLandmarker.create_from_options(fo) as fm, \
         mp_vision.PoseLandmarker.create_from_options(po) as pm, \
         mp_vision.HandLandmarker.create_from_options(ho) as hm:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if real%stride!=0: real+=1; continue
            if max_frames and idx>=max_frames: break
            h0,w0=frame.shape[:2]; sc=width/w0
            left=cv2.resize(frame,None,fx=sc,fy=sc); h,w=left.shape[:2]
            rgb=cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
            img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts=int(real*1000.0/fps)
            right=left.copy()
            face_conf=pose_conf=0.0; n_hands=0; emotion="none"
            smile=frown=browUp=browDown=eyeWide=blink=jawO=noseW=press=cheek=np.nan
            yaw=pitch=np.nan; hand_speed=np.nan; aperture=np.nan; hfd=np.nan; menergy=np.nan
            hfd2=np.nan; face_wn=np.nan; nose2n=None; face_box=None; tif=0.0
            a_rms=a_f0=np.nan; a_speech=0; a_srate=a_pause=np.nan; a_vemo=np.nan
            if audio is not None and idx<len(audio[0]):
                a_rms=float(audio[0][idx]); a_f0=float(audio[1][idx])
                a_speech=int(audio[2][idx]); a_srate=float(audio[3][idx]); a_pause=float(audio[4][idx]); a_vemo=float(audio[5][idx])
            fr=fm.detect_for_video(img,ts)
            if fr.face_landmarks:
                fl=fr.face_landmarks[0]
                face_conf=float(np.mean([getattr(p,"visibility",1.0) or 1.0 for p in fl]))
                for pt in fl: cv2.circle(right,(int(pt.x*w),int(pt.y*h)),1,(0,255,0),-1)
                lm3=np.array([[p.x,p.y,p.z] for p in fl]); ctr=lm3.mean(0); f=lm3[1]-ctr
                yaw=math.atan2(f[0],-f[2]+1e-9); pitch=math.atan2(f[1],-f[2]+1e-9)
                nose=np.array([fl[1].x*w,fl[1].y*h]); ec=np.array([(fl[33].x+fl[263].x)/2*w,(fl[33].y+fl[263].y)/2*h])
                d=nose-ec; nn=np.linalg.norm(d)
                if nn>1e-6:
                    e=nose+d/nn*50; cv2.arrowedLine(right,tuple(nose.astype(int)),tuple(e.astype(int)),(0,0,255),2,tipLength=0.3)
                if fr.face_blendshapes:
                    s={c.category_name:c.score for c in fr.face_blendshapes[0]}
                    smile=max(s.get("mouthSmileLeft",0),s.get("mouthSmileRight",0)); frown=max(s.get("mouthFrownLeft",0),s.get("mouthFrownRight",0))
                    cheek=max(s.get("cheekRaiseLeft",0),s.get("cheekRaiseRight",0))  # orbicularis oculi — Duchenne-маркер
                    browUp=s.get("browInnerUp",0); browDown=max(s.get("browDownLeft",0),s.get("browDownRight",0))
                    eyeWide=max(s.get("eyeWideLeft",0),s.get("eyeWideRight",0)); blink=max(s.get("eyeBlinkLeft",0),s.get("eyeBlinkRight",0))
                    jawO=s.get("jawOpen",0); noseW=s.get("noseWrinkle",0); press=max(s.get("mouthPressLeft",0),s.get("mouthPressRight",0))
                    emo={"joy":smile,"surprise":browUp+eyeWide+jawO,"anger":browDown+frown,"sadness":browUp+frown,"fear":browUp+eyeWide,"disgust":noseW}
                    emotion=max(emo,key=emo.get)
                    face_wn=float(np.hypot(fl[33].x-fl[263].x, fl[33].y-fl[263].y))+1e-9
                    nose2n=(fl[1].x, fl[1].y)
                    xs=[q.x for q in fl]; ys=[q.y for q in fl]
                    cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
                    hw=(max(xs)-min(xs))/2*1.30; hh=(max(ys)-min(ys))/2*1.30
                    face_box=(cx-hw,cy-hh,cx+hw,cy+hh)
            pr=pm.detect_for_video(img,ts)
            if pr.pose_landmarks:
                pl=pr.pose_landmarks[0]; pts=np.array([[p.x*w,p.y*h] for p in pl])
                scale=np.linalg.norm(pts[11]-pts[12])+1e-6
                pose_conf=float(np.mean([getattr(p,"visibility",1.0) or 1.0 for p in pl]))
                for a,b in POSE_LINKS: cv2.line(right,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),(0,255,255),2)
                for p_ in pts: cv2.circle(right,tuple(p_.astype(int)),3,(0,165,255),-1)
                wl=pr.pose_world_landmarks[0] if pr.pose_world_landmarks else None
                vis_w=max(float(getattr(pl[15],"visibility",0) or 0), float(getattr(pl[16],"visibility",0) or 0))
                if wl is not None and vis_w>0.3:
                    head3=np.mean([[wl[i].x,wl[i].y,wl[i].z] for i in (0,7,8)],axis=0)
                    w3l=np.array([wl[15].x,wl[15].y,wl[15].z]); w3r=np.array([wl[16].x,wl[16].y,wl[16].z])
                    sh3=np.linalg.norm(np.array([wl[11].x,wl[11].y,wl[11].z])-np.array([wl[12].x,wl[12].y,wl[12].z]))+1e-9
                    hfd=float(min(np.linalg.norm(w3l-head3),np.linalg.norm(w3r-head3))/sh3)
                else:
                    hfd=np.nan
                if prev_pose is not None: menergy=float(np.median(np.linalg.norm(pts-prev_pose,axis=1)))/scale
                prev_pose=pts
            else: prev_pose=None
            hr=hm.detect_for_video(img,ts)
            if hr.hand_landmarks:
                n_hands=len(hr.hand_landmarks)
                wrists=[np.array([hd[0].x,hd[0].y]) for hd in hr.hand_landmarks]
                hs_scale=float(np.mean([np.linalg.norm(np.array([q[0].x,q[0].y])-np.array([q[9].x,q[9].y])) for q in hr.hand_landmarks]))+1e-6
                if prev_wr is not None and len(wrists)==len(prev_wr):
                    hand_speed=float(max(np.linalg.norm(a-b) for a,b in zip(wrists,prev_wr))/hs_scale)
                prev_wr=wrists
                aps=[np.linalg.norm(np.array([q[8].x,q[8].y])-np.array([q[4].x,q[4].y]))/(np.linalg.norm(np.array([q[0].x,q[0].y])-np.array([q[9].x,q[9].y]))+1e-6) for q in hr.hand_landmarks]
                aperture=float(np.mean(aps))
                if nose2n is not None and not math.isnan(face_wn):
                    hfd2=float(min(min(np.hypot(hd[t].x-nose2n[0], hd[t].y-nose2n[1]) for t in (4,8,12,16,20)) for hd in hr.hand_landmarks)/face_wn)
                if face_box is not None:
                    x0f,y0f,x1f,y1f=face_box
                    tif=1.0 if any((hd[t].x>=x0f and hd[t].x<=x1f and hd[t].y>=y0f and hd[t].y<=y1f) for hd in hr.hand_landmarks for t in (4,8,12,16,20)) else 0.0
                for hd in hr.hand_landmarks:
                    hp=np.array([[p.x*w,p.y*h] for p in hd])
                    for a,b in HAND_LINKS: cv2.line(right,tuple(hp[a].astype(int)),tuple(hp[b].astype(int)),(255,0,255),1)
                    for p_ in hp: cv2.circle(right,tuple(p_.astype(int)),2,(255,0,255),-1)
            else: prev_wr=None
            yaw_hist.append(yaw); pitch_hist.append(pitch)
            ym=np.nanmedian(yaw_hist); pm_=np.nanmedian(pitch_hist)
            yc=yaw-ym if not math.isnan(yaw) else np.nan
            pc=pitch-pm_ if not math.isnan(pitch) else np.nan
            rec={"t":round(real/fps,3),"face_conf":round(face_conf,3),"pose_conf":round(pose_conf,3),"hands":n_hands,"emotion":emotion,
                 **{k:round(float(v),4) if not (isinstance(v,float) and math.isnan(v)) else None for k,v in
                    dict(smile=smile,frown=frown,browUp=browUp,browDown=browDown,eyeWide=eyeWide,blink=blink,jawO=jawO,noseW=noseW,press=press,cheek=cheek,
                         yaw=yaw,pitch=pitch,yc=yc,pc=pc,hand_speed=hand_speed,aperture=aperture,hfd=hfd,hfd2=hfd2,tif=tif,menergy=menergy,
                         rms=a_rms,f0=a_f0,srate=a_srate,pause=a_pause,voicemo=a_vemo).items()}}
            rec={k:(v if v is not None else float("nan")) for k,v in rec.items()}
            rec["speech"]=a_speech
            records.append(rec)
            if make_dashboard:
                hud=[f"t={real/fps:6.1f}s",f"face={face_conf:.2f} pose={pose_conf:.2f} hands={n_hands}",
                     f"emo={emotion} f0={a_f0 if not math.isnan(a_f0) else 0:3.0f}Hz"]
                ov=right.copy()
                cv2.rectangle(ov,(w-260,5),(w-5,5+len(hud)*18+8),(0,0,0),-1)
                cv2.addWeighted(ov,0.5,right,0.5,0,right)
                for i,ln in enumerate(hud): cv2.putText(right,ln,(w-255,20+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
                H_vad[0].append(smile-(frown+browDown+noseW) if not math.isnan(smile) else np.nan)
                H_vad[1].append(jawO+eyeWide if not math.isnan(jawO) else np.nan); H_vad[2].append(0.0)
                H_emo.append(EMO_CODES.get(emotion,np.nan)); H_dyn[0].append(np.nan); H_dyn[1].append(hand_speed)
                H_head[0].append(yc); H_head[1].append(pc); H_hands[0].append(aperture); H_hands[1].append(hfd)
                H_rms.append(a_rms); H_f0.append(a_f0/400.0 if not math.isnan(a_f0) else np.nan); H_vemo.append(a_vemo)
                top=np.hstack([left,right]); TW=top.shape[1]; SH=80
                canvas=np.full((h+7*SH,TW,3),21,dtype=np.uint8); canvas[0:h,:]=top; y0=h
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_vad[0],-1,1,PAL_MUTED[1]); _draw_series(canvas,0,y0,TW,SH,H_vad[1],-1,1,PAL_MUTED[0])
                cv2.putText(canvas,"VAD",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_emo,0,5,PAL_MUTED[3],step=True)
                cv2.putText(canvas,"EMO",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_dyn[1],0,0.5,PAL_MUTED[0])
                cv2.putText(canvas,"DYN",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_head[0],-0.4,0.4,PAL_MUTED[3]); _draw_series(canvas,0,y0,TW,SH,H_head[1],-0.4,0.4,PAL_MUTED[2])
                cv2.putText(canvas,"HEAD",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_hands[0],0,4,PAL_MUTED[2]); _draw_series(canvas,0,y0,TW,SH,H_hands[1],0,4,PAL_MUTED[4])
                cv2.putText(canvas,"HANDS",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_rms,0,0.5,PAL_MUTED[2]); _draw_series(canvas,0,y0,TW,SH,H_f0,0,1,PAL_MUTED[3])
                cv2.putText(canvas,"SPCH",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA); y0+=SH
                _strip_bg(canvas,0,y0,TW,SH)
                _draw_series(canvas,0,y0,TW,SH,H_vemo,0,4,PAL_MUTED[0],step=True)
                cv2.putText(canvas,"VEMO",(4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,MUTED_TXT,1,cv2.LINE_AA)
                if writer is None:
                    Hc,Wc=canvas.shape[:2]
                    writer=cv2.VideoWriter(str(out_dir/"dashboard.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),out_fps,(Wc,Hc))
                writer.write(canvas)
            idx+=1; real+=1
            if progress_cb and n_total: progress_cb(min(1.0,real/n_total), f"кадр {real}/{n_total}")
    cap.release()
    if writer: writer.release()
    with (out_dir/"metrics.jsonl").open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    counters={}; events={}
    for pid,_,fn in PATTERN_DEFS:
        c=e=0; prev=False
        for r in records:
            a=bool(fn(r))
            if a:
                c+=1
                if not prev: e+=1
            prev=a
        counters[pid]=c; events[pid]=e
    stats={}
    for ch in STATS_CH:
        vals=[r[ch] for r in records if not math.isnan(r[ch])]
        stats[ch]={"mean":round(float(np.mean(vals)),4),"std":round(float(np.std(vals)),4),
                   "min":round(float(np.min(vals)),4),"max":round(float(np.max(vals)),4)} if vals else None
    emo_dist={}
    for r in records: emo_dist[r["emotion"]]=emo_dist.get(r["emotion"],0)+1
    nf=len(records); dur=records[-1]["t"]-records[0]["t"] if nf else 0.0
    audio_sum=None; pause_intervals=0; mean_pause=0.0
    if audio is not None and nf:
        rms,f0,speech,srate,pause,voicemo,thr=audio
        rms,f0,speech,srate,pause,voicemo=rms[:nf],f0[:nf],speech[:nf],srate[:nf],pause[:nf],voicemo[:nf]
        ints=[]; st=None
        for i in range(nf):
            if not speech[i]:
                if st is None: st=i
            else:
                if st is not None:
                    if pause[i-1]>=0.5: ints.append(round(float(pause[i-1]),2))
                    st=None
        if st is not None and pause[nf-1]>=0.5: ints.append(round(float(pause[nf-1]),2))
        pause_intervals=len(ints); mean_pause=round(float(np.mean(ints)),2) if ints else 0.0
        vm={}
        for v in voicemo[~np.isnan(voicemo)]: vm[int(v)]=vm.get(int(v),0)+1
        audio_sum={"speech_ratio":round(float(speech.mean()),3),"speech_thr":round(float(thr),4),
                   "pause_intervals_ge05s":pause_intervals,"mean_pause_sec":mean_pause,
                   "speech_rate_syll_per_sec":round(float(np.mean(srate[speech])),2) if speech.any() else 0.0,
                   "f0_mean_hz":round(float(np.nanmean(f0)),1) if not np.all(np.isnan(f0)) else None,
                   "f0_std_hz":round(float(np.nanstd(f0)),1) if not np.all(np.isnan(f0)) else None,
                   "voicemo_dist":vm}
    cues={"duration_sec":round(dur,1),
          "blink_per_min":round(events["P01"]/dur*60,1) if dur>0 else 0,
          "smile_ratio":round(counters["P10"]/nf,3) if nf else 0,
          "tension_ratio":round(counters["P11"]/nf,3) if nf else 0,
          "avoidance_ratio":round(counters["P12"]/nf,3) if nf else 0,
          "hand_face_ratio":round(counters["P06"]/nf,3) if nf else 0,
          "hand_activity_ratio":round(counters["P07"]/nf,3) if nf else 0,
          "head_yaw_std":stats["yc"]["std"] if stats["yc"] else 0,
          "head_pitch_std":stats["pc"]["std"] if stats["pc"] else 0}
    if audio_sum:
        cues["speech_ratio"]=audio_sum["speech_ratio"]
        cues["pause_per_min"]=round(pause_intervals/dur*60,1) if dur>0 else 0
        cues["speech_rate_syll_per_sec"]=audio_sum["speech_rate_syll_per_sec"]
        cues["voice_f0_std_hz"]=audio_sum["f0_std_hz"]
    score=100.0-max(0,abs(cues["blink_per_min"]-20))*1.0-cues["avoidance_ratio"]*60-cues["tension_ratio"]*40-cues["hand_face_ratio"]*30-min(40,abs(cues["head_yaw_std"]-0.1)*120)
    if audio_sum:
        score-=max(0,abs(cues["speech_rate_syll_per_sec"]-4))*3
        score-=min(20,cues["pause_per_min"]*1.5)
    episodes=_extract_episodes(records, fps_proc, fps, video, ep_dir, make_clips, models_dir=models_dir)
    method={"P06":"v4-final hybrid: (кончик пальца кисти внутри 2D-бокса лица x1.30 ИЛИ 3D-дистанция запястье->центр головы < 0.8 ширин плеч) AND кисть детектирована; без visibility-гейта; телефон у лица отсеивается"}
    summary={"method":method,"pipeline_version":PIPELINE_VERSION,"video":Path(video).name,"fps":fps,"n_frames":nf,"duration_sec":round(dur,1),
             "counters":counters,"events":events,"stats":stats,"emotions":emo_dist,
             "audio":audio_sum,"truth_cues":cues,"truth_heuristic":int(max(5,min(95,score))),
             "episodes":episodes,"pattern_names":{pid:nm for pid,nm,_ in PATTERN_DEFS}}
    (out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding="utf-8")
    return summary
