"""Детерминированный поведенческий скоринг v1: оценки 0..100 из метрик MediaPipe.
Формулы открыты (SCORING_METHOD), воспроизводимы, не зависят от LLM.
z(x,a,s) = clip((x-a)/s, -2, 2); score = clip(50 + 8*sum(z), 5, 95)."""
SCORING_METHOD=("z-якоря: smile 0.08/0.06; hand_speed 0.05/0.04; speech_ratio 0.5/0.25; tension 0.15/0.12; "
 "avoidance 0.10/0.08; blink_dev |bpm-20| 0/8; f0_std 25/15; head_yaw_std 0.12/0.08; emo_variety 3/1.5; "
 "fidget(P07) 0.25/0.15; nods(P02) 0.10/0.08; pose_shifts(P09) 0.30/0.20; speech_rate 4/1.5. "
 "E=50+8*(z_smile+z_act+z_speech); N=50+8*(z_tension+z_blinkdev+z_avoid+z_f0std); O=50+8*(z_ystd+z_act+z_emovar); "
 "C=50+8*(-z_fidget+z_speech-z_poses); A=50+8*(z_smile+z_nods-z_tension). MBTI: E_I=E, S_N=O-аналог, T_F=A, J_P=C. "
 "Temperament нормируются к сумме 100. Это операциональные поведенческие оценки, НЕ клинический диагноз.")

def _z(x,a,s):
    if x is None: return 0.0
    return max(-2.0,min(2.0,(x-a)/s))
def _clip(v): return float(max(5,min(95,v)))
def _g(st,ch,key="mean"):
    d=(st or {}).get(ch) or {}
    return d.get(key)

def compute_scores(s):
    c=s.get("counters",{}); st=s.get("stats",{}); cues=s.get("truth_cues",{}); au=s.get("audio") or {}
    nf=max(1,s.get("n_frames",1))
    fr=lambda pid: c.get(pid,0)/nf
    smile=cues.get("smile_ratio",0.0); tension=cues.get("tension_ratio",0.0)
    avoid=cues.get("avoidance_ratio",0.0); handface=cues.get("hand_face_ratio",0.0)
    blink=cues.get("blink_per_min",20.0); ystd=cues.get("head_yaw_std",0.1)
    act=_g(st,"hand_speed") or 0.0; men=_g(st,"menergy") or 0.0
    speech=au.get("speech_ratio") or cues.get("speech_ratio") or 0.0
    rate=au.get("speech_rate_syll_per_sec") or cues.get("speech_rate_syll_per_sec") or 4.0
    f0std=au.get("f0_std_hz") or cues.get("voice_f0_std_hz") or 25.0
    emo_var=len([v for v in (s.get("emotions") or {}).values() if v>0.02*nf])
    nods=fr("P02"); fidget=fr("P07"); poses=fr("P09")
    E=_clip(50+8*(_z(smile,.08,.06)+_z(act,.05,.04)+_z(speech,.5,.25)))
    N=_clip(50+8*(_z(tension,.15,.12)+_z(abs(blink-20),0,8)+_z(avoid,.10,.08)+_z(f0std,25,15)))
    O=_clip(50+8*(_z(ystd,.12,.08)+_z(act,.05,.04)+_z(emo_var,3,1.5)))
    C=_clip(50+8*(-_z(fidget,.25,.15)+_z(speech,.5,.25)-_z(poses,.30,.20)))
    A=_clip(50+8*(_z(smile,.08,.06)+_z(nods,.10,.08)-_z(tension,.15,.12)))
    bf={"openness":O,"conscientiousness":C,"extraversion":E,"agreeableness":A,"neuroticism":N,
        "notes":"операциональные оценки по невербальным маркерам; формулы: scoring.SCORING_METHOD"}
    axes={"E_I":E,"S_N":_clip(50+8*(_z(ystd,.12,.08)+_z(emo_var,3,1.5))),"T_F":A,"J_P":C}
    mbti={"type":("E" if axes["E_I"]>50 else "I")+("N" if axes["S_N"]>50 else "S")+("F" if axes["T_F"]>50 else "T")+("J" if axes["J_P"]>50 else "P"),
          "axes":axes,"notes":"оси = детерминированные композиты; >50 = полюс"}
    if E>55 and N<=55: et=3
    elif E>55 and N>55: et=4
    elif E<=55 and N>55: et=6
    else: et=9
    enn={"type":et,"wing":f"{et}w{et+1 if et%2 else et-1}","score":_clip(55+abs(E-50)/2+abs(N-50)/2),
         "notes":"тип по квадрату E/N; операциональная гипотеза"}
    san=max(2,E*(1-N/100)); chol=max(2,_clip(50+8*(_z(act,.05,.04)+_z(tension,.15,.12)))-0)
    mel=max(2,(N*(1-E/100))/1); phl=max(2,_clip(50-8*(_z(act,.05,.04)+_z(tension,.15,.12)))-0)
    tot=san+chol+mel+phl
    temp={"sanguine":round(san/tot*100),"choleric":round(chol/tot*100),
          "melancholic":round(mel/tot*100),"phlegmatic":round(phl/tot*100),
          "notes":"активность x стабильность, нормировано к 100"}
    hexa={"H":_clip(50-8*_z(avoid,.10,.08)-8*_z(handface,.10,.08)+8*_z(smile,.08,.06)),"E":N,"X":E,"A":A,"C":C,"O":axes["S_N"],
          "notes":"H = низкое избегание/открытость мимики; остальные = композиты"}
    pid5={"negative_affect":N,"detachment":_clip(100-E),"antagonism":_clip(100-A),
          "disinhibition":_clip(100-C),"psychoticism":_clip(50+4*_z(emo_var,3,1.5)),
          "notes":"домены как полярные композиты; скрининговая интерпретация"}
    used=[smile,tension,avoid,act,men,speech,rate,f0std]
    cov=round(sum(1 for v in used if v not in (None,0))/max(1,len(used)),2)
    ae=auto_evidence(s)
    return {"big_five":bf,"mbti":mbti,"enneagram":enn,"temperament":temp,"hexaco":hexa,"pid5":pid5,"auto_evidence":ae,
            "truthfulness":{"score":s.get("truth_heuristic",50),
                            "verdict":"невербальный приор (формула в evidence); не установленная правдивость"},
            "coverage":cov,"scoring_method":SCORING_METHOD,"version":"1.0"}

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
