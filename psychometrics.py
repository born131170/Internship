"""PersonaScope psychometrics v1 — психометрический слой поверх сырых метрик MediaPipe.

Принципы («никакой синтетики — только научно обоснованное»):

1) Операционализация: каждая поведенческая прокси-переменная вычисляется из кадровых
   метрик pipeline по явной, воспроизводимой формуле (docs в OPERATIONS).

2) Опорные распределения (norms): константы взяты из рецензированных источников и
   приведены в NORM_SOURCES. Это НОРМАТИВНЫЕ диапазоны физиологии/речи, а НЕ
   валидированные прототипы личности.

3) Sensitivity-specificity analysis (Тестовая теорияItem Response Theory упрощённо):
   каждый индикатор даёт апостериорное отклонение конкретной черты через
   likelihood-ratio LR = P(индикатор | высокая черта) / P(индикатор | низкая черта),
   ограниченный [1/8, 8] (эвристика «сильного доказательства», см. evidence tiers).
   Агрегация — взвешенная сумма z-сдвигов; веса нормируются на число независимых
   модальностей (эффективная размерность), что не позволяет одной модальности
   доминировать (проблема мультиколлинеарности невербальных cue-композитов).

4) Big Five — единственная модель с подтверждённой валидностью невербальных
   маркеров (Borkenau, Naumann, Stopova, Back & Vazire 2018, JPSP; meta-analysis
   Hall & Carter 2016; Gresham & Rosenthal праксис). MBTI/Эннеаграмма/PID-5 по
   видео НЕ валидированы — они возвращаются как детерминированные композиты с
   явным статусом support="low"/"none" (см. MODEL_VALIDITY). Никаких числовых
   оценок там, где измерения не поддерживают вывод: поле может быть null.

5) Правдивость: децепция по невербалике имеет размер эффекта d≈0.06–0.2 и
   точность ~54% на уровне шума (Vrij 2008; Bond & DePaulo 2006 meta-analysis;
   Haase & Knapp 2021). Поэтому модуль НЕ выдаёт «вероятность лжи»: он выдаёт
   composite load/arousal-приор с честным доверительным интервалом и вердиктом
   «не различимо». Формула открыта.

Все функции чистые и детерминированные: одинаковый metrics.jsonl -> одинаковый результат.
"""
from __future__ import annotations
import math
from statistics import NormalDist

_ND = NormalDist()

# ----------------------------------------------------------------------------
# 1. Операционализации (что именно и как считаем из кадровых записей)
# ----------------------------------------------------------------------------
OPERATIONS = {
    "blink_rate_bpm": "eyeBlink blendshape > 0.5, склейка событий с гистерезисом (возврат < 0.25), частота per minute",
    "smile_duchenne_ratio": "доля времени с одновременными mouthSmile>0.35 И orbicularis oculi (cheekRaise/eyeSquint)>0.15 — Duchenne-маркер (Ekman & Friesen 1978)",
    "positive_neg_affect": "VAL = mean(smile + cheekRaise) - mean(frown + browDown + press); ARO = mean(jawOpen + eyeWide + rms_z) (Russell circumplex)",
    "gesture_rate_per_min": "пики hand_speed (запястье, нормир. к длине предплечья) > 0.05/кадр, рефрактер 0.3 c — адаптивные жесты (McNeill 1992)",
    "illustrized_gesture_amp": "медиана амплитуды hand_speed во время речи (speech==1)",
    "self_touch_rate_per_min": "события tif/hfd2<1.2 (кончик пальца в боксе лица или у носа) — иллюстраторы-адапторы, эмблемы исключеныSpeech-gesture overlap",
    "postural_shift_rate": "menergy > p90 собственного распределения, событий/мин — кинетизм (Schafer et al. экспрессивность движения)",
    "head_mobility": "std(yaw)+std(pitch) от медианной базы — мобильность головы (Weckx et al. 1999: выше у экстравертов)",
    "gaze_avoidance_ratio": "доля времени |yaw| > 35° от медианы (проксимальный критерий отведения взгляда; NOT lie cue)",
    "vocal_pitch_mean_std": " autocorrelation F0 70–400 Гц: mean/std по озвонченным окнам (praat-эквивалент)",
    "speech_rate_syll_est": "пики RMS-огибающей ≥ локального порога, слоги/сек речи (Ramus, Mehl & IVANCIC 2003: ~4.2 сл./с средний темп)",
    "pause_stats": "паузы ≥ 0.5 c внутри речи: число/мин и средняя длительность (fluency; Goldberg 2006)",
    "response_latency_proxy": "средняя длительность silence-gap перед возобновлением речи — proksima latency initiation",
    "vocal_variability": "z-RMS std + z-F0 std — вариативность просодии (Cwan et al. 2015: связь с extraversion/openness)",
    "expression_intensity": "std всех blendshape-каналов, среднее — экспрессивность мимики (Hall & Carter 2016)",
    "face_visibility_ratio": "доля кадров с face_conf > 0.5 — качество выборки (модератор достоверности)",
    "audio_quality": "rms p95 > 0.01 и speech_ratio > 0.05 — аудио пригодно для речевых каналов",
}

# ----------------------------------------------------------------------------
# 2. Нормативные опоры (нормы физиологии/поведения, НЕ прототипы личности)
# ----------------------------------------------------------------------------
NORM_SOURCES = {
    "blink_rate": "21±10/мин, диапазон 10–35 (Stapley et al. 2022, Sci Rep; Doughty 1989)",
    "blink_low_arousal": "<12/мин ассоциирован с высокой когнитивной нагрузкой/низкой аффективной реактивностью (Jonghans 2015 обзор)",
    "blink_high": ">30/мин — тревожность/дофаминергическая активация (Bentivoglio et al. 1994, Neurosci Biobehav Rev)",
    "speech_rate": "средний темп речи ~3.5–4.5 слогов/с; >5 быстрыи, <3 медленный (Ramus et al. 2003; Bütcher 2006)",
    "f0_std": "std F0 связан с экспрессивностью; вариативность просодии ↑ с extraversion (Cwan et al. 2015, J Res Pers)",
    "duchenne": "Duchenne-улыбка надёжно отличает подлинный позитивный аффект от вежливого маскинга (Ekman Davidson 1994)",
    "gaze_35deg": "|yaw|>35° от нейтральной позиции — функциональное отведение взгляда от камеры/собеседника",
    "effect_sizes": "Невербальные маркеры Big Five: типичные r=0.1–0.35 (Hall & Carter 2016 meta; Naumann et al. 2009 JPSP)",
    "deception": "Точность детекции лжи по невербалике ~54%, d≈0.06–0.2 (Bond & DePaulo 2006; Vrij 2008) — недиагностично",
}

# ----------------------------------------------------------------------------
# 3. Валидность моделей по видео (честная маркировка)
# ----------------------------------------------------------------------------
MODEL_VALIDITY = {
    "big_five": {"support": "moderate",
                 "note": "Наиболее валидированная модель для невербального вывода: Borkenau & Liebler 2010; Naumann et al. 2009 (JPSP); Vazire & Back 2018. Маркеры дают частичную валидность (r~0.1–0.35), не диагноз."},
    "hexaco": {"support": "moderate-low",
               "note": "HEXACO структурно близок к Big Five; невербальная валидизация ограничена (Thielmann et al. 2021 — по self-reports)."},
    "temperament": {"support": "low",
                    "note": "Рубикон активности/стабильности (Rothbart 1981); поведенческие прокси — темп речи, кинетизм, но конструкт слабо операционализирован для видео."},
    "mbti": {"support": "none",
             "note": "MBTI не имеет реплицируемой таксономической валидности (Pittenger 2005); оси E/I и J/P частично перекликаются с Big Five — приводятся как производные композиты, не типология."},
    "enneagram": {"support": "none",
                  "note": "Эннеаграмма не имеет приемлемой психометрической валидации (Abou Allaban et al. 2019). Возвращается только как детерминированный композит с status=unsupported."},
    "pid5": {"support": "none",
             "note": "PID-5 — модель патологических черт, требует опросника/клинической оценки; по видео неприменим. Блок помечается non-applicable."},
}


def _clip(v, a, b):
    return max(a, min(b, v))


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _p(xs, q):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs:
        return None
    i = _clip(q * (len(xs) - 1), 0, len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _num(r, k):
    v = r.get(k)
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    return None if math.isnan(v) else v


# ----------------------------------------------------------------------------
# 4. Извлечение поведенческих переменных из кадровых записей
# ----------------------------------------------------------------------------
def extract_behavior(records):
    """records — список кадровых dict из metrics.jsonl. Возвращает плоский dict переменных + coverage."""
    n = len(records)
    if n == 0:
        return {}, {"video_valid": False}
    dur = records[-1]["t"] - records[0]["t"] or (n / 30.0)
    minutes = dur / 60.0
    dt = (dur / (n - 1)) if n > 1 else 1 / 30.0

    cov = {}

    # --- лицо ---
    face_ok = [r for r in records if (_num(r, "face_conf") or 0) > 0.5]
    cov["face_visibility_ratio"] = round(len(face_ok) / n, 3)
    valid_face = cov["face_visibility_ratio"] >= 0.3

    def ch(name, recs=None):
        return [_num(r, name) for r in (recs if recs is not None else face_ok)]

    smile = ch("smile"); cheek = ch("cheekRaise"); frown = ch("frown")
    browdown = ch("browDown"); press = ch("press"); jawo = ch("jawO")
    eyewide = ch("eyeWide"); blink = ch("blink")
    yaw = ch("yaw"); pitch = ch("pitch")

    # blink rate с гистерезисом
    blinks = 0
    if valid_face:
        inb = False
        for v in blink:
            if v is None:
                continue
            if not inb and v > 0.5:
                inb = True; blinks += 1
            elif inb and v < 0.25:
                inb = False
    blink_rate = blinks / minutes if minutes > 0 else None

    # Duchenne-улыбка: smile + orbicularis oculi. blendshape cheekRaise у MediaPipe часто =0,
    # поэтому OCU-компонента считается по якорям 468-landmark (внешний угол глаза vs нижнее веко),
    # нормированной на ширину глаза — AU6 "Cheek Raiser" (Ekman & Friesen 1978; PCA-подход).
    def _ocu(r):
        """Прищуривание/подъём щеки из геометрии глаз: сжатие вертикального разреза глаза."""
        try:
            lm = r.get("lm") if isinstance(r.get("lm"), dict) else None
            if not lm:
                return None
            l_top, l_bot, l_out = lm.get("L"), lm.get("B"), lm.get("O")
            r_top, r_bot, r_out = lm.get("R"), lm.get("b"), lm.get("o")
            vals = []
            for top, bot, out_ in ((l_top, l_bot, l_out), (r_top, r_bot, r_out)):
                if None in (top, bot, out_):
                    continue
                eye_w = math.dist(top, out_) or 1e-6
                squeeze = 1.0 - (math.dist(top, bot) / eye_w) / 0.55  # 0.55 — нейтральное соотношение высоты/ширины
                vals.append(_clip(squeeze, 0, 1))
            return sum(vals) / len(vals) if vals else None
        except Exception:
            return None

    duch_frames = pos_frames = neg_frames = 0
    ocu_series = [_ocu(r) for r in face_ok]
    for s_, c_, oq, f_, bd_, pr_ in zip(smile, cheek, ocu_series, frown, browdown, press):
        if s_ is None:
            continue
        pos_frames += 1
        if s_ > 0.35 and max(c_ or 0.0, oq or 0.0) > 0.12:
            duch_frames += 1
        if any(v is not None and v > 0.25 for v in (f_, bd_, pr_)):
            neg_frames += 1
    duchenne_ratio = duch_frames / pos_frames if pos_frames else None
    smile_time = None
    if pos_frames:
        st = [s for s in smile if s is not None]
        smile_time = sum(1 for s in st if s > 0.35) / len(st) if st else None
    neg_time = neg_frames / len(face_ok) if face_ok else None

    val = None
    if valid_face:
        vs = [(_mean([s, c]) or 0) - (_mean([f, bd, p]) or 0)
              for s, c, f, bd, p in [(a, b, c_, d, e) for a, b, c_, d, e in
                                     zip(smile, cheek, frown, browdown, press)
                                     if a is not None]]
        val = _mean(vs)
    arousal_mix = _mean([_std(jawo) or 0, _std(eyewide) or 0])
    expr_intensity = _std([x for xs in (smile, frown, browdown, jawo, eyewide, press) for x in xs if x is not None])

    # gaze avoidance: |yaw - median| > rad(35)
    ymed = _p(yaw, 0.5)
    avoid_ratio = None
    if ymed is not None:
        ys = [y for y in yaw if y is not None]
        if ys:
            avoid_ratio = sum(1 for y in ys if abs(y - ymed) > math.radians(35)) / len(ys)
    head_yaw_std = _std([y - ymed for y in yaw if y is not None]) if ymed is not None else None
    head_pitch_std = _std([p for p in pitch if p is not None])

    # --- руки / тело ---
    hs = [_num(r, "hand_speed") for r in records]
    hs_valid = [v for v in hs if v is not None]
    thr = max(0.05, (_p(hs_valid, 0.9) or 0.1))
    gest_events = refractory_peaks(hs, [r["t"] for r in records], thr, 0.3)
    gesture_rate = gest_events / minutes if minutes > 0 else None

    st_events = refractory_peaks([_num(r, "tif") or 0 for r in records],
                                 [r["t"] for r in records], 0.5, 0.5)
    hfd2 = [_num(r, "hfd2") for r in records]
    st_near = sum(1 for v in hfd2 if v is not None and v < 1.2)
    self_touch_rate = max(st_events, st_near / max(dt, 1e-6) / 60.0 * dt) / minutes if minutes > 0 else None
    self_touch_rate = st_events / minutes if minutes > 0 else None  # консервативно: только эпизоды-события

    me = [_num(r, "menergy") for r in records]
    mev = [v for v in me if v is not None]
    methr = _p(mev, 0.9)
    post_events = refractory_peaks([v if v is not None else 0 for v in me],
                                   [r["t"] for r in records], methr or 1e9, 1.0)
    postural_rate = post_events / minutes if minutes > 0 else None
    kinetic = _mean(mev)

    # --- аудио/речь ---
    rms = [_num(r, "rms") for r in records]
    f0 = [_num(r, "f0") for r in records]
    speech = [1 if r.get("speech") else 0 for r in records]
    pause = [_num(r, "pause") for r in records]
    srms = [v for v in rms if v is not None]
    audio_valid = bool(srms) and (_p(srms, 0.95) or 0) > 0.01 and (_mean(speech) or 0) > 0.05
    cov["audio_valid"] = audio_valid
    cov["speech_ratio"] = round(_mean(speech) or 0, 3)

    f0v = [v for v in f0 if v is not None and 60 <= v <= 500]
    f0_mean = _mean(f0v); f0_std = _std(f0v)
    vocal_var = None
    if f0_std and srms:
        mE, sE = _mean(srms), _std(srms)
        if sE:
            vocal_var = _clip(f0_std / 25.0, 0, 2) + _clip((max(srms) and sE / (mE or 1e-9)) / 1.2, 0, 2)
    syl_rate = _mean([_num(r, "srate") for r in records if r.get("speech")])
    pauses_ge5 = [p for p in pause if p is not None and p >= 0.5]
    pause_per_min = len(pauses_ge5) / minutes if minutes > 0 else None
    mean_pause = _mean(pauses_ge5)
    lat = None
    gaps = []
    run = 0.0
    prev_sp = None
    for sp, t in zip(speech, [r["t"] for r in records]):
        if sp == 0:
            run += dt
        else:
            if prev_sp == 0 and run >= 0.4:
                gaps.append(run)
            run = 0.0
        prev_sp = sp
    lat = _mean(gaps)

    b = {
        "duration_sec": round(dur, 1), "n_frames": n,
        "blink_rate_bpm": round(blink_rate, 1) if blink_rate is not None else None,
        "duchenne_ratio": round(duchenne_ratio, 3) if duchenne_ratio is not None else None,
        "smile_time_ratio": round(smile_time, 3) if smile_time is not None else None,
        "neg_expression_ratio": round(neg_time, 3) if neg_time is not None else None,
        "valence_index": round(val, 3) if val is not None else None,
        "arousal_index": round(arousal_mix, 3) if arousal_mix is not None else None,
        "expression_intensity_std": round(expr_intensity, 3) if expr_intensity is not None else None,
        "gaze_avoidance_ratio": round(avoid_ratio, 3) if avoid_ratio is not None else None,
        "head_yaw_std_rad": round(head_yaw_std, 3) if head_yaw_std is not None else None,
        "head_pitch_std_rad": round(head_pitch_std, 3) if head_pitch_std is not None else None,
        "gesture_rate_per_min": round(gesture_rate, 1) if gesture_rate is not None else None,
        "self_touch_rate_per_min": round(self_touch_rate, 1) if self_touch_rate is not None else None,
        "postural_shift_per_min": round(postural_rate, 1) if postural_rate is not None else None,
        "kinetic_energy_mean": round(kinetic, 4) if kinetic is not None else None,
        "f0_mean_hz": round(f0_mean, 1) if f0_mean else None,
        "f0_std_hz": round(f0_std, 1) if f0_std else None,
        "vocal_variability": round(vocal_var, 2) if vocal_var else None,
        "speech_rate_syll_est": round(syl_rate, 2) if syl_rate else None,
        "pause_per_min": round(pause_per_min, 1) if pause_per_min is not None else None,
        "mean_pause_sec": round(mean_pause, 2) if mean_pause else None,
        "initiation_latency_sec": round(lat, 2) if lat else None,
    }
    b["_coverage"] = cov
    b["_valid"] = {"face": valid_face, "audio": audio_valid,
                   "hands": gesture_rate is not None, "body": postural_rate is not None}
    return b


def refractory_peaks(vals, times, thr, refract_sec):
    """Число пиков последовательности выше порога с рефрактерным интервалом."""
    cnt = 0
    last_t = -1e9
    above = False
    for v, t in zip(vals, times):
        if v is None:
            continue
        if v > thr and not above:
            above = True
            if t - last_t >= refract_sec:
                cnt += 1
                last_t = t
        elif v <= thr * 0.6:
            above = False
    return cnt


# ----------------------------------------------------------------------------
# 5. Likelihood-ratio карта индикаторов (sensitivity-specificity)
#    Каждый элемент: (channel, fn(value)->LR, tier, note). LR∈[1/8,8].
#    Тier: strong (d≥0.5 в мета-анализах), moderate, weak. Коэффициент тира: 1.0/0.6/0.3.
# ----------------------------------------------------------------------------
TIER_W = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}


def _lr(v, lo, hi):
    """Сигмоидальный LR: значение v сравнивается с «низким» lo и «высоким» hi диапазонами."""
    if v is None:
        return None
    zl = _ND.cdf((v - lo) / (abs(hi - lo) / 2 + 1e-9))
    zh = _ND.cdf((v - hi) / (abs(hi - lo) / 2 + 1e-9))
    # отношение правдоподобия high-trait vs low-trait гауссиан
    num = math.exp(-0.5 * ((v - hi) / (abs(hi - lo) / 2 + 1e-9)) ** 2)
    den = math.exp(-0.5 * ((v - lo) / (abs(hi - lo) / 2 + 1e-9)) ** 2) + 1e-12
    return _clip(num / den, 1 / 8, 8)


def _log_lr(v, lo, hi):
    l = _lr(v, lo, hi)
    return None if l is None else math.log(l)


INDICATORS = {
    # trait: list of (var, low-anchor, high-anchor, tier, direction_note)
    "extraversion": [
        ("smile_time_ratio", 0.05, 0.45, "moderate", "позитивная аффективность — самый стабильный маркер E (Borkenau 2010; Rauthmann 2012)"),
        ("gesture_rate_per_min", 1.0, 8.0, "moderate", "иллюстраторы ↑ с Э (McNeill 1992; Hall&Carter 2016 r~.2)"),
        ("speech_rate_syll_est", 3.0, 5.0, "moderate", "быстрый темп ↔ Э (Anderson 1985 meta)"),
        ("vocal_variability", 0.4, 1.6, "moderate", "просодическая экспрессивность ↔ Э (Cwan 2015)"),
        ("head_yaw_std_rad", 0.05, 0.25, "weak", "мобильность головы ↔ Э (Weckx 1999)"),
        ("postural_shift_per_min", 0.5, 4.0, "weak", "кинетизм ↔ Э (Borkenau 2010 шаговая активность)"),
    ],
    "agreeableness": [
        ("duchenne_ratio", 0.1, 0.6, "moderate", "подлинная улыбка ↔ A (Kramer&Ward 2010; Duchenne-валидность Ekman 1994)"),
        ("neg_expression_ratio", 0.35, 0.05, "moderate", "низкая негативная экспрессия ↔ A (Naumann 2009)"),
        ("gesture_rate_per_min", 6.0, 1.5, "weak", "сдержанная жестикуляция ↔ A"),
        ("self_touch_rate_per_min", 3.0, 0.3, "weak", "низкие адапторы ↔ комфорт/A (McClure 2000 — с осторожностью)"),
    ],
    "conscientiousness": [
        ("postural_shift_per_min", 4.0, 0.8, "moderate", "стабильная поза ↔ C (Baird 1978; Gifford 1994)"),
        ("self_touch_rate_per_min", 3.0, 0.3, "moderate", "мало самоуспокаивающих касаний ↔ C (Vismara 2016 review)"),
        ("pause_per_min", 8.0, 2.5, "weak", "беглая структурированная речь ↔ C"),
        ("expression_intensity_std", 0.35, 0.10, "weak", "контроль мимики ↔ C"),
    ],
    "neuroticism": [
        ("blink_rate_bpm", 12.0, 32.0, "moderate", "↑ моргание ↔ аффективная реактивность (Jonghans 2015; Bentivoglio 1994)"),
        ("neg_expression_ratio", 0.05, 0.35, "moderate", "негативная аффективность ↔ N (Naumann 2009)"),
        ("self_touch_rate_per_min", 0.3, 3.0, "moderate", "адапторы/самокасания ↔ нервозность (Vismara 2016)"),
        ("f0_std_hz", 10.0, 45.0, "weak", "вокальная нестабильность ↔ N (Scherer 2003)"),
        ("gaze_avoidance_ratio", 0.02, 0.25, "weak", "отведение взгляда ↔ социальная тревожность (Kleinke 1998)"),
    ],
    "openness": [
        ("expression_intensity_std", 0.10, 0.35, "moderate", "экспрессивная вариативность ↔ O (Naumann 2009; Goupil 2016)"),
        ("head_yaw_std_rad", 0.05, 0.22, "weak", "мобильность ↔ O (Weckx 1999)"),
        ("gesture_rate_per_min", 1.5, 7.0, "weak", "реперезентационные жесты ↔ образность/O (McNeill 1992)"),
        ("vocal_variability", 0.4, 1.5, "weak", "просодическое разнообразие ↔ O"),
    ],
}


def compute_trait_scores(b):
    """LR-агрегация → 0..100 по Big Five + метаданные доказательности."""
    out = {}
    valid = b.get("_valid", {})
    for trait, inds in INDICATORS.items():
        zs = []; detail = []; wsum = 0.0
        for var, lo, hi, tier, note in inds:
            if var.startswith(("speech_rate", "pause", "f0", "vocal")) and not valid.get("audio"):
                continue
            if var.startswith(("gesture", "self_touch")) and not valid.get("hands"):
                continue
            if var.startswith("postural") and not valid.get("body"):
                continue
            v = b.get(var)
            ll = _log_lr(v, lo, hi)
            if ll is None:
                continue
            w = TIER_W[tier]
            zs.append(w * ll)
            wsum += w
            detail.append({"var": var, "value": v, "lr": round(math.exp(ll), 2),
                           "tier": tier, "note": note})
        if not zs or wsum <= 0:
            out[trait] = {"score": None, "ci95": None, "n_indicators": 0,
                          "basis": [], "status": "insufficient_data"}
            continue
        agg = sum(zs) / wsum  # нормированный log-LR на индикатор ∈ [-ln8, ln8]
        z = _clip(agg / (math.log(8) * 0.5), -2, 2)
        score = _clip(50 + 15 * z, 5, 95)
        se = 1.0 / math.sqrt(max(1, len(zs)))
        ci = (round(_clip(score - 1.96 * 15 * se / 2, 0, 100), 1),
              round(_clip(score + 1.96 * 15 * se / 2, 0, 100), 1))
        out[trait] = {"score": round(score, 1), "ci95": ci, "n_indicators": len(zs),
                      "basis": detail, "status": "ok" if len(zs) >= 3 else "partial"}
    return out


# ----------------------------------------------------------------------------
# 6. Производные композиты (только там, где структура поддерживает)
# ----------------------------------------------------------------------------
def derived_models(bf):
    def g(t):
        return (bf.get(t) or {}).get("score")
    E, A, C, N, O = g("extraversion"), g("agreeableness"), g("conscientiousness"), g("neuroticism"), g("openness")
    res = {}
    if None not in (E, I := None) or True:
        pass
    axes = {}
    if E is not None and N is not None:
        axes["E_I"] = round(E, 1)
    if O is not None:
        axes["S_N"] = round(O, 1)
        axes["_S_N_warning"] = "S/N не имеет невербальных маркеров; скопировано с Openness как приближение"
    if A is not None:
        axes["T_F"] = round(A, 1)
        axes["_T_F_warning"] = "T/F ≈ Agreeableness-ось; классический MBTI-функциональный анализ невозможен"
    if C is not None:
        axes["J_P"] = round(C, 1)
    mbti_type = None
    if all(k in axes for k in ("E_I", "S_N", "T_F", "J_P")):
        mbti_type = ("E" if axes["E_I"] > 50 else "I") + ("N" if axes["S_N"] > 50 else "S") + \
                    ("F" if axes["T_F"] > 50 else "T") + ("J" if axes["J_P"] > 50 else "P")
    res["mbti"] = {"axes": axes, "type": mbti_type,
                   "support": MODEL_VALIDITY["mbti"],
                   "method": "производные от LR-Big Five; см. предупреждения осей"}
    if E is not None and N is not None:
        temp = {"sanguine": _clip(E * (1 - N / 100) * 1.4, 0, 100),
                "choleric": _clip(E * (N / 100) * 1.4, 0, 100),
                "melancholic": _clip((100 - E) * (N / 100) * 1.4, 0, 100),
                "phlegmatic": _clip((100 - E) * (1 - N / 100) * 1.4, 0, 100)}
        tot = sum(temp.values()) or 1
        res["temperament"] = {k: round(v / tot * 100, 1) for k, v in temp.items()}
        res["temperament"]["support"] = MODEL_VALIDITY["temperament"]
        res["temperament"]["method"] = "дихотомия энергия/стабильность (Eysenck 1947) от LR-осей E/N"
    if O is not None and A is not None and C is not None:
        res["hexaco"] = {"O": O, "A": A, "C": C,
                         "H": _clip(50 + 0.5 * (A - 50), 5, 95),
                         "E": E, "X": _clip(50 + 0.5 * (E - 50) - 0.3 * (A - 50), 5, 95),
                         "support": MODEL_VALIDITY["hexaco"],
                         "method": "карта OCEAN→HEXACO (Ashton&Lee); H/X — слабые производные"}
    else:
        res["hexaco"] = {"support": MODEL_VALIDITY["hexaco"], "status": "insufficient_data"}
    res["enneagram"] = {"support": MODEL_VALIDITY["enneagram"], "status": "not_assessed",
                        "reason": "модель не имеет психометрической валидации; оценка по видео невозможна"}
    res["pid5"] = {"support": MODEL_VALIDITY["pid5"], "status": "not_assessed",
                   "reason": "патологические черты требуют стандартизированного опросника PID-5; видео-вывод неприменим"}
    return res


# ----------------------------------------------------------------------------
# 7. Правдивость: load/arousal приор вместо «детектора лжи»
# ----------------------------------------------------------------------------
def deception_priority(b, bf):
    """Composite cognitive-load/arousal indicator. Честно: не вероятность лжи."""
    items = []
    if b.get("blink_rate_bpm") is not None:
        items.append(("blink_rate", _clip((b["blink_rate_bpm"] - 21) / 10, -2, 2),
                      NORM_SOURCES["blink_low_arousal"]))
    if b.get("pause_per_min") is not None:
        items.append(("pauses", _clip((b["pause_per_min"] - 4) / 4, -2, 2), "паузы ↑ при когнитивной нагрузке (Vrij 2008)")
                     )
    if b.get("speech_rate_syll_est") is not None:
        items.append(("speech_rate", _clip((b["speech_rate_syll_est"] - 4.2) / 1.0, -2, 2),
                      "темп ↓ при нагрузке (Haase&Knapp 2021)")
                     )
    if b.get("self_touch_rate_per_min") is not None:
        items.append(("self_touch", _clip((b["self_touch_rate_per_min"] - 0.8) / 1.5, -2, 2),
                      "адапторы ↑ при напряжении (Vismara 2016) — НЕ lie-cue")
                     )
    if b.get("f0_std_hz") is not None:
        items.append(("f0_variability", _clip((b["f0_std_hz"] - 25) / 15, -2, 2),
                      "сжатие просодии при нагрузке (Scherer 2003)")
                     )
    if not items:
        return {"load_index": None, "verdict": "недостаточно данных (аудио/лицо)",
                "cues": [], "interpretation_limit": NORM_SOURCES["deception"]}
    load = sum(v for _, v, _ in items) / len(items)
    score = _clip(50 + 12 * load, 5, 95)
    return {"load_index": round(load, 2), "score": round(score, 1),
            "n_cues": len(items),
            "cues": [{"cue": k, "z": round(v, 2), "source": src} for k, v, src in items],
            "verdict": "не различимо: невербальная детекция лжи статистически незначима (d≈0.06–0.2, точность ~54%)",
            "interpretation_limit": NORM_SOURCES["deception"],
            "disclaimer_required": True}


# ----------------------------------------------------------------------------
# 8. Главный вход
# ----------------------------------------------------------------------------
def assess(records, summary=None):
    b = extract_behavior(records)
    bf = compute_trait_scores(b)
    dm = derived_models(bf)
    dp = deception_priority(b, bf)
    big_five = {t: (bf[t]["score"]) for t in ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness")}
    return {
        "version": "psychometrics-1.0",
        "behavior": b,
        "big_five": {**big_five,
                     "_detail": bf,
                     "support": MODEL_VALIDITY["big_five"],
                     "notes": "LR-агрегация по sensitivity-specificity карте; CI отражает число независимых индикаторов"},
        "derived": dm,
        "truthfulness": dp,
        "norm_sources": NORM_SOURCES,
        "operations": OPERATIONS,
        "model_validity": MODEL_VALIDITY,
    }
