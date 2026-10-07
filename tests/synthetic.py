"""Синтетические кадровые записи и summary для тестов — без MediaPipe и без видео.

Набор воспроизводит случаи, на которых ломалась логика движка:
  * окно улыбки с AU6 при нулевом blendshape cheekRaise (P10D не срабатывал вовсе);
  * отведение взгляда по yaw (P12 с порогом 35° давал 0-4 кадра из 1153);
  * пауза 0.5-3 c внутри речи и длинная тишина, которая паузой не является (P14);
  * моргания как отдельные события (P01) и каналы, которых нет (None/NaN).
"""
from __future__ import annotations
import math

import pipeline
import psychometrics

DT = 0.125          # 8 fps, как в референсных роликах
N = 96              # 12 с


def _stat(vals):
    if not vals:
        return None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def make_records(n=N, dt=DT, with_ocu=True, cheek_value=0.0):
    """Кадровые записи с окнами: улыбка 2-3.5 c, отведение взгляда 5-5.5 c, речь 1-5 c, пауза 5-5.6 c."""
    recs = []
    for i in range(n):
        t = round(i * dt, 3)
        smile_win = 16 <= i < 28
        gaze_win = 40 <= i < 44
        speech = 8 <= i < 40
        if i < 8 or speech:
            pause = 0.0
        else:
            pause = round(min(4.0, (i - 39) * dt), 3)
        pspeech = 1 if (not speech and 40 <= i < 80) else 0
        rec = {
            "t": t,
            "face_conf": 0.9,
            "pose_conf": 0.8,
            "hands": 1 if 4 <= i < 60 else 0,
            "emotion": "joy" if smile_win else "none",
            "smile": 0.62 if smile_win else 0.05,
            "frown": 0.0,
            "browUp": 0.05,
            "browDown": 0.35 if 50 <= i < 56 else 0.02,
            "eyeWide": 0.05,
            "blink": 0.8 if i in (3, 21, 60) else 0.0,
            "jawO": 0.1 if speech else 0.02,
            "noseW": 0.0,
            "press": 0.0,
            "cheek": cheek_value,                       # mediapipe 1.0.0 отдаёт 0.0
            "yaw": math.radians(30.0 if gaze_win else 2.0),
            "pitch": math.radians(4.0),
            "yc": math.radians(30.0 if gaze_win else 2.0),
            "pc": math.radians(1.0),
            "hand_speed": 0.12 if 4 <= i < 12 else (0.02 if 4 <= i < 60 else None),
            "aperture": 1.2,
            "hfd": 1.4,
            "hfd2": 1.5,
            "tif": 0.0,
            "menergy": 0.05 if 30 <= i < 34 else 0.005,
            "rms": 0.03 if speech else 0.0,
            "f0": 180.0 if speech else None,
            "srate": 4.0 if speech else 0.0,
            "pause": pause,
            "pspeech": pspeech,
            "voicemo": 0.0,
            "speech": 1 if speech else 0,
            # геометрия глаза для legacy-проверки AU6 (без канала ocu):
            # разрез глаза 0.01 при ширине 0.04 -> AU6 ~0.55
            "lm": {"L": [0.40, 0.40], "B": [0.40, 0.41], "O": [0.44, 0.40],
                   "R": [0.60, 0.40], "b": [0.60, 0.41], "o": [0.56, 0.40]},
        }
        if with_ocu:
            rec["ocu"] = 0.30 if smile_win else 0.02
        recs.append(rec)
    return recs


def summary_from(records, personality=True, audio=None):
    """Summary той же формы, что пишет analyze_video, но из готовых записей."""
    counters, events, _blinks = pipeline.pattern_stats(records)
    stats = {}
    for ch in pipeline.STATS_CH:
        vals = []
        for r in records:
            v = r.get(ch)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
        stats[ch] = _stat(vals)
    emo = {}
    for r in records:
        emo[r["emotion"]] = emo.get(r["emotion"], 0) + 1
    cues, truth = pipeline.truth_cues(records, counters, events, stats, audio)
    nf = len(records)
    dur = records[-1]["t"] - records[0]["t"] if nf else 0.0
    return {
        "method": {}, "calibration": pipeline.CALIBRATION, "pipeline_version": pipeline.PIPELINE_VERSION,
        "video": "synthetic.mp4", "fps": round(1 / DT, 3), "n_frames": nf, "duration_sec": round(dur, 1),
        "counters": counters, "events": events, "stats": stats, "emotions": emo, "audio": audio,
        "truth_cues": cues, "truth_heuristic": truth, "episodes": [],
        "pattern_names": {pid: nm for pid, nm, _ in pipeline.PATTERN_DEFS},
        "personality": psychometrics.assess(records) if personality else None,
    }


def pattern_fn(pid):
    """Предикат паттерна по коду: pattern_fn('P14')({'pause':1.0,'pspeech':1})."""
    return dict((p, fn) for p, _, fn in pipeline.PATTERN_DEFS)[pid]
