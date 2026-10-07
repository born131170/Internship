#!/usr/bin/env python
"""Оценка детектора касания лица рукой по размеченным роликам.

Запуск:  python tools/eval_touch.py            (порог из pipeline)
         python tools/eval_touch.py --sweep    (плюс перебор порогов для калибровки)

Что делает:
  * берёт разметку data/gt/touch.json (таймкоды) и прогоны data/gt/runs/<label>/metrics.jsonl;
  * считает кадровые precision/recall/F1 и совпадение эпизодов (жадное сопоставление по IoU);
  * отдельно проверяет «дробление»: сколько эпизодов найдено внутри размеченного окна
    (например, в 3.5-7.5 c владелец проекта видит четыре коротких касания);
  * в режиме --sweep перебирает пороги по уже сохранённым каналам touch/twrist,
    то есть без повторного запуска MediaPipe.

Метрики считаются только там, где есть каналы touch/twrist (иначе строка помечается n/a).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402

GT_PATH = ROOT / "tools" / "gt_touch.json"   # разметка версионируется (только таймкоды)
RUNS_PATH = ROOT / "data" / "gt" / "videos.json"
IOU_MATCH = 0.3


def load_json(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def spans_for(records, tau, twrist_tau, min_sec, gap_sec, fps_proc):
    """Локальная копия pipeline.touch_spans с произвольными порогами (для sweep)."""
    on = []
    for r in records:
        t = r.get("touch"); w = r.get("twrist")
        if t is not None:
            on.append(float(t) <= tau)
        elif w is not None:
            on.append(float(w) <= twrist_tau)
        else:
            on.append(False)
    gap = max(1, int(round(gap_sec * fps_proc)))
    min_len = max(1, int(round(min_sec * fps_proc)))
    spans = []; st = None; last = None
    for i, v in enumerate(on):
        if v:
            if st is None:
                st = i
            elif i - last > gap:
                spans.append((st, last)); st = i
            last = i
    if st is not None:
        spans.append((st, last))
    return [(a, b) for a, b in spans if b - a + 1 >= min_len]


def to_intervals(records, spans):
    return [(records[a]["t"], records[b]["t"]) for a, b in spans]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    inter = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_episodes(det, gt, thresh=IOU_MATCH):
    """Жадное сопоставление эпизодов по IoU. Возвращает (пары, непарные det, непарные gt)."""
    pairs = []
    used_d = set(); used_g = set()
    cand = sorted(((iou(d, g), i, j) for i, d in enumerate(det) for j, g in enumerate(gt)), reverse=True)
    for v, i, j in cand:
        if v < thresh or i in used_d or j in used_g:
            continue
        used_d.add(i); used_g.add(j); pairs.append((i, j, v))
    return pairs, [i for i in range(len(det)) if i not in used_d], [j for j in range(len(gt)) if j not in used_g]


def frame_scores(records, det, gt, t_end):
    """Кадровые TP/FP/FN по пересечению с размеченными интервалами."""
    tp = fp = fn = 0
    for r in records:
        t = r["t"]
        in_gt = any(a <= t <= b for a, b in gt)
        in_det = any(a <= t <= b for a, b in det)
        if in_gt and in_det:
            tp += 1
        elif in_det:
            fp += 1
        elif in_gt:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def evaluate(tau=None, twrist_tau=None, min_sec=None, gap_sec=None, verbose=True, exclude=()):
    gt_all = load_json(GT_PATH, {})
    runs = load_json(RUNS_PATH, {})
    if not gt_all or not runs:
        print("нет data/gt/touch.json или data/gt/videos.json")
        return None
    tau = pipeline.TOUCH_MAX_TAU if tau is None else tau
    twrist_tau = pipeline.TWRIST_MAX_TAU if twrist_tau is None else twrist_tau
    min_sec = pipeline.TOUCH_MIN_SEC if min_sec is None else min_sec
    gap_sec = pipeline.TOUCH_GAP_SEC if gap_sec is None else gap_sec

    rows = []
    for fname, meta in runs.items():
        gt = gt_all.get(meta.get("gt") or fname)
        if not gt or fname in exclude:
            continue
        run_dir = ROOT / meta["dir"]
        recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs:
            continue
        dur = recs[-1]["t"] - recs[0]["t"]
        fps_proc = (len(recs) - 1) / dur if dur > 0 else 1.0
        gt_iv = [(a, b) for a, b in gt["windows"] if a <= recs[-1]["t"]]
        det_iv = to_intervals(recs, spans_for(recs, tau, twrist_tau, min_sec, gap_sec, fps_proc))
        tp, fp, fn, prec, rec, f1 = frame_scores(recs, det_iv, gt_iv, recs[-1]["t"])
        pairs, miss_d, miss_g = match_episodes(det_iv, gt_iv)
        e_prec = len(pairs) / len(det_iv) if det_iv else 0.0
        e_rec = len(pairs) / len(gt_iv) if gt_iv else 0.0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if e_prec + e_rec else 0.0
        # «дробление»: сколько эпизодов найдено внутри каждого размеченного окна
        splitting = {f"{a}-{b}": sum(1 for d in det_iv if overlap(d, (a, b)) > 0.3 * (b - a))
                     for a, b in gt_iv if b - a >= 2.0}
        if verbose:
            print(f"\n=== {fname} ({gt.get('label')}): {len(recs)} кадров, {recs[-1]['t']:.1f} c ===")
            print(f"  разметка ({len(gt_iv)}): " + ", ".join(f"{a:g}-{b:g}" for a, b in gt_iv))
            print(f"  найдено  ({len(det_iv)}): " + (", ".join(f"{a:.1f}-{b:.1f}" for a, b in det_iv) or "—"))
            print(f"  кадры: TP={tp} FP={fp} FN={fn} | precision={prec:.2f} recall={rec:.2f} F1={f1:.2f}")
            print(f"  эпизоды: совпало {len(pairs)}/{len(gt_iv)} разметки, лишних {len(miss_d)}, "
                  f"пропущено {len(miss_g)} | precision={e_prec:.2f} recall={e_rec:.2f} F1={e_f1:.2f}")
            if splitting:
                print("  эпизодов внутри окна: " + ", ".join(f"{k}: {v}" for k, v in splitting.items())
                      + f" (ожидалось по разметке: {gt.get('expected_episodes')})")
        rows.append({"video": fname, "frames": len(recs), "gt": gt_iv, "det": det_iv,
                     "frame": {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
                               "recall": round(rec, 3), "f1": round(f1, 3)},
                     "episode": {"matched": len(pairs), "gt": len(gt_iv), "det": len(det_iv),
                                 "precision": round(e_prec, 3), "recall": round(e_rec, 3),
                                 "f1": round(e_f1, 3)},
                     "splitting": splitting})
    if not rows:
        print("нет данных для оценки")
        return None
    macro_f1 = sum(r["episode"]["f1"] for r in rows) / len(rows)
    macro_p = sum(r["episode"]["precision"] for r in rows) / len(rows)
    macro_r = sum(r["episode"]["recall"] for r in rows) / len(rows)
    if verbose:
        print(f"\nИТОГ по эпизодам ({len(rows)} видео): precision={macro_p:.2f} recall={macro_r:.2f} F1={macro_f1:.2f}")
    return {"rows": rows, "macro": {"precision": round(macro_p, 3), "recall": round(macro_r, 3),
                                    "f1": round(macro_f1, 3)},
            "params": {"tau": tau, "twrist_tau": twrist_tau, "min_sec": min_sec, "gap_sec": gap_sec}}


def windows_detail():
    """Что происходило в размеченных окнах: доля кадров с сигналом и значения расстояний.

    Нужно, чтобы отличать «сигнала нет» (кисть/запястье не детектированы) от «сигнал есть,
    но выше порога» — от этого зависит, что крутить: порог или детектор.
    """
    gt_all = load_json(GT_PATH, {})
    runs = load_json(RUNS_PATH, {})
    for fname, meta in runs.items():
        gt = gt_all.get(fname)
        if not gt:
            continue
        recs = [json.loads(l) for l in
                (ROOT / meta["dir"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n=== {fname} ===")
        print(f"{'окно':>14} {'кадров':>7} {'touch есть':>11} {'touch p50':>10} {'twrist есть':>12} {'twrist p50':>11}")
        for a, b in gt["windows"]:
            win = [r for r in recs if a <= r["t"] <= b]
            if not win:
                print(f"{f'{a:g}-{b:g}':>14} {'0':>7}  (вне проанализированного диапазона)")
                continue
            tt = [r["touch"] for r in win if r.get("touch") is not None]
            ww = [r["twrist"] for r in win if r.get("twrist") is not None]
            f = lambda v: f"{sorted(v)[len(v)//2]:.2f}" if v else "—"
            print(f"{f'{a:g}-{b:g}':>14} {len(win):>7} {len(tt)/len(win)*100:>10.0f}% {f(tt):>10} "
                  f"{len(ww)/len(win)*100:>11.0f}% {f(ww):>11}")


def sweep(exclude=("video3.mp4",)):
    """Перебор порогов. video3 исключён по умолчанию: в его размеченных окнах сигнала
    «кисть у лица» нет вовсе (кисть не детектирована, запястье далеко) — вероятно, там
    размечен другой тип эпизода, это отдельный вопрос к владельцу разметки."""
    print(f"\n=== ПЕРЕБОР ПОРОГОВ (без {', '.join(exclude) if exclude else 'исключений'}) ===")
    print(f"{'tau':>5} {'twrist':>7} {'min_s':>6} {'gap_s':>6} {'P':>6} {'R':>6} {'F1':>6}  по видео")
    best = None
    for tau in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
        for twrist_tau in (2.6, 3.0, 3.5):
            for min_sec in (0.4, 0.5, 0.75):
                for gap_sec in (0.25, 0.5):
                    res = evaluate(tau, twrist_tau, min_sec, gap_sec, verbose=False, exclude=exclude)
                    if not res:
                        continue
                    m = res["macro"]
                    per = " ".join(f"{r['video'].replace('.mp4','')}:{r['episode']['f1']:.2f}" for r in res["rows"])
                    print(f"{tau:>5} {twrist_tau:>7} {min_sec:>6} {gap_sec:>6} "
                          f"{m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f}  {per}")
                    if best is None or m["f1"] > best[0]["macro"]["f1"]:
                        best = (res, tau, twrist_tau, min_sec, gap_sec)
    if best:
        res, tau, tw, ms, gs = best
        print(f"\nлучший F1={res['macro']['f1']:.2f} при tau={tau}, twrist_tau={tw}, min_sec={ms}, gap_sec={gs} "
              f"(precision={res['macro']['precision']:.2f}, recall={res['macro']['recall']:.2f})")
        out = ROOT / "data" / "gt" / "sweep_report.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("отчёт:", out.relative_to(ROOT))
        print("\nПроверка выбранных порогов на этом же наборе:")
        evaluate(tau, tw, ms, gs, verbose=True, exclude=exclude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="перебрать пороги и записать отчёт")
    ap.add_argument("--windows", action="store_true", help="показать сигналы внутри размеченных окон")
    ap.add_argument("--json", action="store_true", help="сохранить отчёт по текущим порогам")
    ap.add_argument("--exclude", default="", help="список видео через запятую, которые не учитывать")
    args = ap.parse_args()
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    if args.windows:
        windows_detail()
        return
    res = evaluate(exclude=exclude)
    if res and args.json:
        out = ROOT / "data" / "gt" / "eval_report.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("отчёт:", out.relative_to(ROOT))
    if args.sweep:
        sweep()


if __name__ == "__main__":
    main()
