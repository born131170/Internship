#!/usr/bin/env python
"""Проверка раздела «Поиск идентичных эпизодов» на размеченных роликах.

Запуск:  python tools/eval_search.py            (только строгий режим касаний)
         python tools/eval_search.py --both     (плюс обычный режим для сравнения)

Что делает: берёт слепок из размеченного окна касания одного ролика и ищет им
в своём и в остальных роликах; найденные эпизоды сопоставляются с разметкой
(совпадение = пересечение не меньше 30% размеченного окна). Так проверяется
именно то, на что жаловался владелец проекта: «строгий режим не нашёл ничего»
и «нашёл ~22 неправильных эпизода».
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import snapshots  # noqa: E402

GT_PATH = ROOT / "tools" / "gt_touch.json"   # разметка версионируется (только таймкоды)
RUNS_PATH = ROOT / "data" / "gt" / "videos.json"
CHANNELS = ["touch", "twrist", "hand_speed", "hfd", "hfd2", "tif"]


def load_runs():
    runs = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    out = {}
    for key, meta in runs.items():
        d = ROOT / meta["dir"]
        p = d / "metrics.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs or "touch" not in recs[0]:
            continue
        out[key] = {"meta": meta, "recs": recs, "gt_key": meta.get("gt") or key}
    return out


def matched(det, gt):
    """Эпизод считается найденным, если большая часть ЕГО длительности лежит внутри
    размеченного окна. Разметка владельца — широкие «обёртки» (2-3 c), тогда как детектор
    выдаёт сам момент касания (0.5-1.5 c), поэтому проверять надо вложенность найденного."""
    out = []
    for d in det:
        dur = max(1e-6, d[1] - d[0])
        if any(max(0.0, min(d[1], b) - max(d[0], a)) >= 0.5 * dur for a, b in gt):
            out.append(d)
    return out


def gt_windows(gt, kind="touch"):
    """Окна разметки заданного типа: touch (касание), near (рука рядом без касания),
    negative (заведомо не касание). kind=None — все окна."""
    out = []
    for w in gt.get("windows", []):
        if isinstance(w, (list, tuple)):
            k, a, b = "touch", float(w[0]), float(w[1])
        else:
            k, a, b = w.get("kind", "touch"), float(w["t0"]), float(w["t1"])
        if kind is None or k == kind:
            out.append((a, b))
    return out


def run_eval(strict=True, verbose=True, percentile=None, tau=None):
    gt_all = json.loads(GT_PATH.read_text(encoding="utf-8"))
    if tau is not None:
        import pipeline
        pipeline.TOUCH_MAX_TAU = float(tau)      # touch_frame читает константу во время вызова
    runs = load_runs()
    keys = [k for k in runs if runs[k]["gt_key"] in gt_all]
    results = []
    for src in keys:
        gt_src = gt_windows(gt_all[runs[src]["gt_key"]], "touch")
        a, b = gt_src[0]                       # шаблон — из первого размеченного окна источника
        snap = snapshots.create_snapshot(runs[src]["recs"], a, b, CHANNELS, f"gt_{src}")
        snap["source_video"] = src
        if verbose:
            print(f"\n=== шаблон из {runs[src]['gt_key']} ({a}-{b} c) ===")
        for tgt in keys:
            gt = gt_windows(gt_all[runs[tgt]["gt_key"]], "touch")
            kw = {"top_k": 100, "target_video": tgt}
            if strict:
                kw["face_touch"] = True
            if percentile is not None:
                kw["null_percentile"] = percentile
            res, meta = snapshots.search(runs[tgt]["recs"], snap, **kw)
            det = [(r["t0"], r["t1"]) for r in res]
            hit = matched(det, gt)
            same = "то же видео" if tgt == src else "другое видео"
            f1 = (2 * len(hit) / (len(det) + len(gt))) if (det or gt) else 0.0
            results.append({"src": runs[src]["gt_key"], "tgt": runs[tgt]["gt_key"], "same": tgt == src,
                            "found": len(det), "gt": len(gt), "hit": len(hit), "f1": round(f1, 2),
                            "pattern": meta.get("pattern"), "threshold": meta.get("match_threshold"),
                            "det": det})
            if verbose:
                print(f"  {same:>12} {runs[tgt]['gt_key']:<14} найдено {len(det):>3} из разметки {len(gt):>2}: "
                      f"совпало {len(hit):>2} | паттерн {meta.get('pattern')} порог {meta.get('match_threshold')}")
                if det:
                    print("      " + ", ".join(f"{x[0]:.1f}-{x[1]:.1f}" for x in det[:10]) + (" …" if len(det) > 10 else ""))
    if results:
        tot_det = sum(r["found"] for r in results)
        tot_gt = sum(r["gt"] for r in results)
        tot_hit = sum(r["hit"] for r in results)
        p = tot_hit / tot_det if tot_det else 0.0
        rr = tot_hit / tot_gt if tot_gt else 0.0
        f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
        summary = {"strict": strict, "rows": results, "det": tot_det, "gt": tot_gt, "hit": tot_hit,
                   "params": {"percentile": percentile, "tau": tau},
                   "macro": {"precision": round(p, 3), "recall": round(rr, 3), "f1": round(f1, 3)}}
        for label, subset in (("same_video", [r for r in results if r["same"]]),
                              ("cross_video", [r for r in results if not r["same"]])):
            d = sum(r["found"] for r in subset); g = sum(r["gt"] for r in subset); h = sum(r["hit"] for r in subset)
            summary[label] = {"found": d, "gt": g, "hit": h,
                              "precision": round(h / d, 3) if d else 0.0,
                              "recall": round(h / g, 3) if g else 0.0}
        if verbose:
            print(f"\nИТОГ (строгий режим={strict}): найдено {tot_det}, разметка {tot_gt}, совпало {tot_hit} "
                  f"| precision={p:.2f} recall={rr:.2f} F1={f1:.2f}")
            for label in ("same_video", "cross_video"):
                s = summary[label]
                print(f"  {label:<12} найдено {s['found']:>3}, совпало {s['hit']:>3} из {s['gt']:>3} "
                      f"| precision={s['precision']:.2f} recall={s['recall']:.2f}")
            out = ROOT / "data" / "gt" / ("search_report.json" if strict else "search_report_plain.json")
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
            print("отчёт:", out.relative_to(ROOT))
        return summary
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--both", action="store_true", help="сравнить строгий и обычный режимы")
    ap.add_argument("--percentile", type=float, default=None, help="квантиль нулевого распределения (порог)")
    ap.add_argument("--tau", type=float, default=None, help="порог касания в единицах межзрачкового расстояния")
    ap.add_argument("--grid", action="store_true", help="перебрать порог и tau, показать precision/recall")
    args = ap.parse_args()
    if args.grid:
        print(f"{'tau':>5} {'pct':>5} {'найдено':>8} {'совпало':>8} {'P':>6} {'R':>6} {'F1':>6}")
        for tau in (1.0, 1.5):
            for pct in (2.0, 5.0, 10.0, 20.0):
                r = run_eval(strict=True, verbose=False, percentile=pct, tau=tau)
                if r:
                    print(f"{tau:>5} {pct:>5} {r['det']:>8} {r['hit']:>8} "
                          f"{r['macro']['precision']:>6.2f} {r['macro']['recall']:>6.2f} {r['macro']['f1']:>6.2f}")
        return
    run_eval(strict=True, percentile=args.percentile, tau=args.tau)
    if args.both:
        run_eval(strict=False, percentile=args.percentile, tau=args.tau)


if __name__ == "__main__":
    main()
