"""Тесты v5.2: геометрия касания лица, порог «совпадение / фон», поиск в чужом видео.

Все проверки — на синтетике, без MediaPipe: каналы касания задаются вручную,
поэтому тесты быстрые и детерминированные.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

import pipeline
import snapshots


def _rec(t, touch=None, twrist=None, **kw):
    r = {"t": t, "touch": touch, "twrist": twrist}
    r.update(kw)
    return r


def _rows(n, fps=8.0, touch_windows=(), twrist_windows=()):
    """Кадры с касанием в заданных интервалах (по индексам кадров)."""
    out = []
    for i in range(n):
        t = round(i / fps, 3)
        r = _rec(t)
        for a, b in touch_windows:
            if a <= i < b:
                r["touch"] = 0.2
        for a, b in twrist_windows:
            if a <= i < b:
                r["twrist"] = 2.0
        out.append(r)
    return out


class TestTouchFrame(unittest.TestCase):
    def test_hand_distance_threshold(self):
        self.assertTrue(pipeline.touch_frame(_rec(0.0, touch=pipeline.TOUCH_MAX_TAU - 0.01)))
        self.assertFalse(pipeline.touch_frame(_rec(0.0, touch=pipeline.TOUCH_MAX_TAU + 0.01)))

    def test_wrist_fallback_only_when_hand_missing(self):
        # кисть не найдена -> решает запястье
        self.assertTrue(pipeline.touch_frame(_rec(0.0, touch=None, twrist=pipeline.TWRIST_MAX_TAU - 0.1)))
        self.assertFalse(pipeline.touch_frame(_rec(0.0, touch=None, twrist=pipeline.TWRIST_MAX_TAU + 0.5)))
        # кисть найдена далеко, но запястье близко: приоритет у кисти (её показания точнее)
        self.assertFalse(pipeline.touch_frame(_rec(0.0, touch=5.0, twrist=1.0)))

    def test_nan_and_missing_are_not_touch(self):
        self.assertFalse(pipeline.touch_frame(_rec(0.0, touch=float("nan"), twrist=float("nan"))))
        self.assertFalse(pipeline.touch_frame({"t": 0.0}))

    def test_legacy_records_use_old_rule(self):
        """Старые metrics.jsonl без каналов касания: прежняя оценка (два сигнала из трёх)."""
        two_signals = {"t": 0.0, "hands": 1, "tif": 1.0, "hfd": 0.5, "hfd2": 1.5, "face_conf": 0.9}
        one_signal = {"t": 0.0, "hands": 1, "tif": 1.0, "hfd": 1.4, "hfd2": 1.5, "face_conf": 0.9}
        self.assertTrue(pipeline.touch_frame(two_signals))
        self.assertFalse(pipeline.touch_frame(one_signal))
        self.assertFalse(pipeline.touch_frame({"t": 0.0, "hands": 0, "tif": 1.0, "hfd": 0.5, "hfd2": 0.4}))
        self.assertFalse(pipeline.touch_frame({"t": 0.0, "hands": 1, "tif": 1.0, "hfd": 0.5, "face_conf": 0.1}))


class TestTouchSpans(unittest.TestCase):
    def test_single_span(self):
        recs = _rows(80, touch_windows=[(20, 30)])
        spans = pipeline.touch_spans(recs, 8.0)
        self.assertEqual(spans, [(20, 29)])

    def test_short_blip_is_dropped(self):
        recs = _rows(80, touch_windows=[(20, 22)])          # 2 кадра = 0.25 c
        self.assertEqual(pipeline.touch_spans(recs, 8.0), [])

    def test_gap_is_glued(self):
        # два касания с разрывом 1 кадр (0.125 c < 0.25 c) — один эпизод
        recs = _rows(80, touch_windows=[(20, 26), (27, 33)])
        spans = pipeline.touch_spans(recs, 8.0)
        self.assertEqual(spans, [(20, 32)])

    def test_wrist_fallback_creates_span(self):
        recs = _rows(80, twrist_windows=[(40, 50)])
        self.assertEqual(pipeline.touch_spans(recs, 8.0), [(40, 49)])

    def test_ratio(self):
        recs = _rows(80, touch_windows=[(0, 8)])
        self.assertAlmostEqual(pipeline.touch_ratio(recs, 8.0), 8 / 80, places=3)

    def test_empty(self):
        self.assertEqual(pipeline.touch_spans([]), [])
        self.assertEqual(pipeline.touch_ratio([]), 0.0)


class TestGateAndNull(unittest.TestCase):
    def test_gate_uses_new_channels(self):
        self.assertTrue(snapshots._face_touch_gate(_rec(0.0, touch=0.3)))
        self.assertFalse(snapshots._face_touch_gate(_rec(0.0, touch=3.0)))
        self.assertTrue(snapshots._face_touch_gate(_rec(0.0, twrist=2.0)))

    def test_null_distances_deterministic(self):
        recs = _rows(200, touch_windows=[(50, 60)])
        for r in recs:
            r["smile"] = 0.5 if r["t"] < 5 else 0.05
        snap = {"id": "abcdef1234", "duration": 1.0, "channels": ["smile"],
                "series": {"smile": list(np.linspace(0, 1, snapshots.N))}}
        S = {"smile": np.asarray(snap["series"]["smile"], dtype=float)}
        a = snapshots._null_distances(recs, snap, ["smile"], S, 1.0, 8.0)
        b = snapshots._null_distances(recs, snap, ["smile"], S, 1.0, 8.0)
        self.assertEqual(len(a), snapshots.NULL_WINDOWS)
        self.assertEqual(a, b, "нулевое распределение должно быть детерминированным")
        self.assertTrue(all(x >= 0 for x in a))


class TestCrossVideo(unittest.TestCase):
    """Поиск в ЧУЖОМ видео не должен пересчитывать окно слепка по таймкодам источника."""

    def _records(self, n=240, fps=8.0):
        recs = []
        for i in range(n):
            smile = 0.5 if 5 <= i < 10 else 0.05          # P10 активен вне окна слепка
            yc = 0.05 if 100 <= i < 130 else 0.0          # P03 активен только в окне слепка
            recs.append({"t": round(i / fps, 3), "smile": smile, "yc": yc, "pc": 0.0,
                         "hand_speed": 0.02, "aperture": 1.2, "hfd": 1.4, "hfd2": 1.5,
                         "tif": 0.0, "hands": 0, "menergy": 0.005, "touch": None, "twrist": None,
                         "blink": 0.0, "browDown": 0.0, "press": 0.0, "jawO": 0.0, "eyeWide": 0.0,
                         "frown": 0.0, "browUp": 0.0, "noseW": 0.0, "cheek": 0.0, "ocu": 0.0})
        return recs

    def _snapshot(self):
        bits = {d[0]: i for i, d in enumerate(snapshots.PAT_DEFS)}
        pm = (1 << bits["P10"]) | (1 << bits["P03"])
        return {"id": "snap000001", "name": "t", "source_video": "SOURCE",
                "duration": 3.0, "t0": 12.5, "t1": 15.5, "t0_req": 12.5, "t1_req": 15.5,
                "channels": ["smile"], "series": {"smile": list(np.linspace(-1, 1, snapshots.N))},
                "pattern_mask": pm, "pattern_active": ["P03", "P10"],
                "pattern_frac": {"P10": 0.9, "P03": 0.1}, "touch_ratio": 0.0}

    def test_same_video_uses_window_mask(self):
        recs = self._records()
        _res, meta = snapshots.search(recs, self._snapshot(), top_k=5, null_threshold=False)
        self.assertFalse(meta["cross_video"])
        self.assertEqual(meta["pattern"], "P03", "в своём видео окно слепка берётся по таймкодам")

    def test_other_video_uses_stored_snapshot_fracs(self):
        recs = self._records()
        _res, meta = snapshots.search(recs, self._snapshot(), top_k=5, null_threshold=False,
                                      target_video="TARGET")
        self.assertTrue(meta["cross_video"])
        self.assertEqual(meta["pattern"], "P10",
                         "в чужом видео приоритет у паттерна, перепредставленного в САМОМ слепке")


class TestMatchThreshold(unittest.TestCase):
    def _records(self, n=240, fps=8.0):
        rnd = np.random.RandomState(7)
        return [{"t": round(i / fps, 3), "smile": float(rnd.rand() * 0.2), "yc": 0.05, "pc": 0.0,
                 "hand_speed": 0.02, "aperture": 1.2, "hfd": 1.4, "hfd2": 1.5, "tif": 0.0,
                 "hands": 0, "menergy": 0.005, "touch": None, "twrist": None, "blink": 0.0,
                 "browDown": 0.0, "press": 0.0, "jawO": 0.0, "eyeWide": 0.0, "frown": 0.0,
                 "browUp": 0.0, "noseW": 0.0, "cheek": 0.0, "ocu": 0.0} for i in range(n)]

    def _snapshot(self):
        rnd = np.random.RandomState(11)
        return {"id": "snap000002", "name": "шум", "source_video": "S", "duration": 2.0,
                "t0": 1.0, "t1": 3.0, "channels": ["smile"],
                "series": {"smile": list(rnd.rand(snapshots.N))},
                "pattern_mask": 0, "pattern_active": [], "pattern_frac": {}, "touch_ratio": 0.0}

    def test_noise_search_reports_no_match(self):
        recs = self._records()
        res, meta = snapshots.search(recs, self._snapshot(), top_k=10)
        self.assertIsNotNone(meta["match_threshold"])
        self.assertGreater(meta["null_size"], 0)
        self.assertEqual(res, [], "на шуме порог должен отсечь все «находки»")
        self.assertGreater(meta["dropped_below_threshold"], 0)
        self.assertIn("не найдено", meta.get("note", ""))

    def test_threshold_can_be_disabled(self):
        recs = self._records()
        res, meta = snapshots.search(recs, self._snapshot(), top_k=10, null_threshold=False)
        self.assertTrue(res, "без порога поиск возвращает top-K как раньше")
        self.assertIsNone(meta["match_threshold"])


if __name__ == "__main__":
    unittest.main()
