"""Офлайн-тесты ядра: паттерны, калибровка, слепки, строгий JSON.

Запуск (без сервера, без видео, без MediaPipe):
    python -m unittest discover -s tests -t .
"""
from __future__ import annotations
import json
import math
import unittest

import pipeline
import snapshots
from tests import synthetic


class TestPatternCalibration(unittest.TestCase):
    """Регрессии калибровки v5.1: мёртвые пороги не должны вернуться."""

    def test_p10d_fires_without_cheek_blendshape(self):
        """cheek=0 во всех кадрах (как в mediapipe 1.0.0) — P10D обязан работать по геометрии AU6."""
        fn = synthetic.pattern_fn("P10D")
        self.assertFalse(fn({"smile": 0.62, "cheek": 0.0, "ocu": 0.02}), "слабый AU6 не должен давать Duchenne")
        self.assertTrue(fn({"smile": 0.62, "cheek": 0.0, "ocu": 0.30}), "AU6 по геометрии глаза должен срабатывать")
        self.assertTrue(fn({"smile": 0.62, "cheek": 0.40, "ocu": 0.0}), "blendshape cheekRaise тоже учитывается")
        self.assertFalse(fn({"smile": 0.10, "cheek": 0.0, "ocu": 0.90}), "без улыбки Duchenne нет")

    def test_p12_uses_yaw_only(self):
        """P12 — отведение взгляда по yaw; pitch (кивок) не должен его включать."""
        fn = synthetic.pattern_fn("P12")
        self.assertTrue(fn({"yc": math.radians(30), "pc": 0.0}))
        self.assertFalse(fn({"yc": math.radians(10), "pc": 0.0}))
        self.assertFalse(fn({"yc": 0.0, "pc": math.radians(40)}), "кивок не является отведением взгляда")
        self.assertFalse(fn({"yc": None, "pc": None}))

    def test_p14_requires_speech_context_and_window(self):
        """Пауза в речи: 0.5-3 c и только внутри речевого контекста."""
        fn = synthetic.pattern_fn("P14")
        self.assertTrue(fn({"pause": 1.0, "pspeech": 1}))
        self.assertFalse(fn({"pause": 0.3, "pspeech": 1}), "короче 0.5 c — не пауза")
        self.assertFalse(fn({"pause": 3.5, "pspeech": 1}), "дольше 3 c — отсутствие речи, а не пауза")
        self.assertFalse(fn({"pause": 1.0, "pspeech": 0}), "без речевого контекста паузы нет")
        self.assertTrue(fn({"pause": 1.0}), "легаси-записи без pspeech считаются по окну длительности")

    def test_calibration_travels_into_summary(self):
        s = synthetic.summary_from(synthetic.make_records())
        self.assertEqual(s["calibration"], pipeline.CALIBRATION)
        self.assertEqual(s["calibration"]["p12_gaze_avoidance_yaw_deg"], pipeline.GAZE_AVOIDANCE_YAW_DEG)
        self.assertIn("p10d_note", s["calibration"])
        self.assertTrue(s["pipeline_version"].endswith("calibrated"))

    def test_counters_reflect_windows(self):
        s = synthetic.summary_from(synthetic.make_records())
        c = s["counters"]
        self.assertEqual(c["P01"], 3, "три моргания = три события P01")
        self.assertEqual(s["events"]["P01"], 3)
        self.assertEqual(c["P12"], 4, "окно отведения взгляда 4 кадра")
        self.assertGreater(c["P10D"], 0, "Duchenne должен находиться в окне улыбки")
        self.assertGreater(c["P14"], 0)
        for pid, cnt in c.items():
            self.assertGreaterEqual(cnt, 0)
            self.assertLessEqual(s["events"][pid], cnt, f"{pid}: событий не может быть больше кадров")

    def test_p14_not_triggered_by_long_silence(self):
        """Длинная тишина (pause>=3 c) не должна считаться паузой в речи."""
        recs = synthetic.make_records()
        long_sil = [r for r in recs if (r["pause"] or 0) >= 3.0]
        self.assertTrue(long_sil, "в фикстуре должен быть участок длинной тишины")
        fn = synthetic.pattern_fn("P14")
        self.assertFalse(any(fn(r) for r in long_sil))


class TestBlinkDetector(unittest.TestCase):
    def test_refractory_suppresses_double_count(self):
        recs = [{"t": i * 0.125, "blink": 0.0} for i in range(8)]
        for i in (2, 3):                      # одно моргание, «залипшее» на два кадра
            recs[i]["blink"] = 0.9
        self.assertEqual(len(pipeline.blink_events(recs)), 1)
        for i in (2, 6):
            recs[i]["blink"] = 0.9
        self.assertEqual(len(pipeline.blink_events(recs)), 2)


class TestSnapshotSearch(unittest.TestCase):
    def setUp(self):
        self.recs = synthetic.make_records()
        self.sum = synthetic.summary_from(self.recs)

    def test_create_and_search_contract(self):
        snap = snapshots.create_snapshot(self.recs, 2.0, 3.5,
                                        ["smile", "ocu", "hand_speed", "yc"], "тест")
        self.assertEqual(snap["version"], snapshots.SNAP_VERSION)
        self.assertTrue(snap["series"])
        self.assertEqual(len(snap["series"][snap["channels"][0]]), snapshots.N)
        res, meta = snapshots.search(self.recs, snap, hop=0.25, top_k=5)
        self.assertIsInstance(res, list)
        self.assertIsInstance(meta, dict)
        for key in ("total", "returned", "version", "pattern", "mode"):
            self.assertIn(key, meta)
        self.assertLessEqual(len(res), 5)
        dists = [x["distance"] for x in res]
        self.assertEqual(dists, sorted(dists), "результаты должны быть отсортированы по distance")

    def test_search_is_deterministic(self):
        snap = snapshots.create_snapshot(self.recs, 5.0, 5.5, ["yc", "pc"], "взгляд")
        a, _ = snapshots.search(self.recs, snap, top_k=3)
        b, _ = snapshots.search(self.recs, snap, top_k=3)
        self.assertEqual(a, b)

    def test_snapshot_rejects_short_window(self):
        with self.assertRaises(ValueError):
            snapshots.create_snapshot(self.recs, 1.0, 1.1, ["smile"], "слишком короткое")


class TestStrictJson(unittest.TestCase):
    def test_nan_and_inf_become_null(self):
        payload = {"a": float("nan"), "b": float("inf"), "c": -float("inf"),
                   "d": [1.0, float("nan")], "e": {"f": float("nan")}, "g": "текст"}
        safe = pipeline._json_safe(payload)
        self.assertIsNone(safe["a"]); self.assertIsNone(safe["b"]); self.assertIsNone(safe["c"])
        self.assertIsNone(safe["d"][1]); self.assertIsNone(safe["e"]["f"])
        text = json.dumps(safe, ensure_ascii=False)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text)["g"], "текст")

    def test_old_metrics_files_are_parsable(self):
        """Старые прогоны писались с литеральными NaN: json.loads их принимает,
        а API-слой отдаёт такие значения как null (проверяется в test_api)."""
        import pathlib
        data = pathlib.Path(__file__).resolve().parent.parent / "data"
        for p in sorted(data.glob("*/metrics.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.assertIsInstance(json.loads(line), dict)
            break   # достаточно одного файла: формат одинаковый


if __name__ == "__main__":
    unittest.main()
