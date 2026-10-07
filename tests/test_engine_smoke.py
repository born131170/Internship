"""Опциональный smoke-тест движка: настоящий MediaPipe на синтетическом видео.

По умолчанию пропускается (нужны models/*.task, ~10-20 c). Включить:
    Windows:  set PERSONASCOPE_ENGINE_TEST=1 && python -m unittest discover -s tests -t .
    Linux:    PERSONASCOPE_ENGINE_TEST=1 python -m unittest discover -s tests -t .

Проверяет то, что офлайн-тесты проверить не могут: инициализацию landmarker'ов,
полный проход по кадрам, запись строгого metrics.jsonl и сборку summary.
"""
from __future__ import annotations
import json
import os
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "tests" / "_tmp_engine"


@unittest.skipUnless(os.environ.get("PERSONASCOPE_ENGINE_TEST") == "1",
                     "тяжёлый тест: включите PERSONASCOPE_ENGINE_TEST=1")
class TestEngineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import numpy as np
        shutil.rmtree(TMP, ignore_errors=True)
        (TMP / "out").mkdir(parents=True, exist_ok=True)
        cls.video = TMP / "synthetic.mp4"
        w, h, fps, n = 320, 240, 8, 24
        wr = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for i in range(n):
                frame = np.zeros((h, w, 3), np.uint8)
                frame[:, :, 1] = 40
                cv2.circle(frame, (20 + i * 5, 120), 30, (190, 190, 190), -1)
                wr.write(frame)
        finally:
            wr.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_full_pipeline(self):
        import pipeline
        summary = pipeline.analyze_video(self.video, TMP / "out", stride=2, width=240,
                                        make_dashboard=False, make_clips=False, personality=True)
        self.assertEqual(summary["pipeline_version"], pipeline.PIPELINE_VERSION)
        self.assertIn("calibration", summary)
        self.assertEqual(len(summary["counters"]), len(pipeline.PATTERN_DEFS))
        self.assertEqual(summary["counters"].keys(), summary["events"].keys())
        self.assertIn("ocu", summary["stats"])
        self.assertIn("personality", summary)
        # metrics.jsonl — строгий JSON (его читают jq/JS/сторонние инструменты)
        raw = (TMP / "out" / "metrics.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        self.assertEqual(len(rows), summary["n_frames"])
        self.assertTrue(all("pspeech" in r for r in rows))
        # summary.json перечитывается тем же способом, что и сервером
        saved = json.loads((TMP / "out" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["counters"], summary["counters"])


if __name__ == "__main__":
    unittest.main()
