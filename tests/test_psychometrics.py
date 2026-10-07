"""Тесты психометрического слоя: LR-агрегация, AU6/Duchenne, честность выводов."""
from __future__ import annotations
import math
import unittest

import psychometrics as psy
from tests import synthetic


class TestBehaviorExtraction(unittest.TestCase):
    def setUp(self):
        self.recs = synthetic.make_records()
        self.b = psy.extract_behavior(self.recs)

    def test_channels_present(self):
        b = self.b
        for key in ("blink_rate_bpm", "duchenne_ratio", "smile_time_ratio", "gaze_avoidance_ratio",
                    "gesture_rate_per_min", "self_touch_rate_per_min", "pause_per_min", "_coverage", "_valid"):
            self.assertIn(key, b)
        self.assertAlmostEqual(b["blink_rate_bpm"], 15.2, delta=0.2, msg="3 моргания за ~11.9 c")
        self.assertEqual(b["_coverage"]["face_visibility_ratio"], 1.0)

    def test_duchenne_reads_cheek_channel(self):
        """Регрессия: канал называется cheek (blendshape cheekRaise не пишется вовсе)."""
        recs = synthetic.make_records(with_ocu=False, cheek_value=0.40)
        b = psy.extract_behavior(recs)
        self.assertIsNotNone(b["duchenne_ratio"])
        self.assertGreater(b["duchenne_ratio"], 0.0, "cheek>0.15 в окне улыбки должен давать Duchenne")

    def test_duchenne_from_eye_geometry_without_ocu_channel(self):
        """Без канала ocu AU6 считается по lm — старые metrics.jsonl остаются рабочими."""
        recs = synthetic.make_records(with_ocu=False, cheek_value=0.0)
        b = psy.extract_behavior(recs)
        self.assertIsNotNone(b["duchenne_ratio"])
        self.assertGreater(b["duchenne_ratio"], 0.0)

    def test_gaze_avoidance_uses_shared_threshold(self):
        self.assertAlmostEqual(self.b["gaze_avoidance_ratio"], 4 / 96, places=3)

    def test_missing_channels_do_not_crash(self):
        bare = [{"t": i * 0.125, "face_conf": None, "smile": float("nan")} for i in range(40)]
        b = psy.extract_behavior(bare)
        self.assertFalse(b["_valid"]["face"])

    def test_empty_records(self):
        b, cov = psy.extract_behavior([])
        self.assertEqual(b, {})
        self.assertFalse(cov["video_valid"])


class TestTraitScores(unittest.TestCase):
    def test_scores_and_ci_within_bounds(self):
        b = psy.extract_behavior(synthetic.make_records())
        bf = psy.compute_trait_scores(b)
        for trait, tr in bf.items():
            if tr["score"] is None:
                self.assertEqual(tr["status"], "insufficient_data")
                continue
            self.assertGreaterEqual(tr["score"], 5.0)
            self.assertLessEqual(tr["score"], 95.0)
            lo, hi = tr["ci95"]
            self.assertLessEqual(lo, tr["score"])
            self.assertLessEqual(tr["score"], hi)
            self.assertLessEqual(lo, hi)
            self.assertGreaterEqual(tr["n_indicators"], 1)
            self.assertIn(tr["status"], ("ok", "partial"))

    def test_lr_bounded(self):
        self.assertAlmostEqual(psy._lr(10.0, 0.0, 1.0), 8.0, places=6)
        self.assertAlmostEqual(psy._lr(-10.0, 0.0, 1.0), 1 / 8, places=6)
        self.assertLess(psy._lr(0.05, 0.05, 0.45), psy._lr(0.44, 0.05, 0.45), "высокое значение -> выше LR")

    def test_audio_indicators_skipped_without_audio(self):
        b = psy.extract_behavior(synthetic.make_records())
        b["_valid"]["audio"] = False
        bf = psy.compute_trait_scores(b)
        self.assertFalse(any(d["var"].startswith(("speech_rate", "pause")) for d in bf["conscientiousness"]["basis"]))


class TestAssess(unittest.TestCase):
    def test_assess_shape_and_validity_labels(self):
        res = psy.assess(synthetic.make_records())
        self.assertEqual(res["version"], "psychometrics-1.0")
        for key in ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"):
            self.assertIn(key, res["big_five"])
        self.assertEqual(res["model_validity"]["mbti"]["support"], "none")
        self.assertEqual(res["derived"]["enneagram"]["status"], "not_assessed")
        self.assertEqual(res["derived"]["pid5"]["status"], "not_assessed")
        self.assertEqual(res["derived"]["mbti"]["support"]["support"], "none")
        self.assertIn("_S_N_warning", res["derived"]["mbti"]["axes"])

    def test_deception_is_not_a_lie_probability(self):
        res = psy.assess(synthetic.make_records())
        t = res["truthfulness"]
        self.assertGreaterEqual(t["score"], 5.0)
        self.assertLessEqual(t["score"], 95.0)
        self.assertTrue(t["disclaimer_required"])
        self.assertIn("не различимо", t["verdict"])
        self.assertIn("54%", t["interpretation_limit"])

    def test_deterministic(self):
        recs = synthetic.make_records()
        self.assertEqual(psy.assess(recs), psy.assess(recs))

    def test_pipeline_and_psychometrics_share_au6(self):
        """AU6 считается один раз: формула живёт в pipeline, psychometrics читает канал ocu."""
        import pipeline
        lm = {"L": [0.40, 0.40], "B": [0.40, 0.41], "O": [0.44, 0.40],
              "R": [0.60, 0.40], "b": [0.60, 0.41], "o": [0.56, 0.40]}
        expected = pipeline.ocu_from_landmarks(lm)
        self.assertAlmostEqual(expected, 1.0 - (0.01 / 0.04) / 0.55, places=6)
        self.assertFalse(math.isnan(expected))
        self.assertTrue(math.isnan(pipeline.ocu_from_landmarks(None)))
        # канал ocu имеет приоритет над геометрией (считается в конвейере)
        recs = synthetic.make_records(with_ocu=True)
        recs[16]["lm"] = None
        b = psy.extract_behavior(recs)
        self.assertGreater(b["duchenne_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
