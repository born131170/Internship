"""Регрессии v5.1 (после fix_search_verdict.py).

1) Вердикт LLM печатался по букве на строку: поле cues приходило строкой, а код
   разбирал её как последовательность символов.
2) Поиск эпизодов отдавал окно на всю минуту («00:17-01:10») при слепке 3 с и выбирал
   целевым «вездесущий» паттерн (P03 движения головы) вместо P06 «рука-лицо».

Если патчер ещё не запускался, тесты пропускаются, а не падают.
"""
from __future__ import annotations
import unittest

import snapshots


def _mk_records(n=240, fps=8.0, touch=(92, 109)):
    """P07 (hand_speed>0.01) активен везде, P03 (движения головы) — везде, кроме касания,
    P06 (рука у лица) — только в окне touch."""
    recs = []
    for i in range(n):
        in_touch = touch[0] <= i < touch[1]
        rec = {"t": round(i / fps, 3), "face_conf": 0.9, "pose_conf": 0.8,
               "hands": 1 if in_touch else 0, "emotion": "none",
               "smile": 0.05, "frown": 0.0, "browUp": 0.0, "browDown": 0.0, "eyeWide": 0.0,
               "blink": 0.0, "jawO": 0.0, "noseW": 0.0, "press": 0.0, "cheek": 0.0, "ocu": 0.0,
               "yaw": 0.0 if in_touch else 0.0349, "pitch": 0.0 if in_touch else 0.0175,
               "yc": 0.0 if in_touch else 0.0349, "pc": 0.0 if in_touch else 0.0175,
               "hand_speed": 0.02, "aperture": 1.2, "hfd": 1.4, "hfd2": 1.5,
               "tif": 1.0 if in_touch else 0.0, "menergy": 0.005,
               "rms": 0.0, "f0": None, "srate": 0.0, "pause": 0.0, "pspeech": 0,
               "voicemo": 0.0, "speech": 0}
        recs.append(rec)
    return recs


class TestCueNormalization(unittest.TestCase):
    def setUp(self):
        import llm
        self.llm = llm
        if not hasattr(llm, "_cue_list"):
            self.skipTest("правка _cue_list не применена (запустите fix_search_verdict.py)")

    def test_string_becomes_one_cue(self):
        self.assertEqual(self.llm._cue_list("morganie 21.1/мин норма"),
                         ["morganie 21.1/мин норма"])
        self.assertEqual(self.llm._cue_list("   "), [])
        self.assertEqual(self.llm._cue_list(None), [])
        self.assertEqual(self.llm._cue_list({"cue": "x"}), [])

    def test_char_list_is_glued_back(self):
        """След старого бага, уже сохранённый в llm_result.json."""
        self.assertEqual(self.llm._cue_list(list("моргание 21/мин")), ["моргание 21/мин"])

    def test_normal_list_untouched(self):
        cues = [{"cue": "моргание", "direction": "повышает"}, "паузы ≥0.5 c"]
        self.assertEqual(self.llm._cue_list(cues), cues)

    def test_validate_result_keeps_string_as_single_cue(self):
        from tests import synthetic
        s = synthetic.summary_from(synthetic.make_records())
        parsed = {"truthfulness": {"score": 50, "verdict": "v", "cues": "моргание 21/мин норма"},
                  "evidence": {}}
        out, _w = self.llm.validate_result(parsed, s)
        self.assertEqual(len(out["truthfulness"]["cues"]), 1)
        self.assertEqual(out["truthfulness"]["cues"][0]["cue"], "моргание 21/мин норма")


@unittest.skipUnless(hasattr(snapshots, "_explode_span"),
                     "правка _explode_span не применена (запустите fix_search_verdict.py)")
class TestSearchConcretization(unittest.TestCase):
    def setUp(self):
        self.recs = _mk_records()
        self.mask = snapshots._pattern_mask(self.recs)
        self.fps = 8.0

    def test_long_interval_is_split_into_snapshot_sized_windows(self):
        dur_s = 3.0
        spans = snapshots._explode_span(self.recs, self.mask, ["hand_speed", "tif"],
                                        self.fps, 0, len(self.recs) - 1, dur_s)
        self.assertGreater(len(spans), 1, "интервал на 30 c должен разбиться на несколько окон")
        for a, b in spans:
            self.assertLessEqual(self.recs[b]["t"] - self.recs[a]["t"],
                                 snapshots.MAX_DUR_RATIO * dur_s + 1e-6)

    def test_short_interval_is_kept(self):
        spans = snapshots._explode_span(self.recs, self.mask, ["tif"], self.fps, 92, 108, 3.0)
        self.assertEqual(spans, [(92, 108)])

    def test_search_returns_no_minute_long_window(self):
        """Слепок по «вездесущему» P07: раньше возвращалось одно окно на всё видео."""
        snap = snapshots.create_snapshot(self.recs, 12.5, 13.25, ["hand_speed"], "жест")
        res, meta = snapshots.search(self.recs, snap, top_k=20)
        self.assertTrue(res, "должны найтись кандидаты")
        dur_s = float(snap["duration"])
        for item in res:
            self.assertLessEqual(item["t1"] - item["t0"],
                                 snapshots.MAX_DUR_RATIO * dur_s + 1e-6,
                                 f"окно {item['t0']}-{item['t1']} длиннее слепка более чем в "
                                 f"{snapshots.MAX_DUR_RATIO} раза")

    def test_touch_pattern_wins_over_ubiquitous_head_motion(self):
        """Слепок жеста «рука-лицо»: в окне есть P06 и P07, P03 активен почти везде.

        По доле в окне (старая логика) выигрывал P03/P07 и поиск уходил в движения головы.
        """
        snap = snapshots.create_snapshot(self.recs, 12.5, 13.25,
                                         ["tif", "hand_speed", "yc"], "рука-лицо")
        self.assertIn("P06", snap.get("pattern_active", []),
                      f"P06 должен доминировать в окне слепка: {snap.get('pattern_active')}")
        res, meta = snapshots.search(self.recs, snap, top_k=20)
        self.assertEqual(meta["pattern"], "P06",
                         f"целевой паттерн {meta['pattern']} вместо P06 (касание лица)")
        self.assertFalse(meta["face_touch_gate"],
                         "строгий гейт касания здесь выключен — тестируется выбор паттерна")
        self.assertTrue(res)
        for item in res:
            self.assertLessEqual(item["t1"] - item["t0"], 2.5 * float(snap["duration"]) + 1e-6)


if __name__ == "__main__":
    unittest.main()
