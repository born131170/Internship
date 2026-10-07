"""Тесты скоринга, LLM-контура и конфигурации (без сети)."""
from __future__ import annotations
import json
import os
import shutil
import unittest
from pathlib import Path

import llm
import scoring
from tests import synthetic


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.sum = synthetic.summary_from(synthetic.make_records())
        self.sc = scoring.compute_scores(self.sum)

    def test_truthfulness_single_source(self):
        """И вердикт LLM, и датчик /truth берут одно число из psychometrics."""
        psy_score = round(float(self.sum["personality"]["truthfulness"]["score"]))
        self.assertEqual(self.sc["truthfulness"]["source"], "psychometrics-load")
        self.assertEqual(self.sc["truthfulness"]["score"], psy_score)
        self.assertEqual(self.sc["truthfulness"]["score"], round(self.sc["truthfulness"]["score"]))

    def test_truthfulness_fallback_without_personality(self):
        s = synthetic.summary_from(synthetic.make_records(), personality=False)
        sc = scoring.compute_scores(s)
        self.assertEqual(sc["truthfulness"]["source"], "pipeline-heuristic")
        self.assertEqual(sc["truthfulness"]["score"], s["truth_heuristic"])

    def test_big_five_come_from_psychometrics(self):
        self.assertEqual(self.sc["_engine"], "psychometrics-LR")
        for k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            self.assertGreaterEqual(self.sc["big_five"][k], 5.0)
            self.assertLessEqual(self.sc["big_five"][k], 95.0)

    def test_derived_systems_are_consistent(self):
        sc = self.sc
        E = sc["big_five"]["extraversion"]; A = sc["big_five"]["agreeableness"]
        C = sc["big_five"]["conscientiousness"]; N = sc["big_five"]["neuroticism"]
        O = sc["big_five"]["openness"]
        axes = sc["mbti"]["axes"]
        self.assertEqual((axes["E_I"], axes["S_N"], axes["T_F"], axes["J_P"]), (E, O, A, C))
        self.assertEqual(len(sc["mbti"]["type"]), 4)
        self.assertTrue(set(sc["mbti"]["type"][0]) <= {"E", "I"})
        self.assertEqual(sc["pid5"]["detachment"], round(max(5, min(95, 100 - E)), 1))
        self.assertAlmostEqual(sum(sc["temperament"][k] for k in
                                   ("sanguine", "choleric", "melancholic", "phlegmatic")), 100, delta=2)
        self.assertGreaterEqual(N, 5.0)

    def test_method_text_is_attached(self):
        self.assertTrue(self.sc["scoring_method"].startswith("z-якоря"))
        self.assertEqual(self.sc["version"], "1.2")


class TestLlamaParsing(unittest.TestCase):
    def test_parse_plain_and_fenced(self):
        self.assertEqual(llm.parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(llm.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm.parse_json('Вот результат: {"a": 1} — готово')["a"], 1)
        self.assertIsNone(llm.parse_json("совсем не json"))

    def test_repair_truncated_json(self):
        broken = '{"summary": "ок", "big_five": {"openness": 60, "conscientiousness": 40'
        fixed = llm._repair_json(broken)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed["summary"], "ок")
        self.assertEqual(fixed["big_five"]["openness"], 60)

    def test_validate_drops_unknown_episodes_and_clamps(self):
        s = synthetic.summary_from(synthetic.make_records())
        s["episodes"] = [{"id": "P10_00", "pattern": "P10", "name": "улыбка", "t0": 1.0, "t1": 2.0}]
        parsed = {
            "summary": "текст",
            "big_five": {"openness": 150, "conscientiousness": None},
            "truthfulness": {"score": 99, "verdict": "", "cues": ["строка", {"name": "моргание"}]},
            "evidence": {"big_five": [{"episode_id": "P10_00"}, {"episode_id": "НЕТ_ТАКОГО"}]},
        }
        out, warn = llm.validate_result(parsed, s)
        self.assertEqual(out["big_five"]["openness"], 100.0)
        self.assertEqual(out["big_five"]["conscientiousness"], 50.0, "пропуск заполняется нормой")
        self.assertEqual([e["episode_id"] for e in out["evidence"]["big_five"]], ["P10_00"])
        self.assertTrue(any("несуществующий эпизод" in w for w in warn))
        self.assertTrue(any("вне 0..100" in w for w in warn))
        self.assertEqual(out["truthfulness"]["score"], scoring.compute_scores(s)["truthfulness"]["score"])
        self.assertTrue(any("заменён детерминированным" in w for w in warn))
        self.assertEqual(out["truthfulness"]["cues"], [{"cue": "строка", "direction": ""},
                                                       {"cue": "моргание", "direction": ""}])

    def test_confidence_bounds_and_penalty(self):
        s = synthetic.summary_from(synthetic.make_records())
        full = dict(s)
        parsed = {"summary": "ок", "big_five": {k: 50 for k in
                  ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")},
                  "hexaco": {k: 50 for k in "HEXACO"}, "pid5": {k: 50 for k in
                  ("negative_affect", "detachment", "antagonism", "disinhibition", "psychoticism")},
                  "temperament": {k: 25 for k in ("sanguine", "choleric", "melancholic", "phlegmatic")},
                  "mbti": {"axes": {"E_I": 50, "S_N": 50, "T_F": 50, "J_P": 50}}, "enneagram": {"type": 3},
                  "truthfulness": {"score": 50},
                  "evidence": {k: [{"episode_id": "x"}] for k in
                               ("big_five", "mbti", "enneagram", "temperament", "hexaco", "pid5", "truthfulness")}}
        self.assertEqual(llm.compute_confidence(parsed, []), 1.0)
        self.assertLess(llm.compute_confidence(parsed, ["a", "b", "c"]), 1.0)
        self.assertEqual(llm.compute_confidence(None, []), 0.0)
        # штраф считается по УНИКАЛЬНЫМ предупреждениям и ограничен 0.3
        self.assertAlmostEqual(llm.compute_confidence(parsed, ["x"] * 50), 0.95, places=2)
        self.assertAlmostEqual(llm.compute_confidence(parsed, [f"w{i}" for i in range(50)]), 0.7, places=2)

    def test_evidence_contains_calibration_and_psychometrics(self):
        s = synthetic.summary_from(synthetic.make_records())
        ev = llm.build_evidence(s)
        self.assertIn("psychometrics", ev)
        self.assertIn("truth_heuristic", ev)
        self.assertIn("legacy", ev["truth_heuristic_definition"])


class TestConfigSources(unittest.TestCase):
    """Файлы создаются в tests/_tmp: системный временный каталог недоступен под песочницей."""

    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_tmp" / self._testMethodName
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dotenv_parser(self):
        p = self.tmp / ".env.local"
        p.write_text('# комментарий\nLLM_API_KEY="sk-test"\n\nLLM_MODEL=my-model\nBADLINE\n', encoding="utf-8")
        env = llm._dotenv(p)
        self.assertEqual(env["LLM_API_KEY"], "sk-test")
        self.assertEqual(env["LLM_MODEL"], "my-model")
        self.assertEqual(len(env), 2)
        self.assertEqual(llm._dotenv(self.tmp / "нет-файла"), {})

    def test_env_overrides_json(self):
        (self.tmp / "llm_config.json").write_text(
            json.dumps({"base_url": "http://json/v1", "api_key": "from-json", "model": "json-model"}),
            encoding="utf-8")
        old = {k: os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_TEMPERATURE")}
        try:
            os.environ["LLM_API_KEY"] = "from-env"
            os.environ["LLM_BASE_URL"] = "http://env/v1"
            os.environ["LLM_TEMPERATURE"] = "0.9"
            cfg = llm.load_config(self.tmp)
            self.assertEqual(cfg["api_key"], "from-env")
            self.assertEqual(cfg["base_url"], "http://env/v1")
            self.assertEqual(cfg["temperature"], 0.9)
            self.assertEqual(cfg["model"], "json-model", "не переопределённое поле берётся из json")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_save_roundtrip(self):
        llm.save_config(self.tmp, {"base_url": "http://x/v1", "api_key": "k", "model": "m"})
        cfg = llm.load_config(self.tmp)
        self.assertEqual((cfg["base_url"], cfg["api_key"], cfg["model"]), ("http://x/v1", "k", "m"))


if __name__ == "__main__":
    unittest.main()
