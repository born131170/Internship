"""Тесты HTTP-API на синтетическом прогоне (TestClient, без MediaPipe и без сети).

Данные пишутся в tests/_tmp (переопределение PERSONASCOPE_DATA), реальный data/ не трогается.
"""
from __future__ import annotations
import json
import os
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "tests" / "_tmp"
os.environ["PERSONASCOPE_DATA"] = str(TMP)   # до импорта app

import app as appmod                                  # noqa: E402
from fastapi.testclient import TestClient             # noqa: E402
from tests import synthetic                            # noqa: E402

VID = "testvid01"
VID_PREFIX = "testvid01extra"    # соседний каталог с именем-префиксом (проверка /file)
client = TestClient(appmod.app)


def setUpModule():
    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / "snapshots").mkdir(parents=True, exist_ok=True)   # app создаёт его при импорте
    d = TMP / VID
    (d / "episodes").mkdir(parents=True, exist_ok=True)
    recs = synthetic.make_records()
    lines = [json.dumps(r, ensure_ascii=False) for r in recs]
    lines.append(json.dumps({"t": 99.0, "hand_speed": float("nan"), "note": "старый формат без строгого JSON"}))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(synthetic.summary_from(recs), ensure_ascii=False), encoding="utf-8")
    (d / "episodes" / "P10_00.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    other = TMP / VID_PREFIX
    other.mkdir(parents=True, exist_ok=True)
    (other / "summary.json").write_text(json.dumps({"n_frames": 1}), encoding="utf-8")
    # реестр с идентификатором удалённого прогона: он должен отфильтроваться
    (TMP / "index.json").write_text(json.dumps([{"id": "deleted01", "finished": 1},
                                                {"id": VID, "finished": 2}]), encoding="utf-8")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestBasics(unittest.TestCase):
    def test_index_page(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("PersonaScope", r.text)

    def test_latest_prunes_missing_runs(self):
        r = client.get("/api/videos/latest")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], VID, "запись об удалённом прогоне не должна выдаваться как последняя")

    def test_missing_video_is_404(self):
        self.assertEqual(client.get("/api/videos/неттакого/summary").status_code, 404)

    def test_summary_and_metrics(self):
        s = client.get(f"/api/videos/{VID}/summary").json()
        self.assertTrue(s["pipeline_version"].endswith("calibrated"))
        self.assertIn("calibration", s)
        m = client.get(f"/api/videos/{VID}/metrics")
        self.assertEqual(m.status_code, 200)
        self.assertNotIn("NaN", m.text, "API обязан отдавать NaN как null")
        self.assertTrue(any(row.get("hand_speed") is None for row in m.json()))


class TestDerivedEndpoints(unittest.TestCase):
    def test_truth_uses_psychometric_layer(self):
        t = client.get(f"/api/videos/{VID}/truth").json()
        psy = client.get(f"/api/videos/{VID}/summary").json()["personality"]["truthfulness"]["score"]
        self.assertEqual(t["heuristic"], round(psy))
        self.assertIn("psych", t)
        self.assertIn("blink_per_min", t["cues"])

    def test_scores_match_truth_endpoint(self):
        sc = client.get(f"/api/videos/{VID}/scores").json()
        t = client.get(f"/api/videos/{VID}/truth").json()
        self.assertEqual(sc["truthfulness"]["source"], "psychometrics-load")
        self.assertEqual(sc["truthfulness"]["score"], t["heuristic"],
                         "датчик правдивости и вердикт LLM должны показывать одно число")
        self.assertEqual(sc["_engine"], "psychometrics-LR")


class TestFileEndpoint(unittest.TestCase):
    def test_serves_own_file(self):
        r = client.get(f"/api/videos/{VID}/file", params={"path": "episodes/P10_00.jpg"})
        self.assertEqual(r.status_code, 200)

    def test_blocks_sibling_prefix_directory(self):
        r = client.get(f"/api/videos/{VID}/file", params={"path": f"../{VID_PREFIX}/summary.json"})
        self.assertEqual(r.status_code, 403)

    def test_blocks_escape_from_data_dir(self):
        for path in ("../../app.py", "../llm_config.json", ".././../data/index.json"):
            with self.subTest(path=path):
                self.assertEqual(client.get(f"/api/videos/{VID}/file", params={"path": path}).status_code, 403)


class TestSnapshots(unittest.TestCase):
    def test_create_search_delete(self):
        r = client.post(f"/api/videos/{VID}/snapshots",
                        json={"t0": 2.0, "t1": 3.5, "name": "тест", "channels": ["smile", "ocu", "yc"]})
        self.assertEqual(r.status_code, 200, r.text)
        snap = r.json()
        self.assertTrue(snap["id"])
        self.assertIn("pattern_active", snap)
        lst = client.get("/api/snapshots").json()
        self.assertIn(snap["id"], [x["id"] for x in lst])
        self.assertNotIn("series", lst[0], "список слепков не должен тянуть полные серии")
        res = client.post(f"/api/videos/{VID}/search",
                          json={"snapshot_id": snap["id"], "top_k": 3})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertIn("meta", body)
        self.assertLessEqual(len(body["results"]), 3)
        self.assertEqual(client.delete(f"/api/snapshots/{snap['id']}").status_code, 200)

    def test_upload_validation(self):
        r = client.post("/api/snapshots/upload",
                        files={"file": ("bad.json", "{не json".encode("utf-8"), "application/json")})
        self.assertEqual(r.status_code, 422)
        self.assertIn("error", r.json())

    def test_short_window_rejected(self):
        r = client.post(f"/api/videos/{VID}/snapshots", json={"t0": 1.0, "t1": 1.05, "name": "x"})
        self.assertEqual(r.status_code, 422)


class TestLlmConfig(unittest.TestCase):
    def test_local_client_sees_key(self):
        cfg = client.get("/api/llm/config").json()
        self.assertIn("key_set", cfg)

    def test_non_local_client_does_not_see_key(self):
        self.assertTrue(appmod._is_local(_Req("127.0.0.1")))
        self.assertTrue(appmod._is_local(_Req("::1")))
        self.assertFalse(appmod._is_local(_Req("192.168.1.50")))

    def test_status(self):
        st = client.get("/api/llm/status").json()
        self.assertIn("configured", st)
        self.assertIn("key_set", st)


class _Req:
    """Минимальная заглушка Request: нужен только .client.host."""
    def __init__(self, host):
        self.client = type("C", (), {"host": host})()


if __name__ == "__main__":
    unittest.main()
