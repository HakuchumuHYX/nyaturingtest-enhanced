import importlib.util
import contextlib
import json
import tempfile
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
TOOL_PATH = PLUGIN_DIR / "tools" / "rag_baseline_report.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("rag_baseline_report_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RagBaselineReportTests(unittest.TestCase):
    def test_parse_log_event_extracts_structured_rag_event(self):
        module = _load_tool_module()
        line = '06-08 12:00:00 [INFO] nyaturingtest | {"event":"rag_search","session_id":"100","candidate_count":3}'

        event = module.parse_log_event(line, year=2026)

        self.assertEqual("rag_search", event["event"])
        self.assertEqual("100", event["session_id"])
        self.assertEqual("2026-06-08T12:00:00", event["_timestamp"])
        self.assertEqual("2026-06-08", event["_date"])

    def test_build_report_requires_three_days_two_sessions_and_both_event_types(self):
        module = _load_tool_module()
        lines = []
        for day, session_id in [("08", "100"), ("09", "100"), ("10", "200")]:
            lines.append(
                "06-{day} 12:00:00 [INFO] nyaturingtest | "
                "{{\"event\":\"rag_search\",\"session_id\":\"{session_id}\","
                "\"candidate_count\":4,\"returned_count\":2,\"injected_count\":1,"
                "\"injected_chars\":80,\"adjusted_score_p50\":0.6}}\n".format(
                    day=day,
                    session_id=session_id,
                )
            )
            lines.append(
                "06-{day} 12:00:01 [INFO] nyaturingtest | "
                "{{\"event\":\"rag_prompt_budget\",\"session_id\":\"{session_id}\","
                "\"chat_prompt_total_chars\":1200}}\n".format(
                    day=day,
                    session_id=session_id,
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "output.log"
            log_path.write_text("".join(lines), encoding="utf-8")

            report = module.build_report([log_path], year=2026)

        self.assertTrue(report["ready"])
        self.assertEqual(3, report["days_covered"])
        self.assertEqual(2, report["session_count"])
        self.assertEqual(3, report["rag_search_count"])
        self.assertEqual(1200.0, report["chat_prompt_total_chars"]["p95"])

    def test_cli_can_fail_when_baseline_is_not_ready(self):
        module = _load_tool_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "output.log"
            log_path.write_text(
                '06-08 12:00:00 [INFO] nyaturingtest | {"event":"rag_search","session_id":"100"}\n',
                encoding="utf-8",
            )

            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as capture:
                with contextlib.redirect_stdout(capture):
                    exit_code = module.main([str(log_path), "--year", "2026", "--fail-if-not-ready"])

        self.assertEqual(2, exit_code)

    def test_cli_prints_json_report(self):
        module = _load_tool_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "output.log"
            log_path.write_text("", encoding="utf-8")

            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as capture:
                with contextlib.redirect_stdout(capture):
                    exit_code = module.main([str(log_path), "--year", "2026"])
                capture.seek(0)
                payload = json.loads(capture.read())

        self.assertEqual(0, exit_code)
        self.assertFalse(payload["ready"])
        self.assertIn("no rag_search events", payload["missing"])


if __name__ == "__main__":
    unittest.main()
