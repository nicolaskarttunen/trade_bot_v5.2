"""Oracle preparation checks with synthetic bars, no credentials or sockets."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from model_pipeline import validate_bundle
from tools import prepare_oracle_candidate as candidate

CFG = json.loads(Path(__file__).with_name("config.json").read_text())


def bars(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, .25, n).cumsum()
    return [dict(t=str(t), o=float(c), h=float(c + .3), l=float(c - .3),
                 c=float(c), v=int(v)) for t, c, v in zip(
        pd.date_range("2026-01-05T14:30:00Z", periods=n, freq="5min"),
        close, rng.integers(100, 1000, n))]


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(CFG, history_start="2026-01-01", history_end="2026-02-01")
        self.start, self.end = candidate.bounds(self.cfg)

    def test_every_page_and_exclusive_end_are_requested_with_one_feed(self):
        raw, calls = bars(), []
        def fetch(params):
            calls.append(params)
            return dict(bars={"AAA": raw[:200] if len(calls) == 1 else raw[200:]},
                        next_page_token="page2" if len(calls) == 1 else None)
        result = candidate.download_symbol(None, "AAA", self.cfg, self.start, self.end, "2026-02-01", fetch)
        self.assertEqual(len(result), 400)
        self.assertEqual(calls[1]["page_token"], "page2")
        self.assertEqual(calls[0]["feed"], self.cfg["feed"])
        self.assertEqual(calls[0]["adjustment"], "raw")
        self.assertLess(pd.Timestamp(calls[0]["end"]), self.end)
        self.assertTrue(result.timestamp.is_monotonic_increasing)

    def test_repeated_tokens_duplicates_and_empty_symbol_fail_closed(self):
        raw = bars()
        for reply in (dict(bars={"AAA": raw}, next_page_token="repeated"),
                      dict(bars={"AAA": raw + raw[-1:]}), dict(bars={})):
            with self.subTest(reply_size=len(reply.get("bars", {}).get("AAA", []))), self.assertRaises(ValueError):
                candidate.download_symbol(None, "AAA", self.cfg, self.start, self.end,
                                          "2026-02-01", lambda params: reply)

    def test_retry_is_bounded_and_auth_failure_never_retries(self):
        session = Mock()
        session.get.side_effect = [SimpleNamespace(status_code=429),
                                   SimpleNamespace(status_code=200, json=lambda: {"bars": {}})]
        waits = []
        self.assertEqual(candidate.get_page(session, {}, sleeper=waits.append), {"bars": {}})
        self.assertEqual(waits, [1])
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
        session.reset_mock(side_effect=True)
        session.get.return_value = SimpleNamespace(status_code=403)
        with self.assertRaisesRegex(ValueError, "HTTP 403"):
            candidate.get_page(session, {}, sleeper=waits.append)
        self.assertEqual(session.get.call_count, 1)

    def test_credentials_do_not_mix_pairs_from_different_files(self):
        def values(path, **kwargs):
            return {"ALPACA_API_KEY": "one"} if path.name == ".env.v5" else {"ALPACA_SECRET_KEY": "two"}
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "is_file", return_value=True), \
                patch.object(candidate, "dotenv_values", side_effect=values), \
                patch.object(candidate, "service_credentials", return_value=None):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                candidate.credentials(Path("/unused"))

    @unittest.skipUnless(os.name == "posix", "Oracle reads the same-user Linux process")
    def test_service_credentials_require_expected_process_and_select_only_the_pair(self):
        source, reads = Path("/running-bot"), []
        def read(path):
            reads.append(path.name)
            if path.name == "cmdline":
                return b"/usr/bin/python3.14\0/running-bot/paper_trader_integrated.py\0"
            return b"ALPACA_API_KEY=fake-key\0ALPACA_SECRET_KEY=fake-secret\0OTHER_TOKEN=never-export\0"
        with patch.object(candidate.subprocess, "check_output", return_value="123\n"), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_uid=os.geteuid())), \
                patch.object(Path, "read_bytes", autospec=True, side_effect=read):
            self.assertEqual(candidate.service_credentials(source), ("fake-key", "fake-secret"))
            reads.clear()
            self.assertIsNone(candidate.service_credentials(Path("/different-bot")))
            self.assertEqual(reads, ["cmdline"])


@unittest.skipUnless(os.name == "posix", "Oracle uses a Linux process lock")
class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.output = root / "running-bot", root / "candidate"
        (self.source / "models").mkdir(parents=True)
        self.cfg = dict(CFG, symbols=["AAA", "BBB"], benchmark_symbols=["SPY"],
                        history_start="2026-01-01", history_end="2026-02-01")
        (self.source / "config.json").write_text(json.dumps(self.cfg))
        (self.source / "models/ensemble.pkl").write_bytes(b"legacy artifact must stay unchanged")
        self.calls = []
        self.rows = {s: bars(seed=i + 12) for i, s in enumerate(["AAA", "BBB", "SPY"])}

    def fetch(self, params):
        symbol = params["symbols"]
        self.calls.append(symbol)
        return dict(bars={symbol: self.rows[symbol]}, next_page_token=None)

    def run_candidate(self, fetch=None):
        with redirect_stdout(io.StringIO()), patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            return candidate.run(self.source, self.output, fetch=fetch or self.fetch)

    def test_real_disk_fit_is_valid_complete_resumable_and_never_deployed(self):
        self.cfg["ALPACA_API_KEY"] = "fake-config-key-must-not-be-saved"
        (self.source / "config.json").write_text(json.dumps(self.cfg))
        before = candidate.digest(self.source / "models/ensemble.pkl")
        report = self.run_candidate()
        self.assertEqual(report["status"], "candidate_ready_not_deployed")
        self.assertEqual(set(self.calls), {"AAA", "BBB", "SPY"})
        self.assertEqual(report["prediction_checks"], 2)
        self.assertFalse(report["profitability_validated"])
        self.assertFalse(report["deployment_performed"])
        self.assertNotIn(self.cfg["ALPACA_API_KEY"], (self.output / "status.json").read_text())
        self.assertNotIn(self.cfg["ALPACA_API_KEY"].encode(), (self.output / "candidate.pkl").read_bytes())
        self.assertEqual(before, candidate.digest(self.source / "models/ensemble.pkl"))
        self.assertEqual(json.loads((self.source / "config.json").read_text()), self.cfg)
        with (self.output / "candidate.pkl").open("rb") as model:
            self.assertEqual(validate_bundle(pickle.load(model), self.cfg, live=True), "neutral")
        self.calls.clear()
        with patch.object(candidate, "fit_disk_bundle", side_effect=AssertionError("Must not refit")):
            self.run_candidate()
        self.assertEqual(self.calls, [])
        changed = dict(self.cfg, label_threshold_pct=.003)
        (self.source / "config.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "configuration, source or environment changed"):
            self.run_candidate()

    def test_download_and_completed_matrix_survive_interrupted_fit(self):
        def interrupted(params):
            if params["symbols"] == "BBB":
                raise candidate.CandidateError("Synthetic interruption")
            return self.fetch(params)
        with self.assertRaisesRegex(ValueError, "Synthetic interruption"):
            self.run_candidate(interrupted)
        self.assertEqual(set(json.loads((self.output / "status.json").read_text())["data"]), {"AAA"})
        self.calls.clear()
        with patch.object(candidate, "fit_disk_bundle", side_effect=RuntimeError("Fit interrupted")):
            with self.assertRaisesRegex(RuntimeError, "Fit interrupted"):
                self.run_candidate()
        self.assertEqual(set(self.calls), {"BBB", "SPY"})
        self.calls.clear()
        with patch.object(candidate, "TrainingStore", side_effect=AssertionError("Must reuse matrix")):
            report = self.run_candidate()
        self.assertEqual(self.calls, [])
        self.assertEqual(report["status"], "candidate_ready_not_deployed")

    def test_tampering_or_foreign_outputs_cannot_be_silently_reused(self):
        self.run_candidate()
        path = self.output / "data/AAA_5min.csv"
        with path.open("a") as output:
            output.write("changed")
        with self.assertRaisesRegex(ValueError, "Saved data changed"):
            self.run_candidate()
        foreign = self.output.parent / "foreign"
        foreign.mkdir(); (foreign / "valuable.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "Unrecognized output"):
            candidate.run(self.source, foreign, fetch=self.fetch)
        self.assertEqual((foreign / "valuable.txt").read_text(), "keep")

    def test_concurrent_launches_and_production_output_are_rejected(self):
        with candidate.exclusive(self.output):
            with self.assertRaisesRegex(ValueError, "active process"):
                self.run_candidate()
        with self.assertRaisesRegex(ValueError, "outside the running bot"):
            candidate.run(self.source, self.source / "models/new", fetch=self.fetch)


if __name__ == "__main__":
    unittest.main()
