import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.inspect_cost_margin import describe, run


class CostMarginTests(unittest.TestCase):
    def test_both_fills_and_zero_days_count(self):
        trades = [dict(date="a", qty=10, entry=100, exit=101, pnl=10),
                  dict(date="b", qty=5, entry=101, exit=100, pnl=-5)]
        daily = [dict(date=k, pnl=v) for k, v in zip("abc", (10, -5, 0))]
        r = describe(trades, daily)
        self.assertEqual(r["other_days_usd"], -5)
        self.assertAlmostEqual(r["net_extra_1bp_usd"], 4.6985)
        self.assertAlmostEqual(r["net_extra_3bp_usd"], 4.0955)
        self.assertAlmostEqual(r["extra_break_even_bp_per_side"], 50000 / 3015)
        self.assertIsNone(describe([], [daily[-1]])["extra_break_even_bp_per_side"])
        with self.assertRaises(ValueError):
            describe(trades[:-1], daily)

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "august").mkdir()
            (p / "august/trades.csv").write_bytes(b"altered")
            (p / "report.json").write_text(json.dumps(dict(status="completed_exploratory_trial",
                all_controls_reproduced=True, outputs={"august/trades.csv": hashlib.sha256(b"original").hexdigest()})))
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                run(p)

    def test_complete_report_keeps_both_periods_and_all_arms(self):
        from tools.inspect_cost_margin import PERIODS, ARMS
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            hashes = {}
            for period, n in PERIODS.items():
                (p / period).mkdir()
                files = {
                    "trades": "strategy,date,qty,entry,exit,pnl\n" + "".join(
                        f"{a},1,4,50,51,4\n" for a in ARMS),
                    "daily": "strategy,date,pnl\n" + "".join(
                        f"{a},{i},{4 if i == 1 else 0}\n" for a in ARMS for i in range(1, n+1))}
                for name, content in files.items():
                    rel = f"{period}/{name}.csv"
                    raw = content.encode()
                    (p / rel).write_bytes(raw)
                    hashes[rel] = hashlib.sha256(raw).hexdigest()
            (p / "report.json").write_text(json.dumps(dict(status="completed_exploratory_trial",
                all_controls_reproduced=True, outputs=hashes)))
            results = run(p)
            self.assertEqual(len(results), 8)
            self.assertTrue(all(row["other_days_usd"] == 0 for row in results))
