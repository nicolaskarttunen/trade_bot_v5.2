"""Locked window: real pipeline on synthetic prices, and fail-closed cases."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from tools import run_locked_period as trial
from tools.run_offline_simulation import synthetic_bars
from test_august_simulation import TRAIN


WARM_DAYS=["2026-10-14","2026-10-15","2026-10-16","2026-10-19"]


class LockedPeriodTests(unittest.TestCase):
    def test_calendar_dst_benchmark_and_stock_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"prices.csv"
            x=synthetic_bars(WARM_DAYS+trial.DAYS,1)
            gap=pd.Timestamp("2026-11-02 09:30",tz=trial.NY).tz_convert("UTC")
            self.assertEqual(gap.hour,14)
            self.assertEqual(pd.Timestamp("2026-10-20 09:30",tz=trial.NY).tz_convert("UTC").hour,13)
            x.loc[x.timestamp!=gap].to_csv(path,index=False)
            self.assertEqual(trial.read_extension(path,"NVDA","SPY")[2],[gap.isoformat()])
            with self.assertRaisesRegex(ValueError,"benchmark bars"): trial.read_extension(path,"SPY","SPY")
            x.loc[x.timestamp.dt.tz_convert(trial.NY).dt.strftime("%Y-%m-%d")!="2026-11-03"].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"missing or unexpected"): trial.read_extension(path,"NVDA","SPY")
            x.loc[x.timestamp>=trial.START].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"warm-up"): trial.read_extension(path,"NVDA","SPY")

    def test_fetch_fixed_sip_request_pagination_and_error(self):
        def response(body, code=200):
            m=MagicMock(); m.__enter__.return_value=m; m.status_code=code; m.json.return_value=body
            return m
        bar=dict(t="2026-10-20T13:30:00Z",o=1,h=2,l=1,c=2,v=10)
        body=dict(symbol="SPY",bars=[bar],next_page_token="t2")
        with patch.object(trial.requests,"get",side_effect=[response(body),response(dict(body,next_page_token=None))]) as get:
            x=trial.fetch_bars("SPY",{})
            self.assertEqual(len(x),2); self.assertEqual(get.call_count,2)
            for call in get.call_args_list:
                self.assertEqual(call.args[0],"https://data.alpaca.markets/v2/stocks/SPY/bars")
                self.assertFalse(call.kwargs["allow_redirects"])
                self.assertEqual(call.kwargs["params"]["feed"],"sip")
        with patch.object(trial.requests,"get",return_value=response({},403)):
            with self.assertRaisesRegex(ValueError,"HTTP 403"): trial.fetch_bars("SPY",{})
        with patch.object(trial.requests,"get",side_effect=[response(body),response(body)]):
            with self.assertRaisesRegex(ValueError,"pagination"): trial.fetch_bars("SPY",{})

    def test_all_four_arms_rebuilt_training_and_preflight_failures(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); hist=root/"train"; ext=root/"ext"; hist.mkdir(); ext.mkdir()
            aug=trial.aug; symbols=["NVDA","AMD"]
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(aug.reference,"SYMBOLS",symbols))
            stack.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=symbols)))
            for s,k in [("NVDA",1),("AMD",2),("SPY",3)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(hist/f"{s}_5min.csv",index=False)
                x=synthetic_bars(WARM_DAYS+trial.DAYS,k)
                if s=="SPY":
                    mask=x.timestamp.dt.tz_convert(trial.NY).dt.strftime("%H:%M").eq("09:30")
                    x.loc[mask,"close"]=x.loc[mask,"open"]*1.002
                    x["high"]=np.maximum(x.high,x.close)
                x.to_csv(ext/f"{s}_5min.csv",index=False)
            ref=root/"reference"; aug.run(hist,ref,execute=True,low_memory=True,retain_replay_inputs=True)
            with patch.object(trial.requests,"get",side_effect=AssertionError("offline")),patch.object(trial,"fit_disk_bundle",wraps=trial.fit_disk_bundle) as fit:
                report=trial.run(hist,ref,root/"ok",extension=ext)
                self.assertEqual(fit.call_count,1)
            self.assertEqual(report["status"],"completed_fixed_period_comparison")
            self.assertEqual(report["august_anchor_predictions_checked"],6)
            self.assertEqual(report["spy_allowed_sessions"],14)
            self.assertEqual(len(report["paired_daily_comparisons"]),4)
            self.assertTrue(report["inputs_unchanged"])
            daily=pd.read_csv(root/"ok/daily.csv")
            self.assertEqual(len(daily),56)
            self.assertEqual(set(daily.date),set(trial.DAYS))
            with self.assertRaises(FileExistsError): trial.run(hist,ref,root/"ok",extension=ext)
            with self.assertRaisesRegex(ValueError,"separate"): trial.run(hist,ref,ext/"nested",extension=ext)
            with (hist/"AMD_5min.csv").open("ab") as f: f.write(b"\n")
            with patch.object(trial,"fit_disk_bundle",side_effect=AssertionError("must not fit")):
                with self.assertRaisesRegex(ValueError,"history differs"):
                    trial.run(hist,ref,root/"bad",extension=ext)
            self.assertEqual(json.loads((root/"bad/report.json").read_text())["status"],"failed_not_validated")
            self.assertFalse((root/"bad/summary.csv").exists())
