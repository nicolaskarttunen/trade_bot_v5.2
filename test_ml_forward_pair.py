import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from tools import ml_forward_pair as pair
from tools.run_offline_simulation import synthetic_bars
from test_august_simulation import TRAIN


class ForwardPairTests(unittest.TestCase):
    def test_calendar_and_path_boundary(self):
        self.assertEqual(len(pair.CAL),20)
        self.assertEqual(list(pair.CAL)[:-1],pair.fw.DATES[1:])
        self.assertEqual(list(pair.CAL)[0],"2026-11-11")
        self.assertNotIn("2026-11-26",pair.CAL)
        self.assertEqual(pair.close_time("2026-11-27").tz_convert("UTC").hour,18)
        for a,b in [("s","s"),("s","s/inner"),("s/inner","s")]:
            with self.assertRaises(ValueError): pair.paths(a,b)

    def test_registration_and_twenty_snapshots_with_real_replays(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as st:
            root=Path(tmp); source=root/"old"; state=root/"new"; hist=root/"hist"; hist.mkdir()
            aug,fw=pair.aug,pair.fw; syms=["NVDA","AMD"]
            st.enter_context(redirect_stdout(io.StringIO()))
            st.enter_context(patch.object(aug.reference,"SYMBOLS",syms))
            st.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=syms)))
            clock=st.enter_context(patch.object(fw,"utcnow",return_value=pd.Timestamp("2026-11-09T20:00Z")))
            for s,k in [("NVDA",1),("AMD",2),("SPY",3)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(hist/f"{s}_5min.csv",index=False)
            ref=root/"ref"; aug.run(hist,ref,execute=True,low_memory=True,retain_replay_inputs=True)
            fw.initialize(source,hist,ref)
            st.enter_context(patch.object(fw.locked,"fit_disk_bundle",side_effect=AssertionError("No refit")))
            clock.return_value=pd.Timestamp("2026-11-10T22:00Z"); pair.init(state,source)
            clock.return_value=pd.Timestamp("2026-11-11T14:30Z")
            with self.assertRaisesRegex(ValueError,"closed"): pair.init(root/"toolate",source)
            pair.init(state,source)
            reg=fw.aug.digest(state/"registration.json")
            def fetch(s,start,end):
                day=end.tz_convert(fw.NY).date().isoformat()
                days=pd.bdate_range(end=day,periods=7).strftime("%Y-%m-%d").tolist()
                x=synthetic_bars(days,{"NVDA":1,"AMD":2,"SPY":3}[s])
                return x.loc[x.timestamp<=end]
            (source/"snapshots").mkdir()
            for day in list(pair.CAL)[:-1]:
                clock.return_value=pair.close_time(day)+pd.Timedelta(minutes=45)
                fw.snapshot(source,day,syms,"SPY",fetch)
            before={p.relative_to(source):aug.digest(p) for p in source.rglob("*") if p.is_file()}
            clock.return_value=pd.Timestamp("2026-12-09T21:29Z")
            def nofetch(*args): raise AssertionError("unexpected network")
            self.assertIsNone(pair.update(state,source,fetch=nofetch))
            self.assertFalse((state/"result").exists()); self.assertFalse((state/"final_snapshot").exists())
            clock.return_value=pd.Timestamp("2026-12-10T14:00Z")
            spy=source/"snapshots/2026-11-11/bars.zip"; old=spy.read_bytes()
            spy.write_bytes(old+b"bad")
            with self.assertRaisesRegex(ValueError,"integrity"): pair.update(state,source,fetch=fetch)
            self.assertFalse((state/"result").exists())
            spy.write_bytes(old)
            r=pair.update(state,source,fetch=nofetch)
            self.assertEqual(r["status"],"complete"); self.assertEqual(r["late_sessions"],[pair.LAST])
            self.assertEqual(r["registration_sha256"],reg)
            daily=pd.read_csv(state/"result/daily.csv")
            self.assertEqual(len(daily),40)
            a,b=[daily.loc[daily.strategy==arm] for arm in pair.ARMS]
            self.assertEqual(list(a.date),list(pair.CAL)); self.assertEqual(list(b.date),list(pair.CAL))
            self.assertEqual(a.start_equity.iloc[0],100000); self.assertEqual(b.start_equity.iloc[0],100000)
            self.assertAlmostEqual(r["ml_minus_market_only"]["mean_daily_difference_bps"],
                                   10000*((a.pnl/a.start_equity).mean()-(b.pnl/b.start_equity).mean()))
            self.assertEqual(pair.update(state,source,fetch=nofetch),r)
            self.assertEqual(before,{p.relative_to(source):aug.digest(p) for p in source.rglob("*") if p.is_file()})
            with (state/"result/summary.csv").open("ab") as f: f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): pair.update(state,source,fetch=nofetch)
