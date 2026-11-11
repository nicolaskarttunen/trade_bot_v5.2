"""Fixed-policy selection, backward compatibility and cache-only replay."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from order_trial import CACHE, daily_order, save_cache, load_cache
from test_august_simulation import TRAIN
from test_intraday_replay import CFG, bars, signals
from tools import run_august_simulation as aug
from tools.run_offline_simulation import synthetic_bars
from tools.run_order_trial import experiment, one_arm, STATUS, table


class OrderPolicyTests(unittest.TestCase):
    def test_deterministic_order_not_dependent_on_input_order_or_prices(self):
        symbols=list(aug.reference.SYMBOLS)
        a=daily_order(symbols,"2026-08-03")
        self.assertEqual(a,daily_order(symbols[::-1],"2026-08-03"))
        self.assertEqual(sorted(symbols),sorted(a))
        self.assertNotEqual(a,daily_order(symbols,"2026-08-04"))

    def test_changing_order_only_changes_priority_and_preserves_original_ranks(self):
        day="2026-08-03"; symbols=daily_order(["NVDA","AMD","ORCL"],day)[::-1]
        p=bars(day=day); scores={s:signals(p) for s in symbols}
        args=(scores,{s:p for s in symbols},symbols,dict(CFG,max_trades_per_day=1),{day:"16:00"},{day:True},True)
        a,b=[],[]
        x,_=one_arm(*args,False,a.append)
        y,audit=one_arm(*args,True,b.append)
        self.assertEqual(x[0].symbol.tolist(),[symbols[0]])
        self.assertEqual(y[0].symbol.tolist(),[symbols[-1]])
        self.assertEqual(y[0].symbol_order.tolist(),[3])
        self.assertEqual(y[0].scan_order.tolist(),[1])
        self.assertEqual(audit["issues"],[])
        self.assertEqual((b[0]["symbol_order"],b[0]["scan_order"]),(3,1))
        self.assertEqual(a[0]["qty"],b[0]["qty"])

    def test_cache_roundtrip_preserves_floats_and_refuses_mismatched_universe(self):
        d="2026-08-03"; p=bars(day=d); s=signals(p)
        s["signal_available_at"]=s.timestamp+pd.Timedelta(minutes=5)
        s["ml_prob"]=0.5500000000000002
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/CACHE
            save_cache(path,["NVDA"],{"NVDA":s},{"NVDA":p},{d:True},{d:"16:00"})
            scores,prices,gate=load_cache(path,["NVDA"],{d:"16:00"})
            self.assertEqual(scores["NVDA"].ml_prob.iloc[0],s.ml_prob.iloc[0])
            self.assertEqual(prices["NVDA"].close.tolist(),p.close.tolist())
            self.assertEqual(gate,{d:True})
            with self.assertRaisesRegex(ValueError,"metadata"): load_cache(path,["AMD"],{d:"16:00"})


class OrderTrialIntegrationTests(unittest.TestCase):
    def test_old_run_single_new_fit_then_cache_only_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as st:
            root=Path(tmp); data=root/"data"; data.mkdir()
            symbols=["NVDA","AMD"]
            st.enter_context(redirect_stdout(io.StringIO()))
            st.enter_context(patch.object(aug.reference,"SYMBOLS",symbols))
            st.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=symbols)))
            for sym,k in [("NVDA",1.),("AMD",2.),("SPY",3.)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(data/f"{sym}_5min.csv",index=False)
            old=root/"old"; aug.run(data,old,execute=True,low_memory=True)
            old_hash={p.name:aug.digest(p) for p in old.iterdir() if p.is_file()}
            with self.assertRaisesRegex(ValueError,"no saved signals"):
                experiment(old,None,root/"missing")
            with patch.object(aug,"fit_disk_bundle",wraps=aug.fit_disk_bundle) as fits:
                r=experiment(old,data,root/"trial")
                self.assertEqual(fits.call_count,1)
            self.assertEqual(r["status"],STATUS)
            self.assertEqual(r["training_runs"],1)
            self.assertTrue(r["original_reproduced"])
            self.assertEqual(old_hash,{p.name:aug.digest(p) for p in old.iterdir() if p.is_file()})
            cached=root/"trial"/"reference"
            with patch.object(aug,"fit_disk_bundle",side_effect=AssertionError("no retraining")):
                repeat=experiment(cached,None,root/"repeat")
            self.assertEqual(repeat["training_runs"],0)
            self.assertEqual(r["paired_order_comparison"],repeat["paired_order_comparison"])
            pd.testing.assert_frame_equal(table(root/"repeat/changed_order","trades"),table(root/"trial/changed_order","trades"))
            with self.assertRaises(FileExistsError): experiment(old,data,root/"trial")
            # Different history must be refused before fitting.
            with (data/"AMD_5min.csv").open("ab") as f: f.write(b"\n")
            with patch.object(aug,"fit_disk_bundle",side_effect=AssertionError("no fit")):
                with self.assertRaisesRegex(ValueError,"History differs"):
                    experiment(old,data,root/"different-data")
            # Deliberately give the trial different frozen predictions while
            # keeping the on-disk baseline valid: the paired control must catch it.
            from tools import run_order_trial as trial
            real_load=trial.load_cache
            def opposite(*args):
                scores,prices,gate=real_load(*args)
                saved_count=int(table(cached,"daily").entries.sum())
                for x in scores.values():
                    x["ml_prob"],x["trade_score"]=(0.,0.) if saved_count else (1.,100.)
                return scores,prices,gate
            with patch.object(trial,"load_cache",side_effect=opposite):
                with self.assertRaisesRegex(ValueError,"reproduction differs"):
                    experiment(cached,None,root/"mismatch")
            self.assertFalse((root/"mismatch/changed_order").exists())
            self.assertEqual(json.loads((root/"mismatch/order_report.json").read_text())["status"],"failed_not_validated")


if __name__=="__main__": unittest.main()
