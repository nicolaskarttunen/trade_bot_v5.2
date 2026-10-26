"""Fault tests for read-only historical probes; no live API/credentials."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from tools.probe_missing_bars import GAPS, HistoricalReader, DataReadError, compare_window, run_probe


def fixture(target, include_target=False):
    ts = pd.date_range(target-pd.Timedelta(minutes=10), periods=5, freq="5min")
    frame = pd.DataFrame({"timestamp":ts,"open":100.,"high":101.,"low":99.,"close":100.5,"volume":1000.})
    return frame if include_target else frame[frame.timestamp != target]


def envelope(symbol, frame):
    records=frame.rename(columns={"timestamp":"t","open":"o","high":"h","low":"l","close":"c","volume":"v"}).copy()
    records["t"]=records["t"].astype(str)
    return {"symbol":symbol,"bars":records.to_dict("records")}


class GapProbeTests(unittest.TestCase):
    def test_only_matching_context_yields_candidate(self):
        target=pd.Timestamp(GAPS[0][1]);local=fixture(target);remote=fixture(target,True)
        result,candidate=compare_window(local,remote,target)
        self.assertEqual(result["status"],"candidate_for_review")
        self.assertEqual(candidate.timestamp.tolist(),[target])
        for field in ["open","volume"]:
            bad=remote.copy();bad.loc[bad.index[0],field]+=.1
            result,candidate=compare_window(local,bad,target)
            self.assertEqual(result["status"],"context_mismatch")
            self.assertIsNone(candidate)

    def test_missing_or_incomplete_response_never_invents_bar(self):
        target=pd.Timestamp(GAPS[0][1]);local=fixture(target)
        for remote in [pd.DataFrame(),local]:
            result,candidate=compare_window(local,remote,target)
            self.assertEqual(result["status"],"absent_from_response")
            self.assertIsNone(candidate)
        result,candidate=compare_window(local,fixture(target,True).iloc[[2]],target)
        self.assertEqual(result["status"],"context_mismatch")
        self.assertIsNone(candidate)

    def test_duplicates_and_out_of_window_responses_rejected(self):
        target=pd.Timestamp(GAPS[0][1]);local=fixture(target);remote=fixture(target,True)
        with self.assertRaises(ValueError): compare_window(local,pd.concat([remote,remote]),target)
        remote.loc[0,"timestamp"]=target-pd.Timedelta(minutes=15)
        with self.assertRaises(ValueError): compare_window(local,remote,target)

    def make_sources(self, root, complete=False):
        root.mkdir()
        for symbol in ["EQIX","BLK"]:
            pd.concat([fixture(pd.Timestamp(ts),complete) for s,ts in GAPS if s==symbol]).to_csv(root/f"{symbol}_5min.csv",index=False)

    def test_probe_preserves_sources_records_settings_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";self.make_sources(data)
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
            fetch=Mock(side_effect=lambda s,start,end:envelope(s,fixture(start+pd.Timedelta(minutes=10),True)))
            result=run_probe(data,root/"probe",fetch)
            self.assertEqual(result["status"],"completed")
            self.assertEqual(fetch.call_count,5)
            self.assertTrue(result["source_hashes_unchanged_at_end"])
            self.assertEqual(result["request_settings"]["feed"],"sip")
            self.assertEqual(result["request_settings"]["adjustment"],"raw")
            self.assertFalse(result["original_feed_and_adjustment_proven"])
            self.assertEqual(len(list((root/"probe").glob("*_candidate_ONLY.csv"))),5)
            self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()})
            with self.assertRaises(FileExistsError):run_probe(data,root/"probe",fetch)
            with self.assertRaises(ValueError):run_probe(data,data/"probe",fetch)

    def test_present_targets_skip_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";self.make_sources(data,True)
            fetch=Mock(side_effect=AssertionError("Must not query"))
            result=run_probe(data,root/"probe",fetch)
            fetch.assert_not_called()
            self.assertTrue(all(r["status"]=="already_present_locally" for r in result["gaps"]))

    def test_api_failure_is_reported_without_secret_or_error_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";self.make_sources(data)
            result=run_probe(data,root/"probe",Mock(side_effect=RuntimeError("SECRET_TEST_VALUE")))
            self.assertEqual(result["status"],"failed")
            self.assertNotIn("SECRET_TEST_VALUE",(root/"probe/report.json").read_text())
            self.assertEqual(len(list((root/"probe").glob("*_candidate_ONLY.csv"))),0)

    def test_http_get_only_pagination_and_no_redirects(self):
        reader=HistoricalReader("fake-key","fake-secret")
        first=Mock(status_code=200);first.json.return_value={"symbol":"EQIX","bars":[],"next_page_token":"next"}
        second=Mock(status_code=200);second.json.return_value={"symbol":"EQIX","bars":[],"next_page_token":None}
        with patch("tools.probe_missing_bars.requests.get",side_effect=[first,second]) as get:
            result=reader("EQIX",pd.Timestamp(GAPS[0][1]),pd.Timestamp(GAPS[0][1]))
            self.assertEqual(result,{"symbol":"EQIX","bars":[]})
            self.assertEqual(get.call_count,2)
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertEqual(get.call_args.kwargs["timeout"],(5,20))
            self.assertTrue(get.call_args.args[0].startswith("https://data.alpaca.markets/v2/stocks/"))
        first.close.assert_called_once();second.close.assert_called_once()
        denied=Mock(status_code=403)
        with patch("tools.probe_missing_bars.requests.get",return_value=denied):
            with self.assertRaises(DataReadError):reader("EQIX",pd.Timestamp(GAPS[0][1]),pd.Timestamp(GAPS[0][1]))
        denied.close.assert_called_once()

    def test_env_parser_does_not_expand_or_export_values(self):
        from dotenv import dotenv_values
        from io import StringIO
        import os
        before=dict(os.environ)
        values=dotenv_values(stream=StringIO('ALPACA_API_KEY="fake-key"\nALPACA_SECRET_KEY="literal${NAME}"\n'),interpolate=False)
        self.assertEqual(values["ALPACA_SECRET_KEY"],"literal${NAME}")
        self.assertEqual(dict(os.environ),before)


if __name__=="__main__":unittest.main()
