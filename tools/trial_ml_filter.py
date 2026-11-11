"""Offline historical ablation of stock ML eligibility, with the rolling SPY condition intact."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools import trial_open_priority as prior
from model_pipeline import trade_score
from research_validation import paired_daily_comparison
from tools.inspect_cost_margin import describe

aug, locked, forward = prior.aug, prior.locked, prior.forward
PERIODS = ("august", "october_november")
ARM_ML, ARM_MARKET = "original_ml", "market_only"


def market_rows(scores, market, cfg):
    expected = dict(ml_weight=.45, market_weight=.50, news_weight=.05,
                    probability_threshold=.55, trade_score_threshold=55.)
    if any(cfg[k] != v for k, v in expected.items()) or cfg["benchmark_symbols"][0] != "SPY":
        raise ValueError("Unexpected baseline score contract")
    # Same benchmark clock and close-to-close calculation as technical_features.
    ret = market.set_index("timestamp").close.pct_change(3, fill_method=None)
    if not ret.index.is_monotonic_increasing or ret.index.has_duplicates:
        raise ValueError("Invalid SPY clock")
    selected = {}
    for symbol, x in scores.items():
        m = ret.reindex(x.timestamp).to_numpy()
        p = x.ml_prob.to_numpy()
        if (not np.isfinite(m).all() or not np.isfinite(p).all() or
                not ((p >= 0) & (p <= 1)).all() or not x.news_score.eq(0).all() or
                not np.allclose(trade_score(p, m, 0, cfg), x.trade_score, atol=1e-9, rtol=0)):
            raise ValueError(f"SPY return, prediction or archived score mismatch: {symbol}")
        selected[symbol] = x.loc[m > 0].copy()  # Never use today's close or the next bar.
    return selected


def activity(frames, sessions):
    t, d = frames["trades"], frames["daily"]
    fees = describe(t.to_dict("records"), d.to_dict("records"))
    starts = d.set_index("date").start_equity
    # Entry notional held for the recorded interval; no intrabar or MTM claim.
    minutes = {day: (pd.Timestamp(f"{day} {close}") - pd.Timestamp(f"{day} 09:30")).total_seconds()/60
               for day, close in sessions.items()}
    weighted_pct = turnover = 0.
    if not t.empty:
        cash = t.date.map(starts)
        duration = (pd.to_datetime(t.exit_ts, utc=True) - pd.to_datetime(t.entry_ts, utc=True)).dt.total_seconds()/60
        weighted_pct = float((100*t.entry*t.qty*duration/(cash*t.date.map(minutes))).sum()/len(d))
        turnover = float((100*(t.entry+t.exit)*t.qty/cash).sum()/len(d))
    return dict(avg_session_exposure_pct=weighted_pct, avg_daily_two_sided_turnover_pct=turnover,
                **fees)


def period(reference, arm_name, scores, prices, market, cfg, sessions):
    eligible = market_rows(scores, market, cfg)
    alt = dict(cfg, probability_threshold=0., trade_score_threshold=0.)
    out, audits, metrics = {}, {}, {}
    for arm, rows, config in [(ARM_ML, scores, cfg), (ARM_MARKET, eligible, alt)]:
        print("Replaying", arm, flush=True)
        frames, audit = prior.replay(rows, prices, config, sessions, {}, {}, False, False)
        if arm == ARM_ML:
            prior.verify_control(frames, reference, arm_name)
        out[arm], audits[arm], metrics[arm] = frames, audit, activity(frames, sessions)
    pair = paired_daily_comparison(out[ARM_MARKET]["daily"], out[ARM_ML]["daily"])
    tables = {n: pd.concat([f[n].assign(strategy=a) for a,f in out.items()], ignore_index=True)
              for n in prior.TABLES}
    return tables, dict(ml_minus_market_only=pair, activity=metrics, ledger=audits)


def run(reference, october, frozen, history, output):
    reference, october, frozen, history, output = map(lambda p: Path(p).resolve(),
                                                     (reference, october, frozen, history, output))
    if any(p == output or p in output.parents for p in [reference, october, frozen, history]):
        raise ValueError("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="started", independent_evidence=False, training_runs=0,
                  primary="original_ml minus market_only mean paired daily net return; no opening gate",
                  script_sha256=aug.digest(Path(__file__)), replay_helper_sha256=aug.digest(Path(prior.__file__)))
    watched = {}
    def watch(path, expected=None):
        h = aug.digest(path)
        if expected is not None and h != expected:
            raise ValueError(f"Input hash mismatch: {path.name}")
        watched[path] = h
        return h
    def write():
        aug.write_json(output/"report.json", report)
    write()
    try:
        with ExitStack() as stack:
            forward.offline(stack)
            for path in [reference/"manifest.json", october/"report.json",
                         *(frozen/name for name in ("registration.json","ready.json","model.json"))]:
                watch(path)
            cfg, bundle = forward.load_ready(frozen)
            old = forward.read(october/"report.json")
            manifest = forward.read(reference/"manifest.json")
            if (manifest["status"] != "completed_development_comparison" or manifest["config"] != cfg
                or manifest["windows"] != [list(aug.WINDOW)] or old["status"] != "completed_fixed_period_comparison"
                or old["inputs_unchanged"] is not True or old["initial_config"] != cfg or old["sessions"] != locked.SESSIONS
                or old["reference_manifest_sha256"] != watched[reference/"manifest.json"]
                or old["training_cache_sha256"] != forward.read(frozen/"ready.json")["training_matrix"]):
                raise ValueError("Reference periods, model or settings differ")
            for root, data in [(reference,manifest),(october,old)]:
                for n in prior.TABLES:
                    watch(root/(n+".csv"), data["outputs"][n+".csv"])
            watch(reference/prior.CACHE, manifest["outputs"][prior.CACHE])
            sessions, symbols = aug.load_sessions(), cfg["symbols"]
            path = history/"SPY_5min.csv"
            market,h = aug.read_source(path, old["training_sources"][path.name])
            watched[path] = h
            scores,prices,_ = prior.load_cache(reference/prior.CACHE, symbols, sessions)
            results = [period(reference,prior.AUG_ARMS[0], scores,prices,market,cfg,sessions)]
            del scores, prices, market
            def extension(symbol):
                path = october/"sip_extension"/f"{symbol}_5min.csv"
                x,h,_ = locked.read_extension(path,symbol,"SPY")
                if h != old["extension_sha256"][symbol]:
                    raise ValueError(f"Extension hash differs: {symbol}")
                watched[path] = h
                return x
            market = extension("SPY")
            scores, prices = {}, {}
            for s in symbols:
                x = extension(s)
                scores[s] = locked.score(x,market,cfg,bundle)
                prices[s] = x.loc[x.timestamp >= locked.START,aug.OHLCV]
            results.append(period(october,prior.LOCKED_ARMS[0],scores,prices,market,cfg,locked.SESSIONS))
            if any(aug.digest(p) != h for p,h in watched.items()):
                raise ValueError("Source changed during replay")
            for name,(tables,detail) in zip(PERIODS,results):
                dest = output/name
                dest.mkdir()
                for n,f in tables.items():
                    f.to_csv(dest/(n+".csv"), index=False)
                aug.write_json(dest/"audit.json",detail)
            report.update(status="completed_exploratory_ml_ablation", controls_reproduced=True,
                          verified_inputs={str(p):h for p,h in watched.items()},
                          outputs={p.relative_to(output).as_posix():aug.digest(p) for p in output.glob("*/*")})
            write()
    except BaseException as exc:
        report.update(status="failed_not_validated", error=str(exc))
        write()
        raise
    for name,(tables,detail) in zip(PERIODS,results):
        print(name, flush=True)
        summary=tables["summary"].set_index("strategy")
        extra=pd.DataFrame(detail["activity"]).T
        print(summary[["return_pct","trades","max_dd_pct"]].join(
            extra[["avg_session_exposure_pct","avg_daily_two_sided_turnover_pct","net_extra_3bp_usd"]]).to_string())
        print("ML minus market-only (bp/day):",detail["ml_minus_market_only"])
    print("Status:",report["status"],"Report:",output/"report.json")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, default=Path.home()/"Downloads/trade_bot_v5.2/data_sip")
    p.add_argument("--reference", type=Path, default=BASE/"research_runs/jarjestys-20261109-130806/reference")
    p.add_argument("--october", type=Path, default=BASE/"research_runs/lukittu-20261109-151005")
    p.add_argument("--frozen", type=Path, default=BASE/"research_runs/forward-20261110")
    p.add_argument("--output-dir", type=Path, required=True)
    a=p.parse_args()
    try:
        run(a.reference,a.october,a.frozen,a.history,a.output_dir)
    except (ValueError,KeyError,RuntimeError,OSError) as exc:
        raise SystemExit(f"Stopped: {exc}")
