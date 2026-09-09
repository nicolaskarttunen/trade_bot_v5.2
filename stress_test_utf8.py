import json
from pathlib import Path
import pandas as pd
import numpy as np

from features import FEATURES, enrich, label
from ml_models import LogisticGD, RandomLinearEnsemble, CentroidClassifier


BASE = Path(__file__).resolve().parent
cfg = json.loads((BASE / "config.json").read_text())

data = BASE / "data"
newsdir = BASE / "news"

SYMBOLS = cfg["symbols"]
BENCH = cfg["benchmark_symbols"][0]

WINDOWS = [
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-01-01", "2024-06-30", "2024-07-01", "2025-01-31"),
    ("2023-01-01", "2025-01-31", "2025-02-01", "2025-08-31"),
    ("2023-01-01", "2025-08-31", "2025-09-01", "2026-02-28"),
    ("2023-01-01", "2026-02-28", "2026-03-01", "2026-08-31"),
]

PROB_THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
SCORE_THRESHOLDS = [45, 50, 55, 60]


def load_market():
    m = pd.read_csv(data / f"{BENCH}_5min.csv")
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)

    pc = m["close"].shift(1)

    tr = pd.concat([
        m["high"] - m["low"],
        (m["high"] - pc).abs(),
        (m["low"] - pc).abs()
    ], axis=1).max(axis=1)

    m["atr_pct"] = tr.rolling(14).mean() / m["close"]

    return m


def load_symbol(s, market):
    p = data / f"{s}_5min.csv"

    if not p.exists():
        return None

    nf = newsdir / f"{s}_news.csv"
    news = pd.read_csv(nf) if nf.exists() else None

    x = enrich(
        pd.read_csv(p),
        market,
        news,
        cfg["news_decay_minutes"],
        cfg["news_lookback_minutes"]
    )

    x = label(
        x,
        horizon=cfg["label_horizon_bars"],
        threshold=cfg["label_threshold_pct"]
    )

    x = x.dropna(
        subset=FEATURES + ["atr", "target"]
    ).copy()

    x["symbol"] = s

    return x


def train_ensemble(train):
    X = train[FEATURES].to_numpy()
    y = train["target"].to_numpy().astype(int)

    models = [
        LogisticGD(lr=0.03, epochs=500, l2=0.01),
        RandomLinearEnsemble(n_models=25, seed=42),
        CentroidClassifier()
    ]

    for m in models:
        m.fit(X, y)

    return models


def predict(models, x):
    probs = [
        m.predict_proba(x[FEATURES].to_numpy())[:, 1]
        for m in models
    ]

    return np.mean(probs, axis=0)


def prepare_scores(df, models):
    x = df.copy()

    x["ml_prob"] = predict(models, x)

    x["regime_score"] = 0.5

    x.loc[
        x.market_vs_ema21 > 0.002,
        "regime_score"
    ] += 0.25

    x.loc[
        x.market_vs_ema21 < -0.002,
        "regime_score"
    ] -= 0.25

    x.loc[
        x.market_atr_pct > 0.015,
        "regime_score"
    ] -= 0.10

    return x


def run_test(df, prob_threshold, score_threshold):
    cash = cfg["initial_cash"]

    trades = []
    daily = {}
    daily_start = {}

    slip = (
        cfg["slippage_bps"] +
        cfg["spread_bps"]
    ) / 10000

    df = df.sort_values("timestamp").reset_index(drop=True)

    i = 0

    while i < len(df) - 1:

        r = df.iloc[i]

        day = str(
            pd.Timestamp(r.timestamp).date()
        )

        if daily.get(day, 0) >= cfg["max_trades_per_day"]:
            i += 1
            continue

        if day not in daily_start:
            daily_start[day] = cash

        if cash <= daily_start[day] * (
            1 - cfg["max_daily_loss_pct"]
        ):
            i += 1
            continue

        news_score = float(r.news_score)

        ml_score = max(
            0,
            min(
                100,
                (r.ml_prob - 0.5) * 200
            )
        )

        market_score = max(
            0,
            min(
                100,
                r.regime_score * 100
            )
        )

        news_component = (
            50 +
            50 * max(
                -1,
                min(1, news_score)
            )
        )

        total = (
            cfg["ml_weight"] * ml_score +
            cfg["news_weight"] * news_component +
            cfg["market_weight"] * market_score
        )

        if (
            r.ml_prob < prob_threshold or
            total < score_threshold
        ):
            i += 1
            continue

        risk_multiplier = (
            1.0 +
            (float(r.ml_prob) - 0.52) * 5.0 +
            float(news_score) * 0.25 +
            (float(r.regime_score) - 0.5) * 0.5
        )

        risk_multiplier = max(
            0.5,
            min(2.0, risk_multiplier)
        )

        entry_bar = df.iloc[i + 1]

        entry = float(
            entry_bar.open
        ) * (1 + slip)

        dist = max(
            float(r.atr) *
            cfg["stop_atr_multiplier"],
            entry * 0.003
        )

        stop = entry - dist

        target = (
            entry +
            dist * cfg["reward_risk_ratio"]
        )

        allocation_pct = (
            0.30 -
            (risk_multiplier - 0.5) *
            (0.15 / 1.5)
        )

        allocation_pct = max(
            0.15,
            min(0.30, allocation_pct)
        )

        risk_qty = int(
            (
                cash *
                cfg["risk_per_trade_pct"] *
                risk_multiplier
            ) / dist
        )

        allocation_qty = int(
            (cash * allocation_pct) / entry
        )

        qty = min(
            risk_qty,
            allocation_qty
        )

        if qty < 1:
            i += 1
            continue

        g = df[
            (df.symbol == r.symbol) &
            (df.timestamp >= entry_bar.timestamp)
        ].head(12)

        if len(g) == 0:
            i += 1
            continue

        exitp = None
        reason = "time_exit"

        exit_ts = g.iloc[-1].timestamp

        for _, b in g.iterrows():

            if b.low <= stop:
                exitp = stop * (1 - slip)
                reason = "stop"
                exit_ts = b.timestamp
                break

            if b.high >= target:
                exitp = target * (1 - slip)
                reason = "target"
                exit_ts = b.timestamp
                break

        if exitp is None:
            exitp = (
                float(g.iloc[-1].close) *
                (1 - slip)
            )

        pnl = (
            (exitp - entry) *
            qty -
            cfg["commission_per_trade"]
        )

        cash += pnl

        trades.append({
            "timestamp": entry_bar.timestamp,
            "symbol": r.symbol,
            "pnl": pnl,
            "score": total,
            "ml_prob": r.ml_prob,
            "news_score": news_score,
            "reason": reason
        })

        daily[day] = daily.get(day, 0) + 1

        later = df.index[
            df.timestamp > exit_ts
        ]

        if len(later):
            i = int(later[0])
        else:
            i = len(df)

    return cash, pd.DataFrame(trades)


def evaluate_result(out):
    if len(out) == 0:
        return 0.0, 0.0, 0.0

    win = (
        (out.pnl > 0).mean() *
        100
    )

    gp = out.loc[
        out.pnl > 0,
        "pnl"
    ].sum()

    gl = -out.loc[
        out.pnl < 0,
        "pnl"
    ].sum()

    pf = (
        gp / gl
        if gl
        else float("inf")
    )

    eq = (
        cfg["initial_cash"] +
        out.pnl.cumsum()
    )

    dd = (
        eq / eq.cummax() - 1
    ).min() * 100

    return win, pf, dd


print("\n=== NESTED WALK-FORWARD SENSITIVITY ===")
print("70% train -> 30% validation -> täysin koskematon test")
print("Thresholdit valitaan vain validation-datasta.\n")


market = load_market()

all_data = {}

for s in SYMBOLS:

    print(f"Ladataan {s}...")

    x = load_symbol(s, market)

    if x is not None:
        all_data[s] = x
        print(f"  {len(x):,} riviä")


results = []


for wi, (
    tr_start,
    tr_end,
    te_start,
    te_end
) in enumerate(WINDOWS, 1):

    print("\n" + "=" * 60)
    print(f"WINDOW {wi}")
    print(
        f"Train/validation: "
        f"{tr_start} -> {tr_end}"
    )
    print(
        f"Test: "
        f"{te_start} -> {te_end}"
    )
    print("=" * 60)

    train_frames = []
    test_frames = []

    for s, x in all_data.items():

        train = x[
            (x.timestamp >= pd.Timestamp(
                tr_start,
                tz="UTC"
            )) &
            (x.timestamp <= pd.Timestamp(
                tr_end + " 23:59:59",
                tz="UTC"
            ))
        ].copy()

        train = train.iloc[
            :-cfg["label_horizon_bars"]
        ]

        test = x[
            (x.timestamp >= pd.Timestamp(
                te_start,
                tz="UTC"
            )) &
            (x.timestamp <= pd.Timestamp(
                te_end + " 23:59:59",
                tz="UTC"
            ))
        ].copy()

        if len(train):
            train_frames.append(train)

        if len(test):
            test_frames.append(test)

    train = pd.concat(
        train_frames,
        ignore_index=True
    )

    test = pd.concat(
        test_frames,
        ignore_index=True
    )

    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    # Validation split.
    # Käytetään viimeistä 30 % ajallisesti.
    train = train.sort_values("timestamp")

    split_index = int(
        len(train) * 0.70
    )

    model_train = train.iloc[
        :split_index
    ].copy()

    validation = train.iloc[
        split_index:
    ].copy()

    print(
        f"Model train rows: "
        f"{len(model_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    # Malli koulutetaan vain ensimmäisellä 70 %:lla.
    models = train_ensemble(
        model_train
    )

    validation = prepare_scores(
        validation,
        models
    )

    # Thresholdien valinta tapahtuu AINOASTAAN validationilla.
    best = None

    for pt in PROB_THRESHOLDS:

        for st in SCORE_THRESHOLDS:

            cash, out = run_test(
                validation,
                pt,
                st
            )

            ret = (
                cash /
                cfg["initial_cash"] -
                1
            ) * 100

            if (
                best is None or
                ret > best["return"]
            ):
                best = {
                    "return": ret,
                    "prob": pt,
                    "score": st
                }

    print(
        f"Validationin paras: "
        f"prob>={best['prob']:.2f}, "
        f"score>={best['score']}"
    )

    # Malli opet koulutetaan KAIKELLA training-jaksolla,
    # mutta ei koskaan testijaksolla.
    final_models = train_ensemble(
        train
    )

    test = prepare_scores(
        test,
        final_models
    )

    # Nyt thresholdit ovat lukittu.
    test_cash, test_out = run_test(
        test,
        best["prob"],
        best["score"]
    )

    test_return = (
        test_cash /
        cfg["initial_cash"] -
        1
    ) * 100

    win, pf, dd = evaluate_result(
        test_out
    )

    print("\nTESTITULOS")

    print(
        f"Return:       "
        f"{test_return:.2f}%"
    )

    print(
        f"Trades:       "
        f"{len(test_out)}"
    )

    print(
        f"Win rate:     "
        f"{win:.2f}%"
    )

    print(
        f"Profit factor:"
        f"{pf:.2f}"
    )

    print(
        f"Max drawdown: "
        f"{dd:.2f}%"
    )

    print(
        f"Locked thresholds: "
        f"prob>={best['prob']:.2f}, "
        f"score>={best['score']}"
    )

    if len(test_out):

        print("Syyt:")

        print(
            test_out.reason
            .value_counts()
            .to_string()
        )

    results.append({
        "window": wi,
        "train_start": tr_start,
        "train_end": tr_end,
        "test_start": te_start,
        "test_end": te_end,
        "validation_prob": best["prob"],
        "validation_score": best["score"],
        "return_pct": test_return,
        "trades": len(test_out),
        "win_rate": win,
        "profit_factor": pf,
        "max_drawdown_pct": dd
    })


result_df = pd.DataFrame(results)

print("\n\n=== NESTED WALK-FORWARD YHTEENVETO ===")

print(
    result_df.to_string(
        index=False
    )
)

total_return = (
    (1 + result_df.return_pct / 100).prod()
    - 1
) * 100

final_cash = (
    cfg["initial_cash"] *
    (1 + total_return / 100)
)

profit = (
    final_cash -
    cfg["initial_cash"]
)

print(
    f"\nAlkupääoma: "
    f"${cfg['initial_cash']:,.2f}"
)

print(
    f"Loppupääoma: "
    f"${final_cash:,.2f}"
)

print(
    f"Kokonaisvoitto: "
    f"${profit:,.2f}"
)

print(
    f"Kokonaistuotto: "
    f"{total_return:.2f}%"
)

print(
    f"Testijaksoja voitollisena: "
    f"{(result_df.return_pct > 0).sum()}"
    f"/{len(result_df)}"
)

print(
    f"Kauppoja yhteensä: "
    f"{result_df.trades.sum()}"
)

result_df.to_csv(
    BASE / "nested_sensitivity_results.csv",
    index=False
)

print(
    "\nTallennettu: "
    "nested_sensitivity_results.csv"
)
