"""Shared offline training/prediction contract; no broker access or file IO."""
import numpy as np
import pandas as pd
from features import FEATURES, FEATURE_VERSION, FEATURE_HISTORY_BARS, enrich, completed_bars
from news_features import NEWS_POLICY
from ml_models import LogisticGD, RandomLinearEnsemble, CentroidClassifier


def feature_contract(cfg, news_mode):
    if news_mode not in {"neutral", "received", "publication_proxy"}:
        raise ValueError("Unknown news mode")
    if int(cfg.get("timeframe_minutes", 5)) != 5:
        raise ValueError("Feature contract supports only five-minute bars")
    return {"version": FEATURE_VERSION, "features": list(FEATURES),
            "history_bars": FEATURE_HISTORY_BARS, "bar_minutes": 5,
            "news_policy": NEWS_POLICY, "news_mode": news_mode,
            "news_decay_minutes": float(cfg["news_decay_minutes"]),
            "news_lookback_minutes": float(cfg["news_lookback_minutes"]),
            "benchmark": cfg["benchmark_symbols"][0], "feed": cfg["feed"]}


def validate_bundle(bundle, cfg, live=False):
    contract = bundle.get("feature_contract", {})
    mode = contract.get("news_mode")
    if mode is None or contract != feature_contract(cfg, mode) or bundle.get("features") != FEATURES:
        raise ValueError("Feature contract mismatch. Retrain the entire ensemble; do not reuse legacy artifacts.")
    if live and mode == "publication_proxy":
        raise ValueError("Publication-only news artifacts are research proxies, not live-compatible")
    if not bundle.get("models"):
        raise ValueError("Model bundle is empty")
    return mode


def training_rows(frame, start, cutoff):
    start, cutoff = pd.Timestamp(start), pd.Timestamp(cutoff)
    if start.tzinfo is None or cutoff.tzinfo is None or start >= cutoff:
        raise ValueError("Training bounds must be ordered timezone-aware timestamps")
    # Actual label availability, not a count of rows in a combined symbol table.
    valid = ((frame["timestamp"] >= start) & (frame["timestamp"] < cutoff)
             & (frame["label_end"] <= cutoff) & frame["target"].notna())
    valid &= np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[valid].copy()


def make_models():
    """One fixed definition for RAM and disk-backed full-batch trainers."""
    return [LogisticGD(lr=.03, epochs=500, l2=.01),
            RandomLinearEnsemble(n_models=25, seed=42), CentroidClassifier()]


def fit_bundle(train, cfg, news_mode, metadata=None):
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["target"].to_numpy(dtype=float)
    if len(train) == 0 or not np.isfinite(X).all() or not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires finite features and both binary classes")
    models = make_models()
    for model in models:
        model.fit(X, y.astype(int))
    return {"models": models, "features": list(FEATURES),
            "feature_contract": feature_contract(cfg, news_mode),
            "metadata": metadata or {}}


def predict_bundle(bundle, frame, cfg):
    validate_bundle(bundle, cfg)
    X = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite inference features")
    if len(X) == 0:
        return np.empty(0)
    values = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid model predictions")
    return values


def latest_signal(bars, benchmark, news, bundle, cfg, as_of):
    mode = validate_bundle(bundle, cfg, live=True)
    stock = completed_bars(bars, as_of).tail(FEATURE_HISTORY_BARS)
    market = completed_bars(benchmark, as_of).tail(FEATURE_HISTORY_BARS)
    if len(stock) < FEATURE_HISTORY_BARS or len(market) < FEATURE_HISTORY_BARS:
        return None, None
    expected = pd.Timestamp(as_of).floor("5min") - pd.Timedelta(minutes=5)
    if stock.iloc[-1]["timestamp"] != expected or market.iloc[-1]["timestamp"] != expected:
        return None, None
    x = enrich(stock, market, None if mode == "neutral" else news,
               cfg["news_decay_minutes"], cfg["news_lookback_minutes"],
               require_news_received=mode == "received")
    # Never fall back to an older feature-valid row.
    row = x.iloc[[-1]]
    if not np.isfinite(row[FEATURES].to_numpy(dtype=float)).all():
        return None, None
    return float(predict_bundle(bundle, row, cfg)[0]), row.iloc[0]


def trade_score(probability, market_return, news_score, cfg):
    p, market, news = np.broadcast_arrays(np.asarray(probability, dtype=float),
                                         np.asarray(market_return, dtype=float),
                                         np.asarray(news_score, dtype=float))
    if not all(np.isfinite(v).all() for v in (p, market, news)):
        raise ValueError("Non-finite score input")
    return 100 * (float(cfg["ml_weight"]) * np.clip((p-.5)/.5, 0, 1)
                  + float(cfg["market_weight"]) * (market > 0)
                  + float(cfg["news_weight"]) * (np.clip(news, -1, 1)+1)/2)
