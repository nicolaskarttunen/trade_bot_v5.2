"""Pure, point-in-time news transformation shared by training and inference.

Policy: last available event, exponentially decayed from publication. Events
become available at max(publication, revision, receipt). No receipt -> rejected
in strict mode; publication-only replay must be requested and labelled a proxy.
"""
import numpy as np
import pandas as pd

NEWS_POLICY = "latest_available_decay_v1"


def news_scores(decision_times, news=None, decay_minutes=120, lookback_minutes=360,
                require_received=False):
    times = pd.Series(pd.to_datetime(decision_times, utc=True)).reset_index(drop=True)
    if decay_minutes <= 0 or lookback_minutes < 0:
        raise ValueError("Invalid news decay/lookback")
    if times.isna().any():
        raise ValueError("Missing news decision timestamp")
    if news is None or len(news) == 0:
        return np.zeros(len(times))
    n = pd.DataFrame(news).copy()
    publication = "created_at" if "created_at" in n else "timestamp"
    n["published"] = pd.to_datetime(n[publication], utc=True, errors="coerce")
    n["available"] = n["published"]
    for field in ("updated_at", "received_at"):
        if field in n:
            value = pd.to_datetime(n[field], utc=True, errors="coerce")
            if field == "received_at" and require_received and value.isna().any():
                raise ValueError("News requires received_at for every event")
            n["available"] = pd.concat([n["available"], value], axis=1).max(axis=1)
        elif field == "received_at" and require_received:
            raise ValueError("News requires received_at; publication-only history is a proxy")
    n["sentiment"] = pd.to_numeric(n["sentiment"], errors="coerce")
    if (n[["published", "available"]].isna().any().any()
            or not np.isfinite(n["sentiment"]).all()
            or (n["sentiment"].abs() > 1).any()):
        raise ValueError("Invalid news timestamps or sentiment")
    # Stable ties and duplicate delivery do not reweight the feature.
    n = n.sort_values("available", kind="stable")
    left = pd.DataFrame({"decision": times, "ordinal": np.arange(len(times))})
    merged = pd.merge_asof(left.sort_values("decision"),
                           n[["available", "published", "sentiment"]],
                           left_on="decision", right_on="available", direction="backward")
    age = (merged["decision"] - merged["published"]).dt.total_seconds() / 60
    valid = age.ge(0) & age.le(lookback_minutes)
    merged["score"] = np.where(valid, merged["sentiment"] * np.exp(-age.fillna(0) / decay_minutes), 0.0)
    return merged.sort_values("ordinal")["score"].to_numpy(dtype=float)
