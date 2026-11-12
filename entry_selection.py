"""Cross-sectional entry selection and conditional trade scenarios.

The classifier predicts a short-horizon price event. Its output is not the
probability of reaching the bracket target, so it is never used as that weight.
"""
import math


TOP_ENTRY_COUNT = 5
SELECTION_POLICY = "full_universe_top5_v1"


def valid_opening_selection(selection, symbols):
    """Persisted queues from earlier selection rules need a fresh scan."""
    return (isinstance(selection, dict)
            and selection.get("policy") == SELECTION_POLICY
            and selection.get("symbols") == list(symbols)
            and selection.get("scanned_count") == len(symbols)
            and selection.get("complete") is True
            and type(selection.get("rank")) is int
            and 1 <= selection["rank"] <= TOP_ENTRY_COUNT)


def rank_entry_candidates(candidates, cfg, blocked_symbols=()):
    """Rank every eligible signal before selecting at most five symbols."""
    blocked = set(blocked_symbols)
    eligible = []
    for candidate in candidates:
        probability = float(candidate["probability"])
        score = float(candidate["score"])
        if (candidate["symbol"] not in blocked
                and math.isfinite(probability) and 0 <= probability <= 1
                and math.isfinite(score)
                and probability >= float(cfg["probability_threshold"])
                and score >= float(cfg["trade_score_threshold"])):
            eligible.append(candidate)
    return sorted(eligible, key=lambda item: (
        -float(item["score"]), -float(item["probability"]), item["symbol"]
    ))[:TOP_ENTRY_COUNT]


def long_trade_plan(reference_price, atr, equity, cfg, quantity_cap=None):
    """Use the paper bot's existing sizing and brackets, with modeled costs.

The dollar outcomes are conditional on an exit at the target or the stop.
Gap losses, partial fills and end-of-day exits can have different outcomes.
"""
    price, atr, equity = map(float, (reference_price, atr, equity))
    risk = float(cfg["risk_per_trade_pct"])
    position_cap = float(cfg["max_position_pct"])
    atr_multiplier = float(cfg["stop_atr_multiplier"])
    reward_risk = float(cfg["reward_risk_ratio"])
    values = (price, equity, risk, position_cap, atr_multiplier, reward_risk)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Invalid price, equity or risk settings")
    if not math.isfinite(atr) or atr < 0:
        raise ValueError("Invalid ATR")
    slippage = float(cfg.get("slippage_bps", 0))
    spread = float(cfg.get("spread_bps", 0))
    commission = float(cfg.get("commission_per_trade", 0))
    if not all(math.isfinite(value) and value >= 0 for value in (slippage, spread, commission)):
        raise ValueError("Invalid modeled costs")
    cost_fraction = (slippage + spread) / 10000
    if cost_fraction >= 1:
        raise ValueError("Modeled price costs must be below 100 percent")

    stop_distance = max(atr * atr_multiplier, price * 0.0125)
    stop = round(price - stop_distance, 2)
    target = round(price + stop_distance * reward_risk, 2)
    if not 0 < stop < price < target:
        raise ValueError("Invalid rounded bracket prices")
    quantity = min(int(equity * risk / stop_distance), int(equity * position_cap / price))
    if quantity_cap is not None:
        quantity = min(quantity, max(0, int(quantity_cap)))
    entry_cost = price * quantity * cost_fraction
    # Match intraday_replay: commission_per_trade is one round-trip fee.
    target_cost = entry_cost + target * quantity * cost_fraction + commission
    stop_cost = entry_cost + stop * quantity * cost_fraction + commission
    if quantity == 0:
        target_cost = stop_cost = 0.0
    target_net = quantity * (target - price) - target_cost
    stop_net = quantity * (stop - price) - stop_cost
    position_value = quantity * price
    return {
        "reference_price": price,
        "quantity": quantity,
        "position_value_usd": position_value,
        "stop_distance": stop_distance,
        "stop_price": stop,
        "target_price": target,
        "target_net_pnl_usd": target_net,
        "stop_net_pnl_usd": stop_net,
        "target_net_return_pct": 100 * target_net / position_value if quantity else 0.0,
        "stop_net_return_pct": 100 * stop_net / position_value if quantity else 0.0,
        "target_modeled_cost_usd": target_cost,
        "stop_modeled_cost_usd": stop_cost,
        "slippage_bps_per_side": slippage,
        "spread_bps_per_side": spread,
        "commission_round_trip_usd": commission,
        "expected_pnl_usd": None,
        "outcome_basis": "conditional_target_or_stop_with_modeled_costs",
    }
