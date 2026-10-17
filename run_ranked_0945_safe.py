import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "config.json"
TEST = BASE / "ranked_0945_walkforward_test.py"

# The Windows working copy contains experimental local config changes.
# This wrapper temporarily applies the intended production-comparison risk
# settings, runs the ranked 09:45 walk-forward test, and restores config.json
# byte-for-byte even if the test fails or is interrupted.
TEST_OVERRIDES = {
    "risk_per_trade_pct": 0.01,
    "max_position_pct": 0.10,
    "max_open_positions": 5,
    "max_total_exposure_pct": 0.50,
    "max_daily_loss_pct": 0.03,
    "probability_threshold": 0.55,
    "trade_score_threshold": 55,
    "max_trades_per_day": 5,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
    "ml_weight": 0.45,
    "market_weight": 0.50,
    "news_weight": 0.05,
}

if not CONFIG.exists():
    raise SystemExit(f"Missing {CONFIG}")
if not TEST.exists():
    raise SystemExit(f"Missing {TEST}")

original = CONFIG.read_bytes()

try:
    cfg = json.loads(original.decode("utf-8-sig"))
    cfg.update(TEST_OVERRIDES)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print("Temporary test settings:")
    print(f"  symbols={len(cfg.get('symbols', []))}")
    for key, value in TEST_OVERRIDES.items():
        print(f"  {key}={value}")
    print("\nRunning ranked 09:45 walk-forward test...\n")

    result = subprocess.run([sys.executable, str(TEST)], cwd=BASE)
    raise SystemExit(result.returncode)
finally:
    CONFIG.write_bytes(original)
    print("\nconfig.json restored exactly to its pre-test contents.")
