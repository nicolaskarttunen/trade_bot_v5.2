# Verification, 17 November 2026

- Python 3.12.14, alpaca-py 0.44.0, pandas 2.2.3: 241 tests, zero failures/errors; one isolated-runtime test skipped in the system-site-packages environment.
- Same complete suite on pandas 3.0.0: 241 tests, zero failures/errors, same one skip.
- The skipped independent-runtime packaging test passed separately in an isolated Linux virtual environment (1 test). It checks the real release packaging and no-network system-service startup proof using fake credentials and synthetic two-symbol data.
- All current Python source hashes match both suite reports. Socket connections are blocked by the audit runner. Broker fault tests use actual SQLite and mock broker state; no orders were sent.
- This is local functional evidence, not Oracle Python 3.14 certification, a market-data backtest or evidence of profitability.

Commands (use a virtual environment with the pinned dependencies):

```sh
python tools/run_audit_tests.py --output-dir /new/test-output
python -m unittest -b test_oracle_deployment.RuntimePackagingTests
```

The source reports include hashes and exact dependency versions. The fake-broker EOD cases include lost acknowledgments, 404s, cancel/fill races, partial and residual closes, restarts, foreign orders/positions, nested-fill regression and transactional journal migration.
