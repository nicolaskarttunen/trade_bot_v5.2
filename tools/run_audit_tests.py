"""Explicit offline suite: excludes legacy scripts that perform work on import."""
from contextlib import redirect_stdout, redirect_stderr
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import socket
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
MODULES = ["test_audit_hardening", "test_intraday_replay", "test_pipeline_contract",
           "test_replay_risk", "test_paper_boundaries", "test_driver_end_to_end"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args=parser.parse_args()
    log=io.StringIO()
    with redirect_stdout(log), redirect_stderr(log), patch.object(socket.socket,"connect",side_effect=AssertionError("Network forbidden in offline tests")), patch.object(socket,"create_connection",side_effect=AssertionError("Network forbidden in offline tests")):
        suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"test_results.txt").write_text(log.getvalue())
    source_files=sorted(BASE.glob("*.py"))+sorted((BASE/"tools").glob("*.py"))
    report={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
            "successful":result.wasSuccessful(),"modules":MODULES,"network_blocked":True,
            "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "source_sha256":{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (args.output_dir/"test_results.json").write_text(json.dumps(report,indent=2))
    print(log.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=="__main__":
    main()
