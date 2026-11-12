"""No-network, no-trading startup proof run as opc by the system service manager."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def blocked(*args, **kwargs):
    raise RuntimeError("Network is disabled in the service runtime proof")


def run(release, expected_uid, base_interpreter):
    release = Path(release).resolve()
    prefix = release / ".venv"
    if (release != BASE or os.geteuid() != expected_uid
            or Path(sys.prefix).resolve() != prefix
            or Path(sys.executable).absolute() != prefix / "bin/python"
            or Path("/proc/self/exe").resolve() != Path(base_interpreter).resolve()):
        raise RuntimeError("Unexpected service runtime identity")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    for name in ("numpy", "pandas", "requests", "websocket", "dotenv", "alpaca"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Dependency resolved outside the release runtime")
    from dotenv import dotenv_values
    from model_pipeline import validate_bundle
    expected = dotenv_values(release / ".paper-service.env", interpolate=False)
    names = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    if not all(expected.get(k) and os.environ.get(k) == expected[k] for k in names):
        raise RuntimeError("Service environment differs from the verified credential pair")
    if os.environ.get("ALPACA_PAPER") != "true":
        raise RuntimeError("Service paper flag is not true")
    cfg_path = release / "config.json"
    model_path = release / "models/ensemble.pkl"
    cfg = json.loads(cfg_path.read_text())
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    validate_bundle(model, cfg, live=True)
    # Exercise the release's write access under the inherited service profile,
    # without touching a risk counter, queue, journal, report or trading process.
    with tempfile.TemporaryFile(dir=release) as stream:
        stream.write(b"service-runtime-proof")
        stream.flush()
        os.fsync(stream.fileno())
    proof = dict(uid=os.geteuid(), prefix=str(prefix), base_interpreter=str(Path(base_interpreter).resolve()),
                 environment_matches=True, paper=True, write_test=True, network_blocked=True,
                 model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest())
    path = release / ".service-runtime-proof.json"
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(proof, stream, indent=2)
    return proof


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--expected-uid", type=int, required=True)
    parser.add_argument("--base-interpreter", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.release_dir, args.expected_uid, args.base_interpreter)
    except Exception as error:
        print("SERVICE_RUNTIME_STOPPED", type(error).__name__)
        return 1
    print("SERVICE_RUNTIME_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
