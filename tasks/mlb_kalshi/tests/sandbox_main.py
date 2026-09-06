"""Child-process entry point: run a candidate model.py against fold files with the network disabled.

    python -s -B sandbox_main.py job.json

job.json: {"model": path, "folds": [{"k": id, "train": path, "test": path, "out": name}], "repeat": id}
Writes preds_<k>.npy per fold, preds_<k>_repeat.npy for the repeat fold, and status.json.
Never raises to the parent: any failure is reported in status.json with exit code 0.
"""
import builtins
import json
import sys
import time
import traceback


def _install_guards() -> None:
    """Disable network/process access. Installed after the scientific stack is imported so their
    own import-time use of socket/ssl/asyncio is unaffected."""
    def mark():
        try:
            open("network_attempt", "w").close()
        except OSError:
            pass

    def deny(*_a, **_k):
        mark()
        raise RuntimeError("network access is disabled in the evaluation sandbox")

    import socket
    import _socket

    class DeniedSocket(socket.socket):  # a class, so ssl.SSLSocket(socket) keeps importing
        def __init__(self, *a, **k):
            deny()

    socket.socket = DeniedSocket  # type: ignore[misc]
    socket.SocketType = DeniedSocket  # type: ignore[misc]
    _socket.socket = DeniedSocket  # type: ignore[misc]
    for name in ("create_connection", "getaddrinfo", "create_server", "socketpair", "fromfd"):
        setattr(socket, name, deny)

    import subprocess
    subprocess.Popen = deny  # type: ignore[assignment]
    subprocess.run = deny  # type: ignore[assignment]
    import os
    os.system = deny  # type: ignore[assignment]
    os.fork = deny  # type: ignore[assignment]
    os.execv = deny  # type: ignore[assignment]


def main(job_path: str) -> None:
    job = json.load(open(job_path))
    status = {"ok": False, "error": "", "traceback": "", "elapsed": {}, "n_rows": {}}
    try:
        import numpy as np
        import pandas as pd
        import scipy  # noqa: F401  (pre-import so guards don't interfere with import-time side effects)
        import sklearn  # noqa: F401
        import sklearn.linear_model  # noqa: F401
        import sklearn.ensemble  # noqa: F401

        _install_guards()
        import importlib.util

        spec = importlib.util.spec_from_file_location("candidate_model", job["model"])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["candidate_model"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        predict = getattr(mod, "predict")

        for i, f in enumerate(job["folds"]):
            train = pd.read_parquet(f["train"])
            test = pd.read_parquet(f["test"])
            runs = [f["out"]] + ([f["out"].replace(".npy", "_repeat.npy")] if str(f["k"]) == str(job.get("repeat")) else [])
            for out in runs:
                t0 = time.time()
                p = np.asarray(predict(train.copy(), test.copy()), dtype=np.float64)
                status["elapsed"][out] = round(time.time() - t0, 2)
                np.save(out, p)
            status["n_rows"][str(f["k"])] = int(len(test))
        status["ok"] = True
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit
        status["error"] = f"{type(e).__name__}: {e}"[:2000]
        status["traceback"] = traceback.format_exc()[-4000:]
    finally:
        with open("status.json", "w") as fh:
            json.dump(status, fh)


if __name__ == "__main__":
    main(sys.argv[1])
