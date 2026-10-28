"""Research-only disk-backed full-batch training; no subsampling or SGD.

Preserves the reference models, seeds, bootstrap multiplicities and updates.
Floating-point reductions occur in blocks, so results are numerically close,
not promised bit-identical. Original model classes handle all predictions.
"""
from contextlib import ExitStack
import heapq
import errno
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES
from ml_models import Standardizer
from model_pipeline import make_models, feature_contract

BLOCK_ROWS = 32768
RECORD = np.dtype([("timestamp", "<i8"), ("x", "<f8", (len(FEATURES),)), ("y", "<f8")])


def blocks(path, dtype, rows=BLOCK_ROWS):
    if rows < 1 or Path(path).stat().st_size % np.dtype(dtype).itemsize:
        raise ValueError("Invalid block size or truncated training file")
    with Path(path).open("rb") as f:
        while True:
            block = np.fromfile(f, dtype=dtype, count=rows)
            if not len(block):
                return
            yield block


class TrainingStore:
    """One symbol in RAM at a time, then stable timestamp/symbol-order merge."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.shards = []
        self.rows = 0
        self.last_timestamp = None
        self.path = self.root / "training.bin"

    def append(self, frame):
        if self.path.exists():
            raise ValueError("Training store already finalized")
        ts = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        x, y = frame[FEATURES].to_numpy(dtype=float), frame.target.to_numpy(dtype=float)
        if (not len(frame) or (np.diff(ts) <= 0).any() or not np.isfinite(x).all()
                or not np.isin(y, [0., 1.]).all()):
            raise ValueError("Invalid training shard")
        path = self.root / f"shard-{len(self.shards):03d}.bin"
        with path.open("xb") as f:
            for start in range(0, len(frame), BLOCK_ROWS):
                end = min(start+BLOCK_ROWS, len(frame))
                record = np.empty(end-start, dtype=RECORD)
                record["timestamp"], record["x"], record["y"] = ts[start:end], x[start:end], y[start:end]
                record.tofile(f)
        self.shards.append(path)
        self.rows += len(frame)
        self.last_timestamp = max(self.last_timestamp or int(ts[-1]), int(ts[-1]))

    def finish(self):
        if self.path.exists():
            raise FileExistsError("Training store already finalized")
        if not self.shards:
            raise ValueError("Empty training store")
        # Each shard keeps only 1024 rows in memory. Equal timestamps preserve
        # the symbol order used by concat(...).sort_values(kind='stable').
        with ExitStack() as stack, self.path.open("xb") as out:
            def records(path):
                for block in blocks(path, RECORD, rows=1024):
                    yield from block
            streams = [records(p) for p in self.shards]
            for stream in streams:
                stack.callback(stream.close)
            heap = []
            for i, stream in enumerate(streams):
                row = next(stream, None)
                if row is not None:
                    heapq.heappush(heap, (int(row["timestamp"]), i, row))
            buffer = np.empty(BLOCK_ROWS, dtype=RECORD)
            used, written = 0, 0
            while heap:
                _, i, row = heapq.heappop(heap)
                buffer[used] = row
                used += 1
                if used == len(buffer):
                    buffer.tofile(out)
                    written += used
                    used = 0
                row = next(streams[i], None)
                if row is not None:
                    heapq.heappush(heap, (int(row["timestamp"]), i, row))
            buffer[:used].tofile(out)
            written += used
            if written != self.rows:
                raise ValueError("Training merge lost rows")
        for path in self.shards:
            path.unlink()
        self.shards.clear()
        return self

    def cleanup(self):
        """Release our merged file; never recursively remove unknown files."""
        self.path.unlink(missing_ok=True)
        try:
            self.root.rmdir()
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            return False
        return True


def bootstrap_counts(rng, rows):
    # Exactly the same RNG call/order as RandomLinearEnsemble.fit. A count
    # replaces repeated copies of a sampled row, not the bootstrap itself.
    indices = rng.integers(0, rows, rows)
    return np.bincount(indices, minlength=rows)


def scaler_for(path, columns, counts=None, block_rows=BLOCK_ROWS):
    total = 0.
    sums = np.zeros(len(columns))
    offset = 0
    for b in blocks(path, RECORD, block_rows):
        x = b["x"][:, columns]
        w = np.ones(len(b)) if counts is None else counts[offset:offset+len(b)]
        sums += np.sum(x*w[:, None], axis=0)
        total += w.sum()
        offset += len(b)
    if total <= 0:
        raise ValueError("Empty weighted training sample")
    mean = sums/total
    squared = np.zeros(len(columns))
    offset = 0
    for b in blocks(path, RECORD, block_rows):
        w = np.ones(len(b)) if counts is None else counts[offset:offset+len(b)]
        squared += np.sum((b["x"][:, columns]-mean)**2*w[:, None], axis=0)
        offset += len(b)
    scaler = Standardizer()
    scaler.mean_, scaler.std_ = mean, np.sqrt(squared/total)
    scaler.std_[scaler.std_ < 1e-8] = 1.
    return scaler, total


def fit_logistic(path, model, work_dir, columns=None, counts=None,
                 block_rows=BLOCK_ROWS, progress=lambda message: None):
    columns = np.arange(len(FEATURES)) if columns is None else columns
    model.scaler_, total = scaler_for(path, columns, counts, block_rows)
    dtype = np.dtype([("z", "<f8", (len(columns),)), ("y", "<f8"), ("weight", "<f8")])
    cache = Path(work_dir) / "standardized.bin"
    # Refuse stale/foreign files, and remove only the cache created here.
    with cache.open("xb") as out:
        try:
            offset = 0
            for b in blocks(path, RECORD, block_rows):
                w = np.ones(len(b)) if counts is None else counts[offset:offset+len(b)]
                offset += len(b)
                keep = w > 0
                record = np.empty(int(keep.sum()), dtype=dtype)
                record["z"] = model.scaler_.transform(b["x"][keep][:, columns])
                record["y"], record["weight"] = b["y"][keep], w[keep]
                record.tofile(out)
        except BaseException:
            out.close()
            cache.unlink()
            raise
    try:
        model.w_, model.b_ = np.zeros(len(columns)), 0.
        for epoch in range(model.epochs):
            gradient, bias = np.zeros(len(columns)), 0.
            for b in blocks(cache, dtype, block_rows):
                error = (model.predict_proba_raw(b["z"])-b["y"])*b["weight"]
                gradient += b["z"].T @ error
                bias += error.sum()
            # ONE update after ALL rows; never mini-batch/stochastic descent.
            model.w_ -= model.lr*(gradient/total + model.l2*model.w_)
            model.b_ -= model.lr*bias/total
            if (epoch+1) % 25 == 0 or epoch+1 == model.epochs:
                progress(f"epoch {epoch+1}/{model.epochs}")
    finally:
        cache.unlink()
    return model


def fit_centroid(path, model, block_rows=BLOCK_ROWS):
    columns = np.arange(len(FEATURES))
    model.scaler_, _ = scaler_for(path, columns, block_rows=block_rows)
    sums, sizes = np.zeros((2, len(FEATURES))), np.zeros(2)
    for b in blocks(path, RECORD, block_rows):
        z = model.scaler_.transform(b["x"])
        for cls in (0, 1):
            selected = b["y"] == cls
            sums[cls] += z[selected].sum(axis=0)
            sizes[cls] += selected.sum()
    if (sizes == 0).any():
        raise ValueError("Both classes required")
    model.c0_, model.c1_ = sums/sizes[:, None]
    # Merge within-block population variance using the parallel variance law.
    n, mean, m2 = 0, 0., 0.
    for b in blocks(path, RECORD, block_rows):
        margin = model._margins(model.scaler_.transform(b["x"]))
        bn, bm = len(margin), float(margin.mean())
        delta = bm-mean
        m2 += float(np.sum((margin-bm)**2)) + delta**2*n*bn/(n+bn)
        mean += delta*bn/(n+bn)
        n += bn
    model.margin_scale_ = max(float(np.sqrt(m2/n)), 1e-6)
    return model


def fit_disk_bundle(store, cfg, news_mode, metadata=None, block_rows=BLOCK_ROWS,
                    progress=lambda message: print(message, flush=True)):
    if block_rows < 1 or store.path.stat().st_size != store.rows*RECORD.itemsize:
        raise ValueError("Incomplete disk training set")
    classes = set()
    for b in blocks(store.path, RECORD, block_rows):
        if not np.isfinite(b["x"]).all() or not np.isin(b["y"], [0., 1.]).all():
            raise ValueError("Invalid training values")
        classes.update(np.unique(b["y"]).tolist())
    if classes != {0., 1.}:
        raise ValueError("Both binary classes required")
    models = make_models()
    progress("Training logistic 1/1")
    fit_logistic(store.path, models[0], store.root, block_rows=block_rows, progress=progress)
    ensemble = models[1]
    rng = np.random.default_rng(ensemble.seed)
    ensemble.models_ = []
    for i in range(ensemble.n_models):
        counts = bootstrap_counts(rng, store.rows)
        cols = rng.choice(len(FEATURES), size=max(4, len(FEATURES)//2), replace=False)
        model = ensemble.make_member()
        progress(f"Training bootstrap model {i+1}/{ensemble.n_models}")
        fit_logistic(store.path, model, store.root, cols, counts, block_rows, progress)
        ensemble.models_.append((cols, model))
        del counts
    progress("Training centroid")
    fit_centroid(store.path, models[2], block_rows)
    return {"models": models, "features": list(FEATURES),
            "feature_contract": feature_contract(cfg, news_mode),
            "metadata": dict(metadata or {}, training_backend="disk_full_batch_v1",
                             block_rows=block_rows, training_rows=store.rows,
                             floating_point_reduction_order_changed=True)}
