"""Simulation-only numeric model artifact; decoding never executes arbitrary code."""
import numpy as np
from features import FEATURES
from model_pipeline import make_models, validate_bundle
from ml_models import Standardizer


def encode(bundle):
    main,ensemble,centroid=bundle["models"]
    def scaler(m):
        return dict(mean=m.scaler_.mean_.tolist(), std=m.scaler_.std_.tolist())
    def logistic(m):
        return dict(**scaler(m), w=m.w_.tolist(), b=float(m.b_))
    return dict(version=1,feature_contract=bundle["feature_contract"],features=bundle["features"],
                main=logistic(main),members=[dict(cols=cols.tolist(),**logistic(m)) for cols,m in ensemble.models_],
                centroid=dict(**scaler(centroid),c0=centroid.c0_.tolist(),c1=centroid.c1_.tolist(),
                              scale=float(centroid.margin_scale_)))


def decode(state,cfg):
    if state.get("version")!=1 or state["feature_contract"].get("news_mode")!="neutral":
        raise ValueError("Invalid simulation model contract")
    def vector(x,n):
        a=np.asarray(x,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError("Invalid model vector")
        return a
    def scaler(m,d,n):
        m.scaler_=Standardizer()
        m.scaler_.mean_,m.scaler_.std_=vector(d["mean"],n),vector(d["std"],n)
        if (m.scaler_.std_<=0).any(): raise ValueError("Invalid model scale")
    def logistic(m,d,n):
        scaler(m,d,n); m.w_=vector(d["w"],n); m.b_=float(d["b"])
        if not np.isfinite(m.b_): raise ValueError("Invalid model bias")
    main,ensemble,centroid=make_models()
    n=len(FEATURES); logistic(main,state["main"],n)
    if len(state["members"])!=ensemble.n_models: raise ValueError("Wrong ensemble size")
    ensemble.models_=[]
    for d in state["members"]:
        cols=np.asarray(d["cols"])
        if (cols.dtype.kind not in "iu" or cols.shape!=(max(4,n//2),) or len(set(cols))!=len(cols)
            or (cols<0).any() or (cols>=n).any()): raise ValueError("Invalid ensemble columns")
        m=ensemble.make_member(); logistic(m,d,len(cols)); ensemble.models_.append((cols,m))
    d=state["centroid"]; scaler(centroid,d,n)
    centroid.c0_,centroid.c1_=vector(d["c0"],n),vector(d["c1"],n)
    centroid.margin_scale_=float(d["scale"])
    if not np.isfinite(centroid.margin_scale_) or centroid.margin_scale_<=0: raise ValueError("Invalid margin scale")
    bundle=dict(models=[main,ensemble,centroid],features=state["features"],feature_contract=state["feature_contract"])
    validate_bundle(bundle,cfg)
    return bundle
