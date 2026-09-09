import numpy as np

class Standardizer:
    def fit(self, X):
        X=np.asarray(X,dtype=float); self.mean_=np.nanmean(X,axis=0); self.std_=np.nanstd(X,axis=0); self.std_[self.std_<1e-8]=1.0; return self
    def transform(self,X): return (np.asarray(X,dtype=float)-self.mean_)/self.std_

class LogisticGD:
    def __init__(self, lr=0.03, epochs=500, l2=0.01): self.lr=lr; self.epochs=epochs; self.l2=l2
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float); self.scaler_=Standardizer().fit(X); Z=self.scaler_.transform(X); self.w_=np.zeros(Z.shape[1]); self.b_=0.0
        for _ in range(self.epochs):
            p=self.predict_proba_raw(Z); grad_w=Z.T@(p-y)/len(y)+self.l2*self.w_; grad_b=np.mean(p-y); self.w_-=self.lr*grad_w; self.b_-=self.lr*grad_b
        return self
    def predict_proba_raw(self,Z):
        z=np.clip(Z@self.w_+self.b_,-30,30); return 1/(1+np.exp(-z))
    def predict_proba(self,X):
        p=self.predict_proba_raw(self.scaler_.transform(X)); return np.column_stack([1-p,p])

class RandomLinearEnsemble:
    def __init__(self,n_models=25,seed=42): self.n_models=n_models; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float); rng=np.random.default_rng(self.seed); self.models_=[]
        for i in range(self.n_models):
            idx=rng.integers(0,len(y),len(y)); cols=rng.choice(X.shape[1],size=max(4,X.shape[1]//2),replace=False); m=LogisticGD(lr=0.02,epochs=250,l2=0.03); m.fit(X[idx][:,cols],y[idx]); self.models_.append((cols,m))
        return self
    def predict_proba(self,X):
        probs=[m.predict_proba(np.asarray(X)[:,cols])[:,1] for cols,m in self.models_]; p=np.mean(probs,axis=0); return np.column_stack([1-p,p])

class CentroidClassifier:
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y); self.scaler_=Standardizer().fit(X); Z=self.scaler_.transform(X); self.c0_=np.mean(Z[y==0],axis=0); self.c1_=np.mean(Z[y==1],axis=0); return self
    def predict_proba(self,X):
        Z=self.scaler_.transform(X); d0=np.sum((Z-self.c0_)**2,axis=1); d1=np.sum((Z-self.c1_)**2,axis=1); z=np.clip((d0-d1)/max(np.std(d0-d1),1e-6),-10,10); p=1/(1+np.exp(-z)); return np.column_stack([1-p,p])
