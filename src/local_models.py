"""Value-only learned radial-basis score field (optional; switched off in ACLO and ACLO-NC).

This surrogate fits a smooth radial-basis score from archived *objective values*.
It computes analytic derivatives of that fitted score, NEVER derivative of
expensive objective function.  Provides out-of-sample predictive validation;
features are dimension-normalized [0,1] coordinates.
"""
import numpy as np

class RadialFieldModel:
    def __init__(self, d, archive_capacity=280, n_basis=28):
        self.d=d
        self.capacity=archive_capacity
        self.n_basis=n_basis
        self.Z=np.empty((0,d))
        self.F=np.empty(0)
        self.center=np.empty((0,d))
        self.weights=np.empty(0)
        self.scale=.12
        self.score=float('nan')
        self.fit_count=0

    def add(self, z, f):
        z=np.atleast_2d(z)
        f=np.asarray(f).reshape(-1)
        self.Z=np.vstack((self.Z,z))
        self.F=np.r_[self.F,f]
        if len(self.F)>self.capacity:
            # Deterministic mixture of best, newest and diversity-preserving samples
            n=self.capacity
            best=np.argsort(self.F)[:n//3]
            newest=np.arange(max(0,len(self.F)-n//3),len(self.F))
            # Regular temporal sub-sampling of remaining observations
            slots=np.linspace(0,len(self.F)-1,n,dtype=int)
            idx=np.unique(np.r_[best,newest,slots])
            if len(idx)>n:
                keep=np.unique(np.r_[best,newest]); others=np.setdiff1d(idx,keep)
                idx=np.r_[keep,others[np.linspace(0,len(others)-1,max(0,n-len(keep)),dtype=int)]]
            self.Z=self.Z[idx];self.F=self.F[idx]

    def fit(self):
        self.fit_count+=1
        m=len(self.F)
        if m<40:
            self.score=float('nan');return self.score
        # Deterministic train/test split, stratified by rank (no objective evaluations).
        order=np.argsort(self.F,kind='stable')
        test=order[::6]
        train=np.setdiff1d(np.arange(m),test)
        if len(train)<self.n_basis:
            self.score=float('nan');return self.score
        # Retain elites and diversify the rest of the basis by max-min distance.
        subset=train[np.argsort(self.F[train])[:min(10,self.n_basis//3)]]
        k=min(self.n_basis,len(train))
        pool=train
        selected=list(subset)
        min_dis=np.full(len(pool),np.inf)
        while len(selected)<k:
            q=self.Z[selected[-1]]
            dis=np.mean((self.Z[pool]-q)**2,axis=1)
            min_dis=np.minimum(min_dis,dis)
            min_dis[np.isin(pool,np.array(selected,int))]=-np.inf
            index=int(np.argmax(min_dis))
            if not np.isfinite(min_dis[index]):break
            selected.append(int(pool[index]))
        centres=self.Z[selected]
        pair=np.sqrt(np.mean((centres[:,None,:]-centres[None,:,:])**2,axis=-1))
        nz=pair[np.triu_indices(len(centres),k=1)]
        width=float(np.clip(np.quantile(nz[nz>1e-9],.4) if np.any(nz>1e-9) else .12,.025,.38))
        # Target: bounded log-quality score, normalized to a robust typical gap.
        lower=float(self.F.min())
        gaps=np.maximum(self.F-lower,0)
        delta=max(float(np.percentile(gaps,65)),1e-12)
        quality=-np.log1p(gaps/delta)
        quality=np.maximum(quality,-5)
        quality=(quality-quality.mean())/(quality.std()+1e-9)
        def features(A):
            r2=np.mean((A[:,None,:]-centres[None,:,:])**2,axis=-1)
            return np.exp(-r2/(2*width**2))
        trainfeat=features(self.Z[train])
        y=quality[train]
        mat=trainfeat.T@trainfeat + (0.03*len(train))*np.eye(len(centres))
        coeff=np.linalg.solve(mat,trainfeat.T@y)
        testfeat=features(self.Z[test])
        expected=testfeat@coeff
        actual=quality[test]
        # Correlation is evidence of generalization. Require enough variation.
        if np.std(expected)<1e-9 or np.std(actual)<1e-9:
            score=-1.
        else:
            score=float(np.corrcoef(expected,actual)[0,1])
        self.center=centres;self.weights=coeff;self.scale=width
        self.score=score
        return score

    def predict(self,z):
        z=np.atleast_2d(z)
        if len(self.weights)==0: return np.zeros(len(z))
        diff=self.center[None,:,:]-z[:,None,:]
        radial=np.mean(diff*diff,axis=2)
        return np.exp(-radial/(2*self.scale**2))@self.weights

    def gradient(self,z):
        z=np.atleast_2d(z)
        if len(self.weights)==0:return np.zeros_like(z)
        diff=self.center[None,:,:]-z[:,None,:]
        r2=np.mean(diff*diff,axis=2)
        k=np.exp(-r2/(2*self.scale**2))
        return np.einsum('nk,nkd->nd',k*self.weights[None,:],diff)/(self.d*self.scale**2)

    def steer(self,z,v,length,nsteps=3,angle_limit=.65):
        """Explicit Euler integration of d v/ds = (I-vvT) grad(log n).
        Analytic derivative belongs to the learned surrogate, not the objective.
        """
        z=np.atleast_2d(z)
        v=np.atleast_2d(v).copy()
        x=z.copy()
        step=np.asarray(length).reshape(-1,1)/nsteps
        steering=np.zeros(len(z))
        for _ in range(nsteps):
            grad=self.gradient(x)
            perp=grad-np.sum(grad*v,axis=1,keepdims=True)*v
            norms=np.linalg.norm(perp,axis=1,keepdims=True)
            turn=np.minimum(angle_limit/nsteps,(np.abs(step)*norms).flatten())[:,None]
            # positive score = higher score around a promising region
            tangent=perp/np.maximum(norms,1e-12)
            v=np.cos(turn)*v+np.sin(turn)*tangent
            v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            x=np.clip(x+step*v,0,1)
            steering+=turn.ravel()
        return x,steering

class LocalQuadraticModel:
    """Piecewise, low-rank value-derived score near an incumbent.

    A low-rank coordinate manifold is extracted from evaluated points. We fit a
    second-order response *to observed function values*, never to derivatives.
    Its score is on a log scale. Gradients below are analytic gradients
    of the value-fitted surrogate, not objective gradients.
    """
    def __init__(self, rank=5, observations=140):
        self.rank=rank;self.observations=observations
        self.quality=-1
    def fit(self, z, data_z, data_f):
        if len(data_f)<30:return False
        dist=np.linalg.norm(data_z-z,axis=1)
        ids=np.argsort(dist)[:min(len(data_f),self.observations)]
        Y=data_z[ids]-z
        f=data_f[ids]
        n=len(f)
        # SVD learned from actual local evaluated displacement geometry.
        cov=Y.T@Y
        if Y.shape[1]>128:
            gram=Y@Y.T
            val,V=np.linalg.eigh(gram)
            keep=np.argsort(val)[-self.rank:][::-1]
            U=(V[:,keep].T@Y)/np.sqrt(np.maximum(val[keep],1e-20))[:,None]
        else:
            val,vec=np.linalg.eigh(cov)
            U=vec[:,-self.rank:][:,::-1].T
        k=len(U)
        if k<2:return False
        self.U=U
        t=Y@U.T
        self.scale=max(float(np.quantile(np.linalg.norm(t,axis=1),.7)),.01)
        t=np.clip(t/self.scale,-3,3)
        df=f-f.min()
        y=-np.log1p(df/max(float(np.percentile(df,75)),1e-9))
        self.ref=z.copy()
        self.ymean=float(y.mean());self.ystd=max(float(y.std()),1e-8)
        y=(y-self.ymean)/self.ystd
        ij=np.triu_indices(k)
        self.ij=ij
        def features(t):
            return np.column_stack([np.ones(len(t)),t,t[:,ij[0]]*t[:,ij[1]]])
        feat=features(t)
        # Deterministic holdout, interleaved over *distance* ranking.
        test=np.arange(n)%5==0
        train=~test
        if train.sum()<feat.shape[1]+4:return False
        xx=feat[train];yy=y[train]
        reg=.006*len(yy)*np.eye(xx.shape[1]);reg[0,0]=0.
        beta=np.linalg.solve(xx.T@xx+reg,xx.T@yy)
        yp=feat[test]@beta
        if np.std(yp)<1e-9 or np.std(y[test])<1e-9:self.quality=-1.;return False
        quality=float(np.corrcoef(yp,y[test])[0,1])
        self.quality=quality
        self.beta=beta
        self.H=np.zeros((k,k))
        for ii,(a,b) in enumerate(zip(*ij)):
            if a==b:self.H[a,b]=2*beta[1+k+ii]
            else:self.H[a,b]=self.H[b,a]=beta[1+k+ii]
        return np.isfinite(quality)
    def score(self,x):
        t=(np.atleast_2d(x)-self.ref)@self.U.T/self.scale
        return (self.beta[0]+t@self.beta[1:1+len(self.U)]+.5*np.einsum('nk,kl,nl->n',t,self.H,t))
    def gradient(self,x):
        t=(np.atleast_2d(x)-self.ref)@self.U.T/self.scale
        g=t@self.H+self.beta[1:1+len(self.U)]
        return g@self.U/self.scale
    def propose(self,z,trust=.08,curved=True):
        """Select among line searches in a value-derived metric, using
        a held-out model score; final candidate ALWAYS evaluated on real f."""
        if self.quality<.53:return None,0.
        G=self.gradient(z)[0]
        gn=np.linalg.norm(G)
        if gn<1e-10:return None,0.
        tangent=G/gn
        # Including transverse directions is necessary for nonseparable valleys.
        dirs=[tangent]
        for u in self.U[:min(3,len(self.U))]:
            dirs.extend([.6*tangent+.8*u,.6*tangent-.8*u])
        v=np.array(dirs)
        v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        x=np.repeat(z[None,:],len(v),axis=0)
        tstep=trust/3
        for _ in range(3):
            if curved:
                g=self.gradient(x)
                g-=np.sum(g*v,axis=1,keepdims=True)*v
                ng=np.linalg.norm(g,axis=1,keepdims=True)
                angle=np.minimum(.3,tstep*ng)
                v=np.cos(angle)*v+np.sin(angle)*g/np.maximum(ng,1e-12)
                v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            x=np.clip(x+tstep*v,0,1)
        scores=self.score(x)
        best=int(np.argmax(scores))
        improvement=float(scores[best]-self.score(z)[0])
        if improvement<.035:return None,improvement
        return x[best],improvement
