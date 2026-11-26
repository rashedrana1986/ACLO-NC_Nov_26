"""ACLO: Anchored Curvature-Learning Optimiser.

The base search of aclo_nc.py with curvature learning added to its line-probe
stage.  geometry_mode='off' gives the base search: every added statement is
guarded by `geo is not None` (or only assigns an unused local), and
curvature_frame.py uses its own random generator.  With geometry_mode='frame'
(the ACLO setting, used with gate_mode='skill' and scan_frac_override=0.63) the
directions and steps of the stage-2 probe lines come from a conjugate frame
learned from the curvature of line probes that the search already evaluates
(zero extra objective evaluations).  The modes random_frame, shuffled and
diagonal are control versions; see curvature_frame.py.

Uses only evaluated objective values; no gradient oracle, no finite-
difference gradient vectors, no Hessian inversions, no external local optimizer.
Uses sampled line differences and derivative-free quadratic interpolation. Bounded continuous minimization.
Constrained mode uses a declining epsilon-feasibility filter; all objective
calls (including rejected candidates) count against the exact evaluation budget.

All distances and influence radii are in normalized [0,1]^dimension space.
"""

from dataclasses import dataclass
from local_models import RadialFieldModel, LocalQuadraticModel
import time
import numpy as np

_EPS=1e-12

def _unit(a):
    return a / np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),_EPS)

@dataclass
class Result:
    x: np.ndarray
    fun: float
    violation: float
    nfev: int
    elapsed: float
    history: np.ndarray
    diagnostics: dict


def minimize(fun, bounds, budget=6000, population=48, seed=0,
         constraints=None, equality_tol=1e-5, use_steering=True,
         use_secant_memory=True, use_probe_line_refinement=True, record_every=1,
         expanded_probe_line_trust=True, focus_coordinates=True, adaptive_steering=True, batched_line_scan=False, line_batch_size=4, prioritize_anchor_uncertainty=True, smooth_probe_line_extension=False, outcome_steering_credit=True, curvature_gated_steering=True, conjugate_line_direction=True, centroid_proposals=True, centroid_global_credit=False, geometry_gated_centroid=True, use_radial_field_model=False, field_quality_gate=.22, field_operator_mode=1, use_local_model=False, local_model_curvature=True, retain_all_evaluations=True, bounded_interpolation=True, model_trust=False, quadratic_coupling=True, coupled_probe_line_memory=True, memory_gain_gate=0.25, boundary_probe_line_fallback=True, selective_anchor_restarts=False, per_anchor_basis_mode="off",
         sparse_interactions=False, joint_line_scan=False, interaction_trust=False,
         interaction_control="learned", local_pair_correction=False,
         interaction_evidence_gate=False, geometry_mode="off", geometry_options=None,
         scan_frac_override=None):
    # per_anchor_basis_mode: off, learned, random-control (experimental only)
    """Optimize callable fun(x) or batch fun(X) if fun.vectorized=True.

    constraints=(g,h) where g(x)<=0 and h(x)==0; functions may be
    vectorized (set .vectorized=True). Sum of positive constraint violations
    (equality beyond equality_tol) is minimized lexicographically with f.
    """
    lo,hi=np.asarray(bounds,float).T
    if np.any(hi<=lo): raise ValueError('upper bounds must exceed lower bounds')
    d=len(lo); N=int(population)
    if budget<N*2 or N<10: raise ValueError('budget>=2*population and population>=10 required')
    rng=np.random.default_rng(seed)
    span=hi-lo
    nfev=0
    observed_x=None; observed_f=float("inf"); observed_v=float("inf")
    field=RadialFieldModel(d)
    requires_archive=(use_radial_field_model or use_local_model)
    def eval_batch(Z):
        nonlocal nfev, observed_x, observed_f, observed_v
        Z=np.atleast_2d(Z)
        if nfev+len(Z)>budget:
            raise RuntimeError('the search attempted to exceed the objective evaluation budget')
        X=lo+Z*span
        if getattr(fun,'vectorized',False):
            F=np.asarray(fun(X),float).reshape(-1)
        else:
            F=np.array([fun(xx) for xx in X],dtype=float)
        V=np.zeros(len(X))
        if constraints is not None:
            g,h=constraints
            for cb,equal in ((g,False),(h,True)):
                if cb is None:continue
                if getattr(cb,'vectorized',False):
                    val=np.asarray(cb(X),float)
                else:
                    val=np.array([np.atleast_1d(cb(x)) for x in X])
                val=val.reshape(len(X),-1)
                if equal:V+=np.maximum(np.abs(val)-equality_tol,0).sum(axis=1)
                else:V+=np.maximum(val,0).sum(axis=1)
        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(V)):
            raise ValueError('objective or constraint returned non-finite value')
        nfev+=len(X)
        if retain_all_evaluations:
            bi=rank_indices(F,V)[0]
            if observed_x is None or better(np.array([F[bi]]),np.array([V[bi]]),
                    np.array([observed_f]),np.array([observed_v]),0.)[0]:
                observed_x=Z[bi].copy(); observed_f=float(F[bi]); observed_v=float(V[bi])
        if constraints is None and requires_archive:field.add(Z,F)
        return F,V
    def rank_indices(f,v,eps=0):
        # Feasible-first, and among infeasible points prioritize violation.
        mask=v<=eps
        return np.lexsort((f,np.where(mask,f,v),~mask))
    def better(fa,va,fb,vb,eps):
        af=va<=eps;bf=vb<=eps
        return (af&~bf) | (af&bf&(fa<fb)) | (~af&~bf&((va<vb-1e-12)|((np.abs(va-vb)<=1e-12)&(fa<fb))))
    clock=time.perf_counter()
    Z=rng.random((N,d)); F,V=eval_batch(Z)
    field_steer_count=0;field_move_improvements=0; field_scores=[]
    vel=_unit(rng.standard_normal((N,d)))
    stale=np.zeros(N,int)
    step=np.full(N,.115)
    K=min(8,max(3,N//7))
    order=rank_indices(F,V)[:K]
    anchor=Z[order].copy();lf=F[order].copy();lv=V[order].copy()
    anchor_weight=np.full(K,.50);radius=np.full(K,.14)
    anchor_stall=np.zeros(K,dtype=int)
    anchor_restart_count=0
    diag=np.ones((K,d));axes=_unit(rng.standard_normal((K,d)))
    old_axis=axes.copy();axis_gain=np.zeros(K);success=np.zeros(K)
    # Learned low-rank local secant geometry. O(K * d * rank), never d^2.
    R=4; memory=np.zeros((K,R,d));mem_w=np.zeros((K,R));mem_next=np.zeros(K,int)
    if per_anchor_basis_mode not in ('off','learned','random-control'):
        raise ValueError('Unsupported per_anchor_basis_mode')
    if geometry_mode not in ('off','frame','random_frame','shuffled','diagonal'):
        raise ValueError('Unsupported geometry_mode')
    if geometry_mode!='off' and (batched_line_scan or not bounded_interpolation):
        raise ValueError('geometry_mode requires batched_line_scan=False and bounded_interpolation=True')
    if scan_frac_override is not None and not (0.05<=float(scan_frac_override)<=0.90):
        raise ValueError('scan_frac_override must lie in [0.05, 0.90]')
    local_basis=np.zeros((K,R,d)); local_step=np.full((K,R),.08)
    local_n_basis=np.zeros(K,int)
    # Optional Phase-1 experiments: O(K*d) memory, five FE on selected batches.
    # These are OFF by default and leave the base behaviour intact.
    if interaction_control not in ("learned", "random"):
        raise ValueError("interaction_control must be learned or random")
    if sparse_interactions or joint_line_scan:
        pair_i=rng.integers(0,d,size=K)
        pair_j=(pair_i+1+rng.integers(0,max(1,d-1),size=K))%d if d>1 else pair_i.copy()
        pair_sign=rng.choice([-1.,1.],size=(K,2))
    else:
        # Do not consume RNG values when all experiments are disabled.
        pair_i=np.arange(K)%d
        pair_j=(pair_i+1)%d
        pair_sign=np.ones((K,2))
    pair_step=np.full(K,.045)
    pair_conf=np.zeros(K)
    pair_probe_n=np.zeros(K,int)
    pair_probes=0;pair_joint_trials=0;pair_joint_success=0
    pair_local_trials=0;pair_local_success=0
    interaction_samples=[]
    reward=np.ones(7)*.2
    # Best feasible or least violating candidate is kept from every evaluation.
    bidx=rank_indices(F,V)[0]
    bx=Z[bidx].copy();bf=float(F[bidx]);bv=float(V[bidx])
    hist=[(nfev,bf,bv)]
    steer_events=0; steer_attempts=0; steer_success=0; op_count=np.zeros(7,int); op_good=np.zeros(7,int)
    # Steering reliability is inferred from independently observed probe line outcomes.
    steer_reliability=.50
    # Start with epsilon feasibility if constrained and no feasible points.
    eps0=max(.0,float(np.median(V))) if constraints is not None else 0.
    if eps0<1e-12 and constraints is not None:eps0=float(np.max(V))
    it=0
    # Preallocate part of the evaluation budget for derivative-free line
    # scans, unless constrained feasibility makes a scan inappropriate.
    # The schedule is evidence-based, using held-out line-predictability tests.
    scan_frac=0.0
    probe_done=False
    smoothness_error=float("nan")
    search_end=budget if not use_probe_line_refinement else min(budget,max(N*3,int(.74*budget)//N*N))
    if constraints is not None: search_end=budget
    while nfev<search_end:
        it+=1
        if use_probe_line_refinement and constraints is None and not probe_done and nfev>=max(2*N, int(.12*budget)) and nfev+12<=budget:
            # A quadratic line surrogate is validated on held-out probe lines. All
            # observations are charged; no derivative oracle is used.
            tests=[]
            for q in range(3):
                axis=_unit(rng.normal(size=(1,d)))[0]
                h=.07*np.sqrt(d)
                scales=np.array([-1.,-.5,.5,1.])
                zz=np.clip(bx[None,:]+scales[:,None]*h*axis[None,:],0,1)
                fv,_=eval_batch(zz)
                A=np.column_stack([np.array([-1,1])**2,np.array([-1,1]),np.ones(2)])
                # Three-point interpolant through centre and both endpoints.
                fminus,fplus=fv[0],fv[3]
                predicted=np.array([.375*bf+.375*fminus+.125*fplus,
                                    .375*bf+.125*fminus+.375*fplus])
                # Correct interpolation weights from Lagrange basis at +/-0.5:
                # for t=-0.5: 0.375*f(-1) + .75*f(0) - .125*f(1)
                predicted=np.array([.375*fminus+.75*bf-.125*fplus,
                                    -.125*fminus+.75*bf+.375*fplus])
                denom=max(1e-8,np.std(np.r_[fv,bf]))
                tests.append(float(np.sqrt(np.mean((predicted-fv[[1,2]])**2))/denom))
            smoothness_error=float(np.median(tests))
            # Stable line-scan geometry gets more scan budget. Rough landscapes
            # continue probe line exploration until later, preserving diversity.
            scan_frac=.63 if smoothness_error<.035 else .26
            # Optional override: the probe (and its 12 FE, and smoothness_error used
            # by other gates) is unchanged; only the probe line-stage share is replaced.
            if scan_frac_override is not None:scan_frac=float(scan_frac_override)
            search_end=min(budget,max(nfev+N,int((1-scan_frac)*budget)//N*N))
            probe_done=True
            if retain_all_evaluations and observed_f<bf:
                bx=observed_x.copy(); bf=observed_f; bv=observed_v
            hist.append((nfev,bf,bv))
        batch=min(N,search_end-nfev)
        idx_r=np.arange(batch)
        prog=nfev/budget
        eps=eps0*(max(0,1-prog)**2) if constraints is not None else 0.
        # Rank and location quality combine anchor merit and realized anchor effectiveness.
        indices=rank_indices(lf,lv,eps)
        ranks=np.empty(K,int);ranks[indices]=np.arange(K)
        dists=np.linalg.norm(Z[:batch,None,:]-anchor[None,:,:],axis=2)/np.sqrt(d)
        anchor_score=np.exp(-.42*ranks)[None,:]*(.20+anchor_weight[None,:])*(.30+.70*np.exp(-.5*(dists/(.05+2*radius[None,:]))**2))+.10
        anchor_score/=anchor_score.sum(axis=1,keepdims=True)
        choose=np.minimum(K-1,np.sum(rng.random((batch,1))>np.cumsum(anchor_score,axis=1),axis=1))
        L=anchor[choose]; vec=L-Z[:batch]
        perpendicular=vec-np.sum(vec*vel[:batch],axis=1,keepdims=True)*vel[:batch]
        mag=np.linalg.norm(perpendicular,axis=1)
        strength=(.20+anchor_weight[choose])/(dists[idx_r,choose]+.08)
        pb=1-np.exp(-.48*strength)
        if adaptive_steering: pb*=np.clip(.35+1.30*steer_reliability,.20,1.)
        # Prediction error from held-out quadratic probes gates steering in
        # locally rough geometry rather than forcing all probe lines to steer.
        if curvature_gated_steering and np.isfinite(smoothness_error):
            pb*=np.exp(-2.5*max(0.,smoothness_error-.5))
        theta=np.minimum(.86,.16*strength)
        steer=(rng.random(batch)<pb) & (mag>1e-9) & use_steering
        steered=vel[:batch].copy()
        steered[steer]=np.cos(theta[steer,None])*vel[:batch][steer]+np.sin(theta[steer,None])*_unit(perpendicular[steer])
        steer_events+=int(np.sum(steer))
        # Refresh value-only learned local-model geometry from prior evaluated observations.
        if use_radial_field_model and it%4==0:
            field_scores.append(field.fit())
        # Six operators. The last is learned rank-R trajectory memory.
        logits=2.0*np.clip(reward,-.5,1.5)
        p=np.exp(logits-logits.max());p=.12/7+.88*p/p.sum()
        op=rng.choice(7,size=batch,p=p)
        C=np.empty((batch,d))
        a=(op==0)
        if a.any():C[a]=Z[:batch][a]+np.sqrt(d)*step[:batch][a,None]*steered[a]
        a=(op==1)
        if a.any():
            k=choose[a];C[a]=L[a]+radius[k,None]*rng.standard_normal((a.sum(),d))*np.sqrt(diag[k])
        a=(op==2)
        if a.any():
            k=choose[a];m=a.sum()
            count=max(1,min(d,int(np.sqrt(d))))
            mat=np.zeros((m,d)); ids=rng.integers(0,d,(m,count)); vals=rng.normal(size=(m,count))
            np.put_along_axis(mat,ids,vals,axis=1)
            C[a]=L[a]+(.85*radius[k,None])*mat
        a=(op==3)
        if a.any():
            k=choose[a];m=a.sum()
            C[a]=L[a]+radius[k,None]*(rng.normal(size=(m,1))*axes[k]+.33*rng.normal(size=(m,d))/np.sqrt(d))
        a=(op==4)
        if a.any():
            k=choose[a];m=a.sum();others=rng.integers(0,K,m)
            C[a]=L[a]+rng.uniform(-.6,.9,(m,1))*(L[a]-anchor[others])+.22*radius[k,None]*rng.normal(size=(m,d))
        a=(op==5)
        if a.any():
            k=choose[a];m=a.sum()
            # successful low-rank secant vectors, sampled WITHOUT objective derivatives.
            mm=memory[k]
            ww=mem_w[k]
            selected=rng.integers(0,R,size=m)
            directions=mm[np.arange(m),selected].copy()
            is_valid=ww[np.arange(m),selected]>1e-12
            directions[~is_valid]=axes[k[~is_valid]]
            if not use_secant_memory: directions=axes[k]
            if per_anchor_basis_mode!='off':
                localidx=np.minimum(local_n_basis[k]-1,R-1)
                can=localidx>=0
                select=np.array([rng.integers(0,int(j)+1) if j>=0 else 0 for j in localidx])
                proposal=local_basis[k,select]
                if per_anchor_basis_mode=='random-control':
                    proposal=_unit(rng.normal(size=(m,d)))
                directions=np.where(can[:,None],proposal,directions)
                movement=np.where(can,local_step[k,select],1.5*radius[k])
                C[a]=L[a]+rng.choice([-1.,1.],size=(m,1))*movement[:,None]*rng.uniform(.5,1.5,size=(m,1))*directions+.07*radius[k,None]*rng.normal(size=(m,d))
            else:
                C[a]=L[a]+(1.5*radius[k,None])*rng.uniform(.25,1.5,(m,1))*directions+.07*radius[k,None]*rng.normal(size=(m,d))
        # Coherent centroid: the weighted intersection/focus of multiple
        # productive anchors. Candidate is always verified.
        a=(op==6)
        if a.any():
            k=choose[a];m=a.sum()
            focus_active=(centroid_proposals and (not geometry_gated_centroid or
                          ((not np.isfinite(smoothness_error)) or
                           (smoothness_error<.015 or smoothness_error>.60))))
            if focus_active:
                lex=rank_indices(lf,lv,eps)
                ww=np.exp(-.55*np.arange(K))*(.4+anchor_weight[lex])
                focus=np.average(anchor[lex],axis=0,weights=ww)
                s=rng.uniform(.4,1.5,(m,1))
                C[a]=L[a]+s*(focus[None,:]-L[a])+(.14*radius[k,None])*rng.standard_normal((m,d))/np.sqrt(d)
            else:
                # Default operator: equal-budget ordinary anchor-local search.
                C[a]=L[a]+radius[k,None]*rng.standard_normal((m,d))*np.sqrt(diag[k])
        field_proposed=np.zeros(batch,dtype=bool)
        if use_radial_field_model and field_operator_mode>=1 and np.isfinite(field.score) and field.score>=field_quality_gate:
            # The fitted field is held-out validated.  Proposals are only
            # accepted if its *predicted score* improves over the ordinary probe line.
            eligible=np.isin(op,[0,1,3,5] if field_operator_mode==1 else [0,1,2,3,4,5,6]) & (~(stale[:batch]>=18))
            ix=np.flatnonzero(eligible)
            if len(ix):
                start_probe_line=np.where((op[ix]==0)[:,None],Z[ix],L[ix])
                displacement=C[ix]-start_probe_line
                distance=np.linalg.norm(displacement,axis=1)
                active=distance>1e-8
                ia=ix[active]
                if len(ia):
                    start_probe_line=start_probe_line[active]
                    distance=distance[active]
                    uv=displacement[active]/distance[:,None]
                    curved,turns=field.steer(start_probe_line,uv,distance,nsteps=3)
                    s0=field.predict(np.clip(C[ia],0,1));s1=field.predict(curved)
                    improve=(s1-s0)>0.01
                    improve&=(turns>.0001)
                    # The fitted score must beat both the ordinary probe line and the source.
                    improve&=(s1-field.predict(start_probe_line))>0.007
                    C[ia[improve]]=curved[improve]
                    field_proposed[ia[improve]]=True
                    field_steer_count+=int(improve.sum())
        # Prevent boundary clipping from creating a sticky atom: reflect and clip.
        C=np.where(C<0.,-C,C)
        C=np.where(C>1.,2.-C,C)
        C=np.clip(C,0,1)
        escape=(stale[:batch]>=18)&(rng.random(batch)<.22)
        escape|=(rng.random(batch)<.007)
        C[escape]=rng.random((escape.sum(),d))
        # Independent, optional pair-direction operator. Same candidate-evaluation
        # budget in learned and random-direction controls.
        pair_joint_mask=np.zeros(batch,bool)
        if joint_line_scan and d>=2 and batch>=6:
            sel=(rng.random(batch)<.17)&(~escape)
            if interaction_evidence_gate:
                sel &= (pair_probe_n[choose]>=2)&(pair_conf[choose]>.05)
            if sparse_interactions and it%3==0 and batch>=10:sel[-5:]=False
            for rr in np.flatnonzero(sel):
                kk=choose[rr]
                si,sj=pair_sign[kk]
                v=np.zeros(d);v[pair_i[kk]]=si;v[pair_j[kk]]=sj
                v/=(np.linalg.norm(v)+_EPS)
                radius_joint=(pair_step[kk] if interaction_trust else .045)
                C[rr]=L[rr]+rng.choice([-1.,1.])*rng.uniform(.5,1.5)*radius_joint*v
                pair_joint_mask[rr]=True
        # Explicit objective-value pair interaction: 4 single-coordinate
        # perturbations and one joint trial replace ordinary FE in a batch.
        # No analytical gradient or Hessian oracle is accessed.
        probe_info=None
        if sparse_interactions and d>=2 and batch>=10 and it%3==0:
            kk=int(rank_indices(lf,lv,eps)[(it//3)%min(3,K)])
            if interaction_control=="random" or not np.any(mem_w[kk]>1e-12):
                ii=int(rng.integers(d));jj=int((ii+rng.integers(1,d))%d)
            else:
                direction=axes[kk].copy()
                # Weight successful displacement directions. Keep some pair
                # diversity so early noisy axes do not permanently dominate.
                direction=.6*direction+.4*np.sum(memory[kk]*mem_w[kk,:,None],axis=0)/(mem_w[kk].sum()+_EPS)
                score=np.abs(direction)+.20*rng.random(d)/max(1.,np.sqrt(d))
                order2=np.argsort(-score);ii,jj=int(order2[0]),int(order2[1])
            hh=min(.045,max(.009,.55*radius[kk]))
            probes=np.repeat(anchor[kk][None,:],5,axis=0)
            probes[0,ii]+=hh;probes[1,ii]-=hh
            probes[2,jj]+=hh;probes[3,jj]-=hh
            probes[4,ii]+=hh*pair_sign[kk,0]
            probes[4,jj]+=hh*pair_sign[kk,1]
            ids=np.arange(batch-5,batch)
            C[ids]=np.clip(probes,0,1)
            choose[ids]=kk
            op[ids]=-1
            pair_joint_mask[ids]=False
            escape[ids]=False
            probe_info=(kk,ii,jj,hh,ids,lf[kk])
        if joint_line_scan or sparse_interactions:
            C=np.clip(C,0,1)
        # Optional experimental selective restart: replace only a stale,
        # spatially redundant anchor while retaining the incumbent and all
        # productive anchors. The scout is evaluated in the ordinary batch,
        # so no additional function call is hidden from the budget.
        restart_anchor=-1;restart_probe_line=-1
        if selective_anchor_restarts and it>12 and it%4==0 and batch>=4:
            dd=np.linalg.norm(anchor[:,None,:]-anchor[None,:,:],axis=2)/np.sqrt(d)
            np.fill_diagonal(dd,np.inf)
            eligible=np.flatnonzero((np.min(dd,axis=1)<.055)&(anchor_stall>=10))
            if len(eligible):
                # Avoid replacing the highest-quality anchor, even if clustered.
                rank=rank_indices(lf,lv,eps)
                quality=np.empty(K,int);quality[rank]=np.arange(K)
                elig=eligible[quality[eligible]>0]
                if len(elig):
                    restart_anchor=int(elig[np.argmax(anchor_stall[elig]+quality[elig])])
                    # Find an under-explored basin; maximin scouting is not
                    # a function-evaluation-based selection.
                    probes=rng.random((max(16,4*d),d))
                    scout_dist=np.linalg.norm(probes[:,None,:]-anchor[None,:,:],axis=2)/np.sqrt(d)
                    scout=probes[np.argmax(np.min(scout_dist,axis=1))]
                    restart_probe_line=int(rng.integers(batch))
                    C[restart_probe_line]=scout
                    escape[restart_probe_line]=True
        previous_best_f=bf;previous_best_v=bv
        CF,CV=eval_batch(C)
        if probe_info is not None:
            kk,ii,jj,hh,ids,initial_f=probe_info
            vals=CF[ids]
            # The measured interaction is a mixed finite difference of
            # objective values; no free evaluations or analytic derivatives.
            ia=int(np.argmin(vals[:2]));ja=2+int(np.argmin(vals[2:4]))
            sa=1. if ia==0 else -1.
            sb=1. if ja==2 else -1.
            predicted_joint=float(vals[ia]+vals[ja]-initial_f)
            actual_joint=float(vals[4])
            synergy=predicted_joint-actual_joint
            scale=max(1e-9,abs(initial_f-vals[ia])+abs(initial_f-vals[ja]),abs(initial_f)*.01)
            measured=float(np.clip(synergy/scale,-2.,2.))
            pair_probes+=1;pair_probe_n[kk]+=1
            if interaction_evidence_gate:
                # Preserve the pair that has demonstrated real synergistic gain.
                # The candidate must beat the BEST observed single-coordinate move;
                # a good surrogate interaction score alone is insufficient.
                actual_success=(actual_joint<min(float(vals[ia]),float(vals[ja]),float(initial_f))-1e-12)
                if actual_success and measured>.02 and measured>pair_conf[kk]*.85:
                    pair_i[kk]=ii;pair_j[kk]=jj
                    pair_sign[kk]=[sa,sb]
                    pair_conf[kk]=float(np.clip(measured,0,2))
                    pair_step[kk]=float(np.clip(.85*pair_step[kk]+.15*hh,.002,.17))
                else:
                    pair_conf[kk]=max(0.,pair_conf[kk]*.98)
            else:
                pair_i[kk]=ii;pair_j[kk]=jj
                pair_sign[kk]=[sa,sb]
                pair_conf[kk]=np.clip(.75*pair_conf[kk]+.25*measured,-2.,2.)
                pair_step[kk]=float(np.clip(.85*pair_step[kk]+.15*hh,.002,.17))
            interaction_samples.append((nfev,kk,ii,jj,measured,actual_joint,initial_f))
        if joint_line_scan:
            pair_joint_trials+=int(np.sum(pair_joint_mask))
            good_joint=pair_joint_mask&better(CF,CV,lf[choose],lv[choose],eps)
            pair_joint_success+=int(np.sum(good_joint))
            if interaction_trust:
                for kk in np.unique(choose[pair_joint_mask]):
                    ids=(choose==kk)&pair_joint_mask
                    accepted=good_joint[ids].mean() if np.any(ids) else 0.
                    pair_step[kk]=float(np.clip(pair_step[kk]*np.exp(.22*(accepted-.22)),.002,.18))
        field_move_improvements+=int(np.sum(field_proposed & (CF<lf[choose])))
        if restart_anchor>=0:
            k=restart_anchor; j=restart_probe_line
            anchor[k]=C[j].copy();lf[k]=float(CF[j]);lv[k]=float(CV[j])
            anchor_weight[k]=.15;radius[k]=.25;diag[k]=1.
            axes[k]=_unit(rng.normal(size=(1,d)))[0]
            mem_w[k]=0.;local_n_basis[k]=0;local_step[k]=.08;anchor_stall[k]=0
            anchor_restart_count+=1
        if adaptive_steering:
            # Joint candidates are not credited to the base operator they replaced.
            active=(op==0)&(~escape)&(~pair_joint_mask)
            if active.any():
                n_steered=np.sum(active&steer)
                if n_steered:
                    # A probe line is rewarded for an actual successful movement;
                    # anchor promotion is tracked separately, avoiding rare-event
                    # suppression when only incumbent-over-anchor is counted.
                    target_f=F[:batch] if outcome_steering_credit else lf[choose]
                    target_v=V[:batch] if outcome_steering_credit else lv[choose]
                    success_now=np.sum(active&steer&better(CF,CV,target_f,target_v,eps))
                    steer_reliability=.96*steer_reliability+.04*(success_now/max(1,n_steered))
                    steer_attempts+=int(n_steered);steer_success+=int(success_now)
        # strict incumbent is updated on every evaluated candidate
        ib=rank_indices(np.r_[bf,CF],np.r_[bv,CV])[0]
        if ib>0:bx=C[ib-1].copy();bf=float(CF[ib-1]);bv=float(CV[ib-1])
        # local improvements over actual selected anchor, with equality awareness
        improves=better(CF,CV,lf[choose],lv[choose],eps) & (~escape)
        gain=np.maximum(0,(lf[choose]-CF)/(1+np.abs(lf[choose])))
        # On constraints, successful reductions of infeasibility also earn credit.
        gain+=np.maximum(0,lv[choose]-CV)/(1+np.abs(lv[choose]))
        better_probe_line=better(CF,CV,F[:batch],V[:batch],eps)
        if better_probe_line.any():
            diff=C[better_probe_line]-Z[:batch][better_probe_line]
            vel[:batch][better_probe_line]=_unit(diff)
            Z[:batch][better_probe_line]=C[better_probe_line];F[:batch][better_probe_line]=CF[better_probe_line];V[:batch][better_probe_line]=CV[better_probe_line]
        stale[:batch][better_probe_line|escape]=0
        stale[:batch][~(better_probe_line|escape)]+=1
        if escape.any():vel[:batch][escape]=_unit(rng.normal(size=(escape.sum(),d)))
        step[:batch]=np.clip(step[:batch]*np.exp(.12*(better_probe_line.astype(float)-.25)),.00008,.25)
        anchor_stall+=1
        for kk in range(K):
            ids=(choose==kk)&(~escape)
            if not ids.any():continue
            s=improves&ids
            mean_gain=np.mean(gain[ids]*improves[ids])
            anchor_weight[kk]=np.clip(.95*anchor_weight[kk]+.05*(.08+8*mean_gain),.10,2.0)
            # Success-rate trust radius adaptation avoids premature shrinkage.
            nloc=ids&np.isin(op,[1,2,3,5])
            if nloc.any():radius[kk]=np.clip(radius[kk]*np.exp(.23*(improves[nloc].mean()-.20)),.00003,.32)
            if s.any():
                anchor_stall[kk]=0
                successes=C[s]-L[s]
                raw=_unit(successes)
                weighted=np.average(raw,axis=0,weights=np.maximum(gain[s],1e-6))
                av=_unit(weighted[None,:])[0]
                axes[kk]=_unit((.75*axes[kk]+.25*av)[None,:])[0]
                # diagonal geometric metric: successes shape the anchor's local anisotropy.
                newdiag=np.mean(raw**2,axis=0)*d
                diag[kk]=np.clip(.93*diag[kk]+.07*newdiag,.1,6)
                diag[kk]/=np.mean(diag[kk])
                if use_secant_memory:
                    for vec1,ww in zip(raw,gain[s]):
                        loc=mem_next[kk]%R
                        memory[kk,loc]=vec1
                        mem_w[kk,loc]=ww
                        mem_next[kk]+=1
                    if per_anchor_basis_mode!='off':
                        # Weighted low-rank tangent directions, learned from
                        # successful moves, not objective gradients. R <= 4.
                        valid=mem_w[kk]>1e-12
                        if valid.any():
                            weighted=memory[kk,valid]*np.sqrt(mem_w[kk,valid,None])
                            _,_,vt=np.linalg.svd(weighted,full_matrices=False)
                            r=min(len(vt),R)
                            local_basis[kk,:r]=vt[:r]
                            local_n_basis[kk]=r
                            lengths=np.linalg.norm(successes,axis=1)
                            for j in range(r):
                                projection=np.abs(successes@vt[j])
                                proposed=float(np.average(np.maximum(projection,1e-5),weights=np.maximum(gain[s],1e-9)))
                                local_step[kk,j]=np.clip(.85*local_step[kk,j]+.15*proposed,.0001,.30)
                j=np.flatnonzero(s)[np.argmin(CF[s]+(CV[s]>eps)*1e12)]
                anchor[kk]=C[j];lf[kk]=CF[j];lv[kk]=CV[j]
        # Candidate can become a new diverse anchor even if it had no local credit.
        for ii in rank_indices(np.r_[F[:batch],CF],np.r_[V[:batch],CV],eps)[:4]:
            if ii<batch:point=Z[ii];fp=F[ii];vp=V[ii]
            else:point=C[ii-batch];fp=CF[ii-batch];vp=CV[ii-batch]
            delta=np.linalg.norm(anchor-point,axis=1)/np.sqrt(d)
            near=int(np.argmin(delta))
            if delta[near]<.07:
                if better(np.array([fp]),np.array([vp]),np.array([lf[near]]),np.array([lv[near]]),eps)[0]:
                    anchor[near]=point;lf[near]=fp;lv[near]=vp
            else:
                worst=int(rank_indices(lf,lv,eps)[-1])
                if better(np.array([fp]),np.array([vp]),np.array([lf[worst]]),np.array([lv[worst]]),eps)[0]:
                    anchor[worst]=point;lf[worst]=fp;lv[worst]=vp
                    anchor_weight[worst]=.5;radius[worst]=.14;diag[worst]=1.
                    axes[worst]=_unit(rng.normal(size=(1,d)))[0];mem_w[worst]=0;local_n_basis[worst]=0;local_step[worst]=.08
                    if interaction_evidence_gate:
                        pair_conf[worst]=0.;pair_probe_n[worst]=0;pair_step[worst]=.045
        for j in range(7):
            ids=(op==j)&(~escape)&(~pair_joint_mask)
            if ids.any():
                if j==6 and centroid_proposals and centroid_global_credit and not geometry_gated_centroid:
                    globalsuccess=better(CF[ids],CV[ids],np.full(np.sum(ids),previous_best_f),np.full(np.sum(ids),previous_best_v),eps)
                    credit=np.mean(globalsuccess)
                else:
                    credit=np.mean(improves[ids])
                reward[j]=.93*reward[j]+.07*(.03+credit*1.7)
                op_count[j]+=ids.sum();op_good[j]+=np.sum(improves[ids])
        if it%record_every==0 or nfev==budget:
            hist.append((nfev,bf,bv))
    model_rejections=0; model_trials=0
    scan_count=0;scan_improvements=0;block_scans=0;conj_count=0;conj_success=0
    local_model_attempts=0;local_model_success=0;local_model_quality=[]
    geo=None;stage2_start_fe=-1
    if use_probe_line_refinement and constraints is None and nfev<budget:
        # Derivative-free, bounded line interpolation. Evaluating a
        # probe line in both directions is not a numerical gradient: a 1D quadratic
        # MINIMIZER is inferred from objective values, and checked by an
        # independently evaluated point. Trust region adapts by acceptance.
        hcoord=np.full(d,.045)
        def line_coordinate_order():
            if not prioritize_anchor_uncertainty:
                return rng.permutation(d)
            # Select axes with the largest geometric uncertainty relative to
            # successful anchor centres; this does not use a derivative estimate.
            ids=rank_indices(lf,lv)
            weights=np.exp(-.55*np.arange(K))*(.3+.7*anchor_weight[ids]); weights/=weights.sum()
            centre=np.sum(anchor[ids]*weights[:,None],axis=0)
            spread=np.sqrt(np.sum(weights[:,None]*(anchor[ids]-centre)**2,axis=0))
            importance=np.abs(bx-centre)/(spread+.07)
            # Include stochastic ties and diversity without knowing the function.
            importance=importance/(np.mean(importance)+1e-12)
            return np.argsort(-(importance+.30*rng.standard_normal(d)))
        perm=line_coordinate_order();ci=0
        dir_memory=[]
        local_model=LocalQuadraticModel(rank=5,observations=140)
        local_model_attempts=0;local_model_success=0;local_model_quality=[]
        probe_line_basis={};sweep_gains=np.zeros(d);basis_updates=0;boundary_fallbacks=0
        sweep_anchor=bx.copy();sweep_anchor_f=bf
        conj_count=0;conj_success=0
        stagnant=0
        probe_line_trust=.45
        stage2_start_fe=nfev
        if geometry_mode!='off':
            # ACLO: learns a conjugate frame from the curvature of already-paid line probes.
            from curvature_frame import CurvatureFrame
            geo=CurvatureFrame(d,mode=geometry_mode,seed=seed,**(geometry_options or {}))
        while nfev<budget:
            # Batched line interpolation. Each of B orthogonal probe lines is
            # measured in both directions. Their quadratic vertices are evaluated
            # and a combined move is verified. No gradient oracle is called.
            # All 3B+1 objective evaluations are charged to the budget.
            if batched_line_scan and focus_coordinates and d>=64 and nfev+4<budget:
                if ci>=d:
                    perm=line_coordinate_order();ci=0
                B=min(line_batch_size,d-ci,(budget-nfev-1)//3)
                if B>=2:
                    jj=perm[ci:ci+B];ci+=B
                    base=bx.copy();baseline=bf
                    h=hcoord[jj].copy()
                    ends=np.broadcast_to(base,(2*B,d)).copy()
                    rid=np.arange(B)
                    ends[rid,jj]=np.clip(base[jj]-h,0,1)
                    ends[B+rid,jj]=np.clip(base[jj]+h,0,1)
                    f_end,_=eval_batch(ends)
                    fm=f_end[:B];fp=f_end[B:]
                    curvature=fm+fp-2*baseline
                    trust=np.minimum(8.,.45/np.maximum(h,1e-12))
                    tau=np.zeros(B)
                    good_curvature=curvature>1e-12
                    tau[good_curvature]=np.clip(.5*(fm[good_curvature]-fp[good_curvature])/curvature[good_curvature],
                                               -trust[good_curvature],trust[good_curvature])
                    verts=np.broadcast_to(base,(B,d)).copy()
                    verts[rid,jj]=np.clip(base[jj]+tau*h,0,1)
                    fv,_=eval_batch(verts)
                    options=np.column_stack([fm,fp,fv,np.full(B,baseline)])
                    choice=np.argmin(options,axis=1)
                    minima=options[rid,choice]
                    candidate_coords=np.where(choice==0,ends[rid,jj],np.where(choice==1,ends[B+rid,jj],np.where(choice==2,verts[rid,jj],base[jj])))
                    eligible=minima<baseline-1e-14
                    combined=base.copy()
                    combined[jj[eligible]]=candidate_coords[eligible]
                    fcomb,_=eval_batch(combined[None,:])
                    winner=int(np.argmin(minima))
                    if fcomb[0]<min(baseline,minima[winner]):
                        bx=combined;bf=float(fcomb[0]);scan_improvements+=1
                    elif minima[winner]<baseline:
                        if choice[winner]==0:bx=ends[winner].copy()
                        elif choice[winner]==1:bx=ends[B+winner].copy()
                        elif choice[winner]==2:bx=verts[winner].copy()
                        bf=float(minima[winner]);scan_improvements+=1
                    hcoord[jj]=np.clip(h*np.where(eligible,1.04,.64),1e-8,.20)
                    if bf<baseline-1e-14:
                        delta=bx-base
                        nrm=np.linalg.norm(delta)
                        if nrm>1e-10:
                            dir_memory.append(delta/nrm)
                            if len(dir_memory)>6:dir_memory.pop(0)
                        stagnant=0
                    else:
                        stagnant+=B
                    if stagnant>2*d:
                        hcoord=np.maximum(hcoord,.003);stagnant=0
                    scan_count+=B;block_scans+=1
                    hist.append((nfev,bf,bv))
                    continue
            if ci>=d:
                if use_local_model and nfev<budget and scan_count>=d//2:
                    # Refit from archived objective evaluations at the current
                    # incumbent and verify a candidate line point (one FE).
                    if local_model.fit(bx,field.Z,field.F):
                        local_model_quality.append(local_model.quality)
                        if local_model.quality>=.60 and nfev+3<=budget:
                            # Trust-radius continuation. Candidate points are
                            # predicted without objective calls; all selected
                            # points are independently verified (charged to FE).
                            guesses=[]
                            for trust in [.010,.028,.067]:
                                trial,model_gain=local_model.propose(bx,trust=trust,curved=local_model_curvature)
                                if trial is not None:guesses.append(trial)
                            if guesses:
                                X=np.stack(guesses)
                                FS,_=eval_batch(X)
                                local_model_attempts+=len(X)
                                j=int(np.argmin(FS))
                                if FS[j]<bf:
                                    candidate=X[j]
                                    delta=candidate-bx
                                    if np.linalg.norm(delta)>1e-9:
                                        dir_memory.append(delta/np.linalg.norm(delta))
                                        if len(dir_memory)>6:dir_memory.pop(0)
                                    bx=candidate.copy();bf=float(FS[j]);local_model_success+=1
                # Only use coupled probe lines when independent line probes show
                # a predictable but nontrivial curved local geometry.
                # Near-quadratic separable landscapes retain coordinate scans;
                # very rough landscapes retain broad anchor-based exploration.
                if conjugate_line_direction and ((0. <= smoothness_error < .60) if quadratic_coupling else (.015 < smoothness_error < .60)) and nfev+3<=budget:
                    # Curved-valley tracking: follow the net displacement of
                    # the preceding full coordinate sweep as a new line direction.
                    # All probe line evaluations count against the exact FE budget.
                    displacement=bx-sweep_anchor
                    dis=np.linalg.norm(displacement)
                    if dis>1e-10:
                        direction=displacement/dis
                        h=min(.15*np.sqrt(d),max(.025,1.8*dis))
                        baseline=bf
                        tips=np.clip(bx[None,:]+np.array([-h,h])[:,None]*direction,0,1)
                        fs,_=eval_batch(tips)
                        if geo is not None:
                            geo.observe_points(bx,baseline,direction,tips,fs,kind='conjugate',nfev=nfev)
                        c=fs[0]+fs[1]-2*baseline
                        possible=[(baseline,bx.copy()),(float(fs[0]),tips[0].copy()),(float(fs[1]),tips[1].copy())]
                        if c>1e-12 and nfev<budget:
                            tau=np.clip(.5*(fs[0]-fs[1])/c,-min(8.,.65/h),min(8.,.65/h))
                            vertex=np.clip(bx+tau*h*direction,0,1)
                            fv,_=eval_batch(vertex[None,:])
                            possible.append((float(fv[0]),vertex))
                        fbest,xb=min(possible,key=lambda obj: obj[0])
                        conj_count+=1
                        if fbest<bf-1e-14:
                            bx=xb.copy();bf=float(fbest)
                            conj_success+=1
                            if coupled_probe_line_memory and baseline-fbest>memory_gain_gate*max(1e-14,sweep_anchor_f-baseline) and (geo is None or float(np.max(sweep_gains))>0.):
                                slot=int(np.argmax(sweep_gains))
                                # Bounded low-rank storage; canonical axes remain implicit.
                                if slot in probe_line_basis or len(probe_line_basis)<min(6,max(1,d//2)):
                                    probe_line_basis[slot]=direction.copy();basis_updates+=1
                                    hcoord[slot]=min(.10,max(.005,dis))
                # Controlled 2-variable refinement on a learned anchor pair.
                # The mixed difference estimates local coupling. No external
                # gradient oracle: 5 probes + 1 independently verified trial.
                if local_pair_correction and sparse_interactions and joint_line_scan and d>=2 and nfev+6<=budget:
                    kk=int(np.argmin(np.linalg.norm(anchor-bx[None,:],axis=1)))
                    if pair_probe_n[kk]:
                        ii,jj=int(pair_i[kk]),int(pair_j[kk])
                        hi=min(.035,max(.002, .7*min(bx[ii],1-bx[ii])))
                        hj=min(.035,max(.002, .7*min(bx[jj],1-bx[jj])))
                        if min(bx[ii],1-bx[ii])>.002 and min(bx[jj],1-bx[jj])>.002:
                            old=bx.copy();oldval=float(bf)
                            points=np.repeat(old[None,:],5,axis=0)
                            points[0,ii]+=hi;points[1,ii]-=hi
                            points[2,jj]+=hj;points[3,jj]-=hj
                            points[4,ii]+=hi;points[4,jj]+=hj
                            fv,_=eval_batch(np.clip(points,0,1))
                            gi=(fv[0]-fv[1])/(2*hi)
                            gj=(fv[2]-fv[3])/(2*hj)
                            Hii=(fv[0]+fv[1]-2*oldval)/(hi*hi)
                            Hjj=(fv[2]+fv[3]-2*oldval)/(hj*hj)
                            Hij=(fv[4]-fv[0]-fv[2]+oldval)/(hi*hj)
                            H=np.array([[Hii,Hij],[Hij,Hjj]])
                            eigen,Q=np.linalg.eigh(H)
                            lam=max(1e-6,.001*max(1.,float(np.max(np.abs(eigen)))))
                            Hreg=(Q*np.maximum(eigen,lam))@Q.T
                            v=-np.linalg.solve(Hreg,np.array([gi,gj]))
                            maxnorm=(pair_step[kk] if interaction_trust else .06)
                            nv=float(np.linalg.norm(v))
                            if nv>maxnorm:v*=maxnorm/nv
                            candidate=old.copy();candidate[[ii,jj]]+=v
                            candidate=np.clip(candidate,0,1)
                            ftrial,_=eval_batch(candidate[None,:])
                            pair_local_trials+=1
                            predicted=float(-np.dot([gi,gj],v)-.5*v@H@v)
                            realized=float(oldval-ftrial[0])
                            if realized>0:
                                bx=candidate.copy();bf=float(ftrial[0]);bv=0.
                                pair_local_success+=1
                                dis=bx-old
                                if np.linalg.norm(dis)>1e-12:
                                    dir_memory.append(dis/np.linalg.norm(dis))
                                    if len(dir_memory)>6:dir_memory.pop(0)
                            if interaction_trust:
                                ratio=realized/max(abs(predicted),1e-12)
                                factor=1.22 if ratio>.65 and realized>0 else (.62 if ratio<.10 else 1.)
                                pair_step[kk]=float(np.clip(pair_step[kk]*factor,.002,.18))
                sweep_gains[:]=0.
                sweep_anchor=bx.copy();sweep_anchor_f=bf
                perm=line_coordinate_order();ci=0
            if nfev>=budget:break
            mode=scan_count%7
            geo_probe_line=geo.next_probe_line(bx,bf,hcoord,nfev) if geo is not None else None
            if geo_probe_line is not None:
                # Frame or sensing probe line; the line search below is the base search's own.
                axis=np.asarray(geo_probe_line.direction,dtype=float).copy();h=float(geo_probe_line.step);is_coord=False
                if geo_probe_line.kind=='frame':ci+=1
            elif (focus_coordinates and len(perm)>ci) or mode in (0,1,2,3,4) or not dir_memory:
                axis=np.zeros(d);j=int(perm[ci]);ci+=1;axis[j]=1.
                if coupled_probe_line_memory and j in probe_line_basis:axis=probe_line_basis[j].copy()
                if boundary_probe_line_fallback and j in probe_line_basis:
                    # At a box corner a learned mixed direction can be blocked
                    # in the useful sign. Restore the canonical probe line when
                    # either side is blocked; one-sided straight probes cannot
                    # reliably preserve the learned coupled direction.
                    pos=axis>1e-14;neg=axis< -1e-14
                    upper=np.inf;lower=-np.inf
                    if pos.any():
                        upper=min(upper,float(np.min((1-bx[pos])/axis[pos])))
                        lower=max(lower,float(np.max(-bx[pos]/axis[pos])))
                    if neg.any():
                        upper=min(upper,float(np.min(-bx[neg]/axis[neg])))
                        lower=max(lower,float(np.max((1-bx[neg])/axis[neg])))
                    if min(upper,-lower)<=1e-12:
                        del probe_line_basis[j];axis=np.zeros(d);axis[j]=1.
                        boundary_fallbacks+=1
                h=float(hcoord[j]);is_coord=True
            elif mode==5:
                axis=dir_memory[-1].copy();h=.08;is_coord=False
            else:
                axis=_unit(rng.normal(size=(1,d)))[0];h=.10*np.sqrt(d);is_coord=False
            if geo is not None:
                geo_kind=geo_probe_line.kind if geo_probe_line is not None else ('coord' if is_coord else 'baseline')
                geo_index=geo_probe_line.index if geo_probe_line is not None else -1
            oldx=bx.copy();oldf=bf
            model_ratio=float("nan");model_step=0.
            offsets=np.array([-h,h])
            if bounded_interpolation:
                pos=axis>1e-14;neg=axis< -1e-14
                upper=np.inf;lower=-np.inf
                if pos.any():
                    upper=min(upper,float(np.min((1-oldx[pos])/axis[pos])))
                    lower=max(lower,float(np.max(-oldx[pos]/axis[pos])))
                if neg.any():
                    upper=min(upper,float(np.min(-oldx[neg]/axis[neg])))
                    lower=max(lower,float(np.max((1-oldx[neg])/axis[neg])))
                offsets=np.clip(offsets,lower,upper)
            trials=np.clip(oldx[None,:]+offsets[:,None]*axis,0,1)
            howmany=min(2,budget-nfev)
            tf,tv=eval_batch(trials[:howmany]);scan_count+=1
            for pidx in range(howmany):
                if better(np.array([tf[pidx]]),np.array([tv[pidx]]),
                          np.array([bf]),np.array([bv]),0)[0]:
                    bx=trials[pidx].copy();bf=float(tf[pidx]);bv=float(tv[pidx])
            # Curvature-based interpolation is used only when both sides were
            # evaluated, and its predicted point is independently validated.
            if howmany==2 and nfev<budget:
                fminus,fplus=tf
                curvature=fminus+fplus-2*oldf
                # Fit at the actual scalar offsets after clipping. At a boundary,
                # use a second inward point so three distinct values are available.
                tm=-float(np.dot(trials[0]-oldx,axis))
                tp=float(np.dot(trials[1]-oldx,axis))
                qa=curvature/(2*h*h); qb=(fplus-fminus)/(2*h)
                geo_pts=None
                if bounded_interpolation:
                    if min(tm,tp)<1e-12 and max(tm,tp)>1e-12 and nfev<budget:
                        far=tp if tp>=tm else -tm
                        middle=np.clip(oldx+.5*far*axis,0,1)
                        fmid,_=eval_batch(middle[None,:])
                        if fmid[0]<bf:bx=middle.copy();bf=float(fmid[0]);bv=0.
                        ffar=fplus if far>0 else fminus
                        qa=2*(ffar-2*fmid[0]+oldf)/(far*far)
                        qb=(ffar-oldf)/far-qa*far
                        geo_pts=((0.,.5*far,far),(oldf,float(fmid[0]),float(ffar)))
                    elif min(tm,tp)>1e-12:
                        dm=(oldf-fminus)/tm; dp=(fplus-oldf)/tp
                        qa=(dp-dm)/(tm+tp);qb=dm+qa*tm
                        geo_pts=((-tm,0.,tp),(float(fminus),oldf,float(fplus)))
                    else:
                        qa=0.
                if geo is not None and geo_pts is not None:
                    # Already-paid three-point line measurement (no extra FE).
                    geo.observe(axis,geo_pts[0],geo_pts[1],kind=geo_kind,index=geo_index,nfev=nfev)
                if qa*h*h>5e-13 and nfev<budget:
                    # Wider probe line extrapolation is permitted only when a
                    # separate quadratically-interpolated line held out from
                    # the fit is predictable; the proposed vertex is evaluated.
                    max_width= min(16., probe_line_trust/max(h,1e-12)) if smooth_probe_line_extension and is_coord else min(8., .45/max(h,1e-12))
                    tau=float(np.clip((-qb/(2*qa))/h,
                                          -max_width if expanded_probe_line_trust else -1.5,
                                           max_width if expanded_probe_line_trust else 1.5))
                    if abs(tau)>.005:
                        scalar_step=tau*h
                        if bounded_interpolation:scalar_step=float(np.clip(scalar_step,lower,upper))
                        trial=np.clip(oldx+scalar_step*axis,0,1)
                        fv,_=eval_batch(trial[None,:]);
                        actual_step=float(np.dot(trial-oldx,axis))
                        predicted_gain=-(qb*actual_step+qa*actual_step**2)
                        if predicted_gain>1e-16*max(1.,abs(oldf)):
                            model_ratio=(oldf-fv[0])/predicted_gain
                            model_step=abs(actual_step)
                            model_trials+=1
                        if smooth_probe_line_extension and is_coord:
                            predicted=oldf+.5*tau*(fplus-fminus)+.5*(tau*tau)*curvature
                            expected=oldf-predicted
                            if expected>1e-11:
                                quality=(oldf-fv[0])/expected
                                if quality>.70 and fv[0]<oldf:
                                    probe_line_trust=min(.70,probe_line_trust*1.012)
                                elif quality<.15:
                                    probe_line_trust=max(.25,probe_line_trust*.96)
                        if better(fv,np.array([0.]),np.array([bf]),np.array([bv]),0)[0]:
                            bx=trial.copy();bf=float(fv[0]);bv=0.
            if bf<oldf-1e-14:
                scan_improvements+=1
                move=bx-oldx
                norm=np.linalg.norm(move)
                if norm>1e-10:
                    dir_memory.append(move/norm)
                    if len(dir_memory)>6:dir_memory.pop(0)
                stagnant=0
                if is_coord:
                    hcoord[j]=min(.20,1.08*h)
                    sweep_gains[j]+=oldf-bf
            else:
                stagnant+=1
                if is_coord:hcoord[j]=max(1e-8,.58*h)
            if geo is not None and geo_probe_line is not None:
                geo.after_probe_line(geo_probe_line,improved=bool(bf<oldf-1e-14),displacement=float(np.dot(bx-oldx,axis)))
            if model_trust and is_coord and np.isfinite(model_ratio):
                # Objective values validate the probe line model; no objective gradient.
                # Reduce scale even after improvement when extrapolation is wrong.
                if abs(model_ratio-1.)>.25:
                    hcoord[j]=max(1e-8,.5*h);model_rejections+=1
                elif bf<oldf-1e-14:
                    hcoord[j]=float(np.clip(max(.5*h,2*model_step),1e-8,.20))
            if stagnant>2*d:
                hcoord=np.maximum(hcoord,.003);stagnant=0
            if scan_count%max(1,d//2)==0 or nfev==budget:
                hist.append((nfev,bf,bv))
    # strict best already recorded across ALL evaluations (some early relaxed fits
    # may be infeasible; strict feasibility always takes precedence in incumbent).
    if retain_all_evaluations:
        bx=observed_x.copy();bf=observed_f;bv=observed_v
        if not hist or hist[-1]!=(nfev,bf,bv):hist.append((nfev,bf,bv))
    geo_diag={'geometry_mode':geometry_mode,'stage2_start_fe':int(stage2_start_fe),'scan_frac':float(scan_frac)}
    if geo is not None:geo_diag.update(geo.summary())
    result=Result(lo+bx*span,bf,bv,nfev,time.perf_counter()-clock,np.asarray(hist),
                  {'selective_anchor_restarts':anchor_restart_count,'per_anchor_basis_mode':per_anchor_basis_mode,'basis_anchors_trained':int(np.sum(local_n_basis>0)),'boundary_fallbacks':boundary_fallbacks if 'boundary_fallbacks' in locals() else 0,'basis_updates':basis_updates if use_probe_line_refinement and constraints is None and 'basis_updates' in locals() else 0,'model_trials':model_trials,'model_radius_reductions':model_rejections,'interaction_probes':pair_probes,'interaction_joint_trials':pair_joint_trials,'interaction_joint_success':pair_joint_success,'interaction_local_trials':pair_local_trials,'interaction_local_success':pair_local_success,'interaction_pair_evidence':np.asarray(interaction_samples,dtype=float).reshape((-1,7)), 'interaction_pair_confidence':pair_conf.copy(),'steer_fraction':steer_events/max(1,nfev-N), 'steer_success_rate':steer_success/max(1,steer_attempts),
                   'radius':radius.copy(), 'operator_proportions':op_count/max(1,op_count.sum()),
                   'operator_success':op_good/np.maximum(1,op_count),'uses_external_gradient_oracle':False,'line_scans':scan_count,'line_blocks':block_scans,
                   'scan_improvements':scan_improvements,'local_predictability_error':smoothness_error,
                   'memory_scaling':f'O(population*dimension + K*4*dimension); d={d}',
                   'max_equality_violation_tolerance':equality_tol, 'conjugate_probe_line_trials':conj_count, 'conjugate_probe_line_successes':conj_success,
                   'field_validation_correlation':field.score,'field_updates':field.fit_count,
                   'field_steered_probe_lines':field_steer_count,'field_improving_probe_line_trials':field_move_improvements,
                   'local_model_trials':local_model_attempts if use_probe_line_refinement and constraints is None else 0,
                   'local_model_successes':local_model_success if use_probe_line_refinement and constraints is None else 0,
                   'local_model_cv_mean':float(np.mean(local_model_quality)) if use_probe_line_refinement and constraints is None and local_model_quality else float('nan')})
    result.diagnostics.update(geo_diag)
    return result
