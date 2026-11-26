"""Competitor optimisers used in the ACLO study.

These functions were used in the experiments of the paper; a smoke test called
every function on COCO problems before the experiments started.

Shared contract (identical for every competitor)
    run_X(record, bounds, budget, seed)
* `record` is the measurement wrapper: it counts every evaluation and raises
  budget.BudgetReached instead of ever exceeding `budget`;
* each function spends EXACTLY `budget` evaluations (the caller checks this
  against the recorder and against COCO's own evaluation counter);
* no function sees f_opt, a target value or any COCO internal: only x -> f(x).

Competitors (all with their published default parameters)
    BIPOP-CMA-ES   pycma `cma.fmin2(..., bipop=True)`; x0 ~ U[-4,4]^d,
                   sigma0 = 0.2 x domain width (Hansen's BBOB setting).
    BOBYQA         NLopt LN_BOBYQA (Powell 2009; npt = 2d+1), uniform random
                   restarts from U[-4,4]^d, initial trust radius 0.1 x width.
    L-SHADE        Tanabe & Fukunaga (2014): N_init = 18d, N_min = 4,
                   |A| = 2.6 N, H = 6, p = 0.11, current-to-pbest/1/bin,
                   weighted Lehmer means, terminal CR value, linear
                   population size reduction over the run budget.
    SPSO-2011      Clerc (2011) / Zambrano-Bigiarini et al. (2013): S = 40,
                   w = 1/(2 ln 2), c = 1/2 + ln 2, adaptive random topology
                   (K = 3), hypersphere sampling, confinement x = bound and
                   v = -0.5 v.  Synchronous (generation-wise) update.
BIPOP-CMA-ES, BOBYQA and SPSO-2011 are anytime methods (no budget-dependent
schedule): one run at the largest budget gives valid 100/300/1000xD values.
L-SHADE's population schedule depends on the budget, so it is run separately
at every budget, like ACLO.
"""
from __future__ import annotations

import math

import numpy as np

from budget import BudgetReached

X0_LOW, X0_HIGH = -4.0, 4.0     # standard BBOB start box (all optima lie in [-4,4]^d)
CMA_RESTARTS = 200              # restarts inside one fmin2 call (budget ends the run first)
MAX_OUTER = 1000                # protection against an endless outer loop

SETTINGS = {
    'BIPOP_CMAES': {'library': 'pycma cma.fmin2', 'bipop': True, 'restarts': CMA_RESTARTS,
                    'x0': 'U[-4,4]^d per restart', 'sigma0': '0.2*mean(ub-lb)',
                    'bounds': 'BBOB box via pycma BoundTransform', 'other': 'pycma defaults'},
    'BOBYQA': {'library': 'NLopt LN_BOBYQA', 'npt': '2d+1', 'x0': 'U[-4,4]^d per restart',
               'initial_step': '0.1*min(ub-lb)', 'xtol_rel': 1e-12,
               'restart_rule': 'NLopt stop, or no improvement for 20d+50 evaluations'},
    'LSHADE': {'r_Ninit': 18, 'N_min': 4, 'r_arc': 2.6, 'H': 6, 'p': 0.11,
               'M_F_init': 0.5, 'M_CR_init': 0.5, 'bounds': 'midpoint(parent, violated bound)'},
    'SPSO2011': {'S': 40, 'K': 3, 'w': '1/(2 ln 2)', 'c': '0.5 + ln 2',
                 'topology': 'adaptive random, re-drawn after an iteration without improvement',
                 'confinement': 'x=bound, v=-0.5v', 'update': 'synchronous'},
}


def _box(bounds):
    b = np.asarray(bounds, dtype=float)
    return b[:, 0].copy(), b[:, 1].copy()


def _start_box(lo, hi):
    return np.maximum(lo, X0_LOW), np.minimum(hi, X0_HIGH)


def _fill(record, budget, lo, hi, rng):
    """Spend a leftover too small for one more algorithm step on uniform samples."""
    while record.nfev < budget:
        record(rng.uniform(lo, hi))


# --------------------------------------------------------------------------
# BIPOP-CMA-ES (pycma)
# --------------------------------------------------------------------------
def run_bipop_cmaes(record, bounds, budget, seed):
    import cma
    lo, hi = _box(bounds)
    a, b = _start_box(lo, hi)
    rng = np.random.default_rng(seed)
    sigma0 = 0.2 * float(np.mean(hi - lo))

    def objective(x):
        return float(record(np.asarray(x, dtype=float)))

    def new_x0():
        return rng.uniform(a, b)

    for outer in range(MAX_OUTER):
        if record.nfev >= budget:
            break
        opts = {'seed': int((seed + 104729 * outer) % 2147483646) + 1,
                'bounds': [lo.tolist(), hi.tolist()],
                'verbose': -9, 'verb_disp': 0, 'verb_log': 0}
        before = record.nfev
        try:
            cma.fmin2(objective, new_x0, sigma0, options=opts,
                      restarts=CMA_RESTARTS, bipop=True)
        except BudgetReached:
            break
        if record.nfev == before:
            raise RuntimeError('BIPOP-CMA-ES returned without evaluating any point')
    else:
        raise RuntimeError('BIPOP-CMA-ES outer restart protection reached')
    if record.nfev != budget:
        raise RuntimeError(f'BIPOP-CMA-ES used {record.nfev} of {budget} evaluations')


# --------------------------------------------------------------------------
# BOBYQA (NLopt) with uniform random restarts
# --------------------------------------------------------------------------
def run_bobyqa(record, bounds, budget, seed):
    import nlopt
    lo, hi = _box(bounds)
    a, b = _start_box(lo, hi)
    d = lo.size
    rng = np.random.default_rng(seed)
    npt = 2 * d + 1
    rhobeg = 0.1 * float(np.min(hi - lo))
    stall_limit = 20 * d + 50
    for outer in range(MAX_OUTER * 100):
        remaining = budget - record.nfev
        if remaining < npt + 1:
            break
        opt = nlopt.opt(nlopt.LN_BOBYQA, d)
        opt.set_lower_bounds(lo.copy())
        opt.set_upper_bounds(hi.copy())
        opt.set_initial_step(np.full(d, rhobeg))
        opt.set_xtol_rel(1e-12)
        opt.set_maxeval(int(remaining))
        state = {'best': math.inf, 'stall': 0}

        def objective(x, grad, opt=opt, state=state):
            if record.nfev >= budget:          # cannot happen with set_maxeval; safety only
                opt.force_stop()
                return state['best'] if math.isfinite(state['best']) else 0.0
            v = float(record(np.array(x, dtype=float)))
            best = state['best']
            if not math.isfinite(best) or v < best - 1e-12 * max(1.0, abs(best)):
                state['best'] = v
                state['stall'] = 0
            else:
                state['stall'] += 1
                if state['stall'] >= stall_limit:
                    opt.force_stop()
            return v

        opt.set_min_objective(objective)
        before = record.nfev
        try:
            opt.optimize(rng.uniform(a, b))
        except BudgetReached:
            break
        except Exception:   # nlopt.ForcedStop / RoundoffLimited / failure -> next restart
            pass
        if record.nfev == before:
            raise RuntimeError('BOBYQA returned without evaluating any point')
    else:
        raise RuntimeError('BOBYQA restart protection reached')
    _fill(record, budget, lo, hi, rng)


# --------------------------------------------------------------------------
# L-SHADE
# --------------------------------------------------------------------------
def run_lshade(record, bounds, budget, seed, r_ninit=18.0, n_min=4, r_arc=2.6,
               h_mem=6, p_best=0.11):
    lo, hi = _box(bounds)
    d = lo.size
    rng = np.random.default_rng(seed)
    n_init = int(min(budget, max(n_min, round(r_ninit * d))))
    pop = rng.uniform(lo, hi, (n_init, d))
    fit = np.asarray(record(pop), dtype=float).reshape(-1)
    n = n_init
    m_f = np.full(h_mem, 0.5)
    m_cr = np.full(h_mem, 0.5)          # -1.0 encodes the terminal value
    k_mem = 0
    archive = np.empty((0, d))
    while record.nfev < budget:
        # --- control parameters -------------------------------------------
        r = rng.integers(0, h_mem, n)
        mcr_r = m_cr[r]
        cr = np.clip(rng.normal(mcr_r, 0.1), 0.0, 1.0)
        cr[mcr_r < 0] = 0.0
        f_scale = np.empty(n)
        todo = np.arange(n)
        while todo.size:
            val = m_f[r[todo]] + 0.1 * np.tan(np.pi * (rng.random(todo.size) - 0.5))
            ok = val > 0
            f_scale[todo[ok]] = np.minimum(val[ok], 1.0)
            todo = todo[~ok]
        # --- current-to-pbest/1 with archive ----------------------------
        order = np.argsort(fit, kind='stable')
        n_p = max(2, int(round(p_best * n)))
        pbest = pop[order[rng.integers(0, n_p, n)]]
        ar = np.arange(n)
        r1 = rng.integers(0, n - 1, n)
        r1 = r1 + (r1 >= ar)
        union = np.vstack([pop, archive]) if archive.shape[0] else pop
        n_union = union.shape[0]
        r2 = rng.integers(0, n_union, n)
        bad = (r2 == ar) | (r2 == r1)
        while bad.any():
            r2[bad] = rng.integers(0, n_union, int(bad.sum()))
            bad = (r2 == ar) | (r2 == r1)
        fs = f_scale[:, None]
        mutant = pop + fs * (pbest - pop) + fs * (pop[r1] - union[r2])
        mutant = np.where(mutant < lo, (lo + pop) / 2.0, mutant)
        mutant = np.where(mutant > hi, (hi + pop) / 2.0, mutant)
        # --- binomial crossover -----------------------------------------
        mask = rng.random((n, d)) < cr[:, None]
        mask[ar, rng.integers(0, d, n)] = True
        trial = np.where(mask, mutant, pop)
        # --- evaluation (truncated at the budget) -----------------------
        m = int(min(n, budget - record.nfev))
        tf = np.asarray(record(trial[:m]), dtype=float).reshape(-1)
        better = np.flatnonzero(tf < fit[:m])
        not_worse = np.flatnonzero(tf <= fit[:m])
        if better.size:
            delta = fit[better] - tf[better]
            s_f, s_cr = f_scale[better], cr[better]
            archive = np.vstack([archive, pop[better]])
            w = delta / delta.sum() if delta.sum() > 0 else np.full(better.size, 1.0 / better.size)
            m_f[k_mem] = float(np.sum(w * s_f ** 2) / max(np.sum(w * s_f), 1e-300))
            if m_cr[k_mem] < 0 or float(np.max(s_cr)) == 0.0:
                m_cr[k_mem] = -1.0
            else:
                m_cr[k_mem] = float(np.sum(w * s_cr ** 2) / np.sum(w * s_cr))
            k_mem = (k_mem + 1) % h_mem
        pop[not_worse] = trial[not_worse]
        fit[not_worse] = tf[not_worse]
        if m < n:
            break
        # --- linear population size reduction ---------------------------
        new_n = max(n_min, int(round((n_min - n_init) / budget * record.nfev + n_init)))
        if new_n < n:
            keep = np.argsort(fit, kind='stable')[:new_n]
            pop, fit, n = pop[keep], fit[keep], new_n
        a_max = int(round(r_arc * n))
        if archive.shape[0] > a_max:
            archive = archive[rng.choice(archive.shape[0], a_max, replace=False)]
    _fill(record, budget, lo, hi, rng)


# --------------------------------------------------------------------------
# SPSO-2011
# --------------------------------------------------------------------------
def _random_topology(rng, s, k):
    """links[j, i] is True when particle j informs particle i (each j informs itself)."""
    links = np.eye(s, dtype=bool)
    links[np.repeat(np.arange(s), k), rng.integers(0, s, s * k)] = True
    return links


def run_spso2011(record, bounds, budget, seed, swarm=40, k_inf=3):
    lo, hi = _box(bounds)
    d = lo.size
    rng = np.random.default_rng(seed)
    s = int(min(swarm, budget))
    w = 1.0 / (2.0 * math.log(2.0))
    c = 0.5 + math.log(2.0)
    x = rng.uniform(lo, hi, (s, d))
    v = rng.uniform(lo - x, hi - x)
    fx = np.asarray(record(x), dtype=float).reshape(-1)
    p, fp = x.copy(), fx.copy()
    links = _random_topology(rng, s, k_inf)
    best_so_far = float(fp.min())
    ar = np.arange(s)
    while record.nfev < budget:
        masked = np.where(links, fp[:, None], np.inf)
        lidx = np.argmin(masked, axis=0)
        own = (lidx == ar)[:, None]
        g = np.where(own, x + c * (p - x) / 2.0, x + c * (p + p[lidx] - 2.0 * x) / 3.0)
        radius = np.linalg.norm(g - x, axis=1)
        u = rng.normal(size=(s, d))
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
        xp = g + u * (radius * rng.random(s) ** (1.0 / d))[:, None]
        v = w * v + xp - x
        x = x + v
        out = (x < lo) | (x > hi)
        x = np.clip(x, lo, hi)
        v = np.where(out, -0.5 * v, v)
        m = int(min(s, budget - record.nfev))
        fnew = np.asarray(record(x[:m]), dtype=float).reshape(-1)
        imp = np.flatnonzero(fnew < fp[:m])
        p[imp] = x[imp]
        fp[imp] = fnew[imp]
        if m < s:
            break
        current = float(fp.min())
        if not current < best_so_far:
            links = _random_topology(rng, s, k_inf)
        best_so_far = min(best_so_far, current)
    _fill(record, budget, lo, hi, rng)


RUNNERS = {'bipop': run_bipop_cmaes, 'bobyqa': run_bobyqa,
           'lshade': run_lshade, 'spso': run_spso2011}
