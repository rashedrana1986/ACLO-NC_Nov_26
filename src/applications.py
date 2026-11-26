"""Engineering application problems of the ACLO paper.

A smoke test checked the data and every model (including the published
alpha-pinene optimum) before the experiments started.

All problems are minimisation problems on a box.  The optimiser only sees x -> f(x).

pinene   Chemical-engineering parameter estimation: thermal isomerisation of
         alpha-pinene (Fuguitt & Hawkins 1947; multiresponse analysis by Box, Hunter,
         MacGregor & Erjavec 1973).  Five first-order rate constants of a linear
         five-species ODE system are fitted to 8 x 5 measured compositions (weight %).
         Objective: sum of squared residuals.  Literature optimum: SSE = 19.872.
         Parameters are searched as theta_i * 1e5 in [0, 50].
gr4jcn   Water-resources model calibration: GR4J (Perrin, Michel & Andreassian 2003)
         with a CemaNeige-type degree-day snow module (Valery, Andreassian & Perrin
         2014); 6 parameters.
hbv      Water-resources model calibration: HBV (Bergstrom 1995; HBV-light structure
         of Seibert & Vis 2012) with snow, soil, response and triangular routing
         routines; 14 parameters.
Both hydrological problems use CAMELS-US basin 01031500, a USGS gauge in Maine
(Newman et al. 2015; Addor et al. 2017), daily Daymet forcing and observed runoff as
redistributed in the RRMPG repository (Kratzert, MIT licence).  Warm-up 1980-10-01 to
1981-09-30, calibration 1981-10-01 to 1996-09-30, validation 1996-10-01 to 2010-09-30.
Objective: 1 - KGE (Gupta et al. 2009) on the calibration period; validation KGE and
NSE are reported for the final parameter sets.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:   # still correct, about 100x slower
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

BAD = 1.0e3        # finite objective value for a degenerate simulation

# ---------------------------------------------------------------- alpha-pinene
PINENE_T = np.array([1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0])
PINENE_Y = np.array([
    [88.35, 7.3, 2.3, 0.4, 1.75],
    [76.4, 15.6, 4.5, 0.7, 2.8],
    [65.1, 23.1, 5.3, 1.1, 5.8],
    [50.4, 32.9, 6.0, 1.5, 9.3],
    [37.5, 42.7, 6.0, 1.9, 12.0],
    [25.9, 49.1, 5.9, 2.2, 17.0],
    [14.0, 57.4, 5.1, 2.6, 21.0],
    [4.5, 63.1, 3.8, 2.9, 25.7],
])
PINENE_Y0 = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
PINENE_SCALE = 1e-5
PINENE_F_REF = 19.872                       # Box et al. (1973); COPS benchmark
PINENE_THETA_REF = np.array([5.926, 2.963, 2.047, 27.45, 4.00])   # x = theta * 1e5


def pinene_matrix(theta):
    t1, t2, t3, t4, t5 = theta
    return np.array([
        [-(t1 + t2), 0.0, 0.0, 0.0, 0.0],
        [t1, 0.0, 0.0, 0.0, 0.0],
        [t2, 0.0, -(t3 + t4), 0.0, t5],
        [0.0, 0.0, t3, 0.0, 0.0],
        [0.0, 0.0, t4, 0.0, -t5],
    ])


def pinene_simulate(x):
    from scipy.linalg import expm
    K = pinene_matrix(np.asarray(x, dtype=float) * PINENE_SCALE)
    y = PINENE_Y0.copy()
    out = np.empty_like(PINENE_Y)
    t_prev = 0.0
    for k, t in enumerate(PINENE_T):
        y = expm(K * (t - t_prev)) @ y
        out[k] = y
        t_prev = t
    return out


def pinene_sse(x):
    sim = pinene_simulate(x)
    v = float(np.sum((sim - PINENE_Y) ** 2))
    return v if math.isfinite(v) else BAD


# ---------------------------------------------------------------- hydrology kernels
@njit(cache=False)
def gr4jcn_kernel(x1, x2, x3, x4, ctg, kf, P, T, E, gthr, n):
    Q = np.zeros(n)
    S = 0.3 * x1
    R = 0.5 * x3
    G = 0.0
    etg = 0.0
    n1 = int(math.ceil(x4))
    n2 = int(math.ceil(2.0 * x4))
    uh1 = np.zeros(n1)
    uh2 = np.zeros(n2)
    prev = 0.0
    for i in range(1, n1 + 1):
        t = float(i)
        sh = 1.0 if t >= x4 else (t / x4) ** 2.5
        uh1[i - 1] = sh - prev
        prev = sh
    prev = 0.0
    for i in range(1, n2 + 1):
        t = float(i)
        if t < x4:
            sh = 0.5 * (t / x4) ** 2.5
        elif t < 2.0 * x4:
            sh = 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
        else:
            sh = 1.0
        uh2[i - 1] = sh - prev
        prev = sh
    buf1 = np.zeros(n1)
    buf2 = np.zeros(n2)
    for k in range(n):
        p = P[k]
        temp = T[k]
        e = E[k]
        # snow (CemaNeige-type, one layer)
        fs = (3.0 - temp) / 4.0
        if fs < 0.0:
            fs = 0.0
        elif fs > 1.0:
            fs = 1.0
        psol = p * fs
        pliq = p - psol
        G += psol
        etg = ctg * etg + (1.0 - ctg) * temp
        if etg > 0.0:
            etg = 0.0
        pot = 0.0
        if etg >= 0.0 and temp > 0.0:
            pot = min(G, kf * temp)
        ratio = 1.0
        if gthr > 0.0:
            ratio = min(G / gthr, 1.0)
        melt = (0.9 * ratio + 0.1) * pot
        G -= melt
        pin = pliq + melt
        # production store
        if pin >= e:
            pn = pin - e
            ws = min(pn / x1, 13.0)
            th = math.tanh(ws)
            sr = S / x1
            ps = x1 * (1.0 - sr * sr) * th / (1.0 + sr * th)
            es = 0.0
        else:
            en = e - pin
            pn = 0.0
            ws = min(en / x1, 13.0)
            th = math.tanh(ws)
            sr = S / x1
            es = S * (2.0 - sr) * th / (1.0 + (1.0 - sr) * th)
            ps = 0.0
        S = S - es + ps
        if S < 0.0:
            S = 0.0
        sr = S / x1
        perc = S * (1.0 - (1.0 + (4.0 / 9.0 * sr) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        # unit hydrographs (90% slow branch, 10% direct branch)
        for j in range(n1 - 1):
            buf1[j] = buf1[j + 1] + 0.9 * pr * uh1[j]
        buf1[n1 - 1] = 0.9 * pr * uh1[n1 - 1]
        for j in range(n2 - 1):
            buf2[j] = buf2[j + 1] + 0.1 * pr * uh2[j]
        buf2[n2 - 1] = 0.1 * pr * uh2[n2 - 1]
        q9 = buf1[0]
        q1 = buf2[0]
        # groundwater exchange and routing store
        f_exch = x2 * (R / x3) ** 3.5
        R = R + q9 + f_exch
        if R < 0.0:
            R = 0.0
        qr = R * (1.0 - (1.0 + (R / x3) ** 4) ** -0.25)
        R -= qr
        qd = q1 + f_exch
        if qd < 0.0:
            qd = 0.0
        Q[k] = qr + qd
    return Q


@njit(cache=False)
def hbv_kernel(tt, cfmax, sfcf, cwh, cfr, fc, lp, beta, perc, uzl, k0, k1, k2, maxbas, P, T, E, n):
    Q = np.zeros(n)
    sp = 0.0
    wc = 0.0
    sm = 0.5 * fc
    suz = 0.0
    slz = 0.0
    nb = int(math.ceil(maxbas))
    w = np.zeros(nb)
    prev = 0.0
    for i in range(1, nb + 1):
        x = min(float(i), maxbas)
        if x <= 0.5 * maxbas:
            F = 2.0 * x * x / (maxbas * maxbas)
        else:
            F = 1.0 - 2.0 * (maxbas - x) ** 2 / (maxbas * maxbas)
        w[i - 1] = F - prev
        prev = F
    qbuf = np.zeros(nb)
    for k in range(n):
        p = P[k]
        temp = T[k]
        e = E[k]
        # snow routine
        if temp < tt:
            sp += p * sfcf
            rain = 0.0
        else:
            rain = p
        if temp > tt:
            melt = min(sp, cfmax * (temp - tt))
            sp -= melt
            wc += melt
        else:
            refr = min(wc, cfr * cfmax * (tt - temp))
            sp += refr
            wc -= refr
        wc += rain
        maxwc = cwh * sp
        inflow = 0.0
        if wc > maxwc:
            inflow = wc - maxwc
            wc = maxwc
        # soil routine
        rel = sm / fc
        if rel > 1.0:
            rel = 1.0
        recharge = inflow * rel ** beta
        sm += inflow - recharge
        if sm > fc:
            recharge += sm - fc
            sm = fc
        ea = e * min(sm / (lp * fc), 1.0)
        if ea > sm:
            ea = sm
        sm -= ea
        # response routine
        suz += recharge
        pc = min(perc, suz)
        suz -= pc
        slz += pc
        q0 = k0 * max(suz - uzl, 0.0)
        suz -= q0
        q1 = k1 * suz
        suz -= q1
        q2 = k2 * slz
        slz -= q2
        qg = q0 + q1 + q2
        # triangular routing
        for j in range(nb - 1):
            qbuf[j] = qbuf[j + 1] + qg * w[j]
        qbuf[nb - 1] = qg * w[nb - 1]
        Q[k] = qbuf[0]
    return Q


def kge(sim, obs):
    if sim.size < 2 or not np.all(np.isfinite(sim)):
        return -BAD
    so, ss = float(np.std(obs)), float(np.std(sim))
    mo, ms = float(np.mean(obs)), float(np.mean(sim))
    if so <= 0.0 or ss <= 0.0 or mo <= 0.0:
        return -BAD
    r = float(np.corrcoef(sim, obs)[0, 1])
    if not math.isfinite(r):
        return -BAD
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (ss / so - 1.0) ** 2 + (ms / mo - 1.0) ** 2)


def nse(sim, obs):
    if not np.all(np.isfinite(sim)):
        return -BAD
    den = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - float(np.sum((sim - obs) ** 2)) / den if den > 0 else -BAD


# ---------------------------------------------------------------- CAMELS basin
WARMUP0, CAL0, CAL1, VAL1 = '1980-10-01', '1981-10-01', '1996-09-30', '2010-09-30'


class Basin:
    def __init__(self, data_dir, basin='01031500'):
        import pandas as pd
        data_dir = Path(data_dir)
        f_force = next(data_dir.rglob(f'{basin}_lump_cida_forcing_leap.txt'), None)
        f_out = next(data_dir.rglob(f'{basin}_05_model_output.txt'), None)
        if f_force is None or f_out is None:
            raise FileNotFoundError(f'CAMELS files for basin {basin} not found under {data_dir}')
        fo = pd.read_csv(f_force, sep=r'\s+', header=3)
        fo.index = pd.to_datetime(fo['Year'].astype(str) + '/' + fo['Mnth'].astype(str) + '/'
                                  + fo['Day'].astype(str), format='%Y/%m/%d')
        mo = pd.read_csv(f_out, sep=r'\s+', header=0)
        mo.index = pd.to_datetime(mo['YR'].astype(str) + '/' + mo['MNTH'].astype(str) + '/'
                                  + mo['DY'].astype(str), format='%Y/%m/%d')
        df = pd.DataFrame({'P': fo['prcp(mm/day)'].astype(float),
                           'T': 0.5 * (fo['tmax(C)'].astype(float) + fo['tmin(C)'].astype(float))})
        df = df[~df.index.duplicated(keep='first')].sort_index().loc[WARMUP0:VAL1]
        mo = mo[~mo.index.duplicated(keep='first')]
        # The model-output file may skip a few days (e.g. 29 February): PET is interpolated
        # over such gaps and the observed runoff is left missing there.
        df['PET'] = mo['PET'].astype(float).reindex(df.index).interpolate(limit=3).bfill().ffill()
        df['OBS_RUN'] = mo['OBS_RUN'].astype(float).reindex(df.index)
        if df.index[0] != pd.Timestamp(WARMUP0) or df.index[-1] != pd.Timestamp(VAL1):
            raise ValueError(f'CAMELS record does not cover {WARMUP0}..{VAL1}: '
                             f'{df.index[0].date()}..{df.index[-1].date()}')
        if not np.all(np.diff(df.index.values).astype('timedelta64[D]').astype(int) == 1):
            raise ValueError('CAMELS record has gaps in the daily index')
        self.basin = basin
        self.dates = df.index
        self.P = np.ascontiguousarray(df['P'].values)
        self.T = np.ascontiguousarray(df['T'].values)
        self.E = np.ascontiguousarray(np.clip(df['PET'].values, 0.0, None))
        q = df['OBS_RUN'].values.copy()
        q[q < 0] = np.nan
        self.Q = q
        d = df.index
        self.n_cal = int(np.searchsorted(d, np.datetime64(CAL1), side='right'))   # run length for calibration
        self.cal = (d >= CAL0) & (d <= CAL1) & np.isfinite(q)
        self.val = (d > CAL1) & (d <= VAL1) & np.isfinite(q)
        self.cal_idx = np.flatnonzero(self.cal)
        self.val_idx = np.flatnonzero(self.val)
        fs = np.clip((3.0 - self.T) / 4.0, 0.0, 1.0)
        years = len(df) / 365.25
        self.gthr = 0.9 * float(np.sum(self.P * fs)) / years

    def summary(self):
        q = self.Q[self.cal]
        p = self.P[self.cal]
        e = self.E[self.cal]
        return {'basin': self.basin, 'start': str(self.dates[0].date()), 'end': str(self.dates[-1].date()),
                'n_days': int(len(self.dates)), 'n_cal_obs': int(self.cal.sum()), 'n_val_obs': int(self.val.sum()),
                'mean_P_cal': float(p.mean()), 'mean_PET_cal': float(e.mean()), 'mean_Q_cal': float(q.mean()),
                'runoff_ratio_cal': float(q.mean() / p.mean()), 'snow_threshold_mm': self.gthr,
                'numba': HAVE_NUMBA}


# ---------------------------------------------------------------- problem objects
class Problem:
    """name, dim, lower, upper, names, f(x), report(x)."""

    def __init__(self, name, dim, lower, upper, names, f_ref=None, kind=''):
        self.name, self.dim, self.kind = name, int(dim), kind
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.names = list(names)
        self.f_ref = f_ref
        if self.lower.shape != (self.dim,) or self.upper.shape != (self.dim,) or np.any(self.upper <= self.lower):
            raise ValueError(f'bad bounds for {name}')

    @property
    def bounds(self):
        return list(zip(self.lower.tolist(), self.upper.tolist()))

    def f(self, x):
        raise NotImplementedError

    def report(self, x):
        return {}


class PineneProblem(Problem):
    def __init__(self):
        super().__init__('pinene', 5, [0.0] * 5, [50.0] * 5,
                         ['k1_x1e5', 'k2_x1e5', 'k3_x1e5', 'k4_x1e5', 'k5_x1e5'],
                         f_ref=PINENE_F_REF, kind='chemical kinetics')

    def f(self, x):
        return pinene_sse(x)

    def report(self, x):
        sim = pinene_simulate(x)
        rmse = np.sqrt(np.mean((sim - PINENE_Y) ** 2, axis=0))
        return {'sse': pinene_sse(x), **{f'rmse_y{i + 1}': float(v) for i, v in enumerate(rmse)}}


class HydroProblem(Problem):
    def __init__(self, name, basin, dim, lower, upper, names):
        super().__init__(name, dim, lower, upper, names, kind='hydrological model calibration')
        self.b = basin

    def simulate(self, x, n):
        raise NotImplementedError

    def f(self, x):
        sim = self.simulate(x, self.b.n_cal)
        idx = self.b.cal_idx
        return 1.0 - kge(sim[idx], self.b.Q[idx])

    def report(self, x):
        sim = self.simulate(x, len(self.b.Q))
        out = {}
        for tag, idx in (('cal', self.b.cal_idx), ('val', self.b.val_idx)):
            out[f'kge_{tag}'] = kge(sim[idx], self.b.Q[idx])
            out[f'nse_{tag}'] = nse(sim[idx], self.b.Q[idx])
        return out


class GR4JCN(HydroProblem):
    def __init__(self, basin):
        super().__init__('gr4jcn', basin, 6, [10.0, -10.0, 10.0, 0.5, 0.0, 0.0],
                         [1500.0, 5.0, 500.0, 4.0, 1.0, 10.0], ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf'])

    def simulate(self, x, n):
        x = np.asarray(x, dtype=float)
        return gr4jcn_kernel(x[0], x[1], x[2], x[3], x[4], x[5], self.b.P, self.b.T, self.b.E,
                             self.b.gthr, int(n))


class HBV(HydroProblem):
    def __init__(self, basin):
        super().__init__('hbv', basin, 14,
                         [-2.5, 0.5, 0.4, 0.0, 0.0, 50.0, 0.3, 1.0, 0.0, 0.0, 0.05, 0.01, 0.001, 1.0],
                         [2.5, 10.0, 1.4, 0.2, 0.1, 700.0, 1.0, 6.0, 6.0, 100.0, 0.9, 0.5, 0.2, 6.0],
                         ['TT', 'CFMAX', 'SFCF', 'CWH', 'CFR', 'FC', 'LP', 'BETA', 'PERC', 'UZL',
                          'K0', 'K1', 'K2', 'MAXBAS'])

    def simulate(self, x, n):
        x = np.asarray(x, dtype=float)
        return hbv_kernel(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11],
                          x[12], x[13], self.b.P, self.b.T, self.b.E, int(n))


PROBLEM_NAMES = ('pinene', 'gr4jcn', 'hbv')
PARAM_NAMES = {'pinene': ['k1_x1e5', 'k2_x1e5', 'k3_x1e5', 'k4_x1e5', 'k5_x1e5'],
               'gr4jcn': ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf'],
               'hbv': ['TT', 'CFMAX', 'SFCF', 'CWH', 'CFR', 'FC', 'LP', 'BETA', 'PERC', 'UZL',
                       'K0', 'K1', 'K2', 'MAXBAS']}
_BASIN = {}


def make_problem(name, data_dir=None):
    if name == 'pinene':
        return PineneProblem()
    if data_dir is None:
        raise ValueError('hydrological problems need the CAMELS data folder')
    key = str(Path(data_dir).resolve())
    if key not in _BASIN:
        _BASIN[key] = Basin(data_dir)
    if name == 'gr4jcn':
        return GR4JCN(_BASIN[key])
    if name == 'hbv':
        return HBV(_BASIN[key])
    raise ValueError(f'unknown problem {name}')


def data_checks(data_dir):
    """Blocking sanity checks of the data and models (no optimisation)."""
    checks, info = {}, {}
    sums = PINENE_Y.sum(axis=1)
    checks['pinene_rows_sum_to_100'] = bool(np.all(np.abs(sums - 100.0) < 0.6))
    sse_ref = pinene_sse(PINENE_THETA_REF)
    info['pinene_sse_at_literature_optimum'] = sse_ref
    checks['pinene_literature_optimum_reproduced'] = bool(19.80 <= sse_ref <= 20.00)
    b = Basin(data_dir)
    s = b.summary()
    info['basin'] = s
    checks['basin_has_calibration_data'] = s['n_cal_obs'] > 4000 and s['n_val_obs'] > 3000
    checks['basin_water_balance_plausible'] = 0.1 < s['runoff_ratio_cal'] < 1.2
    for name in ('gr4jcn', 'hbv'):
        p = make_problem(name, data_dir)
        mid = 0.5 * (p.lower + p.upper)
        v = p.f(mid)
        rep = p.report(mid)
        info[f'{name}_mid_box'] = {'objective': v, **rep}
        checks[f'{name}_objective_finite'] = bool(math.isfinite(v) and v < BAD)
    return {'passed': all(checks.values()), 'checks': checks, 'info': info}
