"""Curvature-sensing conjugate frame for the stage-2 probe lines of ACLO.

Idea
----
Every stage-2 probe line of the base search already evaluates f at three
collinear points.  For a quadratic restricted to that line, the second divided
difference of those three values equals u^T H u exactly (u = unit direction of
the probe line), at any position and any step length.  This module

1. stores these already-paid curvature measurements (no extra evaluations),
2. fits a symmetric metric M by weighted least squares (dense mode,
   d <= dense_max_dim) or a low-rank frame by streaming deflation (factored
   mode, larger d),
3. tests every new measurement against the CURRENT fit before using it
   (a chronological held-out test, also free), and
4. only while that test passes, proposes probe-line directions along the
   eigen-frame of M; otherwise the base search's own coordinate probe lines
   are used.

It never calls the objective and never sees a gradient or Hessian oracle.  It
does not change the evaluation budget: it only chooses the direction and step
of probe lines that the base search evaluates anyway.

Modes (identical data, identical sensing design, identical gate)
    frame         probe-line frame = eigenvectors of the honest fit   (ACLO)
    random_frame  Haar-random orthonormal frame, same refresh times   (control)
    shuffled      eigenvectors of a fit to permuted curvature labels  (control)
    diagonal      coordinate axes, i.e. a diagonal metric only        (control)
Factored mode (d > dense_max_dim) supports 'frame' and 'random_frame' only.
The factored mode failed a development test (detection of the dominant
curvature directions at chance level at 160 and 640 dimensions); it is not
used for any result of the paper (all BBOB results are for d <= 40) and must
be redesigned before any large-scale use.

The least-squares fit is scale-free (no ridge, which biased large-curvature
entries in an earlier version), and the gate must beat a trivial isotropic
curvature predictor on the same measurements.

Theory: with exact line searches along an orthonormal frame Q in random order,
a quadratic contracts by about exp(-(1 - eta_J)) per sweep,
eta_J = ||D^-1/2 Q^T H Q D^-1/2 - I||_2, D = diag(Q^T H Q); see
`conjugacy_defect`.  Eigenvalue errors do not matter, only the frame does.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

try:  # QR-based least squares keeps one power of the condition number
    from scipy.linalg import lstsq as _scipy_lstsq
except Exception:  # pragma: no cover
    _scipy_lstsq = None

_EPS = float(np.finfo(float).eps)
MODES = ('frame', 'random_frame', 'shuffled', 'diagonal')
FACTORED_MODES = ('frame', 'random_frame')
_ERR_CAP = 200000


# --------------------------------------------------------------------------
# Measurement and diagnostics helpers
# --------------------------------------------------------------------------
def second_divided_difference(ts, fs, ulp_factor=16.0):
    """Return (s, noise) for three collinear samples f(x + t_k u).

    s is exact for quadratics: s = u^T H u (u unit, t in units of u).
    `noise` bounds the roundoff error of s when each f carries
    ulp_factor*eps*|f| absolute error.
    """
    t = np.asarray(ts, dtype=float).reshape(-1)
    f = np.asarray(fs, dtype=float).reshape(-1)
    if t.size != 3 or f.size != 3 or not (np.all(np.isfinite(t)) and np.all(np.isfinite(f))):
        return float('nan'), float('inf')
    order = np.argsort(t)
    ta, tb, tc = t[order]
    fa, fb, fc = f[order]
    left, right, span = tb - ta, tc - tb, tc - ta
    if min(left, right) <= 1e-300:
        return float('nan'), float('inf')
    s = 2.0 * ((fc - fb) / right - (fb - fa) / left) / span
    delta = ulp_factor * _EPS * max(abs(fa), abs(fb), abs(fc), 1e-300)
    noise = 4.0 * delta * (1.0 / right + 1.0 / left) / span
    return float(s), float(noise)


def haar_orthogonal(n, rng, k=None):
    """Haar-distributed orthonormal n x k matrix (k=n by default)."""
    k = n if k is None else int(k)
    g = rng.standard_normal((n, k))
    q, r = np.linalg.qr(g)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    return q * sign[None, :]


def conjugacy_defect(Q, H):
    """eta_J of frame Q (d x k, completed to d x d) for a symmetric PD H."""
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    d = H.shape[0]
    if Q.ndim != 2 or Q.shape[0] != d:
        raise ValueError('frame and matrix dimensions differ')
    if Q.shape[1] < d:
        P = np.eye(d) - Q @ Q.T
        w, V = np.linalg.eigh(P)
        Q = np.column_stack([Q, V[:, w > 0.5]])
    A = Q.T @ H @ Q
    dg = np.diag(A)
    if np.any(dg <= 0):
        return float('inf')
    s = 1.0 / np.sqrt(dg)
    Abar = A * s[:, None] * s[None, :]
    return float(np.linalg.norm(Abar - np.eye(Q.shape[1]), 2))


@dataclass
class ProbeLine:
    direction: np.ndarray
    step: float
    kind: str          # 'frame' or 'sense'
    index: int = -1    # column of the probe line frame for 'frame' probe lines


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------
class CurvatureFrame:
    """Learns a conjugate probe-line frame from the curvatures of the search's own line probes."""

    def __init__(self, d, mode='frame', seed=0, dense_max_dim=40, rank_max=None,
                 window_factor=1.5, forgetting=0.5,
                 gate_error=0.25, gate_close_factor=1.6, gate_skill_ratio=0.5,
                 gate_mode='absolute', skill_open=0.75, skill_close=0.9,
                 gate_window=None,
                 min_measurements=None, sensing_period_closed=2,
                 sensing_period_open=4, whitened_sensing=0.5,
                 refit_fraction=0.25, roundoff_tolerance=1e-3,
                 step_bounds=(1e-8, 0.2), stall_reset_step=0.003,
                 add_ratio=10.0, oja_rate=0.5, record_events=False,
                 event_cap=20000):
        d = int(d)
        if d < 1:
            raise ValueError('dimension must be positive')
        if mode not in MODES:
            raise ValueError(f'unknown geometry mode {mode!r}')
        self.d = d
        self.mode = mode
        self.dense = d <= int(dense_max_dim)
        if not self.dense and mode not in FACTORED_MODES:
            raise ValueError(f'mode {mode!r} is defined only for d <= dense_max_dim={dense_max_dim}')
        if int(sensing_period_closed) < 1 or int(sensing_period_open) < 1:
            raise ValueError('sensing periods must be >= 1')
        # Own generator: the generator of the base search is never touched by this module.
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 7919]))
        self.forgetting = float(forgetting)
        self.gate_error = float(gate_error)
        self.gate_close_factor = float(gate_close_factor)
        self.gate_skill_ratio = float(gate_skill_ratio)
        # Gate options.  'absolute' (default) is the earlier absolute-error gate.
        # 'skill' drops the absolute error threshold: on T_osz functions the
        # curvature carries irreducible noise (development runs: errors 0.28-0.53, gate
        # flapping 69-141 times per run), yet the frame is what matters.  It
        # opens when the model's error is <= skill_open x the isotropic
        # predictor's error and closes when it exceeds skill_close x it.
        if gate_mode not in ('absolute', 'skill'):
            raise ValueError("gate_mode must be 'absolute' or 'skill'")
        self.gate_mode = gate_mode
        self.skill_open = float(skill_open)
        self.skill_close = float(skill_close)
        self.sense_closed = int(sensing_period_closed)
        self.sense_open = int(sensing_period_open)
        self.whitened = float(whitened_sensing)
        self.refit_fraction = float(refit_fraction)
        self.tol = float(roundoff_tolerance)
        self.h_lo, self.h_hi = float(step_bounds[0]), float(step_bounds[1])
        self.stall_reset_step = float(stall_reset_step)
        self.add_ratio = float(add_ratio)
        self.oja_rate = float(oja_rate)
        gw = int(gate_window) if gate_window else int(min(64, max(16, d)))
        self.errs = deque(maxlen=gw)
        self.errs_all = []
        # Errors of the trivial isotropic predictor on the same measurements:
        # the gate must beat it, not just an absolute threshold.
        self.iso_errs = deque(maxlen=gw)
        self.iso_all = []
        self.shadow_errs = deque(maxlen=gw)
        self.shadow_all = []
        self.gate_open = False
        self.min_meas = int(min_measurements) if min_measurements else (d + 1 if self.dense else 2 * d)
        self.n_meas = 0
        self.counts = dict(rejected_roundoff=0, rejected_offline=0, rejected_nonfinite=0,
                           refits=0, frame_probe_lines=0, frame_success=0, sense_probe_lines=0,
                           sense_success=0, gate_open_events=0, gate_close_events=0,
                           promotions=0, demotions=0)
        self.first_open_fe = -1
        self.fit_seconds = 0.0
        self.frame_overlap = []
        self._probe_line_counter = 0
        self._perm = []
        self._hc = None
        self.hdir = None
        self._stall = 0
        self.events = [] if record_events else None
        self.event_cap = int(event_cap)
        if self.dense:
            self._init_dense(window_factor)
        else:
            self._init_factored(rank_max)

    # ------------------------------------------------------------------ init
    def _init_dense(self, window_factor):
        d = self.d
        self.p = d * (d + 1) // 2
        iu = np.triu_indices(d)
        self._iu0, self._iu1 = iu
        self._off = iu[0] != iu[1]
        self.W = int(window_factor * self.p) + d
        self._dirs = np.zeros((self.W, d))
        self._s = np.zeros(self.W)
        self._stamp = np.zeros(self.W)
        self._filled = 0
        self._head = 0
        self._since_fit = 0
        self.M = None           # honest fit (used for prediction and gate)
        self.M_shadow = None    # fit to permuted labels ('shuffled' arm only)
        self.eigval = None
        self.eigvec = None
        self.Qprobe_line = np.eye(d)   # frame used for probe lines

    def _init_factored(self, rank_max):
        d = self.d
        self.r_max = int(rank_max) if rank_max else int(max(1, min(32, d // 8)))
        self.Q = np.zeros((d, 0))
        self.sigma = np.zeros(0)
        self.sigma_n = np.zeros(0, dtype=int)
        self.m1 = None
        self.n_perp = 0
        self._since_promo = 0
        self.k_sketch = 2
        self.Vt = self._orth_complement(self.rng.standard_normal((d, self.k_sketch)))
        self.Vb = self._orth_complement(self.rng.standard_normal((d, self.k_sketch)))
        self.m2t = np.zeros(self.k_sketch)
        self.m2b = np.zeros(self.k_sketch)
        self.Qprobe_line = np.zeros((d, 0))

    # ------------------------------------------------------------ prediction
    def predict(self, u):
        """Predicted u^T H u for a unit direction, or None before any fit."""
        if self.dense:
            if self.M is None:
                return None
            return float(u @ self.M @ u)
        if self.m1 is None:
            return None
        r = self.Q.shape[1]
        if r:
            a = self.Q.T @ u
            rest = max(0.0, 1.0 - float(a @ a))
            return float(self.sigma @ (a * a) + self.m1 * rest)
        return float(self.m1)

    def predict_isotropic(self):
        """Trivial baseline: mean curvature of the current fit, any direction."""
        if self.dense:
            return None if self.M is None else float(np.trace(self.M) / self.d)
        return None if self.m1 is None else float(self.m1)

    def _has_frame(self):
        return (self.M is not None) if self.dense else (self.Q.shape[1] >= 1)

    # ------------------------------------------------------------ observing
    def observe(self, direction, ts, fs, kind='probe_line', index=-1, nfev=-1):
        """Record a three-point line measurement; returns s or None."""
        u = np.asarray(direction, dtype=float).reshape(-1)
        nrm = float(np.linalg.norm(u)) if u.size else 0.0
        if u.size != self.d or not np.isfinite(nrm) or nrm < 1e-12:
            self.counts['rejected_nonfinite'] += 1
            return None
        u = u / nrm
        ts = np.asarray(ts, dtype=float) * nrm
        s, noise = second_divided_difference(ts, fs)
        if not np.isfinite(s):
            self.counts['rejected_nonfinite'] += 1
            return None
        if abs(s) * self.tol < noise:
            self.counts['rejected_roundoff'] += 1
            return None
        pred = self.predict(u)
        err = None
        if pred is not None and np.isfinite(pred):
            err = abs(s - pred) / (abs(s) + abs(pred) + 1e-300)
            self.errs.append(err)
            if len(self.errs_all) < _ERR_CAP:
                self.errs_all.append(err)
            iso = self.predict_isotropic()
            if iso is not None and np.isfinite(iso):
                ie = abs(s - iso) / (abs(s) + abs(iso) + 1e-300)
                self.iso_errs.append(ie)
                if len(self.iso_all) < _ERR_CAP:
                    self.iso_all.append(ie)
        if self.dense and self.mode == 'shuffled' and self.M_shadow is not None:
            sp = float(u @ self.M_shadow @ u)
            se = abs(s - sp) / (abs(s) + abs(sp) + 1e-300)
            self.shadow_errs.append(se)
            if len(self.shadow_all) < _ERR_CAP:
                self.shadow_all.append(se)
        self.n_meas += 1
        if self.dense:
            self._store_dense(u, s)
            self._maybe_fit_dense()
        else:
            self._update_factored(u, s, kind)
        self._update_gate(nfev)
        if self.events is not None and len(self.events) < self.event_cap:
            self.events.append((int(nfev), str(kind), float(s),
                                float('nan') if pred is None else float(pred),
                                float('nan') if err is None else float(err),
                                int(self.gate_open)))
        return s

    def observe_points(self, x0, f0, direction, points, fvals, kind='probe_line', nfev=-1):
        """Measurement from x0 and two evaluated points that must lie on the line."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = np.asarray(direction, dtype=float).reshape(-1)
        nrm = float(np.linalg.norm(u))
        if nrm < 1e-12 or u.size != self.d:
            self.counts['rejected_nonfinite'] += 1
            return None
        u = u / nrm
        P = np.atleast_2d(np.asarray(points, dtype=float))
        if P.shape[0] != 2:
            return None
        disp = P - x0[None, :]
        t = disp @ u
        resid = np.linalg.norm(disp - t[:, None] * u[None, :], axis=1)
        if np.any(resid > 1e-9 * (1.0 + np.abs(t))):
            self.counts['rejected_offline'] += 1   # clipping moved a point off the line
            return None
        fv = np.asarray(fvals, dtype=float).reshape(-1)
        return self.observe(u, np.r_[0.0, t], np.r_[float(f0), fv[:2]], kind=kind, nfev=nfev)

    # ---------------------------------------------------------- dense model
    def _store_dense(self, u, s):
        i = self._head
        self._dirs[i] = u
        self._s[i] = s
        self._stamp[i] = self.n_meas
        self._head = (i + 1) % self.W
        self._filled = min(self._filled + 1, self.W)
        self._since_fit += 1

    def _maybe_fit_dense(self):
        need = max(1, self.d // 2, int(self.refit_fraction * self._filled))
        if self._filled >= self.d + 1 and self._since_fit >= need:
            self._fit_dense()

    def _solve_dense(self, F, s, age):
        # Scale-free fit: a ridge used in an earlier version acted in absolute curvature
        # units while rows are relative-error weighted, so it overwhelmed the
        # data on large-curvature entries (discus, kappa=1e6 failed; cigar
        # passed only because its isotropic prior is nearly correct).  Now the
        # fit is scale-free: least squares for the DEVIATION from an isotropic
        # prior, minimum-norm in the unidentified directions, no ridge, so
        # identified components carry no prior bias at all.
        p = self.p
        smax = float(np.max(np.abs(s))) if s.size else 0.0
        c = np.abs(s) + 1e-9 * smax + 1e-300          # relative-error weighting
        w = np.sqrt(self.forgetting ** (age / max(1.0, float(p)))) / c
        A = F * w[:, None]
        prior = np.zeros(p)
        prior[~self._off] = float(np.median(s))      # isotropic prior for unseen structure
        b = (s - F @ prior) * w
        try:
            if _scipy_lstsq is not None:
                delta = _scipy_lstsq(A, b, lapack_driver='gelsy', check_finite=False)[0]
            else:
                delta = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            return None
        theta = prior + delta
        if not np.all(np.isfinite(theta)):
            return None
        M = np.zeros((self.d, self.d))
        M[self._iu0, self._iu1] = theta
        return M + M.T - np.diag(np.diag(M))

    def _fit_dense(self):
        t0 = time.perf_counter()
        n = self._filled
        U = self._dirs[:n]
        s = self._s[:n]
        age = self.n_meas - self._stamp[:n]
        F = U[:, self._iu0] * U[:, self._iu1]
        F[:, self._off] *= 2.0
        M_new = self._solve_dense(F, s, age)
        if M_new is not None:
            lam, V = np.linalg.eigh(M_new)
            self.M, self.eigval, self.eigvec = M_new, lam, V
            Qnew = None
            if self.mode == 'frame':
                Qnew = V
            elif self.mode == 'random_frame':
                Qnew = haar_orthogonal(self.d, self.rng)
            elif self.mode == 'diagonal':
                Qnew = np.eye(self.d)
            else:  # shuffled: same design rows, permuted curvature labels
                Ms = self._solve_dense(F, s[self.rng.permutation(n)], age)
                if Ms is not None:
                    self.M_shadow = Ms
                    Qnew = np.linalg.eigh(Ms)[1]
            if Qnew is not None:
                self._set_probe_line_frame(Qnew)
            self.counts['refits'] += 1
        self._since_fit = 0
        self.fit_seconds += time.perf_counter() - t0

    # ------------------------------------------------------- factored model
    def _orth_complement(self, V):
        V = np.asarray(V, dtype=float)
        if self.Q.shape[1]:
            V = V - self.Q @ (self.Q.T @ V)
        q, _ = np.linalg.qr(V)
        return q

    def _update_factored(self, u, s, kind):
        r = self.Q.shape[1]
        a = self.Q.T @ u if r else np.zeros(0)
        if r and float(np.max(np.abs(a))) > 0.999:
            i = int(np.argmax(np.abs(a)))      # a probe line along frame column i
            beta = 1.0 if self.sigma_n[i] == 0 else (0.5 if self.sigma_n[i] < 4 else 0.25)
            self.sigma[i] = (1.0 - beta) * self.sigma[i] + beta * s
            self.sigma_n[i] += 1
            self._maybe_demote(i)
            return
        if kind != 'sense' or (r and float(a @ a) > 1e-6):
            return                              # only isotropic complement samples feed moments
        n = self.d - r
        if n < 3:
            return
        self.n_perp += 1
        self._since_promo += 1
        beta = 1.0 / max(16.0, float(n))
        self.m1 = s if self.m1 is None else (1.0 - beta) * self.m1 + beta * s
        scale = abs(self.m1) if self.m1 else abs(s)
        z = float(np.clip(s / max(scale, 1e-300), -1e3, 1e3))
        g = float(np.sign(z) * min(0.25, self.oja_rate * abs(z) / n))
        wt = u @ self.Vt
        wb = u @ self.Vb
        self.Vt = self._orth_complement(self.Vt + g * np.outer(u, wt))   # Oja: top eigenvectors
        self.Vb = self._orth_complement(self.Vb - g * np.outer(u, wb))   # anti-Oja: bottom
        beta2 = 1.0 / max(64.0, 4.0 * n)
        self.m2t = (1.0 - beta2) * self.m2t + beta2 * s * (u @ self.Vt) ** 2
        self.m2b = (1.0 - beta2) * self.m2b + beta2 * s * (u @ self.Vb) ** 2
        self._maybe_promote()

    def _maybe_promote(self):
        r = self.Q.shape[1]
        n = self.d - r
        if r >= self.r_max or n < 4 or self.m1 is None or self._since_promo < 2 * n:
            return
        if self.m1 <= 0:
            return
        # Unbiased Rayleigh estimates for w uniform on the complement sphere:
        # E[s (w.v)^2] = (2 v'Av + tr A)/(n(n+2)),  E[s] = tr A / n.
        mu_t = 0.5 * (n * (n + 2) * self.m2t - n * self.m1)
        mu_b = 0.5 * (n * (n + 2) * self.m2b - n * self.m1)
        cand = None
        jt = int(np.argmax(mu_t))
        jb = int(np.argmin(mu_b))
        if mu_t[jt] > self.add_ratio * self.m1:
            cand = (self.Vt[:, jt].copy(), float(mu_t[jt]))
        elif mu_b[jb] < self.m1 / self.add_ratio:
            cand = (self.Vb[:, jb].copy(), float(mu_b[jb]))
        if cand is None:
            return
        q = cand[0] - self.Q @ (self.Q.T @ cand[0]) if r else cand[0]
        nq = float(np.linalg.norm(q))
        if nq < 1e-9:
            return
        self.Q = np.column_stack([self.Q, q / nq])
        self.sigma = np.r_[self.sigma, cand[1]]
        self.sigma_n = np.r_[self.sigma_n, 0]
        self.m1 = (n * self.m1 - cand[1]) / (n - 1)
        self.Vt = self._orth_complement(self.rng.standard_normal((self.d, self.k_sketch)))
        self.Vb = self._orth_complement(self.rng.standard_normal((self.d, self.k_sketch)))
        self.m2t[:] = 0.0
        self.m2b[:] = 0.0
        self._since_promo = 0
        self.counts['promotions'] += 1
        self._refresh_factored_frame()

    def _maybe_demote(self, i):
        if self.m1 is None or self.m1 <= 0 or self.sigma_n[i] < 1:
            return
        ratio = self.sigma[i] / self.m1
        if 1.0 / 3.0 < ratio < 3.0:          # not distinguishable from the complement
            n = self.d - self.Q.shape[1]
            removed = float(self.sigma[i])
            keep = np.ones(self.Q.shape[1], dtype=bool)
            keep[i] = False
            self.Q = self.Q[:, keep]
            self.sigma = self.sigma[keep]
            self.sigma_n = self.sigma_n[keep]
            self.m1 = (n * self.m1 + removed) / (n + 1)
            self.counts['demotions'] += 1
            self._refresh_factored_frame()

    def _refresh_factored_frame(self):
        r = self.Q.shape[1]
        if self.mode == 'frame':
            Qnew = self.Q.copy()
        else:
            Qnew = haar_orthogonal(self.d, self.rng, r) if r else np.zeros((self.d, 0))
        self._set_probe_line_frame(Qnew)

    # ------------------------------------------------------- frame and steps
    def _steps_from_coordinates(self, Q):
        hc = np.full(self.d, 0.045) if self._hc is None else np.asarray(self._hc, dtype=float)
        h = np.sqrt((Q ** 2).T @ (hc ** 2))
        return np.clip(h, self.h_lo, self.h_hi)

    def _set_probe_line_frame(self, Qnew):
        Qnew = np.asarray(Qnew, dtype=float)
        k_new = Qnew.shape[1]
        if self.hdir is not None and k_new and self.Qprobe_line.shape[1] and self.hdir.size == self.Qprobe_line.shape[1]:
            C = Qnew.T @ self.Qprobe_line
            O2 = C ** 2
            rows = np.clip(O2.sum(axis=1), 0.0, 1.0)
            hc_map = self._steps_from_coordinates(Qnew)
            h2 = O2 @ (self.hdir ** 2) + (1.0 - rows) * hc_map ** 2
            self.hdir = np.clip(np.sqrt(h2), self.h_lo, self.h_hi)
            self.frame_overlap.append(float(np.mean(np.max(np.abs(C), axis=1))))
        elif self.hdir is not None:
            self.hdir = self._steps_from_coordinates(Qnew) if k_new else None
        self.Qprobe_line = Qnew
        self._perm = []

    def _update_gate(self, nfev):
        ready = (len(self.errs) == self.errs.maxlen and self.n_meas >= self.min_meas
                 and self._has_frame())
        if not ready:
            return
        med = float(np.median(self.errs))
        med_iso = float(np.median(self.iso_errs)) if len(self.iso_errs) else 0.0
        if self.gate_mode == 'skill':
            want_open = med <= self.skill_open * med_iso
            want_close = med > self.skill_close * med_iso
        else:
            want_open = med < self.gate_error and med <= self.gate_skill_ratio * med_iso
            want_close = (med > self.gate_close_factor * self.gate_error
                          or med > 0.8 * med_iso)
        if not self.gate_open and want_open:
            self.gate_open = True
            self.counts['gate_open_events'] += 1
            if self.first_open_fe < 0:
                self.first_open_fe = int(nfev)
            if self.hdir is None or self.hdir.size != self.Qprobe_line.shape[1]:
                self.hdir = self._steps_from_coordinates(self.Qprobe_line)
            self._perm = []
        elif self.gate_open and want_close:
            self.gate_open = False
            self.counts['gate_close_events'] += 1

    # --------------------------------------------------------------- probe lines
    def measurement_floor(self, u, fx):
        """Smallest symmetric step whose curvature survives roundoff."""
        pred = self.predict(u)
        if pred is None or not np.isfinite(pred) or abs(pred) < 1e-300:
            return self.h_lo
        delta = 16.0 * _EPS * max(abs(float(fx)), 1e-300)
        return 1.5 * math.sqrt(4.0 * delta / (self.tol * abs(pred)))

    def sensing_direction(self):
        """Isotropic or curvature-whitened sensing direction (unit)."""
        g = self.rng.standard_normal(self.d)
        if self.dense:
            if self.eigvec is not None and self.rng.random() < self.whitened:
                lam = np.abs(self.eigval)
                lam = np.maximum(lam, max(float(lam.max()), 1e-300) * 1e-12)
                g = self.eigvec @ ((self.eigvec.T @ g) / np.sqrt(lam))
        elif self.Q.shape[1]:
            g = g - self.Q @ (self.Q.T @ g)
        nrm = float(np.linalg.norm(g))
        return None if nrm < 1e-12 else g / nrm

    def _sensing_probe_line(self, fx):
        u = self.sensing_direction()
        if u is None:
            return None
        if self.gate_open and self.hdir is not None and self.hdir.size:
            base = float(np.median(self.hdir))
        elif self._hc is not None:
            base = float(np.median(np.asarray(self._hc, dtype=float)))
        else:
            base = 0.045
        h = float(np.clip(max(base, self.measurement_floor(u, fx)), self.h_lo, self.h_hi))
        self.counts['sense_probe_lines'] += 1
        return ProbeLine(u, h, 'sense', -1)

    def _frame_probe_line(self, fx):
        k = self.Qprobe_line.shape[1]
        if k == 0:
            return self._sensing_probe_line(fx)
        if self.hdir is None or self.hdir.size != k:
            self.hdir = self._steps_from_coordinates(self.Qprobe_line)
        if not self._perm:
            self._perm = list(self.rng.permutation(k))
        i = int(self._perm.pop())
        self.counts['frame_probe_lines'] += 1
        return ProbeLine(self.Qprobe_line[:, i].copy(), float(self.hdir[i]), 'frame', i)

    def next_probe_line(self, x, fx, hcoord, nfev=-1):
        """ProbeLine for the next stage-2 line search, or None for the base search's own probe line."""
        self._hc = hcoord
        self._probe_line_counter += 1
        period = self.sense_open if self.gate_open else self.sense_closed
        if self._probe_line_counter % period == 0:
            return self._sensing_probe_line(fx)
        if not self.gate_open:
            return None
        return self._frame_probe_line(fx)

    def after_probe_line(self, probe_line, improved, displacement):
        """Step-length update after the base search has evaluated the probe line."""
        if probe_line is None:
            return
        if probe_line.kind == 'frame' and self.hdir is not None and 0 <= probe_line.index < self.hdir.size:
            i = probe_line.index
            h = float(self.hdir[i])
            if improved:
                self.hdir[i] = min(self.h_hi, max(1.08 * h, 0.5 * abs(float(displacement))))
                self.counts['frame_success'] += 1
                self._stall = 0
            else:
                self.hdir[i] = max(self.h_lo, 0.58 * h)
                self._stall += 1
            if self._stall > 2 * self.d:
                self.hdir = np.maximum(self.hdir, self.stall_reset_step)
                self._stall = 0
        elif probe_line.kind == 'sense' and improved:
            self.counts['sense_success'] += 1

    # ------------------------------------------------------------ telemetry
    def summary(self):
        def med(values):
            return float(np.median(values)) if len(values) else float('nan')
        out = {
            'geo_mode': self.mode,
            'geo_gate_mode': self.gate_mode,
            'geo_dense': int(self.dense),
            'geo_rank': int(self.d if self.dense else self.Q.shape[1]),
            'geo_measurements': int(self.n_meas),
            'geo_first_open_fe': int(self.first_open_fe),
            'geo_gate_open_final': int(self.gate_open),
            'geo_err_median_all': med(self.errs_all),
            'geo_err_median_last': med(list(self.errs)),
            'geo_iso_err_median_all': med(self.iso_all),
            'geo_shadow_err_median': med(self.shadow_all),
            'geo_frame_overlap_median': med(self.frame_overlap),
            'geo_fit_seconds': float(self.fit_seconds),
            'geo_condition_est': float('nan'),
            'geo_negative_eigs': -1,
        }
        for key, value in self.counts.items():
            out['geo_' + key] = int(value)
        if self.dense and self.eigval is not None:
            lam = np.abs(self.eigval)
            pos = lam[lam > 0]
            out['geo_condition_est'] = float(pos.max() / pos.min()) if pos.size else float('nan')
            out['geo_negative_eigs'] = int(np.sum(self.eigval <= 0))
        elif not self.dense and self.sigma.size and self.m1:
            ratios = np.abs(self.sigma) / max(abs(self.m1), 1e-300)
            out['geo_condition_est'] = float(max(ratios.max(), 1.0) / max(min(ratios.min(), 1.0), 1e-300))
            out['geo_negative_eigs'] = int(np.sum(self.sigma <= 0))
        return out
