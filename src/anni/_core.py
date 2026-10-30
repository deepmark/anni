"""ANNI's computation, in PyTorch.

Steps: presentation level (one gain for both signals) -> alignment (constant delay) -> outer/middle-ear filter ->
auditory-nerve model (polarity-averaged, delay-compensated) -> 1.9 ms bins -> pattern and texture observers ->
200 ms windows -> D_peak, D_med.
"""
import cmath
import math
from functools import lru_cache

import torch

from . import _auditory_model

SR = 44100                     # the model's input rate
FS_N = 11025                   # nerve output rate (stride 4)
P_REF = 20e-6                  # Pa at 0 dB SPL
DT = torch.float32
# calibration constants: internal-noise floor r0 (spikes/s), pattern gain kP, texture gain kS, texture floor e0
CONSTANTS = dict(r0=66.56840796512564, kP=7.440094426024921, kS=0.05248517778798395, e0=2.3703160271327422e-05)


def next_fast_len(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


# ------------------------------------------------------------------------------------------------ presentation

def tq(f):
    """Terhardt threshold in quiet (dB SPL)."""
    f = torch.clamp(torch.as_tensor(f, dtype=DT), min=1.0) / 1000.0
    return 3.64 * f ** -0.8 - 6.5 * torch.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4


def ear_filter(x, pad=8192):
    """Zero-phase outer/middle-ear filter: gain clip(-(Tq(f) - Tq(1 kHz)), -100, 20) dB, H(0) = 0; x [..., T]."""
    T = x.shape[-1]
    n = next_fast_len(T + 2 * pad)
    xp = torch.nn.functional.pad(x, (pad, n - T - pad))
    f = torch.fft.rfftfreq(n, 1.0 / SR, dtype=DT)
    H = 10 ** (torch.clamp(-(tq(f) - tq(1000.0)), -100.0, 20.0) / 20.0)
    H[0] = 0.0
    return torch.fft.irfft(torch.fft.rfft(xp, dim=-1) * H.to(device=x.device, dtype=x.dtype), n, dim=-1)[..., pad:pad + T]


def active_rms(x, frame=0.02, rel_db=40.0):
    """RMS over 20 ms frames whose power is within rel_db of the 99th-percentile frame power (mean removed)."""
    x = x - x.mean()
    L = int(SR * frame)
    n = x.shape[-1] // L
    e = (x[:n * L].reshape(n, L) ** 2).mean(1)
    thr = torch.quantile(e, 0.99) * 10 ** (-rel_db / 10)
    return torch.sqrt(e[e > thr].mean())


def pa_gain(x, level=65.0):
    """Pa per digital unit that puts x's active level at `level` dB SPL."""
    return P_REF * 10 ** (level / 20) / active_rms(x)


def shift(y, tau, pad=4096):
    """y(t + tau) by an FFT phase ramp (zero-padded)."""
    T = y.shape[-1]
    N = next_fast_len(T + 2 * pad)
    yp = torch.nn.functional.pad(y, (pad, N - T - pad))
    f = torch.fft.rfftfreq(N, dtype=DT, device=y.device)
    ramp = torch.exp(2j * math.pi * f * torch.as_tensor(tau, dtype=DT))
    return torch.fft.irfft(torch.fft.rfft(yp) * ramp, N)[..., pad:pad + T]


def align(x, y, max_lag=441, min_shift=0.05):
    """Removes a constant delay of up to max_lag samples: integer lag by FFT cross-correlation (largest |c|),
    parabolic sub-sample refinement, phase-ramp shift of y. x and y have equal length. Returns (y, tau, rho), tau
    being the shift applied in samples (0.0 when the delay found is below min_shift and y is left as it is)."""
    n = next_fast_len(2 * x.shape[-1])
    with torch.no_grad():
        c = torch.fft.irfft(torch.conj(torch.fft.rfft(x, n)) * torch.fft.rfft(y, n), n)
        lags = torch.cat([torch.arange(0, max_lag + 1), torch.arange(-max_lag, 0)])
        cv = c[lags % n]
        i = int(torch.argmax(cv.abs()))
        l0 = int(lags[i])
        sgn = 1.0 if float(cv[i]) >= 0 else -1.0
        cm, c0, cp = sgn * float(c[(l0 - 1) % n]), sgn * float(cv[i]), sgn * float(c[(l0 + 1) % n])
        den = cm - 2 * c0 + cp
        tau = l0 + (0.5 * (cm - cp) / den if den != 0 else 0.0)
        rho = float(cv[i]) / math.sqrt(float((x ** 2).sum() * (y ** 2).sum()) + 1e-30)
    if abs(tau) < min_shift:
        return y, 0.0, rho
    return shift(y, tau), float(tau), rho


# ------------------------------------------------------------------------------------------------ auditory nerve

def analytic(g):
    """Analytic signal along the last axis."""
    N = g.shape[-1]
    h = torch.zeros(N, dtype=DT)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return torch.fft.ifft(torch.fft.fft(g, dim=-1) * h, dim=-1)


class Nerve:
    """Saddler's simple auditory-nerve model in a full-band configuration: 44.1 kHz in, 128 ERB-spaced fibres from
    80 Hz to 16 kHz, 4th-order gammatone filters (1 ERB), IHC low-pass (nominal 3 kHz, order 7), sigmoid rate-level
    function (0 dB SPL threshold, 80 dB range, max 250 spikes/s), 11025 Hz output. FIR filtering by FFT."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        cfs = _auditory_model.erbspace(80.0, 16000.0, 128)
        g = torch.as_tensor(_auditory_model.gammatone_filterbank_fir(SR, cfs, fir_dur=0.05, bw_mult=1.0), dtype=DT)
        lp = torch.as_tensor(_auditory_model.ihc_lowpass_filter_fir(SR, 0.05, cutoff=3000.0, order=7), dtype=DT)
        self.cfs = torch.as_tensor(cfs, dtype=DT)
        self.g, self.lp = g.to(device), lp.to(device)
        self.stride = SR // FS_N
        tau_g = torch.argmax(analytic(g).abs(), dim=-1).to(DT) / SR          # gammatone envelope peak
        tau_l = (torch.arange(len(lp), dtype=DT) * lp).sum() / lp.sum() / SR  # low-pass centroid
        self.delays = torch.round((tau_g + tau_l) * FS_N).long()              # per-fibre delay, output samples

    def forward(self, x):
        """x [B, T] (Pa) -> firing rates [B, 128, ceil(T / 4)] (spikes/s)."""
        T = x.shape[-1]
        n1 = next_fast_len(T + self.g.shape[-1] - 1)
        n2 = next_fast_len(T + self.lp.shape[-1] - 1)
        G = torch.fft.rfft(self.g, n=n1)
        LP = torch.fft.rfft(self.lp, n=n2)
        out = []
        for xb in x:                                       # one row at a time keeps the memory down
            y = torch.fft.irfft(torch.fft.rfft(xb, n=n1)[None, :] * G, n=n1)[:, :T]
            y = torch.relu(y)
            y = torch.fft.irfft(torch.fft.rfft(y, n=n2) * LP, n=n2)[:, :T:self.stride]
            out.append(torch.relu(y))
        return _auditory_model.rate_level(torch.stack(out))

    def compensate(self, r):
        """Removes each fibre's own delay, so all fibres line up in time."""
        cols = []
        for c in range(r.shape[-2]):
            s = int(self.delays[c])
            cols.append(torch.nn.functional.pad(r[..., c, s:], (0, s)))
        return torch.stack(cols, dim=-2)

    def rates(self, p):
        """p [B, T] pressure (Pa) at 44.1 kHz -> ear filter -> polarity-averaged, delay-compensated rates, returned
        on the CPU for the observers."""
        x = ear_filter(p.to(device=self.device, dtype=DT))
        r = self.forward(torch.cat([x, -x]))
        r = 0.5 * (r[:len(x)] + r[len(x):])
        return self.compensate(r).cpu()


# ------------------------------------------------------------------------------------------------ observers

def _bilinear(zs, ps, k, fs):
    fs2 = 2.0 * fs
    zd = [(fs2 + z) / (fs2 - z) for z in zs]
    pd = [(fs2 + p) / (fs2 - p) for p in ps]
    zd += [-1.0] * (len(ps) - len(zs))
    num = 1.0
    for z in zs:
        num *= (fs2 - z)
    den = 1.0
    for p in ps:
        den *= (fs2 - p)
    return zd, pd, k * (num / den).real


def _poly2(roots):
    r1, r2 = roots
    return [1.0, (-(r1 + r2)).real, (r1 * r2).real]


def butter2(kind, wn, fs):
    """One second-order Butterworth section (b, a), as scipy.signal.butter designs it: 'lowpass' (order 2, wn Hz) or
    'bandpass' (order 1, wn = (lo, hi) Hz)."""
    fsn = 2.0
    warp = lambda w: 2 * fsn * math.tan(math.pi * (2 * w / fs) / fsn)        # noqa: E731
    if kind == "lowpass":
        p_proto = [complex(-math.sqrt(0.5), math.sqrt(0.5)), complex(-math.sqrt(0.5), -math.sqrt(0.5))]
        wo = warp(wn)
        zs, ps, k = [], [wo * p for p in p_proto], wo ** 2
    elif kind == "bandpass":
        lo, hi = warp(wn[0]), warp(wn[1])
        bw, wo = hi - lo, math.sqrt(lo * hi)
        disc = cmath.sqrt((-bw / 2) ** 2 - wo ** 2)
        zs, ps, k = [0.0], [-bw / 2 + disc, -bw / 2 - disc], bw
    else:
        raise ValueError(kind)
    zd, pd, kd = _bilinear(zs, ps, k, fsn)
    return [kd * c for c in _poly2(zd)], _poly2(pd)


@lru_cache(maxsize=64)
def _impulse(sec, n):
    """Impulse response (n samples) of one biquad (b0, b1, b2, a0, a1, a2), from rest."""
    b0, b1, b2, _, a1, a2 = sec
    h = [0.0] * n
    y1 = y2 = 0.0
    for i in range(n):
        y0 = (b0 if i == 0 else b1 if i == 1 else b2 if i == 2 else 0.0) - a1 * y1 - a2 * y2
        h[i] = y0
        y2, y1 = y1, y0
    return tuple(h)


class Observer:
    """Compares two neurograms: the pattern observer (rate changes slower than ~3 Hz) and the texture observer
    (changes in rate fluctuation, octave bands 4-128 Hz), evidence in d'^2 units pooled over fibres and 200 ms."""

    def __init__(self, cfs, bin_n=21, bands=(4., 8., 16., 32., 64., 128.), win_s=0.1995, pad_s=1.0,
                 trim_s=(0.1, 0.05)):
        self.bin_n, self.dt = bin_n, bin_n / FS_N
        fsb = 1.0 / self.dt
        secs = [butter2("lowpass", 2.0 * math.sqrt(2.0), fsb)]
        secs += [butter2("bandpass", (fc / math.sqrt(2), fc * math.sqrt(2)), fsb) for fc in bands]
        self.secs = [tuple(b) + tuple(a) for b, a in secs]
        fk = torch.tensor(bands, dtype=DT)
        self.W = int(round(win_s / self.dt))
        self.padc = int(round(pad_s / self.dt))
        cfs = torch.as_tensor(cfs, dtype=DT)
        self.dz = float(torch.diff(torch.as_tensor(_auditory_model.freq2erb(cfs.numpy()))).mean())  # fibre spacing
        self.mask = (fk[:, None] <= cfs[None, :] / 4.0).to(DT)      # a band counts only up to CF / 4
        self.trim_s = trim_s

    def bin(self, r):
        nb = r.shape[-1] // self.bin_n
        return r[..., :nb * self.bin_n].reshape(*r.shape[:-1], nb, self.bin_n).mean(-1)

    def split(self, a):
        """Causal filter bank along the bin axis (start-padded with the first bin): [1 + K, ..., nb]."""
        ap = torch.cat([a[..., :1].expand(*a.shape[:-1], self.padc), a], -1)
        L = ap.shape[-1]
        n = next_fast_len(2 * L)
        A = torch.fft.rfft(ap, n)
        out = []
        for sec in self.secs:
            h = torch.tensor(_impulse(sec, L), dtype=a.dtype, device=a.device)
            out.append(torch.fft.irfft(A * torch.fft.rfft(h, n), n)[..., :L][..., self.padc:])
        return torch.stack(out)

    def movmean(self, a):
        W = self.W
        c = torch.cumsum(torch.nn.functional.pad(a, (1, 0)), -1)
        return (c[..., W:] - c[..., :-W]) / W

    def valid(self, npos, nb):
        """Windows clear of the trimmed start and end of the signal."""
        t0b, t1b = math.ceil(self.trim_s[0] / self.dt), math.ceil(self.trim_s[1] / self.dt)
        pos = torch.arange(npos)
        return (pos >= t0b) & (pos + self.W <= nb - t1b)

    def window_d2(self, Rx, Ry, p):
        """Total evidence (d'^2) in each 200 ms window: pattern + texture."""
        rbar = 0.5 * (Rx + Ry)
        db, xb, yb = self.split(Ry - Rx), self.split(Rx), self.split(Ry)
        d2P = p["kP"] * self.dz * self.dt * db[0] ** 2 / (rbar + p["r0"])
        P_w = self.movmean(d2P.sum(0)) * self.W
        mx, my = self.movmean(Rx), self.movmean(Ry)
        ex = self.movmean(xb[1:] ** 2) / (mx + p["r0"]) ** 2
        ey = self.movmean(yb[1:] ** 2) / (my + p["r0"]) ** 2
        lr2 = p["kS"] * self.dz * torch.log((ey + p["e0"]) / (ex + p["e0"])) ** 2 * self.mask[:, :, None]
        return P_w + lr2.sum((0, 1))


def scores(ob, Rx, Ry, p=CONSTANTS):
    """D_peak (most noticeable 200 ms window) and D_med (median window) as tensors."""
    bx, by = ob.bin(Rx), ob.bin(Ry)
    tot = ob.window_d2(bx, by, p)
    ok = ob.valid(len(tot), bx.shape[-1])
    d2 = tot[ok]
    return dict(D_peak=torch.sqrt(d2.max()), D_med=torch.median(torch.sqrt(torch.clamp(d2, min=0))))
