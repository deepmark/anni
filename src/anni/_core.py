"""ANNI's computation, in PyTorch.

Steps: presentation level (one gain for both signals) -> alignment (constant delay) -> outer/middle-ear filter ->
auditory-nerve model (polarity-averaged, delay-compensated) -> 1.9 ms bins (plus the power of fast fluctuations,
256-1024 Hz, measured on 0.27 ms sub-bins) -> pattern and texture observers -> 200 ms windows -> D_peak, D_med.
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
# long signals are processed in blocks, so memory does not grow with their length (the scores do not change)
NERVE_BLOCK_S = 2.0            # nerve model: seconds per block
OBSERVER_BLOCK_S = 10.0        # observers: seconds of 200 ms window positions per block
OBSERVER_CONTEXT_S = 2.0       # observers: history before each block, for their slow filters to settle
# calibration constants: internal-noise floor r0 (spikes/s), pattern gain kP, texture gain kS, texture floor e0,
# fast-fluctuation gain kF
CONSTANTS = dict(r0=60.250680087717825, kP=6.882786932864671, kS=0.04863147509425055, e0=2.6535338234460336e-05,
                 kF=0.03465053496767872)
# fast fluctuations: octave bands above the texture observer's, on sub-bins of 3 nerve samples (0.27 ms), weighted by
# a first-order low-pass at 150 Hz, the limit on following fast amplitude changes that Kohlrausch, Fassel & Dau (2000)
# place after the cochlea
FAST_BANDS = (256., 512., 1024.)
FAST_CUTOFF = 150.0            # Hz
FINE_N = 3                     # nerve samples per sub-bin
FAST_HISTORY_S = 0.05          # history the fast band filters need (their impulse responses are below 1e-9 after it)


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
        self._spectra = None                                                  # filter spectra of the last length

    def forward(self, x):
        """x [B, T] (Pa) -> firing rates [B, 128, ceil(T / 4)] (spikes/s)."""
        T = x.shape[-1]
        n1 = next_fast_len(T + self.g.shape[-1] - 1)
        n2 = next_fast_len(T + self.lp.shape[-1] - 1)
        if self._spectra is None or self._spectra[0] != (n1, n2):      # all full blocks share one length
            self._spectra = ((n1, n2), torch.fft.rfft(self.g, n=n1), torch.fft.rfft(self.lp, n=n2))
        _, G, LP = self._spectra
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
        on the CPU for the observers. Whole signal at once; ANNI itself uses binned_rates."""
        x = ear_filter(p.to(device=self.device, dtype=DT))
        r = self.forward(torch.cat([x, -x]))
        r = 0.5 * (r[:len(x)] + r[len(x):])
        return self.compensate(r).cpu()

    def binned_rates(self, p, bin_n, block_s=NERVE_BLOCK_S, fast=None):
        """p [T] pressure (Pa) at 44.1 kHz -> ear filter -> polarity-averaged, delay-compensated rates averaged in
        bins of bin_n output samples, [128, nb], on the CPU. The same as binning rates(p), but computed in blocks of
        about block_s seconds, so memory does not grow with the length of p. This is exact: an output sample depends
        only on the last len(g) + len(lp) input samples (two FIR filters; the rest is sample by sample), so each block
        gets that much input before it, and the largest fibre delay after it.

        With fast (an Observer), also returns the power of the fast fluctuation bands per bin, [K, F, nb], for the F
        fibres that use them (Observer.fast_power), computed in the same blocks: each block's sub-bins get the fast
        filters' history before them, so the result equals Observer.sub_bin_fast on the whole signal."""
        x = ear_filter(p.to(device=self.device, dtype=DT))
        T = x.shape[-1]
        st = self.stride
        N = -(-T // st)                                                # output samples of forward() for all of x
        nb = N // bin_n
        left = -(-(self.g.shape[-1] + self.lp.shape[-1] - 2) // st) * st     # input history, a multiple of the stride
        right = st * (int(self.delays.max()) + 1)                     # input after the block, for the delay shift
        per_block = max(1, int(round(block_s * FS_N / bin_n)))       # bins per block
        delays = self.delays.to(self.device)
        hist = fast.fast_hist * fast.fine_n if fast is not None else 0     # output samples of fast-filter history
        out, out_fast = [], []
        for k0 in range(0, nb, per_block):
            k1 = min(nb, k0 + per_block)
            o0, o1 = k0 * bin_n, k1 * bin_n                            # output samples of this block
            oh = max(0, o0 - hist)                                     # ... and of the history before it
            a, b = max(0, st * oh - left), min(T, st * o1 + right)     # input samples it needs
            seg = x[a:b]
            r = self.forward(torch.stack([seg, -seg]))
            r = 0.5 * (r[0] + r[1])                                    # [128, ceil((b - a) / stride)]
            idx = torch.arange(oh, o1, device=self.device)[None, :] + delays[:, None] - a // st
            inside = idx < r.shape[-1]                                 # past the end of the signal: zero, as in compensate
            rc = torch.gather(r, 1, idx.clamp(max=r.shape[-1] - 1)) * inside
            out.append(rc[:, o0 - oh:].reshape(rc.shape[0], k1 - k0, bin_n).mean(-1).cpu())
            if fast is not None:
                rf = rc[fast.fast_fibres]
                fine = rf.reshape(rf.shape[0], -1, fast.fine_n).mean(-1)
                skip = (o0 - oh) // fast.fine_n
                if skip < fast.fast_hist:                              # the signal's start: pad with the first sub-bin
                    fine = torch.cat([fine[:, :1].expand(-1, fast.fast_hist - skip), fine], -1)
                    skip = fast.fast_hist
                out_fast.append(fast.fast_power(fine, skip).cpu())
        binned = torch.cat(out, -1) if out else torch.zeros(len(self.cfs), 0, dtype=DT)
        if fast is None:
            return binned
        fp = (torch.cat(out_fast, -1) if out_fast
              else torch.zeros(len(FAST_BANDS), len(self.cfs[fast.fast_fibres]), 0, dtype=DT))
        return binned, fp


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
    (changes in rate fluctuation, octave bands 4-128 Hz, and fast bands 256-1024 Hz from sub-bins), evidence in d'^2
    units pooled over fibres and 200 ms."""

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
        # fast fluctuations: band-passed on sub-bins of FINE_N nerve samples; their power is averaged back into the bins
        if bin_n % FINE_N:
            raise ValueError(f"bin_n must be a multiple of {FINE_N}")
        self.fine_n, self.ratio = FINE_N, bin_n // FINE_N
        fsf = FS_N / FINE_N
        self.fast_hist = int(round(FAST_HISTORY_S * fsf))            # sub-bins of history the band filters need
        fast_secs = [butter2("bandpass", (fc / math.sqrt(2), fc * math.sqrt(2)), fsf) for fc in FAST_BANDS]
        self.fast_h = torch.tensor([_impulse(tuple(b) + tuple(a), self.fast_hist) for b, a in fast_secs], dtype=DT)
        ff = torch.tensor(FAST_BANDS, dtype=DT)
        f0 = int(torch.nonzero(cfs >= 4.0 * FAST_BANDS[0])[0])       # fibres below this use no fast band (CF / 4)
        self.fast_fibres = slice(f0, None)
        self.fast_mask = ((ff[:, None] <= cfs[None, f0:] / 4.0).to(DT)                 # the CF / 4 rule, as above
                          / (1.0 + (ff[:, None] / FAST_CUTOFF) ** 2))                  # and the 150 Hz low-pass

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

    def fast_power(self, fine, skip):
        """Sub-binned rates [..., F, n] of the fibres that use the fast bands (fast_fibres), the first `skip` sub-bins
        being history, -> power of each fast band (causal band-pass, then squared), averaged into the bins that follow
        the history: [K, ..., F, (n - skip) // ratio]."""
        n = fine.shape[-1]
        L = next_fast_len(n + self.fast_hist)
        h = torch.fft.rfft(self.fast_h.to(fine.device, fine.dtype), L)
        h = h.reshape(h.shape[0], *[1] * (fine.dim() - 1), h.shape[-1])
        band = torch.fft.irfft(torch.fft.rfft(fine, L)[None] * h, L)[..., skip:n]
        m = band.shape[-1] // self.ratio
        return (band[..., :m * self.ratio] ** 2).reshape(*band.shape[:-1], m, self.ratio).mean(-1)

    def sub_bin_fast(self, r):
        """Fast band power [K, ..., F, nb] for whole-signal rates r [..., 128, N] (as bin(r) gives the bins):
        sub-bins of the fibres that use the fast bands, start-padded with the first one for the filters' history."""
        nb = r.shape[-1] // self.bin_n
        r = r[..., self.fast_fibres, :nb * self.bin_n]
        fine = r.reshape(*r.shape[:-1], nb * self.ratio, self.fine_n).mean(-1)
        fine = torch.cat([fine[..., :1].expand(*fine.shape[:-1], self.fast_hist), fine], -1)
        return self.fast_power(fine, self.fast_hist)

    def movmean(self, a):
        W = self.W
        c = torch.cumsum(torch.nn.functional.pad(a, (1, 0)), -1)
        return (c[..., W:] - c[..., :-W]) / W

    def valid(self, npos, nb):
        """Windows clear of the trimmed start and end of the signal."""
        t0b, t1b = math.ceil(self.trim_s[0] / self.dt), math.ceil(self.trim_s[1] / self.dt)
        pos = torch.arange(npos)
        return (pos >= t0b) & (pos + self.W <= nb - t1b)

    def window_d2(self, Rx, Ry, p, Fx=None, Fy=None):
        """Total evidence (d'^2) in each 200 ms window: pattern + texture. Rx, Ry: [..., 128, nb] (a batch works
        too); Fx, Fy: their fast band power [K, ..., F, nb] (fast_power), or None to leave the fast bands out;
        returns [..., nb - W + 1]."""
        rbar = 0.5 * (Rx + Ry)
        db, xb, yb = self.split(Ry - Rx), self.split(Rx), self.split(Ry)
        d2P = p["kP"] * self.dz * self.dt * db[0] ** 2 / (rbar + p["r0"])
        P_w = self.movmean(d2P.sum(-2)) * self.W
        mx, my = self.movmean(Rx), self.movmean(Ry)
        ex = self.movmean(xb[1:] ** 2) / (mx + p["r0"]) ** 2
        ey = self.movmean(yb[1:] ** 2) / (my + p["r0"]) ** 2
        mask = self.mask.to(ex.device, ex.dtype).reshape(self.mask.shape[0], *[1] * (ex.dim() - 3), -1, 1)
        lr2 = p["kS"] * self.dz * torch.log((ey + p["e0"]) / (ex + p["e0"])) ** 2 * mask
        tot = P_w + lr2.sum((0, -2))
        if Fx is not None:                                            # fast bands: the same comparison, own gain
            fx = self.movmean(Fx) / (mx[..., self.fast_fibres, :] + p["r0"]) ** 2
            fy = self.movmean(Fy) / (my[..., self.fast_fibres, :] + p["r0"]) ** 2
            fm = self.fast_mask.to(fx.device, fx.dtype).reshape(self.fast_mask.shape[0], *[1] * (fx.dim() - 3), -1, 1)
            tot = tot + (p["kF"] * self.dz * torch.log((fy + p["e0"]) / (fx + p["e0"])) ** 2 * fm).sum((0, -2))
        return tot


def window_d2_blocks(ob, bx, by, p=CONSTANTS, fx=None, fy=None, block_s=OBSERVER_BLOCK_S, context_s=OBSERVER_CONTEXT_S):
    """Observer.window_d2 over binned rates [128, nb] (and fast band power [K, 128, nb], if given), computed in blocks
    of about block_s seconds of window positions, so memory does not grow with the length. Each block starts context_s
    seconds early, so that the observers' slow filters have settled: what is left of the history before that is below
    1e-7 of it (the slowest filter, the 4 Hz band, decays about e^-9 per second). The first block starts at the
    signal's start, as before."""
    nb = bx.shape[-1]
    npos = nb - ob.W + 1
    per_block = max(1, int(round(block_s / ob.dt)))
    ctx = int(round(context_s / ob.dt))
    out = []
    for q0 in range(0, max(npos, 0), per_block):
        q1 = min(npos, q0 + per_block)
        s0 = q0 - ctx if q0 > ctx else 0
        e = q1 + ob.W - 1
        f = (fx[..., s0:e], fy[..., s0:e]) if fx is not None else (None, None)
        out.append(ob.window_d2(bx[:, s0:e], by[:, s0:e], p, *f)[q0 - s0:])
    return torch.cat(out) if out else torch.zeros(0, dtype=DT)


def scores(ob, bx, by, p=CONSTANTS, fx=None, fy=None, **blocks):
    """D_peak (most noticeable 200 ms window) and D_med (median window) as tensors, from binned rates (and fast band
    power). blocks are passed to window_d2_blocks (one huge block, e.g. block_s=1e9, computes all windows at once)."""
    tot = window_d2_blocks(ob, bx, by, p, fx, fy, **blocks)
    ok = ob.valid(len(tot), bx.shape[-1])
    d2 = tot[ok]
    return dict(D_peak=torch.sqrt(d2.max()), D_med=torch.median(torch.sqrt(torch.clamp(d2, min=0))))
