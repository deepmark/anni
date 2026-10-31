"""Adapted from https://github.com/msaddler/auditory_machine_learning, under the following licence:

MIT License

Copyright (c) 2026 Mark R. Saddler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math

import numpy as np
import soxr
import torch


def freq2erb(freq):
    """Frequency in Hz to ERB-number."""
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """ERB-number to frequency in Hz."""
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(start, stop, num):
    """num frequencies (Hz) evenly spaced on the ERB-number scale."""
    return erb2freq(np.linspace(freq2erb(start), freq2erb(stop), num=num))


def gammatone_filterbank_fir(sr, cfs, fir_dur=0.05, order=4, bw_mult=1.0):
    """Impulse responses of a gammatone filterbank, shape (n_filters, int(sr * fir_dur))."""
    cfs = np.array(cfs).reshape([-1])
    bw = 2 * np.pi * bw_mult * (24.7 + cfs / 9.265)
    wc = 2 * np.pi * cfs
    t = np.arange(0, fir_dur, 1 / sr)
    a = 2 / math.factorial(order - 1) / np.abs(1 / bw ** order + 1 / (bw + 2j * wc) ** order) / sr
    return a[:, None] * t ** (order - 1) * np.exp(-bw[:, None] * t[None, :]) * np.cos(wc[:, None] * t[None, :])


def ihc_lowpass_filter_fir(sr, fir_dur, cutoff=3e3, order=7):
    """Impulse response of the inner-hair-cell low-pass filter (from bez2018model/model_IHC_BEZ2018.c)."""
    sr_bez2018 = 100e3

    def resample(x):
        return soxr.resample(x, in_rate=sr_bez2018, out_rate=sr, quality="VHQ")

    n_taps = int(sr_bez2018 * fir_dur)
    while len(resample(np.zeros(n_taps))) % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr_bez2018 - 2 * np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * np.hanning(n_taps)
    fir = resample(fir)
    fir[fir < 0] = 0
    return fir / fir.sum()


class GradientStableSigmoid(torch.autograd.Function):
    """Sigmoid with a NaN-free gradient (avoids overflow in the rate-level function)."""

    @staticmethod
    def forward(ctx, x, k, x0):
        ctx.save_for_backward(x, k, x0)
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    @staticmethod
    def backward(ctx, grad_output):
        x, k, x0 = ctx.saved_tensors
        grad = k * torch.exp(-k * (x - x0))
        grad = grad / (torch.exp(-k * (x - x0)) + 1.0) ** 2
        grad = torch.nan_to_num(grad, nan=0.0, posinf=None, neginf=None)
        return grad_output * grad, None, None


def rate_level(x, rate_spont=0.0, rate_max=250.0, threshold=0.0, dynamic_range=80.0, dynamic_range_interval=0.95):
    """Sigmoid rate-level function: half-wave rectified, low-passed subbands (Pa) -> firing rate (spikes/s)."""
    y_threshold = (1 - dynamic_range_interval) / 2
    k = math.log((1 / y_threshold) - 1) / (dynamic_range / 2)
    x0 = threshold - (math.log((1 / y_threshold) - 1) / (-k))
    k_t = torch.tensor(k, dtype=x.dtype, device=x.device)
    x0_t = torch.tensor(x0, dtype=x.dtype, device=x.device)
    s = GradientStableSigmoid.apply(20 * torch.log10(x / 20e-6), k_t, x0_t)
    return rate_spont + (rate_max - rate_spont) * s
