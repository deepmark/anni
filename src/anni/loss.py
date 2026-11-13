"""ANNI as a training loss (experimental)."""
import math

import torch

from . import _core


class ANNILoss(torch.nn.Module):
    """Differentiable ANNI, for training: zero while the test signal is at most `target` noticeable, and growing
    quadratically above it.

    It runs the same model as ANNI (playback level, ear filter, nerve model, pattern and texture observers,
    200 ms windows), with two differences that gradients need:

    - No alignment: the two signals must already line up sample by sample (as a watermarked signal and its
      original do). No resampling either: the signals must be at 44.1 kHz. As in ANNI, each signal's DC offset is
      removed and a silent reference is rejected (its playback level cannot be set).
    - The peak and median over the windows are replaced by one smooth summary, the p-mean of the window scores
      (mean of d'^p, to the power 1/p). p=1 is the plain mean (the typical moment), larger p moves it towards the
      most noticeable moment, and p=inf is the peak itself. With many windows a finite p stays below the peak, so
      the target refers to this summary, not to D_peak.

    Args:
        target: the noticeability (in d') that is still allowed; the loss is zero at or below it. d' = 1 is a
            just-noticeable change (about 3 in 4 correct picks), 0.5 about 64 %.
        p: the p-mean's exponent, 1 or more (float("inf") for the peak).
        level: playback level of the reference in dB SPL, as in ANNI. A higher level is a safety margin for louder
            listening: parts of the change that are below the threshold of hearing at 65 dB become audible louder.
        polarities: 2 (default) runs each signal as it is and upside down and averages, exactly as ANNI does; 1
            runs it as it is, with about half the memory.
        device: "cpu" or "cuda" (.to() and .cuda() work too). The signals must be on the same device.

    Example:
        >>> loss_fn = ANNILoss(target=0.5, p=6)
        >>> loss = loss_fn(reference, watermarked)    # [batch, time] at 44.1 kHz
        >>> loss.backward()
    """

    def __init__(self, target=0.5, p=6.0, level=65.0, polarities=2, device="cpu"):
        super().__init__()
        if not target >= 0:
            raise ValueError(f"target must be 0 or more; got {target}")
        if not p >= 1:
            raise ValueError(f"p must be 1 or more; got {p}")
        if polarities not in (1, 2):
            raise ValueError(f"polarities must be 1 or 2; got {polarities}")
        self.target, self.p, self.level, self.polarities = float(target), float(p), float(level), polarities
        self._nerve = _core.Nerve(device=device)
        self._observer = _core.Observer(self._nerve.cfs)

    def _apply(self, fn, *args, **kwargs):
        """Lets .to(), .cuda() and .cpu() move the nerve model to another device (its tensors are not registered
        buffers). Only the device follows: the model stays in 32-bit even after .half() or a bfloat16 conversion."""
        super()._apply(fn, *args, **kwargs)
        nv, ob = self._nerve, self._observer
        dev = fn(nv.g).device
        nv.g, nv.lp, ob.mask = nv.g.to(dev), nv.lp.to(dev), ob.mask.to(dev)
        nv.device, nv._spectra = dev, None
        return self

    def forward(self, reference, test, sr=_core.SR):
        """Mean over the batch of relu(score - target)^2, score being soft_score(reference, test)."""
        return (torch.relu(self.soft_score(reference, test, sr) - self.target) ** 2).mean()

    def soft_score(self, reference, test, sr=_core.SR):
        """The p-mean of the window scores (in d'), per example: [batch]. Differentiable with respect to test."""
        d2 = self._window_d2(reference, test, sr).clamp_min(0) + 1e-12
        m = d2.amax(-1, keepdim=True)
        if math.isinf(self.p):
            return torch.sqrt(m[..., 0])
        m = m.detach()                    # scaling by the largest window keeps large p finite (same value and gradient)
        return torch.sqrt(m[..., 0]) * (d2 / m).pow(self.p / 2).mean(-1).pow(1 / self.p)

    def window_scores(self, reference, test, sr=_core.SR):
        """d' of every 200 ms window that ANNI counts, per example: [batch, windows]. For inspection."""
        return torch.sqrt(self._window_d2(reference, test, sr).clamp_min(0))

    def _window_d2(self, reference, test, sr):
        x, y = self._check(reference, test, sr)
        with torch.no_grad():
            silent = (x.amax(-1) == x.amin(-1)).nonzero().flatten().tolist()   # silent or constant, as ANNI checks
        x = x - x.mean(-1, keepdim=True)                               # each signal's DC offset removed, as in ANNI
        y = y - y.mean(-1, keepdim=True)
        with torch.no_grad():                                          # the reference needs no gradient
            if silent:
                raise ValueError(f"reference is silent (or constant) in batch example(s) {silent}: its playback "
                                 "level cannot be set; leave such examples out of the batch")
            gain = torch.stack([_core.pa_gain(r, self.level) for r in x])[:, None]
            rx = self._rates(gain * x)
        ry = self._rates(gain * y)                                     # one gain for both, set by the reference
        d2 = self._observer.window_d2(rx, ry, _core.CONSTANTS)
        ok = self._observer.valid(d2.shape[-1], rx.shape[-1]).to(d2.device)
        return d2[..., ok]

    def _rates(self, p):
        """Pressure [batch, time] -> ear filter -> nerve model (one or both polarities) -> delay compensation ->
        1.9 ms bins: [batch, 128, bins]."""
        x = _core.ear_filter(p)
        if self.polarities == 2:
            r = self._nerve.forward(torch.cat([x, -x]))
            r = 0.5 * (r[:len(x)] + r[len(x):])
        else:
            r = self._nerve.forward(x)
        return self._observer.bin(self._nerve.compensate(r))

    def _check(self, reference, test, sr):
        if sr != _core.SR:
            raise ValueError(f"ANNILoss runs at {_core.SR} Hz without resampling; got sr={sr}")
        x, y = (torch.as_tensor(s) for s in (reference, test))
        if x.shape != y.shape:
            raise ValueError(f"reference and test must have the same shape; got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.dim() == 1:
            x, y = x[None], y[None]
        if x.dim() != 2:
            raise ValueError(f"signals must be [time] or [batch, time]; got {tuple(x.shape)}")
        if x.shape[-1] < 0.5 * _core.SR:
            raise ValueError("signals must be at least 0.5 s long")
        dev = self._nerve.device
        for s in (reference, test):
            if isinstance(s, torch.Tensor) and not _same_device(s.device, dev):
                raise ValueError(f"the signals are on {s.device} but ANNILoss runs on {dev}: create it with "
                                 f"device={str(s.device)!r} or move it with .to()")
        return x.to(device=dev, dtype=_core.DT), y.to(device=dev, dtype=_core.DT)


def _same_device(a, b):
    """Whether two devices are the same ("cuda" matches "cuda:0")."""
    return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)
