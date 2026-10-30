"""The ANNI measure."""
import warnings

import numpy as np
import soxr
import torch

from . import _core

MIN_SECONDS = 0.5


class ANNI:
    """Auditory Nerve Noticeability Index: how noticeable the difference between a reference and a test signal
    is, in d' units (0 = no difference; larger = more noticeable).

    Args:
        level: playback level of the reference, in dB SPL of its active (non-silent) parts. Default 65 dB SPL,
            about conversational speech.
        device: "cpu" (default) or "cuda".

    Example:
        >>> anni = ANNI()
        >>> anni(reference, test, sr=44100)
        {'D_peak': 1.49, 'D_med': 0.81}
    """

    def __init__(self, level=65.0, device="cpu"):
        self.level = float(level)
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError(f'device must be "cpu" or "cuda"; got {device!r}')
        self._nerve = _core.Nerve(device=device)
        self._observer = _core.Observer(self._nerve.cfs)

    def __call__(self, reference, test, sr):
        """Scores `test` against `reference`.

        Args:
            reference, test: mono signals (1-D NumPy arrays or torch tensors) at sample rate `sr`.
            sr: sample rate in Hz. The model runs at 44.1 kHz; other rates are resampled, with a warning.

        Returns:
            {"D_peak": float, "D_med": float}: D_peak is the most noticeable 200 ms moment, D_med the typical one.
        """
        with torch.no_grad():
            px, py = self._prepare(reference, test, sr)
            d = self._forward(px, py)
        return {k: float(v) for k, v in d.items()}

    def _prepare(self, reference, test, sr):
        """Inputs -> pressure signals (Pa) at 44.1 kHz: offsets removed, equal length, resampling, constant-delay
        alignment, one playback gain for both (so a level change in `test` is kept and scored). Warns about every
        change it makes to the inputs."""
        if not sr > 0:
            raise ValueError(f"sr must be a positive sample rate; got {sr}")
        x, y = self._check(reference, "reference"), self._check(test, "test")
        x, y, dc = x - x.mean(), y - y.mean(), y.mean() - x.mean()
        if len(x) != len(y):
            warnings.warn(f"reference and test differ in length ({len(x)} vs {len(y)} samples); the longer one is "
                          "cut to the shorter.", stacklevel=3)
            m = min(len(x), len(y))
            x, y = x[:m], y[:m]
        if len(x) < MIN_SECONDS * sr:
            raise ValueError(f"signals must be at least {MIN_SECONDS} s long")
        if np.ptp(x) == 0.0:
            raise ValueError("reference is silent (or constant)")
        if abs(dc) > 1e-3 * np.sqrt(np.mean(x ** 2)):
            warnings.warn(f"test has a constant (DC) offset of {dc:+.3g} relative to reference; removed, as it is "
                          "inaudible.", stacklevel=3)
        if sr != _core.SR:
            warnings.warn(f"resampling from {sr:g} Hz to {_core.SR} Hz, the model's input rate.", stacklevel=3)
            x, y = (soxr.resample(s, sr, _core.SR, quality="VHQ") for s in (x, y))
        x, y = torch.as_tensor(x, dtype=_core.DT), torch.as_tensor(y, dtype=_core.DT)
        y, tau, rho = _core.align(x, y)
        shifted = f"test was shifted by {tau:+.2f} samples ({1000 * tau / _core.SR:+.3f} ms) to line up with reference"
        if abs(rho) < 0.9:                    # the delay and polarity estimates are unreliable too: one warning only
            warnings.warn(f"reference and test correlate weakly (|rho| = {abs(rho):.2f} < 0.9) after alignment: they "
                          "may be too different for a sample-by-sample comparison, so the scores may not be "
                          "meaningful." + (f" Also, {shifted}." if tau != 0.0 else ""), stacklevel=3)
        else:
            if tau != 0.0:
                warnings.warn(shifted + ".", stacklevel=3)
            if rho < 0:
                warnings.warn("test is polarity-inverted relative to reference.", stacklevel=3)
        g = _core.pa_gain(x, self.level)
        return g * x, g * y

    def _forward(self, px, py):
        """Pressure signals -> {"D_peak", "D_med"} tensors."""
        R = self._nerve.rates(torch.stack([px, py]))
        return _core.scores(self._observer, R[0], R[1])

    @staticmethod
    def _check(s, name):
        """A mono, finite NumPy array (float64 for the input steps; the model itself runs in 32-bit)."""
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        s = np.asarray(s, dtype=np.float64)
        if s.ndim != 1:
            raise ValueError(f"{name} must be mono (1-D); got shape {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"{name} contains NaN or infinite values")
        return s
