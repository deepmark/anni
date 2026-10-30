"""Controlled changes to a speech recording, scored by ANNI, PESQ and ViSQOL.

Applies simple changes to a speech recording and scores each with PESQ (wideband), ViSQOL (speech and audio mode)
and ANNI. The changed versions are written next to the original (by default to examples/out/) for listening.

usage:  pip install anni soundfile pesq "visqol-python[lattice]"
        python examples/demo.py speech.wav [output_folder]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from anni import ANNI

SR = 44100


def hiss(x, rng):
    """Faint hiss between 10 and 16 kHz, 30 dB below the speech."""
    X = np.fft.rfft(rng.standard_normal(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < 10000) | (f > 16000)] = 0
    n = np.fft.irfft(X, len(x))
    return x + n / np.sqrt(np.mean(n ** 2)) * np.sqrt(np.mean(x ** 2)) * 10 ** (-30 / 20)


def burst(x, rng):
    """One 10 ms noise burst, 6 dB below the speech level, in the quietest moment (a pause) of the recording."""
    n = int(0.010 * SR)
    hop = int(0.010 * SR)
    edge = min(int(0.5 * SR), len(x) // 4)                  # keep away from the ends, also for short clips
    e = [np.sum(x[i:i + 4 * hop] ** 2) for i in range(edge, len(x) - edge - 4 * hop, hop)]
    t0 = edge + int(np.argmin(e)) * hop + hop
    y = x.copy()
    y[t0:t0 + n] += np.sqrt(np.mean(x ** 2)) * 10 ** (-6 / 20) * rng.standard_normal(n) * np.hanning(n)
    return y


def brighter(x, rng):
    """Tone colour: a 10 dB boost around 3 kHz (one octave wide)."""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = 10 ** (10 / 20 * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / 3000) / 0.5) ** 2))
    return np.fft.irfft(np.fft.rfft(x) * g, len(x))


def notch(x, rng):
    """The 1-3 kHz band turned down by 15 dB."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f >= 1000) & (f <= 3000)] *= 10 ** (-15 / 20)
    return np.fft.irfft(X, len(x))


def echo(x, rng):
    """Echo hiding: an echo after 1 ms, 6 dB quieter (it colours the tone like a comb filter)."""
    d = int(0.001 * SR)
    y = x.copy()
    y[d:] += 10 ** (-6 / 20) * x[:-d]
    return y


def clicks(x, rng):
    """Frame-boundary clicks: a one-sample spike every 80 ms, about 18 dB below the speech level."""
    y = x.copy()
    p = int(0.080 * SR)
    y[p::p] += 4 * 10 ** (-30 / 20) * np.sqrt(np.mean(x ** 2))
    return y


def lowpass(x, rng):
    """Everything above 7 kHz removed."""
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / SR) > 7000] = 0
    return np.fft.irfft(X, len(x))


CHANGES = {  # file name: (label, change)
    "1_hiss": ("hiss 10-16 kHz", hiss), "2_burst": ("10 ms burst", burst), "3_brighter": ("+10 dB at 3 kHz", brighter),
    "4_notch": ("-15 dB at 1-3 kHz", notch), "5_echo_hiding": ("echo after 1 ms", echo),
    "6_frame_clicks": ("clicks every 80 ms", clicks), "7_lowpass": ("low-pass 7 kHz", lowpass)}


def score_all(x, y, anni, visqol_speech, visqol_audio):
    from pesq import pesq
    x16, y16 = (soxr.resample(s, SR, 16000) for s in (x, y))
    x48, y48 = (soxr.resample(s, SR, 48000) for s in (x, y))
    return dict(pesq=pesq(16000, x16, y16, "wb"),
                visqol_speech=visqol_speech.measure_from_arrays(x16, y16, 16000).moslqo,
                visqol_audio=visqol_audio.measure_from_arrays(x48, y48, 48000).moslqo,
                **anni(x, y, sr=SR))


def main(path, out=None):
    from visqol import VisqolApi
    warnings.simplefilter("ignore")
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    vs, va = VisqolApi(), VisqolApi()
    vs.create(mode="speech", use_lattice_model=True)        # the mapper of the published table
    va.create(mode="audio")
    anni = ANNI()
    out = Path(out) if out else Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "0_original.wav", x, SR)
    print(f"{'change':20s} {'PESQ':>6s} {'ViSQOL speech':>14s} {'ViSQOL audio':>13s} {'ANNI D_peak':>12s} {'D_med':>6s}")
    for fname, (label, change) in CHANGES.items():
        y = np.clip(change(x, np.random.default_rng(0)), -1, 1)    # the same samples are written and scored
        sf.write(out / f"{fname}.wav", y, SR)
        s = score_all(x, y, anni, vs, va)
        print(f"{label:20s} {s['pesq']:6.2f} {s['visqol_speech']:14.2f} {s['visqol_audio']:13.2f} "
              f"{s['D_peak']:12.2f} {s['D_med']:6.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
