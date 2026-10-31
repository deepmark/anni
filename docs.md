# How ANNI works, function by function

The public interface is in [`measure.py`](src/anni/measure.py), the computation in [`_core.py`](src/anni/_core.py), and
the auditory-nerve model components in [`_auditory_model.py`](src/anni/_auditory_model.py).

```
measure.py:          reference, test -> checks, resampling, alignment, playback level -> sound pressure (Pa)
_core.py:            ear_filter -> Nerve.rates -> Observer.window_d2 -> scores -> D_peak, D_med
_auditory_model.py:  used by Nerve: cochlear (gammatone) filters, hair-cell low-pass, firing-rate curve
```

## measure.py: the public interface

| Function | What it does |
|---|---|
| `ANNI(level=65.0, device="cpu")` | Builds the nerve model and the observers once. `level` is the playback level of the reference in dB SPL; `device` is "cpu" or "cuda" (an NVIDIA GPU). |
| `ANNI.__call__(reference, test, sr)` | Scores `test` against `reference` and returns `{"D_peak": ..., "D_med": ...}`. |
| `_check` | Converts an input to a 1-D NumPy array and rejects multichannel or non-finite input. |
| `_prepare` | Turns the inputs into sound-pressure signals at 44.1 kHz: removes each signal's constant (DC) offset, cuts them to equal length, resamples to 44.1 kHz, removes a constant delay (`align`) and applies one playback gain to both (`pa_gain`). |
| `_forward` | Runs the nerve model on both signals and returns the scores. |

The order matters: the reference sets the playback level and the test signal is the one shifted into alignment,
so swapping the two arguments can change the scores slightly.

## _auditory_model.py: the auditory-nerve components

Adapted from Mark R. Saddler's [auditory_machine_learning](https://github.com/msaddler/auditory_machine_learning)
(MIT License; the full text is in the file header).

| Function | What it does |
|---|---|
| `freq2erb`, `erb2freq` | Convert between frequency in Hz and the ERB scale, the ear's frequency scale: equal steps on it correspond to equal stretches of the cochlea. |
| `erbspace` | Frequencies evenly spaced on the ERB scale. Used to choose the 128 fibres' tuning frequencies from 80 Hz to 16 kHz: dense at low frequencies, sparse at high ones, as in the cochlea. |
| `gammatone_filterbank_fir` | The cochlear filters: one band-pass filter per fibre that keeps only frequencies near its tuning frequency (4th-order gammatone, 1 ERB wide, about 130 Hz at 1 kHz), as 50 ms impulse responses. |
| `ihc_lowpass_filter_fir` | The inner-hair-cell low-pass filter. Hair cells cannot follow fast vibrations, so the signal is smoothed: nominal cutoff 3 kHz, 7th order (in practice −3 dB at about 1.7 kHz). Designed at 100 kHz as in the Bruce, Erfani and Zilany (2018) model, then resampled to 44.1 kHz. |
| `rate_level` | The firing-rate curve: turns the hair-cell signal's level (dB SPL) into spikes per second. About 6 spikes/s at 0 dB SPL, 125 at 40 dB (the midpoint) and 244 at 80 dB, never more than 250. This is where loud sounds saturate. |

## _core.py: ANNI's computation

### Constants

| Name | Value |
|---|---|
| `SR` | 44100 Hz, the model's input rate |
| `FS_N` | 11025 Hz, the nerve output rate |
| `P_REF` | 20 µPa, the pressure at 0 dB SPL |
| `DT` | 32-bit floating point |
| `CONSTANTS` | the four calibrated constants: r₀ (internal-noise floor, 66.6 spikes/s), k_P (pattern gain), k_S (texture gain) and e₀ (a small floor in the texture comparison) |

### Helpers

| Function | What it does |
|---|---|
| `next_fast_len(n)` | The smallest length ≥ n with only the factors 2, 3 and 5, for fast FFTs. |
| `tq(f)` | Terhardt's threshold of hearing in quiet: the quietest audible level at each frequency, in dB SPL. |

### Preparing the signals

| Function | What it does |
|---|---|
| `active_rms(x)` | The level of the signal's active parts: the signal is cut into 20 ms frames and only frames within 40 dB of the loudest (99th percentile) count, so pauses do not lower the level. |
| `pa_gain(x, level)` | The gain that turns digital samples into sound pressure (Pa) so that `x` plays at `level` dB SPL. |
| `align(x, y)` | Removes a constant delay of up to ±441 samples (±10 ms): cross-correlation finds the best whole-sample lag, a parabola through the peak refines it to a fraction of a sample, and `shift` moves `y`. Returns the aligned signal, the shift applied (0 if below 0.05 samples) and the correlation ρ. |
| `shift(y, tau)` | Delays or advances a signal by `tau` samples, fractions included, with a phase ramp in the frequency domain. |
| `ear_filter(x)` | The outer and middle ear: a zero-phase filter whose gain is the threshold of hearing turned upside down, relative to 1 kHz and limited to −100 to +20 dB. It boosts the 2–5 kHz region, where hearing is most sensitive, and cuts low and very high frequencies. |

### The auditory nerve: class `Nerve`

| Method | What it does |
|---|---|
| `__init__` | Builds the cochlear and hair-cell filters, stores the 128 tuning frequencies and computes each fibre's delay: the time to the peak of its cochlear filter's response plus the low-pass filter's delay. Low-frequency fibres respond later. |
| `forward(x)` | The model, for each signal: cochlear filters → keep the positive half-waves (the hair cell responds to movement in one direction) → hair-cell low-pass, keeping every 4th sample (44.1 → 11.025 kHz) → positive half-waves → firing-rate curve. Output: firing rate in spikes/s for 128 fibres over time. |
| `compensate(r)` | Shifts each fibre earlier by its own delay, so that all fibres line up in time. |
| `rates(p)` | The full front end: `ear_filter` → `forward` on the signal and on its upside-down copy, averaged (polarity averaging, so the comparison concerns firing rates rather than the exact waveform phase) → `compensate`. |

### Filter design for the observers

| Function | What it does |
|---|---|
| `butter2(kind, wn, fs)` | Designs one second-order Butterworth section, low-pass or band-pass, exactly as `scipy.signal.butter` does. |
| `_impulse(sec, n)` | The impulse response of one filter section, so filtering can be done as a fast FFT convolution. |

### The observers: class `Observer`

| Method | What it does |
|---|---|
| `__init__` | Sets up 1.9 ms time bins and a 200 ms window (105 bins); seven filters (a 2.8 Hz low-pass for the pattern observer, octave band-passes at 4, 8, 16, 32, 64 and 128 Hz for the texture observer); and a mask that lets each fibre use only fluctuation bands up to a quarter of its tuning frequency. |
| `bin(r)` | Averages the firing rates into 1.9 ms bins (21 samples). |
| `split(a)` | Runs a signal through all seven filters. |
| `movmean(a)` | A 200 ms moving average. |
| `valid(npos, nb)` | Marks the windows that count: the first 0.1 s and the last 0.05 s are skipped, while the filters settle. |
| `window_d2(Rx, Ry, p)` | The comparison. For every 200 ms window it adds up the evidence (in d′² units) from two observers, summed over fibres and weighted by the spacing between fibres. **Pattern:** (slow change in firing rate)² ÷ (average firing rate + r₀). A busier fibre fires more irregularly, so the same change counts less; this is how masking enters. **Texture:** (logarithm of the ratio of fluctuation strength between the two signals)², i.e. a change in roughness or flutter. |

### The result

| Function | What it does |
|---|---|
| `scores(ob, Rx, Ry)` | D for each window is the square root of its total evidence. `D_peak` is the largest D over the windows, `D_med` the median. |
