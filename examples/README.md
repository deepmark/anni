# Comparison with PESQ and ViSQOL on controlled changes

[`audio/`](audio/) holds a 6 s speech recording (`0_original.wav`) and seven changed versions of it. Listen to each
next to the original, preferably on headphones. [`demo.py`](demo.py) makes the changes and scores them with PESQ
(wideband), ViSQOL (speech and audio mode) and ANNI:

```bash
pip install anni soundfile pesq visqol-python
python examples/demo.py speech.wav [output_folder]
```

Scores of the files in `audio/`. The unchanged recording scores at or near each measure's best value (PESQ 4.64,
ViSQOL audio 4.73, ANNI 0). Bold marks a measure that misses or barely reacts to the change.

| File | Change | PESQ-wb | ViSQOL speech | ViSQOL audio | ANNI D_peak | ANNI D_med |
|---|---|---|---|---|---|---|
| `1_hiss` | faint hiss at 10–16 kHz, 30 dB below the speech | **4.64** | **4.62** | 3.13 | 2.96 | 1.42 |
| `2_burst` | one 10 ms noise burst in a pause | 4.05 | **4.64** | **4.73** | 15.71 | **0.00** |
| `3_brighter` | brighter tone: +10 dB around 3 kHz | **4.64** | 3.95 | **4.70** | 11.67 | 7.02 |
| `4_notch` | duller tone: the 1–3 kHz band 15 dB down | **4.37** | 3.33 | **4.64** | 14.75 | 5.90 |
| `5_echo_hiding` | echo after 1 ms, 6 dB quieter | **4.56** | 3.47 | **4.64** | 4.95 | 2.91 |
| `6_frame_clicks` | a tiny click every 80 ms | 4.13 | **4.62** | **4.68** | 3.14 | **0.08** |
| `7_lowpass` | everything above 7 kHz removed | **4.43** | **4.60** | 2.59 | 13.14 | 4.95 |