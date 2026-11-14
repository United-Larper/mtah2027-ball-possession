"""Audio helpers: decode, level statistics, whistle-candidate detection, loudness envelopes.

Everything here returns aggregates or time stamps of detector output computed locally; nothing is
transcribed or exported.
"""
from __future__ import annotations

import subprocess
import wave

import numpy as np
from scipy import signal
from scipy.fft import rfft

from .paths import asset_paths, cache_dir

SR = 16000
FPS = 25
WIN = 1024           # 64 ms
HOP = 160            # 10 ms
FRAME_SAMPLES = SR // FPS   # 640 samples per video frame


def wav_path(source: str):
    return cache_dir() / f"audio_{source}_16k.wav"


def decode(source: str):
    """Decode the source video's audio to mono 16 kHz 16-bit WAV under cache/ (skipped if present)."""
    out = wav_path(source)
    if not out.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(asset_paths(source)["video"]),
                        "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def read_wav(source: str) -> np.ndarray:
    """float32 samples in [-1, 1)."""
    with wave.open(str(decode(source)), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------------------- A1
def level_stats(x: np.ndarray) -> dict:
    """Duration, 1 s RMS dBFS percentiles (1/10/50/90/99, over non-silent windows), share of digital-silence
    windows (all samples exactly zero), and long-term octave-band levels (dB re full scale, mean-square)."""
    n = (x.size // SR) * SR
    w = x[:n].reshape(-1, SR).astype(np.float64)
    rms = np.sqrt((w ** 2).mean(1))
    silent = (w == 0).all(1)
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    nz = db[~silent]
    f, pxx = signal.welch(x, fs=SR, nperseg=4096, noverlap=2048, scaling="density")
    df = f[1] - f[0]
    bands = [(44.2, 88.4), (88.4, 176.8), (176.8, 353.6), (353.6, 707), (707, 1414), (1414, 2828),
             (2828, 5657), (5657, 8000.1)]
    centres = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    spec = {}
    for c, (lo, hi) in zip(centres, bands):
        m = (f >= lo) & (f < hi)
        spec[c] = float(10 * np.log10(max(pxx[m].sum() * df, 1e-20)))
    return dict(duration_s=x.size / SR, n_windows=int(rms.size), silent_share=float(silent.mean()),
                rms_pcts=dict(zip([1, 10, 50, 90, 99], np.percentile(nz, [1, 10, 50, 90, 99]).tolist())),
                octave_db=spec)


# ---------------------------------------------------------------------------------------- A2
LO_BAND = (2500.0, 4500.0)
REF_BAND = (1000.0, 7000.0)


def stft_features(x: np.ndarray, block: int = 40000):
    """Per 10 ms hop: (tonality_dB, band_share, peak_freq_Hz).

    tonality = 10 log10( max bin power in 2.5-4.5 kHz / median bin power in 1-7 kHz )
    band_share = sum power 2.5-4.5 kHz / sum power 1-7 kHz.  Hann window, 1024 samples (64 ms), hop 160 (10 ms).
    """
    win = signal.windows.hann(WIN, sym=False).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    lo = (freqs >= LO_BAND[0]) & (freqs <= LO_BAND[1])
    ref = (freqs >= REF_BAND[0]) & (freqs <= REF_BAND[1])
    n_frames = 1 + (x.size - WIN) // HOP
    ton = np.empty(n_frames, np.float32)
    share = np.empty(n_frames, np.float32)
    pk = np.empty(n_frames, np.float32)
    lo_idx = np.flatnonzero(lo)
    for a in range(0, n_frames, block):
        b = min(a + block, n_frames)
        seg = x[a * HOP: (b - 1) * HOP + WIN]
        fr = np.lib.stride_tricks.sliding_window_view(seg, WIN)[::HOP][: b - a]
        p = np.abs(rfft(fr * win, axis=1, workers=-1)) ** 2
        pl = p[:, lo]
        am = pl.argmax(1)
        peak = pl[np.arange(b - a), am]
        med = np.median(p[:, ref], axis=1)
        ton[a:b] = 10 * np.log10((peak + 1e-12) / (med + 1e-12))
        share[a:b] = pl.sum(1) / (p[:, ref].sum(1) + 1e-12)
        pk[a:b] = freqs[lo_idx[am]]
    return ton, share, pk


def robust_stats(v: np.ndarray) -> tuple[float, float]:
    m = float(np.median(v))
    return m, float(np.median(np.abs(v - m)))


def detect(ton, share, pk, k: float, min_frames: int = 15, merge_s: float = 0.4):
    """Candidate whistles. A hop is 'on' when tonality > median + k*MAD AND band_share > median + k*MAD
    (median/MAD of each feature over the whole file). Runs shorter than min_frames (0.15 s) are dropped and
    runs separated by less than merge_s are merged. Returns list of (start_s, end_s, peak_freq_hz)."""
    mt, dt = robust_stats(ton)
    ms, ds = robust_stats(share)
    on = (ton > mt + k * dt) & (share > ms + k * ds)
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_frames
    s, e = s[keep], e[keep]
    merged = []
    for a, b in zip(s, e):
        if merged and (a - merged[-1][1]) * HOP / SR < merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        f = float(np.median(pk[a:b]))
        out.append((a * HOP / SR, (b * HOP + WIN) / SR, f))
    return out, dict(med_ton=mt, mad_ton=dt, med_share=ms, mad_share=ds)


# ---------------------------------------------------------------------------------------- A5
def frame_envelope(x: np.ndarray, n_frames: int, half_win: int = 800, hp_frames: int = 250) -> np.ndarray:
    """Log RMS envelope (dB) per video frame f: window centred on f/25 s (default 100 ms = samples
    [f*640-800, f*640+800)), then high-passed by subtracting a centred moving average of hp_frames (10 s)."""
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    f = np.arange(n_frames)
    lo = np.clip(f * FRAME_SAMPLES - half_win, 0, x.size)
    hi = np.clip(f * FRAME_SAMPLES + half_win, 0, x.size)
    ms = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    env = 10 * np.log10(ms + 1e-12)
    k = hp_frames
    c = np.concatenate([[0.0], np.cumsum(env)])
    a = np.clip(f - k // 2, 0, n_frames)
    b = np.clip(f + k // 2 + 1, 0, n_frames)
    return env - (c[b] - c[a]) / np.maximum(b - a, 1)
