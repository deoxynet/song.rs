//! Vowel-formant modem: F2-only formant detection with melodic (non-data) pitch.
//!
//! Each symbol carries 2 bits via vowel identity (F2 formant position).
//! Pitch (F0) carries NO data — the encoder plays a melody (see
//! `protocol::MELODY`) and the primary detection path is F0-agnostic,
//! so the receiver does not need to know the melody.
//!
//! - Vowel (2 bits): F2 formant position in the 1000-2500 Hz band
//! - Pitch: free aesthetic channel, D lydian arpeggio

use crate::protocol::*;

use std::f64::consts::PI;

/// Vowel formant parameters.
#[derive(Debug, Clone, Copy)]
pub struct VowelParams {
    pub f1: f64,
    pub f2: f64,
}

/// The 4-vowel alphabet with hybrid F1+F2 detection.
///
/// Primary detection uses F2-only (1000-2500 Hz) which survives VoIP processing.
/// Hybrid mode adds F1 (300-850 Hz) as secondary signal for clean channels.
///
/// F2 positions chosen so the F2-band spectral centroid stays close to the
/// vowel center at every melody note (D3, G#3, A3, C#4). Vowel 0 (1074 Hz)
/// is the sensitive one: each melody note has a harmonic within ~40 Hz of it.
pub const VOWELS: [VowelParams; NUM_VOWELS] = [
    VowelParams { f1: 416.0, f2: 1074.0 },  // 0 — low F1, low F2
    VowelParams { f1: 624.0, f2: 1420.0 },  // 1 — mid-high F1, mid-low F2
    VowelParams { f1: 520.0, f2: 1663.0 },  // 2 — mid-low F1, mid-high F2
    VowelParams { f1: 832.0, f2: 2252.0 },  // 3 — high F1, high F2
];

/// Start preamble: vowels 0 and 3 have maximum F2 separation.
/// Symbols ARE vowel indices (pitch carries no data).
pub const PREAMBLE_START: [usize; PREAMBLE_LEN] = [0, 3, 0, 3];

/// End preamble: inverted pattern.
pub const PREAMBLE_END: [usize; PREAMBLE_LEN] = [3, 0, 3, 0];

// --- Synthesis ---

/// High-frequency rolloff knee (Hz). Harmonics above this are attenuated.
const HF_ROLLOFF_KNEE: f64 = 2400.0;

/// High-frequency rolloff rate (Hz). Lower = steeper rolloff.
const HF_ROLLOFF_RATE: f64 = 600.0;

/// Compute harmonic amplitudes shaped by the formant envelope.
///
/// Returns amplitudes for harmonics H1..H(NUM_HARMONICS) at the given F0.
/// Harmonics above HARMONIC_CEILING get zero amplitude (keeps 8 kHz mode
/// below Nyquist and bounds the loop for low melody notes).
/// Applies a gentle high-frequency rolloff above 2400 Hz to prevent clipping.
pub fn harmonic_amplitudes(f1: f64, f2: f64, f0: f64) -> [f64; NUM_HARMONICS] {
    let mut amps = [0.0f64; NUM_HARMONICS];
    for h in 0..NUM_HARMONICS {
        let freq = f0 * (h + 1) as f64;
        if freq > HARMONIC_CEILING {
            break;
        }
        // Dual formant shaping: F1 (narrow) + F2 (wide for robustness)
        let g1 = (-(freq - f1).powi(2) / (2.0 * BW1 * BW1)).exp();
        let g2 = (-(freq - f2).powi(2) / (2.0 * BW2 * BW2)).exp();
        // Combine formants: F2 is primary, F1 adds coloring
        let formant_gain = g2.max(g1 * 0.5);

        // Gentle high-frequency rolloff above knee
        let hf_atten = if freq > HF_ROLLOFF_KNEE {
            (-((freq - HF_ROLLOFF_KNEE) / HF_ROLLOFF_RATE)).exp()
        } else {
            1.0
        };

        amps[h] = formant_gain * hf_atten;
    }
    let max = amps.iter().cloned().fold(0.0f64, f64::max);
    if max > 0.0 {
        for a in &mut amps {
            *a /= max;
        }
    }
    amps
}

/// Synthesize one symbol slot: SYMBOL_TOTAL_SAMPLES of audio.
///
/// The first SAMPLES_PER_SYMBOL are the voiced classification window with
/// raised-cosine attack/release inside it; the trailing GUARD_SAMPLES are
/// true silence. The silent guard is load-bearing robustness, not a styling
/// choice: speech denoisers (RNNoise) rely on the rhythmic gaps to classify
/// the signal as speech — filling the guard with a release tail (even 60% of
/// it) makes them suppress the modem as stationary noise (measured: 2-4
/// RNNoise robustness tests fail with any tail in the guard).
///
/// `f0` is the melody note and `pos` the symbol position within the
/// transmission (see `protocol::melody_f0`); `pos` keeps the vibrato phase
/// coherent across symbol boundaries.
pub fn synthesize_symbol(vowel_idx: usize, f0: f64, volume: f64, pos: usize) -> Vec<f64> {
    let vowel = &VOWELS[vowel_idx];
    let amps = harmonic_amplitudes(vowel.f1, vowel.f2, f0);

    // These ramps are timing infrastructure, not styling: the streaming
    // clock tracker probes guard-adjacent energy within ~2.5 ms of each
    // boundary, and BOTH edge gradients are its signal. Measured: a 13 ms
    // attack fails positive-offset correction detection and a 10 ms
    // release fails the negative side (stream_clock_corrections test).
    // Soften the sound via vibrato depth and HF rolloff instead.
    let attack_samples = (SAMPLE_RATE * 0.010) as usize; // 10 ms
    let release_samples = (SAMPLE_RATE * 0.008) as usize; // 8 ms, ends at voiced edge
    let mut out = vec![0.0f64; SYMBOL_TOTAL_SAMPLES];

    // Global time offset of this slot, for transmission-coherent vibrato
    let t_slot = (pos * SYMBOL_TOTAL_SAMPLES) as f64 / SAMPLE_RATE;

    // First pass: synthesize with per-harmonic phase integration so vibrato
    // modulates frequency without phase jumps. Harmonic phases start
    // aligned at zero. Schroeder-style phase dispersal sounds less buzzy
    // and was tried, but the echo clock estimator correlates the smoothed
    // power envelope, and dispersal weakens the pitch-rate envelope
    // structure it depends on (measured: echo_clock_recovery_preserves_
    // the_next_melody fails with dispersed phases). Do not disperse.
    let mut phases = [0.0f64; NUM_HARMONICS];
    for i in 0..SYMBOL_TOTAL_SAMPLES {
        let t_global = t_slot + i as f64 / SAMPLE_RATE;
        let vib = 1.0 + VIBRATO_DEPTH * (2.0 * PI * VIBRATO_RATE * t_global).sin();

        let mut sample = 0.0f64;
        for h in 0..NUM_HARMONICS {
            if amps[h] == 0.0 {
                continue;
            }
            sample += amps[h] * phases[h].sin();
            phases[h] += 2.0 * PI * f0 * (h + 1) as f64 * vib / SAMPLE_RATE;
        }

        out[i] = sample;
    }

    // Normalize to prevent clipping (harmonics can sum > 1.0)
    let peak = out.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    if peak > 0.0 {
        let norm = 1.0 / peak;
        for s in &mut out {
            *s *= norm;
        }
    }

    // Second pass: envelope — raised-cosine attack, full sustain through the
    // classification window, raised-cosine release across the guard.
    for i in 0..SYMBOL_TOTAL_SAMPLES {
        let env = if i >= SAMPLES_PER_SYMBOL {
            0.0
        } else if i < attack_samples {
            let t = i as f64 / attack_samples as f64;
            0.5 * (1.0 - (PI * t).cos())
        } else if i >= SAMPLES_PER_SYMBOL - release_samples {
            let t = (i - (SAMPLES_PER_SYMBOL - release_samples)) as f64 / release_samples as f64;
            0.5 * (1.0 + (PI * t).cos())
        } else {
            1.0
        };

        out[i] *= volume * env;
    }

    out
}

// --- Detection ---

/// Detect F0 from a power spectrum by testing candidate F0 values.
///
/// Scores each candidate by summing spectral power at expected harmonic bins.
/// Returns the best-fit F0. Only feeds the secondary F1+F2 detection path;
/// the primary F2-centroid path is F0-agnostic.
pub fn detect_f0(spectrum: &[f32]) -> f64 {
    let mut best_f0 = F0_CANDIDATES[0];
    let mut best_score = 0.0f64;

    for &f0 in &F0_CANDIDATES {
        let mut score = 0.0f64;
        for h in 1..=NUM_HARMONICS {
            let freq = f0 * h as f64;
            if freq > HARMONIC_CEILING {
                break;
            }
            let bin = (freq / HZ_PER_BIN).round() as usize;
            if bin >= spectrum.len() {
                break;
            }
            // Sum power in ±2 bin window
            for offset in 0..=2usize {
                if offset == 0 {
                    score += spectrum[bin] as f64;
                } else {
                    if bin >= offset {
                        score += spectrum[bin - offset] as f64 * 0.5;
                    }
                    if bin + offset < spectrum.len() {
                        score += spectrum[bin + offset] as f64 * 0.5;
                    }
                }
            }
        }
        if score > best_score {
            best_score = score;
            best_f0 = f0;
        }
    }

    best_f0
}

/// Detect F2 formant frequency by finding spectral centroid in F2 band.
///
/// Uses weighted centroid of spectral power in F2 band (1000-2500 Hz).
/// Energy spreads across harmonics within the F2 envelope. The template
/// and adaptive-centroid retries account for the remaining pitch dependence.
pub fn detect_formants(spectrum: &[f32], _f0: f64) -> f64 {
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = (F2_HI / HZ_PER_BIN).floor() as usize;

    if bin_hi >= spectrum.len() || bin_lo >= bin_hi {
        return (F2_LO + F2_HI) / 2.0; // Fallback to center
    }

    // Find spectral centroid (power-weighted frequency average)
    let mut power_sum = 0.0f64;
    let mut weighted_sum = 0.0f64;

    for bin in bin_lo..=bin_hi.min(spectrum.len() - 1) {
        let freq = bin as f64 * HZ_PER_BIN;
        let power = spectrum[bin] as f64;
        power_sum += power;
        weighted_sum += freq * power;
    }

    if power_sum > 0.0 {
        weighted_sum / power_sum
    } else {
        (F2_LO + F2_HI) / 2.0
    }
}

/// Detect F1 and F2 formants using harmonic peak detection.
///
/// This is the secondary detection path for hybrid mode. It finds the
/// strongest harmonic in each formant band, which works well on clean
/// channels but may fail when F1 is crushed by VoIP processing.
pub fn detect_formants_f1f2(spectrum: &[f32], f0: f64) -> (f64, f64) {
    let mut best_f1_power = 0.0f64;
    let mut best_f1_freq = (F1_LO + F1_HI) / 2.0;
    let mut best_f2_power = 0.0f64;
    let mut best_f2_freq = (F2_LO + F2_HI) / 2.0;

    for h in 1..=NUM_HARMONICS {
        let freq = f0 * h as f64;
        if freq > HARMONIC_CEILING {
            break;
        }
        let bin = (freq / HZ_PER_BIN).round() as usize;
        if bin >= spectrum.len() {
            break;
        }

        // Sum power in 3-bin window around harmonic
        let mut power = 0.0f64;
        for offset in 0..=2usize {
            if offset == 0 {
                power += spectrum[bin] as f64;
            } else {
                if bin >= offset && bin - offset < spectrum.len() {
                    power += spectrum[bin - offset] as f64;
                }
                if bin + offset < spectrum.len() {
                    power += spectrum[bin + offset] as f64;
                }
            }
        }

        if freq >= F1_LO && freq <= F1_HI && power > best_f1_power {
            best_f1_power = power;
            best_f1_freq = freq;
        }
        if freq >= F2_LO && freq <= F2_HI && power > best_f2_power {
            best_f2_power = power;
            best_f2_freq = freq;
        }
    }

    (best_f1_freq, best_f2_freq)
}

/// Expected F2-band spectral centroids per vowel, averaged over melody notes.
///
/// The measured centroid is NOT the nominal F2: the 1000-2500 Hz band cuts
/// the lower Gaussian tail of vowel 0 (F2=1074), biasing its centroid up to
/// ~1180 Hz. Classifying against the centroid the synthesis model actually
/// produces (instead of nominal F2) centers each vowel in its own decision
/// region — vowel 0 gains ~60 Hz of boundary margin against vowel 1.
pub const EXPECTED_F2_CENTROIDS: [f64; NUM_VOWELS] = [1177.0, 1426.0, 1663.0, 2214.0];

/// Classify detected F2-band centroid to the nearest vowel, returning both
/// the best and runner-up vowel (for Chase-style second-choice retry).
pub fn classify_vowel_f2_ranked(f2: f64, centers: &[f64; NUM_VOWELS]) -> (usize, usize, f64) {
    let mut best_idx = 0;
    let mut second_idx = 0;
    let mut best_dist = f64::MAX;
    let mut second_dist = f64::MAX;

    for (i, &center) in centers.iter().enumerate() {
        let dist = (f2 - center).abs() / 200.0; // Normalize by spacing

        if dist < best_dist {
            second_dist = best_dist;
            second_idx = best_idx;
            best_dist = dist;
            best_idx = i;
        } else if dist < second_dist {
            second_dist = dist;
            second_idx = i;
        }
    }

    let confidence = if best_dist > 0.0 {
        (second_dist - best_dist) / best_dist
    } else {
        second_dist
    };

    (best_idx, second_idx, confidence)
}

/// Classify detected F2-band centroid to the nearest vowel (primary classifier).
///
/// Compares against EXPECTED_F2_CENTROIDS; robust to VoIP F1 crushing.
pub fn classify_vowel_f2(f2: f64, centers: &[f64; NUM_VOWELS]) -> (usize, f64) {
    let (best, _, conf) = classify_vowel_f2_ranked(f2, centers);
    (best, conf)
}

/// Classify detected (F1, F2) to the nearest vowel (2D classification),
/// returning best and runner-up.
pub fn classify_vowel_f1f2_ranked(f1: f64, f2: f64) -> (usize, usize, f64) {
    let mut best_idx = 0;
    let mut second_idx = 0;
    let mut best_dist = f64::MAX;
    let mut second_dist = f64::MAX;

    for (i, v) in VOWELS.iter().enumerate() {
        let d1 = (f1 - v.f1) / 100.0; // F1 spacing ~100 Hz
        let d2 = (f2 - v.f2) / 200.0; // F2 spacing ~200 Hz
        let dist = (d1 * d1 + d2 * d2).sqrt();

        if dist < best_dist {
            second_dist = best_dist;
            second_idx = best_idx;
            best_dist = dist;
            best_idx = i;
        } else if dist < second_dist {
            second_dist = dist;
            second_idx = i;
        }
    }

    let confidence = if best_dist > 0.0 {
        (second_dist - best_dist) / best_dist
    } else {
        second_dist
    };

    (best_idx, second_idx, confidence)
}

/// Classify detected (F1, F2) to the nearest vowel (2D classification).
///
/// Uses Euclidean distance in normalized F1/F2 space.
/// This is the secondary classifier for hybrid mode, used when F1 is available.
pub fn classify_vowel_f1f2(f1: f64, f2: f64) -> (usize, f64) {
    let (best, _, conf) = classify_vowel_f1f2_ranked(f1, f2);
    (best, conf)
}

/// Full result of classifying one voiced window.
#[derive(Debug, Clone, Copy)]
pub struct SymbolClass {
    /// Best vowel decision.
    pub vowel: usize,
    /// Runner-up vowel (second choice for Chase-style retry decoding).
    pub second: usize,
    /// Normalized decision confidence in 0..1.
    pub confidence: f64,
    /// Raw F2-band spectral centroid (Hz) — channel-shaped, used for
    /// structural (channel-agnostic) preamble detection.
    pub centroid: f64,
}

/// Classify a symbol (= vowel) from a power spectrum using hybrid detection,
/// returning the full decision (best, runner-up, confidence, raw centroid).
///
/// Runs two parallel detection paths:
/// 1. F2-only (primary): robust to VoIP F1 crushing
/// 2. F1+F2 (secondary): better accuracy on clean channels
///
/// Picks the result with higher confidence, combining the best of both.
/// Pitch plays no role: it carries no data.
pub fn classify_symbol_full(spectrum: &[f32], centers: &[f64; NUM_VOWELS]) -> SymbolClass {
    let f0 = detect_f0(spectrum);

    // Path 1: F2-only detection (VoIP-safe, primary)
    let f2_centroid = detect_formants(spectrum, f0);
    let (vowel_f2, second_f2, conf_f2) = classify_vowel_f2_ranked(f2_centroid, centers);
    let conf_f2_norm = normalize_confidence(conf_f2);

    // Path 2: F1+F2 detection (secondary, for clean channels)
    let (f1_peak, f2_peak) = detect_formants_f1f2(spectrum, f0);
    let (vowel_f1f2, _second_f1f2, conf_f1f2) = classify_vowel_f1f2_ranked(f1_peak, f2_peak);
    let conf_f1f2_norm = normalize_confidence(conf_f1f2);

    // Hybrid decision logic:
    // - If both paths agree: use result with boosted confidence
    // - If they disagree: prefer F2-only (more reliable under distortion)
    //   unless F1+F2 confidence is significantly higher
    let (vowel_idx, second_idx, vowel_conf) = if vowel_f2 == vowel_f1f2 {
        // Agreement: boost confidence
        (vowel_f2, second_f2, (conf_f2_norm + conf_f1f2_norm) / 2.0 + 0.1)
    } else if conf_f1f2_norm > conf_f2_norm + 0.15 {
        // F1+F2 has significantly higher confidence, trust it.
        // The disagreeing F2 decision is the natural second choice.
        (vowel_f1f2, vowel_f2, conf_f1f2_norm)
    } else {
        // Disagreement: prefer F2-only (VoIP-safe default); the F1+F2
        // decision is the natural second choice.
        (vowel_f2, vowel_f1f2, conf_f2_norm)
    };

    SymbolClass {
        vowel: vowel_idx,
        second: second_idx,
        confidence: vowel_conf.min(1.0),
        centroid: f2_centroid,
    }
}

/// Classify a symbol (= vowel) from a power spectrum using hybrid detection.
pub fn classify_symbol(spectrum: &[f32], centers: &[f64; NUM_VOWELS]) -> (usize, f64) {
    let c = classify_symbol_full(spectrum, centers);
    (c.vowel, c.confidence)
}

/// Map raw vowel confidence (0..∞) into 0..1 for stable mixing.
#[inline]
fn normalize_confidence(raw: f64) -> f64 {
    let raw = raw.max(0.0);
    raw / (raw + 1.0)
}

// --- Channel equalization from the calibration pilot ---
//
// A static spectral filter (VoIP "voice optimization", Teams-style F1 crush
// with in-band tilt) shifts every F2-band centroid wholesale, which a
// per-vowel center offset (calibrate_centers) cannot absorb once the tilt is
// steep. The pilot symbols are KNOWN, so the receiver can compare each
// received pilot spectrum against the spectrum the encoder is known to have
// produced and estimate a smooth per-frequency channel gain across the F2
// band. Dividing data spectra by that gain undoes arbitrary static shaping
// before centroid classification. Used strictly as a decode RETRY, like
// center calibration — RS self-verification gates it.

/// EQ estimation bucket width (Hz).
const EQ_BUCKET_HZ: f64 = 100.0;

/// Minimum populated buckets for a usable EQ estimate.
const EQ_MIN_BUCKETS: usize = 8;

/// Deepest channel notch the EQ will invert (dB). Bounds noise amplification
/// where the channel crushed the band to nothing.
const EQ_FLOOR_DB: f64 = -60.0;

/// Reference (transmit-side) power spectra of the pilot symbols, unit volume.
/// Pilot symbol k is vowel PILOT[k] at melody position PREAMBLE_LEN + k.
fn expected_pilot_spectra(profile: EccProfile) -> &'static Vec<Vec<f32>> {
    use std::sync::OnceLock;
    static SPECTRA: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    static LEGACY_SPECTRA: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    let cache = if profile == EccProfile::Legacy { &LEGACY_SPECTRA } else { &SPECTRA };
    cache.get_or_init(|| {
        PILOT
            .iter()
            .enumerate()
            .map(|(k, &vowel)| {
                let pos = PREAMBLE_LEN + k;
                let slot = synthesize_symbol(vowel, pilot_f0(k, profile), 1.0, pos);
                let voiced: Vec<f32> =
                    slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| s as f32).collect();
                let mut spectrum = vec![0.0f32; voiced.len()];
                crate::fft::power_spectrum(&voiced, &mut spectrum);
                spectrum
            })
            .collect()
    })
}

/// Estimate a per-FFT-bin equalization multiplier from received pilot
/// spectra. `measured` holds (pilot_index, received_power_spectrum) pairs.
///
/// Returns a multiplier vector (apply as `psd[bin] * eq[bin]`) covering the
/// F2 band, or None when the pilot didn't yield enough usable gain samples.
pub fn estimate_eq(measured: &[(usize, Vec<f32>)]) -> Option<Vec<f64>> {
    estimate_eq_with_profile(measured, EccProfile::Standard)
}

/// Estimate EQ against the pilot melody used by the selected frame format.
pub fn estimate_eq_with_profile(measured: &[(usize, Vec<f32>)], profile: EccProfile) -> Option<Vec<f64>> {
    let refs = expected_pilot_spectra(profile);
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = (F2_HI / HZ_PER_BIN).floor() as usize;
    let n_buckets = ((F2_HI - F2_LO) / EQ_BUCKET_HZ).ceil() as usize;

    let mut sums = vec![0.0f64; n_buckets];
    let mut counts = vec![0usize; n_buckets];

    for (k, psd) in measured {
        let Some(expected) = refs.get(*k) else { continue };
        if psd.len() <= bin_hi || expected.len() <= bin_hi {
            continue;
        }
        let e_max = expected[bin_lo..=bin_hi]
            .iter()
            .cloned()
            .fold(0.0f32, f32::max) as f64;
        if e_max <= 0.0 {
            continue;
        }
        for bin in bin_lo..=bin_hi {
            let ev = expected[bin] as f64;
            // Only harmonic peaks carry gain information.
            if ev < 0.02 * e_max {
                continue;
            }
            let rv = psd[bin] as f64;
            if rv <= 0.0 {
                continue;
            }
            let freq = bin as f64 * HZ_PER_BIN;
            let b = (((freq - F2_LO) / EQ_BUCKET_HZ) as usize).min(n_buckets - 1);
            sums[b] += (rv / ev).ln();
            counts[b] += 1;
        }
    }

    if counts.iter().filter(|&&c| c > 0).count() < EQ_MIN_BUCKETS {
        return None;
    }

    // Bucket means in log domain; fill empty buckets by linear interpolation
    // between populated neighbors (edge-hold at the ends).
    let mut gain_ln = vec![f64::NAN; n_buckets];
    for b in 0..n_buckets {
        if counts[b] > 0 {
            gain_ln[b] = sums[b] / counts[b] as f64;
        }
    }
    let first = gain_ln.iter().position(|v| !v.is_nan())?;
    let last = gain_ln.iter().rposition(|v| !v.is_nan())?;
    for b in 0..first {
        gain_ln[b] = gain_ln[first];
    }
    for b in last + 1..n_buckets {
        gain_ln[b] = gain_ln[last];
    }
    let mut b = first;
    while b < last {
        if gain_ln[b + 1].is_nan() {
            let next = (b + 2..=last).find(|&i| !gain_ln[i].is_nan()).unwrap_or(last);
            let span = (next - b) as f64;
            for i in b + 1..next {
                let t = (i - b) as f64 / span;
                gain_ln[i] = gain_ln[b] * (1.0 - t) + gain_ln[next] * t;
            }
            b = next;
        } else {
            b += 1;
        }
    }

    // 3-tap smoothing.
    let smoothed: Vec<f64> = (0..n_buckets)
        .map(|i| {
            let lo = i.saturating_sub(1);
            let hi = (i + 1).min(n_buckets - 1);
            (gain_ln[lo] + gain_ln[i] + gain_ln[hi]) / 3.0
        })
        .collect();

    // Normalize so the strongest bucket has unity gain, then floor the
    // inversion depth to bound noise amplification.
    let g_max = smoothed.iter().cloned().fold(f64::MIN, f64::max);
    // These are power ratios (received PSD / expected PSD), so dB uses
    // 10*log10. Using 20 here silently halves the specified EQ range.
    let floor_ln = EQ_FLOOR_DB / 10.0 * std::f64::consts::LN_10;
    let normalized: Vec<f64> = smoothed
        .iter()
        .map(|&g| (g - g_max).max(floor_ln))
        .collect();

    // Per-bin multiplier: inverse gain, linearly interpolated between bucket
    // centers in log domain.
    let mut eq = vec![1.0f64; FFT_SIZE];
    for bin in bin_lo..=bin_hi.min(FFT_SIZE - 1) {
        let freq = bin as f64 * HZ_PER_BIN;
        let x = (freq - F2_LO) / EQ_BUCKET_HZ - 0.5;
        let i0 = (x.floor().max(0.0) as usize).min(n_buckets - 1);
        let i1 = (i0 + 1).min(n_buckets - 1);
        let t = (x - i0 as f64).clamp(0.0, 1.0);
        let g = normalized[i0] * (1.0 - t) + normalized[i1] * t;
        eq[bin] = (-g).exp();
    }
    Some(eq)
}

/// F2-band spectral centroid of an equalized spectrum.
pub fn detect_formants_eq(spectrum: &[f32], eq: &[f64]) -> f64 {
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = (F2_HI / HZ_PER_BIN).floor() as usize;

    if bin_hi >= spectrum.len() || bin_lo >= bin_hi {
        return (F2_LO + F2_HI) / 2.0;
    }

    let mut power_sum = 0.0f64;
    let mut weighted_sum = 0.0f64;
    for bin in bin_lo..=bin_hi.min(spectrum.len() - 1).min(eq.len() - 1) {
        let freq = bin as f64 * HZ_PER_BIN;
        let power = spectrum[bin] as f64 * eq[bin];
        power_sum += power;
        weighted_sum += freq * power;
    }

    if power_sum > 0.0 {
        weighted_sum / power_sum
    } else {
        (F2_LO + F2_HI) / 2.0
    }
}

/// Classify one voiced window through the channel EQ: equalize the F2 band,
/// take the centroid, classify against the DEFAULT expected centroids
/// (equalization restores the transmit-side spectrum shape). Pure F2 path —
/// the F1 band is not equalized and stays out of the decision.
pub fn classify_symbol_eq(spectrum: &[f32], eq: &[f64]) -> SymbolClass {
    let centroid = detect_formants_eq(spectrum, eq);
    let (vowel, second, conf) = classify_vowel_f2_ranked(centroid, &EXPECTED_F2_CENTROIDS);
    SymbolClass {
        vowel,
        second,
        confidence: normalize_confidence(conf),
        centroid,
    }
}

/// Chromatic note bank for matched-filter templates: every 12-TET note
/// from B2 to F#4 — the modem's melodic register. The default melody's
/// notes (D3, G#3, C#4) are members, and so is any expressive
/// `encode_with_pitch` track in this compass, so template classification
/// needs no knowledge of what tune the encoder sang. The top five notes
/// (D4..F#4) have only ~4-6 harmonics in the F2 band — usable, but with
/// thinner margins than the octave below; expressive tracks should prefer
/// the middle of the register for notes that must survive harsh channels.
/// (Below B2 the 26-harmonic synthesis can no longer reach vowel 3's
/// formant; above F#4 the harmonic grid is too sparse to separate vowels
/// 0 and 1.)
const TEMPLATE_NOTES: [f64; 20] = [
    123.47, 130.81, 138.59, // B2..C#3
    146.83, 155.56, 164.81, 174.61, 185.00, 196.00, // D3..G3
    207.65, 220.00, 233.08, 246.94, 261.63, 277.18, // G#3..C#4
    293.66, 311.13, 329.63, 349.23, 369.99, // D4..F#4
];

/// Transmit-side F2-band power spectra for every (note, vowel) pair.
fn expected_vowel_spectra() -> &'static Vec<[Vec<f32>; NUM_VOWELS]> {
    use std::sync::OnceLock;
    static SPECTRA: OnceLock<Vec<[Vec<f32>; NUM_VOWELS]>> = OnceLock::new();
    SPECTRA.get_or_init(|| {
        TEMPLATE_NOTES
            .iter()
            .map(|&f0| {
                std::array::from_fn(|vowel| {
                    let slot = synthesize_symbol(vowel, f0, 1.0, 0);
                    let voiced: Vec<f32> =
                        slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| s as f32).collect();
                    let mut spectrum = vec![0.0f32; voiced.len()];
                    crate::fft::power_spectrum(&voiced, &mut spectrum);
                    spectrum
                })
            })
            .collect()
    })
}

/// Fixed F2-band template amplitudes and cumulative norms. Only the
/// observation changes between calls: recomputing thousands of template
/// square roots per classified window wastes receiver callback time. Store
/// all four vowels together at each bin so independent dot products can be
/// accumulated side by side without reordering any vowel's arithmetic.
struct NoteAmplitudeTemplates {
    values: Vec<[f64; NUM_VOWELS]>,
    norms: Vec<[f64; NUM_VOWELS]>,
}

fn expected_vowel_amplitudes() -> &'static Vec<NoteAmplitudeTemplates> {
    use std::sync::OnceLock;
    static AMPLITUDES: OnceLock<Vec<NoteAmplitudeTemplates>> = OnceLock::new();
    AMPLITUDES.get_or_init(|| {
        let lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
        let hi = (F2_HI / HZ_PER_BIN).floor() as usize;
        expected_vowel_spectra().iter().map(|refs| {
            let mut values = Vec::with_capacity(hi - lo + 1);
            let mut norms = Vec::with_capacity(hi - lo + 1);
            let mut norm_sq = [0.0f64; NUM_VOWELS];
            for bin in lo..=hi {
                let amplitudes: [f64; NUM_VOWELS] =
                    std::array::from_fn(|v| (refs[v][bin] as f64).max(0.0).sqrt());
                for v in 0..NUM_VOWELS {
                    norm_sq[v] += amplitudes[v] * amplitudes[v];
                }
                values.push(amplitudes);
                // Every lane retains the original increasing-bin sum,
                // including each partial-band norm.
                norms.push(norm_sq.map(f64::sqrt));
            }
            NoteAmplitudeTemplates { values, norms }
        }).collect()
    })
}

/// Expected F2-band centroids per (note, vowel) — what the synthesis model
/// actually produces at each F0. The static EXPECTED_F2_CENTROIDS are an
/// average over the default melody; per-note tables remove the residual
/// F0-dependent centroid bias (up to ~60 Hz), which matters both for
/// expressive `encode_with_pitch` tracks and at the margins of harsh
/// channels.
fn expected_centroids_per_note() -> &'static Vec<[f64; NUM_VOWELS]> {
    use std::sync::OnceLock;
    static CENTROIDS: OnceLock<Vec<[f64; NUM_VOWELS]>> = OnceLock::new();
    CENTROIDS.get_or_init(|| {
        expected_vowel_spectra()
            .iter()
            .map(|refs| {
                std::array::from_fn(|v| {
                    let f0 = detect_f0(&refs[v]);
                    detect_formants(&refs[v], f0)
                })
            })
            .collect()
    })
}

/// Detect which bank note a window was sung at, from the equalized F2-band
/// spectrum: comb-score each candidate F0's harmonics inside the band.
/// Pitch carries no data, but knowing it sharpens vowel classification.
pub fn detect_note(spectrum: &[f32], eq: &[f64]) -> usize {
    let mut best = 0;
    let mut best_score = f64::MIN;
    for (idx, &f0) in TEMPLATE_NOTES.iter().enumerate() {
        let mut score = 0.0f64;
        let mut count = 0usize;
        let mut h = (F2_LO / f0).ceil() as usize;
        loop {
            let freq = f0 * h as f64;
            if freq > F2_HI {
                break;
            }
            let bin = (freq / HZ_PER_BIN).round() as usize;
            if bin + 1 >= spectrum.len() || bin + 1 >= eq.len() {
                break;
            }
            for b in bin.saturating_sub(1)..=bin + 1 {
                score += spectrum[b] as f64 * eq[b];
            }
            count += 1;
            h += 1;
        }
        if count > 0 {
            // Normalize by comb size so dense low-F0 combs (which cover
            // more of the band) don't win by default.
            score /= count as f64;
            if score > best_score {
                best_score = score;
                best = idx;
            }
        }
    }
    best
}

/// F0-adaptive centroid classification: detect the window's note, then
/// classify the equalized centroid against THAT note's expected per-vowel
/// centroids. Keeps the centroid's robustness to spectral warping while
/// removing the F0-dependent bias a static center table carries — the rung
/// that lets the encoder sing arbitrary melodies without costing margin.
pub fn classify_symbol_f0_adaptive(spectrum: &[f32], eq: &[f64]) -> SymbolClass {
    let note = detect_note(spectrum, eq);
    let centers = &expected_centroids_per_note()[note];
    let centroid = detect_formants_eq(spectrum, eq);
    let (vowel, second, conf) = classify_vowel_f2_ranked(centroid, centers);
    SymbolClass {
        vowel,
        second,
        confidence: normalize_confidence(conf),
        centroid,
    }
}

/// Matched-filter classification: correlate the equalized F2-band spectrum
/// jointly against synthesis templates for every (note, vowel) pair and
/// take the best-correlating vowel.
///
/// Beats centroid classification when additive in-band interference (e.g.
/// spectral leakage from LOUD neighboring windows under heavy channel tilt)
/// drags the centroid — the interference lives on the *neighbor's* F0
/// harmonic grid, while the true template lives on this window's own grid,
/// so correlation rejects what a 1-D centroid cannot. Joint maximization
/// over the note bank keeps this F0-agnostic (pitch carries no data, and
/// the sender may be playing an expressive track the receiver has never
/// heard of). `eq` is the channel EQ (pass all-ones when unknown).
pub fn classify_symbol_template(spectrum: &[f32], eq: &[f64]) -> SymbolClass {
    let bank = expected_vowel_amplitudes();
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = ((F2_HI / HZ_PER_BIN).floor() as usize)
        .min(spectrum.len().saturating_sub(1))
        .min(eq.len().saturating_sub(1));

    let centroid = detect_formants_eq(spectrum, eq);
    if bin_lo >= bin_hi {
        let (vowel, second, conf) = classify_vowel_f2_ranked(centroid, &EXPECTED_F2_CENTROIDS);
        return SymbolClass {
            vowel,
            second,
            confidence: normalize_confidence(conf),
            centroid,
        };
    }

    // Amplitude (sqrt-power) domain tames the band's dynamic range so the
    // correlation isn't owned by the loudest harmonics alone.
    let es: Vec<f64> = (bin_lo..=bin_hi)
        .map(|b| (spectrum[b] as f64 * eq[b]).max(0.0).sqrt())
        .collect();
    let es_norm = es.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Per vowel: the best correlation across all candidate notes.
    let mut scores = [0.0f64; NUM_VOWELS];
    for refs in bank {
        let dots = template_dots(&es, &refs.values);
        for (v, score) in scores.iter_mut().enumerate() {
            let denom = es_norm * refs.norms[bin_hi - bin_lo][v];
            if denom > 0.0 {
                *score = score.max(dots[v] / denom);
            }
        }
    }

    let mut best = 0;
    let mut second = 0;
    let mut s1 = f64::MIN;
    let mut s2 = f64::MIN;
    for (v, &s) in scores.iter().enumerate() {
        if s > s1 {
            s2 = s1;
            second = best;
            s1 = s;
            best = v;
        } else if s > s2 {
            s2 = s;
            second = v;
        }
    }

    // Margin of correlation relative to remaining headroom.
    let confidence = ((s1 - s2) / (1.0 - s2).max(1e-9)).clamp(0.0, 1.0);
    SymbolClass {
        vowel: best,
        second,
        confidence,
        centroid,
    }
}

/// Number of notes in the template bank (chromatic B2..F#4).
pub(crate) const BANK_NOTE_COUNT: usize = TEMPLATE_NOTES.len();

/// Nearest template-bank index for a synthesis frequency (log-distance).
pub(crate) fn bank_note_index(hz: f64) -> usize {
    let mut best = 0;
    let mut best_dist = f64::MAX;
    for (idx, &note) in TEMPLATE_NOTES.iter().enumerate() {
        let dist = (hz / note).ln().abs();
        if dist < best_dist {
            best_dist = dist;
            best = idx;
        }
    }
    best
}

/// Per-note melody evidence: for each bank note, the best normalized
/// template correlation across the four vowels over the F2 band. Cosine
/// normalization cancels channel gain. This feeds the decoder's
/// melody-derived LENGTH hint only — vowel decisions never consult it,
/// and pitch still carries no data.
pub(crate) fn note_match_scores(spectrum: &[f32]) -> [f64; BANK_NOTE_COUNT] {
    let bank = expected_vowel_amplitudes();
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = ((F2_HI / HZ_PER_BIN).floor() as usize).min(spectrum.len().saturating_sub(1));
    let mut out = [0.0f64; BANK_NOTE_COUNT];
    if bin_lo >= bin_hi {
        return out;
    }
    let es: Vec<f64> = (bin_lo..=bin_hi)
        .map(|b| (spectrum[b] as f64).max(0.0).sqrt())
        .collect();
    let es_norm = es.iter().map(|x| x * x).sum::<f64>().sqrt();
    if es_norm <= 0.0 {
        return out;
    }
    for (score, refs) in out.iter_mut().zip(bank) {
        let dots = template_dots(&es, &refs.values);
        for (v, &dot) in dots.iter().enumerate() {
            let denom = es_norm * refs.norms[bin_hi - bin_lo][v];
            if denom > 0.0 {
                *score = score.max(dot / denom);
            }
        }
    }
    out
}

// Vectorize across independent vowels, never across the frequency-bin sum.
// Separate multiply and add operations preserve the scalar rounding order;
// fused multiply-add or a horizontal reduction would change confidences.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn template_dots(observed: &[f64], templates: &[[f64; NUM_VOWELS]]) -> [f64; NUM_VOWELS] {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};
    const _: () = assert!(NUM_VOWELS == 4);
    let mut dots = [0.0; NUM_VOWELS];
    // SAFETY: the cfg requires NEON support. Each input row and the output
    // contain four f64s, making both two-lane loads/stores valid. Iterating
    // with zip limits observation access to the common slice length.
    unsafe {
        let mut low = vdupq_n_f64(0.0);
        let mut high = vdupq_n_f64(0.0);
        for (&value, row) in observed.iter().zip(templates) {
            let value = vdupq_n_f64(value);
            low = vaddq_f64(low, vmulq_f64(value, vld1q_f64(row.as_ptr())));
            high = vaddq_f64(high, vmulq_f64(value, vld1q_f64(row.as_ptr().add(2))));
        }
        vst1q_f64(dots.as_mut_ptr(), low);
        vst1q_f64(dots.as_mut_ptr().add(2), high);
    }
    dots
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[inline]
fn template_dots(observed: &[f64], templates: &[[f64; NUM_VOWELS]]) -> [f64; NUM_VOWELS] {
    template_dots_portable(observed, templates)
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
#[inline]
fn template_dots_portable(observed: &[f64], templates: &[[f64; NUM_VOWELS]]) -> [f64; NUM_VOWELS] {
    let mut dots = [0.0; NUM_VOWELS];
    for (&value, row) in observed.iter().zip(templates) {
        for v in 0..NUM_VOWELS {
            dots[v] += value * row[v];
        }
    }
    dots
}

// --- 2-bit packing ---

/// Pack bytes into 2-bit symbol indices (4 symbols per byte, MSB first).
pub fn bytes_to_symbols(bytes: &[u8]) -> Vec<usize> {
    let total_bits = bytes.len() * 8;
    let n_symbols = (total_bits + BITS_PER_SYMBOL - 1) / BITS_PER_SYMBOL;
    let mut symbols = Vec::with_capacity(n_symbols);

    let mut bit_pos = 0;
    for _ in 0..n_symbols {
        let mut val = 0usize;
        for b in 0..BITS_PER_SYMBOL {
            let byte_idx = (bit_pos + b) / 8;
            let bit_idx = 7 - ((bit_pos + b) % 8); // MSB first
            if byte_idx < bytes.len() {
                val |= (((bytes[byte_idx] >> bit_idx) & 1) as usize) << (BITS_PER_SYMBOL - 1 - b);
            }
        }
        symbols.push(val);
        bit_pos += BITS_PER_SYMBOL;
    }

    symbols
}

/// Unpack 2-bit symbol indices back into bytes.
///
/// Extracts the bitstream from symbol indices and packs into bytes.
/// `num_bytes` specifies how many output bytes to produce.
pub fn symbols_to_bytes(symbols: &[usize], num_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; num_bytes];

    let mut bit_pos = 0;
    for &sym in symbols {
        for b in 0..BITS_PER_SYMBOL {
            let byte_idx = (bit_pos + b) / 8;
            let bit_idx = 7 - ((bit_pos + b) % 8);
            if byte_idx < num_bytes {
                let bit_val = (sym >> (BITS_PER_SYMBOL - 1 - b)) & 1;
                bytes[byte_idx] |= (bit_val as u8) << bit_idx;
            }
        }
        bit_pos += BITS_PER_SYMBOL;
    }

    bytes
}

/// Compute number of symbols needed to encode n_bytes.
pub fn symbols_for_bytes(n_bytes: usize) -> usize {
    (n_bytes * 8 + BITS_PER_SYMBOL - 1) / BITS_PER_SYMBOL
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference matcher retains the original per-call template square
    // roots, norm accumulation, score ordering, and confidence calculation.
    fn uncached_template_classifier(spectrum: &[f32], eq: &[f64]) -> SymbolClass {
        let bank = expected_vowel_spectra();
        let lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
        let hi = ((F2_HI / HZ_PER_BIN).floor() as usize)
            .min(spectrum.len().saturating_sub(1)).min(eq.len().saturating_sub(1));
        let centroid = detect_formants_eq(spectrum, eq);
        if lo >= hi {
            let (vowel, second, confidence) = classify_vowel_f2_ranked(centroid, &EXPECTED_F2_CENTROIDS);
            return SymbolClass { vowel, second, confidence: normalize_confidence(confidence), centroid };
        }
        let observation: Vec<f64> = (lo..=hi)
            .map(|b| (spectrum[b] as f64 * eq[b]).max(0.0).sqrt()).collect();
        let norm = observation.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut scores = [0.0f64; NUM_VOWELS];
        for refs in bank {
            for (v, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0;
                let mut template_norm_sq = 0.0;
                for (k, b) in (lo..=hi).enumerate() {
                    let amplitude = (refs[v][b] as f64).max(0.0).sqrt();
                    dot += observation[k] * amplitude;
                    template_norm_sq += amplitude * amplitude;
                }
                let denominator = norm * template_norm_sq.sqrt();
                if denominator > 0.0 { *score = score.max(dot / denominator); }
            }
        }
        let (mut vowel, mut second) = (0, 0);
        let (mut first_score, mut second_score) = (f64::MIN, f64::MIN);
        for (v, &score) in scores.iter().enumerate() {
            if score > first_score {
                second_score = first_score;
                second = vowel;
                first_score = score;
                vowel = v;
            } else if score > second_score {
                second_score = score;
                second = v;
            }
        }
        SymbolClass { vowel, second, centroid,
            confidence: ((first_score - second_score) / (1.0 - second_score).max(1e-9)).clamp(0.0, 1.0) }
    }

    fn assert_same_class(actual: SymbolClass, expected: SymbolClass) {
        assert_eq!(actual.vowel, expected.vowel);
        assert_eq!(actual.second, expected.second);
        assert_eq!(actual.confidence.to_bits(), expected.confidence.to_bits());
        assert_eq!(actual.centroid.to_bits(), expected.centroid.to_bits());
    }

    #[test]
    fn template_dots_preserve_native_and_portable_arithmetic_order() {
        for kind in 0..4 {
            let observed: Vec<f64> = (0..160).map(|i| match kind {
                0 => 0.0,
                1 => (i as f64 + 1.0) / 37.0,
                // Cancellation exposes accidental reordering or fused
                // arithmetic that ordinary positive spectra could miss.
                2 => [1e16, 1.0, -1e16, -1.0, 0.125][i % 5],
                _ => [1e-100, 1e100, -1e-100, -1e100, 1.0][i % 5],
            }).collect();
            let templates: Vec<[f64; NUM_VOWELS]> = (0..160).map(|i| {
                std::array::from_fn(|v| [0.0, -0.0, 0.1, -0.3, 1.0000000000000002][(i + v) % 5])
            }).collect();
            for observed_len in [0, 1, 2, 3, 7, 8, 9, 31, 75, 76, 77, 155] {
                for template_len in [0, 1, 7, 8, 9, 76, 155] {
                    let observed = &observed[..observed_len];
                    let templates = &templates[..template_len];
                    let expected: [f64; NUM_VOWELS] = std::array::from_fn(|v| {
                        observed.iter().zip(templates).fold(0.0, |dot, (&value, row)| dot + value * row[v])
                    });
                    assert_eq!(template_dots(observed, templates).map(f64::to_bits), expected.map(f64::to_bits));
                    assert_eq!(template_dots_portable(observed, templates).map(f64::to_bits), expected.map(f64::to_bits));
                }
            }
        }
    }

    #[test]
    fn interleaved_templates_preserve_every_amplitude_and_prefix_norm() {
        let lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
        let hi = (F2_HI / HZ_PER_BIN).floor() as usize;
        let bank = expected_vowel_amplitudes();
        assert_eq!(bank.len(), TEMPLATE_NOTES.len());
        for (note, spectra) in expected_vowel_spectra().iter().enumerate() {
            assert_eq!(bank[note].values.len(), hi - lo + 1);
            assert_eq!(bank[note].norms.len(), hi - lo + 1);
            for v in 0..NUM_VOWELS {
                let mut norm_sq = 0.0f64;
                for bin in lo..=hi {
                    let amplitude = (spectra[v][bin] as f64).max(0.0).sqrt();
                    norm_sq += amplitude * amplitude;
                    assert_eq!(bank[note].values[bin - lo][v].to_bits(), amplitude.to_bits());
                    assert_eq!(bank[note].norms[bin - lo][v].to_bits(), norm_sq.sqrt().to_bits());
                }
            }
        }
    }

    #[test]
    fn interleaved_templates_preserve_mixtures_and_every_partial_band() {
        let bank = expected_vowel_spectra();
        let hi = (F2_HI / HZ_PER_BIN).floor() as usize;
        let flat_eq = vec![1.0; FFT_SIZE];
        // Exercise each partial spectrum and EQ boundary, not only the
        // complete band used by normal decoding. Silence also tests ties.
        for length in 0..=hi + 2 {
            for vowel in 0..NUM_VOWELS {
                let spectrum = &bank[7][vowel][..length];
                assert_same_class(classify_symbol_template(spectrum, &flat_eq),
                    uncached_template_classifier(spectrum, &flat_eq));
                if length > 0 {
                    assert_same_class(classify_symbol_template(&bank[7][vowel], &flat_eq[..length]),
                        uncached_template_classifier(&bank[7][vowel], &flat_eq[..length]));
                }
            }
            let silence = vec![0.0; length];
            assert_same_class(classify_symbol_template(&silence, &flat_eq),
                uncached_template_classifier(&silence, &flat_eq));
        }
        // Competing notes/vowels and almost-equal blends stress ranking,
        // runner-up selection, and confidence near decision boundaries.
        for note in 0..TEMPLATE_NOTES.len() {
            for vowel in 0..NUM_VOWELS {
                for weight in [0.001, 0.25, 0.4999, 0.5, 0.5001, 0.75, 0.999] {
                    let spectrum: Vec<f32> = bank[note][vowel].iter()
                        .zip(&bank[(note + 7) % bank.len()][(vowel + 1) % NUM_VOWELS])
                        .map(|(&a, &b)| ((1.0 - weight) * a as f64 + weight * b as f64) as f32)
                        .collect();
                    for tilt in [-50.0, 0.0, 40.0] {
                        let eq: Vec<f64> = (0..FFT_SIZE).map(|bin| {
                            let t = ((bin as f64 * HZ_PER_BIN - F2_LO) / (F2_HI - F2_LO)).clamp(0.0, 1.0);
                            10.0f64.powf(tilt * t / 10.0)
                        }).collect();
                        assert_same_class(classify_symbol_template(&spectrum, &eq),
                            uncached_template_classifier(&spectrum, &eq));
                    }
                }
            }
        }
    }

    #[test]
    fn cached_templates_preserve_all_classification_fields_bit_for_bit() {
        for midi in 47..=66 {
            for detune in [-0.3, 0.0, 0.3] {
                let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                for vowel in 0..NUM_VOWELS {
                    let slot = synthesize_symbol(vowel, hz, 0.5, midi as usize);
                    let voiced: Vec<f32> = slot[..SAMPLES_PER_SYMBOL].iter().map(|&x| x as f32).collect();
                    let mut spectrum = vec![0.0; voiced.len()];
                    crate::fft::power_spectrum(&voiced, &mut spectrum);
                    for tilt in [-50.0, 0.0, 40.0] {
                        let eq: Vec<f64> = (0..spectrum.len()).map(|bin| {
                            let t = ((bin as f64 * HZ_PER_BIN - F2_LO) / (F2_HI - F2_LO)).clamp(0.0, 1.0);
                            10.0f64.powf(tilt * t / 10.0)
                        }).collect();
                        assert_same_class(classify_symbol_template(&spectrum, &eq),
                            uncached_template_classifier(&spectrum, &eq));
                    }
                }
            }
        }
    }

    #[test]
    fn cached_templates_preserve_partial_bands_silence_and_interference() {
        let mut state = 18573u32;
        for spectrum_len in [0, 1, 50, 51, 70, 125, 126, FFT_SIZE] {
            for eq_len in [1, 50, 51, 70, 125, 126, FFT_SIZE] {
                for kind in 0..3 {
                    let spectrum: Vec<f32> = (0..spectrum_len).map(|bin| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        match kind {
                            0 => 0.0,
                            1 => (state as f64 / u32::MAX as f64) as f32,
                            _ => if bin % 7 == 0 { 100.0 } else { 0.001 },
                        }
                    }).collect();
                    let eq: Vec<f64> = (0..eq_len).map(|bin| 10.0f64.powf((bin % 9) as f64 - 4.0)).collect();
                    assert_same_class(classify_symbol_template(&spectrum, &eq),
                        uncached_template_classifier(&spectrum, &eq));
                }
            }
        }
    }

    #[test]
    fn test_bit_packing_roundtrip() {
        for len in 1..=20 {
            let bytes: Vec<u8> = (0..len).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let symbols = bytes_to_symbols(&bytes);
            assert!(symbols.iter().all(|&s| s < NUM_SYMBOLS));
            let back = symbols_to_bytes(&symbols, len);
            assert_eq!(bytes, back, "packing roundtrip failed for len {len}");
        }
    }

    #[test]
    fn test_symbols_for_bytes_exact() {
        // 2 bits per symbol: exactly 4 symbols per byte, no padding slop
        for n in 1..=64 {
            assert_eq!(symbols_for_bytes(n), n * 4);
        }
    }

    #[test]
    fn test_melody_notes_are_candidates() {
        // Every melody note must be an F0 detection candidate
        for pos in 0..MELODY.len() {
            let f0 = melody_f0(pos);
            assert!(
                F0_CANDIDATES.iter().any(|&c| (c - f0).abs() < 0.01),
                "melody note {f0} missing from F0_CANDIDATES"
            );
        }
    }

    #[test]
    fn test_eq_recovers_large_power_tilt() {
        let attenuate = |psd: &[f32]| -> Vec<f32> {
            psd.iter().enumerate().map(|(bin, &power)| {
                let freq = bin as f64 * HZ_PER_BIN;
                let db = -50.0 * ((F2_HI - freq) / (F2_HI - F2_LO)).clamp(0.0, 1.0);
                power * 10.0f64.powf(db / 10.0) as f32
            }).collect()
        };
        let measured: Vec<_> = expected_pilot_spectra(EccProfile::Standard).iter().enumerate()
            .map(|(i, psd)| (i, attenuate(psd))).collect();
        let eq = estimate_eq(&measured).unwrap();
        assert!(eq[52] / eq[121] > 10_000.0, "50 dB power tilt must not clip at 30 dB");
        for (note, spectra) in expected_vowel_spectra().iter().enumerate() {
            for (vowel, spectrum) in spectra.iter().enumerate() {
                let expected = detect_formants(spectrum, 0.0);
                let recovered = detect_formants_eq(&attenuate(spectrum), &eq);
                assert!((recovered - expected).abs() < 65.0,
                    "note {note}, vowel {vowel}: centroid {expected:.1} recovered as {recovered:.1}");
            }
        }
    }

    #[test]
    fn test_all_vowels_classify_at_all_melody_notes() {
        use crate::fft;
        for &f0 in &F0_CANDIDATES {
            for vowel in 0..NUM_VOWELS {
                let slot = synthesize_symbol(vowel, f0, 0.5, 0);
                let voiced: Vec<f32> =
                    slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| s as f32).collect();
                let mut spectrum = vec![0.0f32; voiced.len()];
                fft::power_spectrum(&voiced, &mut spectrum);
                let (detected, conf) = classify_symbol(&spectrum, &EXPECTED_F2_CENTROIDS);
                assert_eq!(
                    detected, vowel,
                    "vowel {vowel} at F0={f0} misclassified as {detected} (conf {conf:.2})"
                );
            }
        }
    }
}
