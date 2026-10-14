//! Decoder: f32 audio samples → payload (vowel modem, streaming).
//!
//! Symbols are vowel indices; pitch carries no data and is ignored entirely
//! by the primary (F2-centroid) detection path.

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::reed_solomon::{PrefixDecoder, ReedSolomon, RetrySyndromes};

/// Preamble vowel patterns for start detection (robust to misalignment).
const PREAMBLE_START_VOWELS: [usize; PREAMBLE_LEN] = [0, 3, 0, 3];

/// Preamble vowel patterns for end detection.
const PREAMBLE_END_VOWELS: [usize; PREAMBLE_LEN] = [3, 0, 3, 0];

/// Number of history windows to keep during Listening for alignment recovery.
///
/// Must cover preamble + pilot + margin: under channel tilt the Listening
/// lock can fire LATE — the pilot itself contains a [2,3,2,3] run that looks
/// structurally like a preamble — and the recording must still retain the
/// true preamble and pilot for analyze()'s candidate scan and EQ estimation.
const HISTORY_WINDOWS: usize = PREAMBLE_LEN + PILOT_LEN + 4;

/// Keep the ordinary two-slot alignment search available just before the
/// next song's boundary, even when the previous frame had no end marker.
const DELIVERY_CONTEXT_WINDOWS: usize = 2;

/// A quiet vowel is still voiced. Spectral shaping can make a run of low
/// vowels much quieter than its neighbors, so relative level alone cannot
/// distinguish payload from the silence after a transmission. Check for a
/// periodic waveform over the supported singing register before finalizing.
pub(crate) fn has_voiced_periodicity(samples: &[f32]) -> bool {
    // Synthesis is band-limited below 4 kHz; this also gives both sample-rate
    // modes the same work and lag resolution (one sample at 8 kHz).
    let stride = (SAMPLE_RATE / 8000.0) as usize;
    let rate = SAMPLE_RATE / stride as f64;
    let first_lag = (rate / 380.0).floor() as usize;
    let last_lag = (rate / 120.0).ceil() as usize;
    let count = samples.len() / stride;
    if count <= 2 * last_lag {
        return false;
    }
    let mean = (0..count).map(|i| samples[i * stride] as f64).sum::<f64>() / count as f64;
    for lag in first_lag..=last_lag {
        let mut dot = 0.0;
        let mut left = 0.0;
        let mut right = 0.0;
        let mut left_sum = 0.0;
        let mut right_sum = 0.0;
        for i in lag..count {
            let a = samples[i * stride] as f64 - mean;
            let b = samples[(i - lag) * stride] as f64 - mean;
            dot += a * b;
            left += a * a;
            right += b * b;
            left_sum += a;
            right_sum += b;
        }
        // Center each overlap so a DC offset cannot pass as a sung note.
        let n = (count - lag) as f64;
        dot -= left_sum * right_sum / n;
        left -= left_sum * left_sum / n;
        right -= right_sum * right_sum / n;
        if left > 0.0 && right > 0.0 && dot > 0.35 * (left * right).sqrt() {
            return true;
        }
    }
    false
}

/// Minimum low→high centroid swing (Hz) for structural preamble detection.
///
/// The preamble alternates vowels 0 and 3 (~1 kHz nominal centroid
/// separation). Exact vowel matching fails when a channel filter tilts the
/// F2 band and shifts every centroid; the *alternation structure* survives
/// any monotone spectral shaping, so a lo-hi-lo-hi centroid pattern with a
/// healthy swing is accepted as a preamble even when classification against
/// the default centers disagrees. A channel that compresses the 0↔3 swing
/// below 250 Hz cannot carry data regardless.
const STRUCT_MIN_SWING_HZ: f64 = 250.0;

/// Maximum centroid spread (Hz) between the two windows carrying the same
/// preamble vowel in the structural pattern check.
const STRUCT_MAX_SPREAD_HZ: f64 = 200.0;

/// Structural match for the start preamble [0, 3, 0, 3]: lo-hi-lo-hi.
pub(crate) fn structural_start_match(c: &[f64]) -> bool {
    c.len() == PREAMBLE_LEN
        && c[1] > c[0] + STRUCT_MIN_SWING_HZ
        && c[3] > c[2] + STRUCT_MIN_SWING_HZ
        && (c[0] - c[2]).abs() < STRUCT_MAX_SPREAD_HZ
        && (c[1] - c[3]).abs() < STRUCT_MAX_SPREAD_HZ
}

/// Structural match for the end preamble [3, 0, 3, 0]: hi-lo-hi-lo.
fn structural_end_match(c: &[f64]) -> bool {
    c.len() == PREAMBLE_LEN
        && c[0] > c[1] + STRUCT_MIN_SWING_HZ
        && c[2] > c[3] + STRUCT_MIN_SWING_HZ
        && (c[0] - c[2]).abs() < STRUCT_MAX_SPREAD_HZ
        && (c[1] - c[3]).abs() < STRUCT_MAX_SPREAD_HZ
}

/// Limit blind/pattern candidates to a short prefix. Later starts require
/// the sliding sync/header check; an unrestricted pattern search mostly
/// surfaces preamble-shaped data runs, especially under channel tilt.
const PREAMBLE_SCAN_WINDOWS: usize = 24;

/// Sliding header-search span, also retained across a failed recording's
/// deadline. Include sync, the encoded length, and alignment margin after
/// every candidate position so a new frame can straddle a rollover.
const HEADER_PROBE_WINDOWS: usize = PREAMBLE_SCAN_WINDOWS + PREAMBLE_LEN + PILOT_LEN
    + ENCODED_DATA_OFFSET * (8 / BITS_PER_SYMBOL) + 2;

/// Failure-only, frame-anchored clock grids; one best phase per nominal slot.
const MAX_CLOCK_CANDIDATES: usize = 32;
/// Final, CRC-checked phase search when echoes obscure guard timing.
const MAX_ENVELOPE_CLOCK_PHASES: usize = 32;
/// Extra non-finalizing probes prompted by quiet tails, per recording.
const MAX_QUIET_COMPLETION_PROBES: usize = 4;

/// Candidate start-preamble positions in a classified window stream.
///
/// Exact vowel-pattern matches come first, then structural centroid matches
/// in position order (the true preamble is the earliest plausible match by
/// construction). The structural test here is deliberately LOOSE — every
/// candidate is verified by an RS-gated decode attempt, so a false candidate
/// costs compute, never correctness; a missed true candidate costs the
/// message.
fn preamble_candidates(symbols: &[usize], centroids: &[f64], max_scan: usize) -> Vec<usize> {
    let scan = max_scan.min(PREAMBLE_SCAN_WINDOWS);
    let mut out = Vec::new();
    for pos in 0..scan {
        if symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_START_VOWELS {
            out.push(pos);
            if out.len() >= 2 {
                break;
            }
        }
    }
    for pos in 0..scan {
        if out.len() >= 5 {
            break;
        }
        let c = &centroids[pos..pos + PREAMBLE_LEN];
        let loose = c[1] > c[0] + 150.0
            && c[3] > c[2] + 150.0
            && (c[0] - c[2]).abs() < 400.0
            && (c[1] - c[3]).abs() < 400.0;
        if loose && !out.contains(&pos) {
            out.push(pos);
        }
    }
    out
}

/// Confidence-weighted per-symbol vote across classification streams.
///
/// The rungs of the retry ladder fail at DIFFERENT positions (centroid vs
/// template vs per-note centers have distinct failure modes), so a majority
/// vote across them has a substantially lower symbol error rate than any
/// single rung when the channel is at the edge of all of them.
fn ensemble_vote(streams: &[(&[usize], &[f64])]) -> (Vec<usize>, Vec<f64>, Vec<usize>) {
    let n = streams.iter().map(|(s, _)| s.len()).min().unwrap_or(0);
    let mut syms = Vec::with_capacity(n);
    let mut confs = Vec::with_capacity(n);
    let mut seconds = Vec::with_capacity(n);
    for i in 0..n {
        let mut score = [0.0f64; NUM_SYMBOLS];
        for (s, c) in streams {
            if s[i] < NUM_SYMBOLS {
                // Every vote counts at least a little, so a confident
                // minority can't be silently outvoted by two zero-confidence
                // agreeing streams — and vice versa.
                score[s[i]] += c[i].max(0.05);
            }
        }
        let mut best = 0usize;
        let mut second = 0usize;
        let mut s1 = f64::MIN;
        let mut s2 = f64::MIN;
        for (v, &sc) in score.iter().enumerate() {
            if sc > s1 {
                s2 = s1;
                second = best;
                s1 = sc;
                best = v;
            } else if sc > s2 {
                s2 = sc;
                second = v;
            }
        }
        syms.push(best);
        seconds.push(second);
        let s2 = s2.max(0.0);
        confs.push(((s1 - s2) / (s1 + s2 + 1e-9)).clamp(0.0, 1.0));
    }
    (syms, confs, seconds)
}

// Classifiers inspect the formant bands and harmonics up to the ceiling,
// with ±2 neighboring bins. Allow for rounding, too. Retain the full FFT
// output only if it is already shorter than this prefix.
const RETAINED_SPECTRUM_BINS: usize = {
    let upper = if HARMONIC_CEILING > F2_HI { HARMONIC_CEILING } else { F2_HI };
    let needed = (upper / HZ_PER_BIN) as usize + 4;
    if needed < SAMPLES_PER_SYMBOL { needed } else { SAMPLES_PER_SYMBOL }
};

/// Spectra for one immutable recording slice and one sample alignment.
/// Compute each voiced window once, retain only classifier-readable bins,
/// and drop all results when the alignment changes. Borrowing the recording
/// prevents accidentally reusing spectra after an append, rollover, or reset.
struct AlignedSpectra<'a> {
    audio: &'a [f32],
    offset: usize,
    clock_starts: Option<Vec<usize>>,
    windows: Vec<Option<[f32; RETAINED_SPECTRUM_BINS]>>,
    note_rows: Vec<Option<[f64; formant::BANK_NOTE_COUNT]>>,
    scratch: Vec<f32>,
    frequency_ratio: f64,
    #[cfg(test)]
    transforms: usize,
}

/// Normalize the F2 template band within the retained spectrum. Frequencies
/// beyond that cache are unavailable, rather than extrapolated.
fn normalize_frequency_bins(raw: &[f32], ratio: f64) -> [f32; RETAINED_SPECTRUM_BINS] {
    let mut warped = [0.0; RETAINED_SPECTRUM_BINS];
    for (bin, power) in warped.iter_mut().enumerate() {
        let position = bin as f64 * ratio;
        let lo = position as usize;
        if lo + 1 < raw.len() {
            let fraction = (position - lo as f64) as f32;
            *power = raw[lo] * (1.0 - fraction) + raw[lo + 1] * fraction;
        }
    }
    warped
}

/// Estimate one stable clock from the repeating voiced/guard amplitude
/// envelope. An eight-slot estimate disambiguates a 24-slot refinement.
/// Short captures may fall back to 16-, 12-, then eight-slot spans if a longer
/// measurement fails its stability checks; no threshold is relaxed.
/// reducing pitch/vowel-dependent bias without selecting a neighboring beat.
/// Both halves must independently support the same bounded rate; neither
/// this estimate nor its rhythm correlation constitutes a frame anchor.
fn envelope_clock_period(audio: &[f32]) -> Option<f64> {
    if let Some(period) = envelope_clock_period_with_lags(audio, 8, 24) { return Some(period); }
    // Fewer than four long-lag repeats per half: note/vowel changes can
    // obscure a long-span correlation even when shorter spans agree. Keep
    // the original estimate (including rejection) for longer recordings.
    let short_limit = 8 * 24 * (SYMBOL_TOTAL_SAMPLES + crate::audio::CLOCK_RADIUS);
    if audio.len() >= short_limit { return None; }
    envelope_clock_period_with_lags(audio, 8, 16)
        .or_else(|| envelope_clock_period_with_lags(audio, 4, 12))
        .or_else(|| envelope_clock_period_with_lags(audio, 4, 8))
}

fn envelope_clock_period_with_lags(audio: &[f32], multiple: usize, refinement: usize) -> Option<f64> {
    let step = SAMPLE_RATE as usize / 2000;
    let middle = multiple * SYMBOL_TOTAL_SAMPLES / step;
    let radius = multiple * crate::audio::CLOCK_RADIUS / step;
    if audio.len() / step < 4 * refinement * (SYMBOL_TOTAL_SAMPLES + crate::audio::CLOCK_RADIUS) / step {
        return None;
    }
    let raw: Vec<_> = audio.chunks_exact(step).map(|chunk|
        chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / step as f64).collect();
    let smooth: Vec<_> = raw.windows(6).map(|w| w.iter().sum::<f64>() / 6.0).collect();
    let estimate = |envelope: &[f64], middle: usize, radius: usize, multiple: usize| -> Option<f64> {
        let mean = envelope.iter().sum::<f64>() / envelope.len() as f64;
        let energy: Vec<_> = envelope.iter().map(|x| x - mean).collect();
        let lo = middle - radius;
        let scores: Vec<_> = (lo..=middle + radius).map(|lag| {
            let (mut dot, mut a, mut b) = (0.0, 0.0, 0.0);
            for (&x, &y) in energy.iter().zip(&energy[lag..]) {
                dot += x * y;
                a += x * x;
                b += y * y;
            }
            dot / (a.sqrt() * b.sqrt())
        }).collect();
        if scores.iter().any(|score| !score.is_finite()) { return None; }
        let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))?;
        if best == 0 || best + 1 == scores.len() || scores[best] < 0.2 { return None; }
        let (a, b, c) = (scores[best - 1], scores[best], scores[best + 1]);
        let curvature = a - 2.0 * b + c;
        if curvature >= 0.0 { return None; }
        let fraction = 0.5 * (a - c) / curvature;
        if fraction.abs() > 0.5 { return None; }
        Some(((lo + best) as f64 + fraction) * step as f64 / multiple as f64)
    };
    let split = smooth.len() / 2;
    let refine = |envelope: &[f64]| -> Option<f64> {
        let coarse = estimate(envelope, middle, radius, multiple)?;
        let multiple = refinement;
        let middle = (coarse * multiple as f64 / step as f64).round() as usize;
        let radius = multiple * GUARD_SAMPLES / (40 * step);
        estimate(envelope, middle, radius, multiple)
    };
    let left = refine(&smooth[..split])?;
    let right = refine(&smooth[split..])?;
    let nominal = SYMBOL_TOTAL_SAMPLES as f64;
    let period = 0.5 * (left + right);
    let drift = period - nominal;
    if drift.abs() <= nominal * 0.005 || drift.abs() > crate::audio::CLOCK_RADIUS as f64
        || (left - nominal) * (right - nominal) <= 0.0
        || (left - right).abs() > GUARD_SAMPLES as f64 / 40.0 {
        return None;
    }
    Some(period)
}

/// A checked result and the end of the last voiced window that carried it.
/// The recording can already contain the beginning of another transmission.
struct DecodedFrame {
    payload: Vec<u8>,
    audio_end: usize,
}

/// A verified clock recovery and the anchored observations that produced it.
type ClockRecovery = (DecodedFrame, Vec<usize>, Vec<f64>);

/// One non-nominal clock phase verified by the bounded header probe.
#[derive(Clone, Copy)]
struct ClockHint {
    score: f64,
    offset: usize,
    length: usize,
}

impl<'a> AlignedSpectra<'a> {
    fn new(audio: &'a [f32], offset: usize, max_windows: usize) -> Self {
        let available = audio.len().checked_sub(offset)
            .and_then(|len| len.checked_sub(SAMPLES_PER_SYMBOL))
            .map_or(0, |tail| tail / SYMBOL_TOTAL_SAMPLES + 1);
        Self {
            audio,
            offset,
            clock_starts: None,
            windows: (0..available.min(max_windows)).map(|_| None).collect(),
            note_rows: Vec::new(),
            scratch: vec![0.0; SAMPLES_PER_SYMBOL],
            frequency_ratio: 1.0,
            #[cfg(test)]
            transforms: 0,
        }
    }

    fn clocked(audio: &'a [f32], offset: usize, max_windows: usize) -> Self {
        let mut starts = Vec::new();
        let mut start = offset;
        let mut period = SYMBOL_TOTAL_SAMPLES as f64;
        let mut rms_ema = 0.0;
        while audio.len().saturating_sub(start) >= SAMPLES_PER_SYMBOL && starts.len() < max_windows {
            starts.push(start);
            if audio.len() - start < SYMBOL_TOTAL_SAMPLES { break; }
            let window = &audio[start..start + SYMBOL_TOTAL_SAMPLES];
            let voiced = &window[..SAMPLES_PER_SYMBOL];
            let rms = (voiced.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / voiced.len() as f64).sqrt();
            let silent = rms_ema > 0.0 && rms < 0.3 * rms_ema && !has_voiced_periodicity(voiced);
            let evidence = (!silent).then(|| crate::audio::guard_clock_evidence(window)).flatten();
            // Smooth phase corrections so a noisy guard cannot pull the
            // whole next window off-grid. Learn the rate before subsequent
            // weak/obscured guards need to coast on it.
            let step = evidence.map_or_else(|| period.round() as usize,
                |measured| (0.5 * measured as f64 + 0.5 * period).round() as usize);
            if let Some(measured) = evidence {
                period += 0.1 * (measured as f64 - period);
            }
            if !silent {
                rms_ema = if rms_ema == 0.0 { rms } else { 0.9 * rms_ema + 0.1 * rms };
            }
            start += step;
        }
        let mut spectra = Self::new(audio, offset, 0);
        spectra.windows.resize(starts.len(), None);
        spectra.clock_starts = Some(starts);
        spectra
    }

    /// A sample-clock mismatch scales formant frequencies as well as note
    /// timing. Reuse the cached FFTs for one frequency-normalized retry when
    /// both halves of this complete frame agree on a non-nominal clock.
    fn frequency_compensated(&mut self, frame_start: usize, frame_end: usize) -> Option<Self> {
        if self.frequency_ratio != 1.0 { return None; }
        let starts = self.clock_starts.as_ref()?;
        let first = frame_start.checked_add(PREAMBLE_LEN + PILOT_LEN)?;
        let last = frame_end.checked_sub(1)?;
        if last >= starts.len() || last < first + 16 { return None; }
        let nominal = SYMBOL_TOTAL_SAMPLES as f64;
        let period = (starts[last] - starts[first]) as f64 / (last - first) as f64;
        let drift = period - nominal;
        if drift.abs() <= 0.005 * nominal || drift.abs() > crate::audio::CLOCK_RADIUS as f64 {
            return None;
        }
        let middle = (first + last) / 2;
        let left = (starts[middle] - starts[first]) as f64 / (middle - first) as f64;
        let right = (starts[last] - starts[middle]) as f64 / (last - middle) as f64;
        if (left - nominal) * drift <= 0.0 || (right - nominal) * drift <= 0.0
            || (left - right).abs() > 0.5 * drift.abs()
            || (left - right).abs() > crate::audio::CLOCK_RADIUS as f64 / 2.0 {
            return None;
        }
        let ratio = nominal / period;
        let mut corrected = Self::new(self.audio, self.offset, 0);
        corrected.clock_starts = Some(starts[..frame_end].to_vec());
        corrected.frequency_ratio = ratio;
        corrected.windows = self.windows[..frame_end].iter()
            .map(|window| window.as_ref().map(|raw| normalize_frequency_bins(raw, ratio))).collect();
        Some(corrected)
    }

    fn uniform_clocked(audio: &'a [f32], offset: usize, period: f64) -> Self {
        let nominal = SYMBOL_TOTAL_SAMPLES as f64;
        let radius = crate::audio::CLOCK_RADIUS as f64;
        if !period.is_finite() || !(nominal - radius..=nominal + radius).contains(&period) {
            return Self::new(audio, offset, 0);
        }
        let mut starts = Vec::new();
        loop {
            let start = (offset as f64 + starts.len() as f64 * period).round() as usize;
            if audio.len().saturating_sub(start) < SAMPLES_PER_SYMBOL { break; }
            starts.push(start);
        }
        let mut spectra = Self::new(audio, offset, 0);
        spectra.windows.resize(starts.len(), None);
        spectra.clock_starts = Some(starts);
        spectra
    }

    fn get(&mut self, index: usize) -> Option<&[f32]> {
        let window = self.windows.get_mut(index)?;
        if window.is_none() {
            let start = self.clock_starts.as_ref().map_or_else(
                || self.offset + index * SYMBOL_TOTAL_SAMPLES, |starts| starts[index],
            );
            fft::power_spectrum(&self.audio[start..start + SAMPLES_PER_SYMBOL], &mut self.scratch);
            let mut retained = [0.0; RETAINED_SPECTRUM_BINS];
            retained.copy_from_slice(&self.scratch[..RETAINED_SPECTRUM_BINS]);
            if self.frequency_ratio != 1.0 {
                retained = normalize_frequency_bins(&retained, self.frequency_ratio);
            }
            *window = Some(retained);
            #[cfg(test)]
            {
                self.transforms += 1;
            }
        }
        window.as_ref().map(|bins| bins.as_slice())
    }

    /// Per-note melody-match row for one window, cached alongside the
    /// spectrum. Feeds only the melody-derived length hint; vowel
    /// classification never reads it.
    fn note_scores(&mut self, index: usize) -> Option<[f64; formant::BANK_NOTE_COUNT]> {
        if index >= self.windows.len() {
            return None;
        }
        if self.note_rows.len() < self.windows.len() {
            self.note_rows.resize(self.windows.len(), None);
        }
        if self.note_rows[index].is_none() {
            let row = formant::note_match_scores(self.get(index)?);
            self.note_rows[index] = Some(row);
        }
        self.note_rows[index]
    }

    fn decoded_frame(&self, payload: Vec<u8>, data_start: usize, profile: EccProfile) -> DecodedFrame {
        let last = data_start + formant::symbols_for_bytes(profile.encoded_bytes(payload.len())) - 1;
        // Every successful RS interpretation includes its entire codeword.
        assert!(last < self.windows.len());
        let start = self.clock_starts.as_ref().map_or_else(
            || self.offset + last * SYMBOL_TOTAL_SAMPLES, |starts| starts[last],
        );
        DecodedFrame { payload, audio_end: start + SAMPLES_PER_SYMBOL }
    }

    fn classify_full(&mut self, index: usize, centers: &[f64; NUM_VOWELS]) -> Option<formant::SymbolClass> {
        self.get(index).map(|spectrum| formant::classify_symbol_full(spectrum, centers))
    }

    fn classify_eq(&mut self, index: usize, eq: &[f64]) -> Option<formant::SymbolClass> {
        self.get(index).map(|spectrum| formant::classify_symbol_eq(spectrum, eq))
    }

    fn classify_template(&mut self, index: usize, eq: &[f64]) -> Option<formant::SymbolClass> {
        self.get(index).map(|spectrum| formant::classify_symbol_template(spectrum, eq))
    }

    fn classify_f0a(&mut self, index: usize, eq: &[f64]) -> Option<formant::SymbolClass> {
        self.get(index).map(|spectrum| formant::classify_symbol_f0_adaptive(spectrum, eq))
    }

    fn pilot_centers(&mut self, pilot_start: usize) -> Option<[f64; NUM_VOWELS]> {
        let mut measured = Vec::with_capacity(PILOT_LEN);
        for (k, &vowel) in PILOT.iter().enumerate() {
            let spectrum = self.get(pilot_start + k)?;
            let f0 = formant::detect_f0(spectrum);
            measured.push((vowel, formant::detect_formants(spectrum, f0)));
        }
        calibrate_centers(&measured)
    }

    fn pilot_eq(&mut self, pilot_start: usize, profile: EccProfile) -> Option<Vec<f64>> {
        let mut measured = Vec::with_capacity(PILOT_LEN);
        for k in 0..PILOT_LEN {
            measured.push((k, self.get(pilot_start + k)?.to_vec()));
        }
        formant::estimate_eq_with_profile(&measured, profile)
    }
}

/// Streaming decoder for the 2-channel orthogonal vocal modem.
pub struct Decoder {
    state: DecoderState,
    sample_buffer: Vec<f32>,
    recent_vowels: Vec<usize>,
    recent_symbols: Vec<usize>,
    recorded_audio: Vec<f32>,
    symbols_recorded: usize,
    max_record_symbols: usize,
    /// Ring buffer of recent windows during Listening for alignment recovery.
    history: Vec<Vec<f32>>,
    /// Already-delivered lookbehind, used only as contiguous analysis
    /// context for the next acquisition, never replayed through Listening.
    delivery_context: Vec<f32>,
    /// Accumulated classified symbols during Receiving (for best-effort preview).
    preview_symbols: Vec<usize>,
    /// Counter for consecutive silence windows (for gap detection).
    silence_count: usize,
    /// Counter for consecutive active windows without preamble match.
    active_count: usize,
    /// ECC policy (must match the sender's).
    profile: EccProfile,
    /// Active F2 classification centers (pilot-calibrated when possible).
    centers: [f64; NUM_VOWELS],
    /// Pilot windows still expected at the start of Receiving.
    pilot_remaining: usize,
    /// Measured pilot windows: (pilot index, centroid, power spectrum).
    pilot_measured: Vec<(usize, f64, Vec<f32>)>,
    /// Pilot-derived channel EQ (F2-band inverse gain). Like `calibrated`,
    /// used strictly as a decode RETRY after default and calibrated-center
    /// classification fail — it can rescue a tilted channel but must never
    /// preempt a decode that already works.
    eq: Option<Vec<f64>>,
    /// Centroids parallel to `recent_vowels` (Listening) for structural
    /// preamble detection under channel tilt.
    recent_centroids: Vec<f64>,
    /// Centroids of recent Receiving windows for structural end detection.
    recent_rx_centroids: Vec<f64>,
    /// Known start-preamble position in the recording (window index), when
    /// the Listening lock saw the preamble directly: history_len - 4.
    preamble_pos_hint: Option<usize>,
    /// Pilot-calibrated centers, used only as a decode RETRY — primary
    /// classification always uses the defaults (a pilot measured through a
    /// denoiser's warm-up ramp can pass the sanity gate yet still be worse).
    calibrated: Option<[f64; NUM_VOWELS]>,
    /// Best-aligned data-region (symbols, confidences) from the last decode
    /// attempt (success or failure), for cross-reception soft combining.
    last_attempt: Option<(Vec<usize>, Vec<f64>)>,
    /// Optional payload validator (e.g. the transfer layer's CRC check).
    /// A trial decode that fails validation is treated as NO decode: the
    /// reception continues instead of terminating on plausible garbage —
    /// without this, a brute-force/Chase "success" mid-burst would reset
    /// the decoder and split the reception into unusable fragments.
    validator: Option<fn(&[u8]) -> bool>,
    /// Running RMS average of voiced windows during Receiving.
    rx_rms_ema: f64,
    /// Consecutive near-silent windows during Receiving.
    rx_silence_run: usize,
    /// A tentative energy drop can prompt checked completion, never finalization.
    rx_quiet_probe_run: usize,
    quiet_completion_probes: usize,
    /// Candidate completion windows inferred from a sliding sync/header scan.
    /// Timing hints only: a failed checked decode never ends reception.
    frame_end_hints: Option<Vec<usize>>,
    /// A failed timing probe must not hide a later sender behind its slow-clock deadline.
    rescan_after_failed_probe: bool,
    /// Sync positions discovered after an initial acquisition on background.
    /// Kept as alternatives; no original audio is discarded on a header guess.
    frame_start_hints: Vec<usize>,
    /// Intact headers found by the phased scan, keyed by frame start.
    /// None records conflicting supported lengths, not a guessed length.
    frame_length_hints: std::collections::BTreeMap<usize, Option<usize>>,
    /// Best locally supported drifting-clock phase, never an unchecked length.
    frame_clock_hint: Option<ClockHint>,
    #[cfg(test)]
    trial_length_searches: std::cell::Cell<usize>,
    #[cfg(test)]
    trial_failure_records: std::cell::Cell<usize>,
    #[cfg(test)]
    clock_grids: usize,
    #[cfg(test)]
    nominal_grids: usize,
    #[cfg(test)]
    priority_clock_grids: usize,
    #[cfg(test)]
    envelope_clock_phases: std::cell::Cell<usize>,
    #[cfg(test)]
    clock_candidate_peak: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    Listening,
    Receiving,
    Analyzing,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Receive CRC-protected Standard messages, without an unchecked fallback.
    pub fn new() -> Self {
        Self::with_profile(EccProfile::Standard)
    }

    /// Create a decoder with an explicit ECC profile (must match the sender).
    pub fn with_profile(profile: EccProfile) -> Self {
        Self {
            profile,
            centers: formant::EXPECTED_F2_CENTROIDS,
            pilot_remaining: 0,
            pilot_measured: Vec::new(),
            eq: None,
            recent_centroids: Vec::new(),
            recent_rx_centroids: Vec::new(),
            preamble_pos_hint: None,
            calibrated: None,
            last_attempt: None,
            validator: None,
            rx_rms_ema: 0.0,
            rx_silence_run: 0,
            rx_quiet_probe_run: 0,
            quiet_completion_probes: 0,
            frame_end_hints: None,
            rescan_after_failed_probe: false,
            frame_start_hints: Vec::new(),
            frame_length_hints: std::collections::BTreeMap::new(),
            frame_clock_hint: None,
            #[cfg(test)]
            trial_length_searches: std::cell::Cell::new(0),
            #[cfg(test)]
            trial_failure_records: std::cell::Cell::new(0),
            #[cfg(test)]
            clock_grids: 0,
            #[cfg(test)]
            nominal_grids: 0,
            #[cfg(test)]
            priority_clock_grids: 0,
            #[cfg(test)]
            envelope_clock_phases: std::cell::Cell::new(0),
            #[cfg(test)]
            clock_candidate_peak: 0,
            state: DecoderState::Listening,
            sample_buffer: Vec::new(),
            recent_vowels: Vec::new(),
            recent_symbols: Vec::new(),
            recorded_audio: Vec::new(),
            symbols_recorded: 0,
            max_record_symbols: 0,
            history: Vec::new(),
            delivery_context: Vec::new(),
            preview_symbols: Vec::new(),
            silence_count: 0,
            active_count: 0,
        }
    }

    pub fn reset(&mut self) {
        let validator = self.validator;
        *self = Self::with_profile(self.profile);
        self.validator = validator;
    }

    /// Install a payload validator (e.g. a CRC check). Decode results that
    /// fail it continue through erasure, alternate-length, and Chase retries.
    /// Mid-reception trial decodes keep receiving instead of terminating on
    /// plausible garbage, and analysis never returns a rejected payload.
    /// The predicate may be called repeatedly, including for the same bytes.
    pub fn set_payload_validator(&mut self, f: fn(&[u8]) -> bool) {
        self.validator = Some(f);
    }

    /// True while the decoder has locked onto a transmission (i.e. it is
    /// receiving or analyzing, not idle-listening for a preamble).
    pub fn is_receiving(&self) -> bool {
        self.state != DecoderState::Listening
    }

    /// Most recently retained start-anchored (symbols, confidences), for
    /// soft combining across repeated transmissions. An unanchored trial
    /// neither publishes an observation nor replaces an existing one.
    pub fn last_attempt(&self) -> Option<&(Vec<usize>, Vec<f64>)> {
        self.last_attempt.as_ref()
    }

    /// TransferRx takes ownership exactly once, so a later failure with no
    /// new anchored observation cannot harvest a previous reception again.
    pub(crate) fn take_last_attempt(&mut self) -> Option<(Vec<usize>, Vec<f64>)> {
        self.last_attempt.take()
    }

    /// Best-effort preview of symbols accumulated during the current Receiving phase.
    ///
    /// Converts classified (but uncorrected) symbols to raw bytes. Returns None
    /// if the decoder is not currently receiving or has too few symbols.
    pub fn preview_bytes(&self) -> Option<Vec<u8>> {
        if self.state != DecoderState::Receiving || self.preview_symbols.is_empty() {
            return None;
        }
        let max_bytes = (self.preview_symbols.len() * BITS_PER_SYMBOL) / 8;
        if max_bytes == 0 {
            return None;
        }
        Some(formant::symbols_to_bytes(&self.preview_symbols, max_bytes))
    }

    /// Feed audio, preserving its sample timing. NaN and infinities are
    /// treated as zero-valued dropouts; finite samples are not altered.
    pub fn decode(&mut self, samples: &[f32]) -> Result<Option<Vec<u8>>, crate::Error> {
        crate::audio::append_finite_samples(&mut self.sample_buffer, samples);

        while self.sample_buffer.len() >= SYMBOL_TOTAL_SAMPLES {
            let window: Vec<f32> = self.sample_buffer[..SYMBOL_TOTAL_SAMPLES].to_vec();
            self.sample_buffer.drain(..SYMBOL_TOTAL_SAMPLES);

            let result = self.process_symbol_window(&window)?;
            if result.is_some() {
                return Ok(result);
            }
        }

        Ok(None)
    }

    fn process_symbol_window(
        &mut self,
        window: &[f32],
    ) -> Result<Option<Vec<u8>>, crate::Error> {
        match self.state {
            DecoderState::Listening => {
                let measured = self.measure_window_vowel(window);

                // Track silence for gap detection. When we see sustained silence,
                // clear state so the next audio frame gets clean preamble detection.
                if measured.is_none() {
                    self.silence_count += 1;
                    self.active_count = 0;
                    // After four windows of silence (240 ms at either sample rate),
                    // clear history and recent_vowels to reset alignment.
                    if self.silence_count >= 4 {
                        self.history.clear();
                        self.delivery_context.clear();
                        self.recent_vowels.clear();
                        self.recent_centroids.clear();
                    }
                } else {
                    self.silence_count = 0;
                    self.active_count += 1;
                }

                // Keep history for alignment recovery
                self.history.push(window.to_vec());
                if self.history.len() > HISTORY_WINDOWS {
                    self.history.remove(0);
                    // Sliding history is no longer adjacent to delivery.
                    self.delivery_context.clear();
                }

                // Only track non-silence vowels for preamble detection.
                // This makes detection robust to misaligned windows at frame boundaries.
                if let Some((vowel, centroid)) = measured {
                    self.recent_vowels.push(vowel);
                    self.recent_centroids.push(centroid);
                    if self.recent_vowels.len() > PREAMBLE_LEN {
                        self.recent_vowels
                            .drain(..self.recent_vowels.len() - PREAMBLE_LEN);
                        self.recent_centroids
                            .drain(..self.recent_centroids.len() - PREAMBLE_LEN);
                    }

                    if self.recent_vowels.len() == PREAMBLE_LEN
                        && (self.recent_vowels[..] == PREAMBLE_START_VOWELS[..]
                            || structural_start_match(&self.recent_centroids))
                    {
                        self.start_receiving(true);
                    }
                }

                // Fallback: if we've seen many active windows without preamble match,
                // force into receiving mode. The multi-offset analysis will find alignment.
                // This handles misaligned frame boundaries (e.g., after dropped frame).
                // Threshold must be higher than preamble length (4 windows) to avoid false triggers.
                // 12 windows (720 ms) leave time for direct preamble detection.
                if self.active_count >= 12 && self.history.len() >= HISTORY_WINDOWS {
                    self.start_receiving(false);
                }

                Ok(None)
            }
            DecoderState::Receiving => {
                self.recorded_audio.extend_from_slice(window);
                self.symbols_recorded += 1;

                // Calibration pilot: measure each vowel's centroid through the
                // live channel; classify subsequent data against measured
                // centers instead of the static expected ones.
                if self.pilot_remaining > 0 {
                    let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
                    let mut spectrum = vec![0.0f32; voiced.len()];
                    fft::power_spectrum(voiced, &mut spectrum);
                    let f0 = formant::detect_f0(&spectrum);
                    let centroid = formant::detect_formants(&spectrum, f0);
                    let idx = PILOT_LEN - self.pilot_remaining;
                    self.pilot_measured.push((idx, centroid, spectrum));
                    self.pilot_remaining -= 1;
                    if self.pilot_remaining == 0 {
                        let pairs: Vec<(usize, f64)> = self
                            .pilot_measured
                            .iter()
                            .map(|&(k, c, _)| (PILOT[k], c))
                            .collect();
                        self.calibrated = calibrate_centers(&pairs);
                        let spectra: Vec<(usize, Vec<f32>)> = self
                            .pilot_measured
                            .iter()
                            .map(|(k, _, s)| (*k, s.clone()))
                            .collect();
                        self.eq = formant::estimate_eq_with_profile(&spectra, self.profile);
                    }
                    return Ok(None);
                }

                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];

                // End-of-transmission fallback: sustained silence. Without
                // this, a reception whose end preamble is corrupted would sit
                // in Receiving until the (large) recording buffer fills —
                // fatal for repeat-transmission soft combining.
                let rms = {
                    let sum_sq: f64 =
                        voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
                    (sum_sq / voiced.len() as f64).sqrt()
                };
                self.rx_quiet_probe_run = if self.rx_rms_ema > 0.0 && rms < 0.6 * self.rx_rms_ema {
                    self.rx_quiet_probe_run.saturating_add(1)
                } else { 0 };
                // Profile-dependent run length: the transfer flow (Paranoid)
                // needs corrupted receptions to finalize quickly between
                // repeats (4 windows ≈ 240 ms). The standard decoder demands
                // a longer run (6 windows ≈ 360 ms) because denoiser gain
                // dips and deep fading legitimately silence ~240 ms
                // mid-message — but a reception whose end preamble was
                // missed must STILL finalize on trailing silence, or it
                // sits in Receiving until the recording buffer fills.
                // A start-shaped outro fragment can acquire on its final
                // voiced window. Exact silence must release that acquisition
                // even if no later voice initialized the Receiving-level EMA.
                let is_silent = rms == 0.0 || (self.rx_rms_ema > 0.0
                    && rms < 0.3 * self.rx_rms_ema
                    && !has_voiced_periodicity(voiced));
                if is_silent {
                    self.rx_silence_run += 1;
                    let needed = if self.profile == EccProfile::Paranoid { 4 } else { 6 };
                    let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                    if self.rx_silence_run >= needed {
                        if self.symbols_recorded <= min_data_symbols + PREAMBLE_LEN + PILOT_LEN {
                            // No room beyond sync and a length header: this
                            // acquisition cannot contain a payload observation.
                            // Resume listening without inventing a failed copy
                            // or discarding unread samples from a later sender.
                            self.state = DecoderState::Listening;
                            self.recent_vowels.clear();
                            self.recent_centroids.clear();
                            return Ok(None);
                        }
                        self.state = DecoderState::Analyzing;
                        return self.analyze(true);
                    }
                } else {
                    self.rx_silence_run = 0;
                    self.rx_rms_ema = if self.rx_rms_ema == 0.0 {
                        rms
                    } else {
                        0.9 * self.rx_rms_ema + 0.1 * rms
                    };
                }

                // Filtering/denoising tails may stay periodic, and phone
                // noise may exceed the strict silence threshold. A sustained
                // energy drop near a supported frame's possible endpoint can
                // still prompt one CRC-checked probe. A quiet musical phrase
                // is never discarded when that probe fails.
                // With no trustworthy length, a three-slot gap may be the
                // only completion cue before another sender begins. Keep
                // the existing four-slot timing for known-length frames.
                if (self.rx_quiet_probe_run == 4 && self.has_possible_complete_frame())
                    || (self.rx_quiet_probe_run == 3 && self.can_probe_sync_only_tail())
                {
                    self.quiet_completion_probes += 1;
                    if let Ok(Some(payload)) = self.analyze(false) { return Ok(Some(payload)); }
                    self.rescan_after_failed_probe = true;
                }

                if is_silent && self.profile == EccProfile::Paranoid {
                    // Keep silence out of preview/end-detection streams,
                    // but only after the bounded, checked completion probe.
                    // A failed third-slot probe still reaches the original
                    // fourth-slot finalization above on the next window.
                    return Ok(None);
                }

                let class = self.classify_voiced_full(voiced);
                self.preview_symbols.push(class.vowel);
                self.recent_symbols.push(class.vowel);
                self.recent_rx_centroids.push(class.centroid);
                if self.recent_symbols.len() > PREAMBLE_LEN {
                    self.recent_symbols
                        .drain(..self.recent_symbols.len() - PREAMBLE_LEN);
                    self.recent_rx_centroids
                        .drain(..self.recent_rx_centroids.len() - PREAMBLE_LEN);
                }

                let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                let end_detected = self.recent_symbols.len() == PREAMBLE_LEN
                    && (self.recent_symbols[..] == PREAMBLE_END_VOWELS[..]
                        || structural_end_match(&self.recent_rx_centroids));
                if end_detected
                    && self.symbols_recorded > min_data_symbols + PREAMBLE_LEN
                {
                    let trial_n = self.symbols_recorded - PREAMBLE_LEN;
                    if trial_n >= min_data_symbols {
                        if let Some(frame) = self.trial_decode(trial_n) {
                            return Ok(Some(self.finish_frame(frame)));
                        }
                        // Trial decode failed — likely misaligned due to history
                        // prepend, or a false positive vowel/symbol match from data.
                        // Don't transition to Analyzing on first false positive;
                        // continue receiving to find the real end preamble.
                    }
                }

                // Denoisers can leave periodic residual noise after the
                // frame, defeating both the end preamble and silence test.
                // Probe at header-derived completion times, but never end
                // a reception on a length guess alone. This path requires
                // built-in CRC or an explicit caller-supplied validator.
                if self.profile == EccProfile::Standard || self.validator.is_some() {
                    let prefix_windows = HEADER_PROBE_WINDOWS;
                    let needs_probe = self.rescan_after_failed_probe
                        || self.frame_end_hints.as_ref().map_or(true, Vec::is_empty);
                    if needs_probe && self.symbols_recorded >= prefix_windows
                        && (self.symbols_recorded - prefix_windows) % 16 == 0
                    {
                        let candidates = self.predict_frame_ends(prefix_windows);
                        for &(start, end, length) in &candidates {
                            if !self.frame_start_hints.contains(&start) {
                                self.frame_start_hints.push(start);
                            }
                            self.max_record_symbols = self.max_record_symbols.max(end);
                            if let Some(length) = length {
                                self.remember_frame_length(start, length);
                            }
                        }
                        // A sync-only candidate reserves enough room for
                        // a maximum frame, but it is not a length lock.
                        // Keep scanning for later headers in that case.
                        let new_hints: Vec<_> = candidates.into_iter()
                            .filter_map(|(_, end, length)| length.is_some().then_some(end)).collect();
                        if self.rescan_after_failed_probe {
                            let found_header = !new_hints.is_empty();
                            let hints = self.frame_end_hints.get_or_insert_with(Vec::new);
                            hints.extend(new_hints);
                            hints.sort_unstable();
                            hints.dedup();
                            if found_header { self.rescan_after_failed_probe = false; }
                        } else {
                            self.frame_end_hints = Some(new_hints);
                        }
                    }
                    if self.frame_end_hints.as_ref()
                        .is_some_and(|hints| hints.contains(&self.symbols_recorded))
                    {
                        if let Ok(Some(payload)) = self.analyze(false) {
                            return Ok(Some(payload));
                        }
                        self.rescan_after_failed_probe = true;
                    }
                    if let Some(hints) = &mut self.frame_end_hints {
                        hints.retain(|&end| end > self.symbols_recorded);
                    }
                }

                if self.symbols_recorded >= self.max_record_symbols {
                    self.state = DecoderState::Analyzing;
                    let result = self.analyze(true);
                    if result.is_err() {
                        // A bounded noise acquisition can expire while a
                        // real preamble/header is arriving. Retain overlap
                        // for the next acquisition instead of cutting the
                        // new frame in half. This is not a payload success:
                        // the failed attempt remains available to callers,
                        // and every later delivery still passes validation.
                        let start = self.recorded_audio.len()
                            .saturating_sub(HEADER_PROBE_WINDOWS * SYMBOL_TOTAL_SAMPLES);
                        self.history = self.recorded_audio[start..]
                            .chunks_exact(SYMBOL_TOTAL_SAMPLES)
                            .map(|slot| slot.to_vec()).collect();
                        let previous_rms = self.rx_rms_ema;
                        self.start_receiving(false);
                        // Silence immediately after rollover must still
                        // release the receiver, even before another voiced
                        // window arrives to initialize its level estimate.
                        self.rx_rms_ema = previous_rms;
                    }
                    return result;
                }

                Ok(None)
            }
            DecoderState::Analyzing => Ok(None),
        }
    }

    /// Measure a Listening window: Some((vowel, F2 centroid)), or None when
    /// the window is silent.
    fn measure_window_vowel(&self, window: &[f32]) -> Option<(usize, f64)> {
        let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];

        let rms = {
            let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / voiced.len() as f64).sqrt()
        };
        if rms < 1e-6 {
            return None;
        }

        // Hann-windowed spectrum, same as the Receiving path: measurably more
        // accurate than the raw spectrum under reverb + denoiser distortion.
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        let f0 = formant::detect_f0(&spectrum);
        let f2 = formant::detect_formants(&spectrum, f0);
        let (vowel_idx, _) = formant::classify_vowel_f2(f2, &self.centers);
        Some((vowel_idx, f2))
    }

    fn classify_voiced_full(&self, voiced: &[f32]) -> formant::SymbolClass {
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        formant::classify_symbol_full(&spectrum, &self.centers)
    }

    fn start_receiving(&mut self, saw_preamble: bool) {
        self.state = DecoderState::Receiving;
        self.recent_vowels.clear();
        self.recent_centroids.clear();
        self.recent_symbols.clear();
        self.recent_rx_centroids.clear();
        self.preview_symbols.clear();
        // Only the clean preamble-match path knows the pilot comes next; the
        // fallback path leaves calibration to analyze()'s position math.
        self.pilot_remaining = if saw_preamble { PILOT_LEN } else { 0 };
        self.pilot_measured.clear();
        self.calibrated = None;
        self.eq = None;
        self.rx_rms_ema = 0.0;
        self.rx_silence_run = 0;
        self.rx_quiet_probe_run = 0;
        self.quiet_completion_probes = 0;
        self.frame_end_hints = None;
        self.rescan_after_failed_probe = false;
        self.frame_start_hints.clear();
        self.frame_length_hints.clear();
        self.frame_clock_hint = None;

        // Prepend history windows to recording for alignment recovery
        self.recorded_audio.clear();
        let n_history = self.history.len() + self.delivery_context.len() / SYMBOL_TOTAL_SAMPLES;
        self.recorded_audio.extend_from_slice(&self.delivery_context);
        self.delivery_context.clear();
        for w in &self.history {
            self.recorded_audio.extend_from_slice(w);
        }
        self.symbols_recorded = n_history;
        self.history.clear();
        // The lock fired on the window that completed the preamble pattern,
        // so the preamble occupies the last PREAMBLE_LEN history windows.
        self.preamble_pos_hint =
            saw_preamble.then(|| n_history.saturating_sub(PREAMBLE_LEN));

        let max_total_bytes = self.profile.encoded_bytes(MAX_LENGTH_VARIABLE);
        let max_data_syms = formant::symbols_for_bytes(max_total_bytes);
        let clock_slack = crate::audio::clock_slack_windows(max_data_syms + PREAMBLE_LEN + PILOT_LEN);
        self.max_record_symbols =
            max_data_syms + clock_slack + PREAMBLE_LEN + PILOT_LEN + n_history + 10;

        self.recorded_audio
            .reserve(self.max_record_symbols * SYMBOL_TOTAL_SAMPLES);
    }

    /// Return already-consumed lookahead ahead of unread callback audio.
    /// Keep the exact suffix, including the last guard/end marker: those can
    /// be absent or damaged, so they cannot define how much audio to discard.
    /// Only a successful codeword establishes this boundary; failed probes
    /// keep their recording and retry evidence unchanged.
    fn finish_frame(&mut self, frame: DecodedFrame) -> Vec<u8> {
        // A valid codeword's last analysis window can end just before the
        // next preamble. No next-frame samples are lost, but noisy fast
        // songs may need an alignment starting slightly BEFORE that point.
        // Retain real, already-consumed samples separately from lookahead;
        // they are not classified twice by the streaming acquisition path.
        self.delivery_context.clear();
        if self.profile == EccProfile::Standard || self.validator.is_some() {
            let context = (frame.audio_end / SYMBOL_TOTAL_SAMPLES).min(DELIVERY_CONTEXT_WINDOWS)
                * SYMBOL_TOTAL_SAMPLES;
            self.delivery_context.extend_from_slice(&self.recorded_audio[frame.audio_end - context..frame.audio_end]);
        }
        let mut suffix = self.recorded_audio.split_off(frame.audio_end);
        suffix.append(&mut self.sample_buffer);
        self.sample_buffer = suffix;
        self.recorded_audio.clear();
        self.symbols_recorded = 0;
        self.state = DecoderState::Listening;
        self.history.clear();
        self.recent_vowels.clear();
        self.recent_centroids.clear();
        self.recent_symbols.clear();
        self.recent_rx_centroids.clear();
        self.active_count = 0;
        self.silence_count = 0;
        self.rx_quiet_probe_run = 0;
        self.quiet_completion_probes = 0;
        self.frame_start_hints.clear();
        self.frame_length_hints.clear();
        self.frame_clock_hint = None;
        self.frame_end_hints = None;
        self.rescan_after_failed_probe = false;
        frame.payload
    }

    fn trial_decode(&mut self, n_data_symbols: usize) -> Option<DecodedFrame> {
        let mut failures = FailedPayloadCache::default();
        self.trial_decode_cached(n_data_symbols, Some(&mut failures))
    }

    fn trial_decode_cached(
        &mut self, n_data_symbols: usize, mut failures: Option<&mut FailedPayloadCache>,
    ) -> Option<DecodedFrame> {
        #[cfg(test)]
        {
            self.trial_length_searches.set(0);
            self.trial_failure_records.set(0);
        }
        let mut spectra = AlignedSpectra::new(&self.recorded_audio, 0, n_data_symbols);
        let mut symbols = Vec::with_capacity(n_data_symbols);
        let mut confidences = Vec::with_capacity(n_data_symbols);
        let mut seconds = Vec::with_capacity(n_data_symbols);
        let mut centroids = Vec::with_capacity(n_data_symbols);

        let centers = self.centers;
        for i in 0..n_data_symbols {
            let class = spectra.classify_full(i, &centers)?;
            symbols.push(class.vowel);
            confidences.push(class.confidence);
            seconds.push(class.second);
            centroids.push(class.centroid);
        }

        // Skip past start preamble + pilot if present in the recorded symbols
        // (history windows prepended in start_receiving may include them).
        // Several candidate positions are tried — exact vowel matches first,
        // then structural centroid matches (tilt survivors) by quality.
        let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let sync_len = PREAMBLE_LEN + PILOT_LEN;
        let scan_end = symbols.len().saturating_sub(min_data_symbols + sync_len);
        let cand_positions = self.candidate_positions(&symbols, &centroids, scan_end);

        // Soft-combining alignment must come from a STRONG anchor (the
        // recorded lock position or an exact preamble match) — loose/blind
        // decode candidates would hand the combiner misaligned streams.
        let anchored = self.frame_start_hints.first().copied().or(self.preamble_pos_hint).or_else(|| {
            (0..scan_end.min(PREAMBLE_SCAN_WINDOWS))
                .find(|&pos| symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_START_VOWELS)
        });
        if let Some(anchor) = anchored {
            let anchor_start = anchor + sync_len;
            if anchor_start < symbols.len() {
                self.last_attempt = Some((
                    symbols[anchor_start..].to_vec(),
                    confidences[anchor_start..].to_vec(),
                ));
            }
        }

        // No preamble located: single attempt from the start of the region.
        if cand_positions.is_empty() {
            return self.try_decode_trial_symbols_cached(&symbols, &confidences, &seconds, None, failures.as_deref_mut())
                .map(|payload| spectra.decoded_frame(payload, 0, self.profile));
        }

        // A pristine header supplies the trial's default length lock, also
        // when the phased scan found it on a different alignment. Specific
        // frame hints below override that default for later senders; each
        // candidate's classifier rungs retain the same validation gates.
        let locked = self.locked_frame_len(&mut spectra, &cand_positions, &symbols, &confidences)
            .or_else(|| self.common_hinted_length(&cand_positions));
        if self.profile == EccProfile::Legacy && locked.is_none() && self.validator.is_none() {
            return None;
        }
        for &pos in &cand_positions {
            // A later frame may have a different length from a failed one.
            // A conflicting hint deliberately falls back to length search.
            let locked = self.frame_length_hints.get(&pos).copied().unwrap_or(locked);
            let data_start = pos + sync_len;
            if let Some(payload) = self.try_decode_trial_symbols_cached(
                &symbols[data_start..],
                &confidences[data_start..],
                &seconds[data_start..],
                locked,
                failures.as_deref_mut(),
            ) {
                return Some(spectra.decoded_frame(payload, data_start, self.profile));
            }
            // With a locked length, a region too short to hold the frame
            // cannot decode at ANY rung — skip the expensive
            // re-classification sweeps (this is the common case for false
            // end-preamble fires mid-reception).
            if let Some(l) = locked {
                let need =
                    formant::symbols_for_bytes(self.profile.encoded_bytes(l));
                if symbols.len().saturating_sub(data_start) < need {
                    continue;
                }
            }
            // Retry with pilot-calibrated centers (absorbs mild channel tilt).
            let cal_centers = self
                .calibrated
                .or_else(|| spectra.pilot_centers(pos + PREAMBLE_LEN));
            if let Some(centers) = cal_centers {
                let mut cal_syms = symbols[data_start..].to_vec();
                let mut cal_confs = confidences[data_start..].to_vec();
                let mut cal_seconds = seconds[data_start..].to_vec();
                for (k, i) in (data_start..symbols.len()).enumerate() {
                    if let Some(class) = spectra.classify_full(i, &centers) {
                        cal_syms[k] = class.vowel;
                        cal_confs[k] = class.confidence;
                        cal_seconds[k] = class.second;
                    }
                }
                if let Some(payload) = self.try_decode_trial_symbols_cached(
                    &cal_syms,
                    &cal_confs,
                    &cal_seconds,
                    locked,
                    failures.as_deref_mut(),
                ) {
                    return Some(spectra.decoded_frame(payload, data_start, self.profile));
                }
            }
            // Retry through the pilot-derived channel EQ (absorbs heavy tilt).
            let eq = spectra
                .pilot_eq(pos + PREAMBLE_LEN, self.profile)
                .or_else(|| self.eq.clone());
            if let Some(eq) = eq {
                let n = symbols.len() - data_start;
                let classify_stream =
                    |f: &mut dyn FnMut(usize) -> Option<formant::SymbolClass>| {
                        let mut cs = symbols[data_start..].to_vec();
                        let mut cc = confidences[data_start..].to_vec();
                        let mut c2 = seconds[data_start..].to_vec();
                        for k in 0..n {
                            if let Some(class) = f(data_start + k) {
                                cs[k] = class.vowel;
                                cc[k] = class.confidence;
                                c2[k] = class.second;
                            }
                        }
                        (cs, cc, c2)
                    };
                // Equalized centroid, matched-filter templates (reject
                // in-band leakage a centroid cannot), and F0-adaptive
                // centroids (per-note centers for expressive melodies).
                let eq_s = classify_stream(&mut |i| spectra.classify_eq(i, &eq));
                let tm_s = classify_stream(&mut |i| spectra.classify_template(i, &eq));
                let fa_s = classify_stream(&mut |i| spectra.classify_f0a(i, &eq));
                for (cs, cc, c2) in [&eq_s, &tm_s, &fa_s] {
                    if let Some(payload) =
                        self.try_decode_trial_symbols_cached(cs, cc, c2, locked, failures.as_deref_mut())
                    {
                        return Some(spectra.decoded_frame(payload, data_start, self.profile));
                    }
                }
                // Ensemble: the rungs fail at different positions; a
                // confidence-weighted vote across them beats each alone at
                // the edge of the channel.
                let (vs, vc, v2) = ensemble_vote(&[
                    (&symbols[data_start..], &confidences[data_start..]),
                    (&eq_s.0, &eq_s.1),
                    (&tm_s.0, &tm_s.1),
                    (&fa_s.0, &fa_s.1),
                ]);
                if let Some(payload) =
                    self.try_decode_trial_symbols_cached(&vs, &vc, &v2, locked, failures.as_deref_mut())
                {
                    return Some(spectra.decoded_frame(payload, data_start, self.profile));
                }
            }
        }
        None
    }

    fn has_possible_complete_frame(&self) -> bool {
        (self.profile == EccProfile::Standard || self.validator.is_some())
            && self.quiet_completion_probes < MAX_QUIET_COMPLETION_PROBES
            && self.frame_length_hints.iter().any(|(&start, &length)| {
                length.is_some_and(|length| {
                    let symbols = PREAMBLE_LEN + PILOT_LEN
                        + formant::symbols_for_bytes(self.profile.encoded_bytes(length));
                    self.symbols_recorded >= start + symbols - crate::audio::clock_slack_windows(symbols)
                })
            })
    }

    /// A fully checked sync sequence can survive a damaged/low-confidence
    /// header. Quiet tails may prompt bounded analysis of such a recording,
    /// but never invent a length, override retained header evidence, or end
    /// an unsuccessful reception. CRC/caller validation remains mandatory.
    fn can_probe_sync_only_tail(&self) -> bool {
        if (self.profile != EccProfile::Standard && self.validator.is_none())
            || self.quiet_completion_probes >= MAX_QUIET_COMPLETION_PROBES
            || !self.frame_length_hints.is_empty()
        { return false; }
        let minimum = PREAMBLE_LEN + PILOT_LEN
            + formant::symbols_for_bytes(self.profile.encoded_bytes(1));
        self.frame_start_hints.iter().any(|&start| {
            self.symbols_recorded >= start + minimum - crate::audio::clock_slack_windows(minimum)
        })
    }

    /// Find (start, deadline, supported length) hints in a bounded sliding prefix.
    /// A pristine header uses the same guarded length detection as final
    /// analysis; exact sync without a length only reserves a maximum frame.
    /// Check all phases because codecs introduce fractional-symbol delay.
    fn predict_frame_ends(&mut self, prefix_windows: usize) -> Vec<(usize, usize, Option<usize>)> {
        let mut nominal_ends = Vec::new();
        let mut clock_delays: std::collections::BTreeMap<(usize, usize), (f64, isize, usize)> =
            std::collections::BTreeMap::new();
        let sync_len = PREAMBLE_LEN + PILOT_LEN;
        let header_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let prefix_start = self.symbols_recorded.saturating_sub(prefix_windows);
        // One maximum-length false acquisition plus one real frame. Late
        // headers may extend the deadline, but never permit unbounded audio.
        let max_frame = formant::symbols_for_bytes(self.profile.encoded_bytes(MAX_LENGTH_VARIABLE)) + sync_len;
        let recording_limit = 2 * (max_frame + crate::audio::clock_slack_windows(max_frame)
            + HISTORY_WINDOWS + 10);
        for phase in 0..16 {
            let phase_offset = phase * SYMBOL_TOTAL_SAMPLES / 16;
            let offset = prefix_start * SYMBOL_TOTAL_SAMPLES + phase_offset;
            let mut spectra = AlignedSpectra::new(&self.recorded_audio, offset, prefix_windows.saturating_sub(1));
            let classes: Vec<_> = (0..prefix_windows.saturating_sub(1))
                .filter_map(|i| spectra.classify_full(i, &self.centers))
                .collect();
            let symbols: Vec<_> = classes.iter().map(|c| c.vowel).collect();
            let confidence: Vec<_> = classes.iter().map(|c| c.confidence).collect();
            let centroids: Vec<_> = classes.iter().map(|c| c.centroid).collect();
            let scan_end = symbols.len().saturating_sub(sync_len + header_symbols);
            // Positions are relative to this sliding prefix, not the start
            // of the original (possibly noise-triggered) recording.
            let mut positions = preamble_candidates(&symbols, &centroids, scan_end);
            for pos in 0..scan_end.min(PREAMBLE_SCAN_WINDOWS) {
                if !positions.contains(&pos) {
                    positions.push(pos);
                }
            }
            for pos in positions {
                let length = self.locked_frame_len(&mut spectra, &[pos], &symbols, &confidence);
                if length.is_none() && !self.strong_sync_at(&mut spectra, pos, &symbols) {
                    continue;
                }
                let start = prefix_start + pos;
                let len = length.unwrap_or(MAX_LENGTH_VARIABLE);
                let frame_symbols = sync_len + formant::symbols_for_bytes(self.profile.encoded_bytes(len));
                let data_end = start + frame_symbols;
                // Include the end preamble and two alignment/codec windows.
                let end = data_end + usize::from(phase_offset > 0) + PREAMBLE_LEN + 2;
                nominal_ends.push((start, end, length));
                if let Some(length) = length {
                    if let Some(estimate) = self.clock_probe_shift(
                        offset + pos * SYMBOL_TOTAL_SAMPLES, prefix_windows.saturating_sub(pos),
                        frame_symbols, length,
                    ) {
                        let estimate = (estimate.0, estimate.1, offset + pos * SYMBOL_TOTAL_SAMPLES);
                        clock_delays.entry((start, length))
                            .and_modify(|previous| {
                                if estimate.0 < previous.0 { *previous = estimate; }
                            })
                            .or_insert(estimate);
                    }
                }
            }
        }
        let mut ends = Vec::new();
        for (start, nominal_end, length) in nominal_ends {
            let frame_symbols = sync_len + formant::symbols_for_bytes(
                self.profile.encoded_bytes(length.unwrap_or(MAX_LENGTH_VARIABLE)));
            let delay = length.and_then(|length| clock_delays.get(&(start, length)))
                .map_or(0, |estimate| estimate.1);
            // Slow frames can postpone the primary probe within its old
            // fallback bound. Fast frames gain an earlier probe while
            // retaining BOTH original deadlines if the clock changes later.
            let primary = nominal_end.saturating_add_signed(delay);
            for end in [primary, if delay < 0 { nominal_end } else { primary },
                nominal_end + crate::audio::clock_slack_windows(frame_symbols)] {
                if end > self.symbols_recorded && end <= recording_limit {
                    ends.push((start, end, length));
                    if let Some(length) = length {
                        if let Some(&(score, _, offset)) = clock_delays.get(&(start, length)) {
                            if self.frame_clock_hint.is_none_or(|previous| score < previous.score) {
                                self.frame_clock_hint = Some(ClockHint { score, offset, length });
                            }
                        }
                    }
                }
            }
        }
        ends.sort_unstable();
        ends.dedup();
        ends
    }

    /// A bounded completion-probe shift, not a length lock. Locally verified
    /// headers and clear, consistently moving guards must support the rate.
    fn clock_probe_shift(
        &self, offset: usize, max_windows: usize, frame_symbols: usize, length: usize,
    ) -> Option<(f64, isize)> {
        let maximum_delay = crate::audio::clock_slack_windows(frame_symbols);
        if maximum_delay <= PREAMBLE_LEN + 2 { return None; }
        let mut spectra = AlignedSpectra::clocked(&self.recorded_audio, offset, max_windows);
        let starts = spectra.clock_starts.as_ref()?;
        // Leave acquisition out of the rate estimate, then require two
        // independently measured halves of at least eight intervals each.
        let first = PREAMBLE_LEN + PILOT_LEN;
        let last = starts.len().checked_sub(1)?;
        if last < first + 16 || last >= frame_symbols { return None; }
        let intervals = (last - first) as f64;
        let nominal = SYMBOL_TOTAL_SAMPLES as f64;
        let radius = crate::audio::CLOCK_RADIUS as f64;
        let period = (starts[last] - starts[first]) as f64 / intervals;
        let margin = 2.0 * radius / intervals;
        let drift = period - nominal;
        if drift.abs() <= margin || drift.abs() <= nominal * 0.005 { return None; }
        let middle = (first + last) / 2;
        let left = (starts[middle] - starts[first]) as f64 / (middle - first) as f64;
        let right = (starts[last] - starts[middle]) as f64 / (last - middle) as f64;
        if (left - nominal) * drift <= 0.0 || (right - nominal) * drift <= 0.0
            || (left - right).abs() > radius / 2.0
            || (left - right).abs() > 0.5 * drift.abs() { return None; }

        let mut score = 0.0;
        let mut clear_guards = 0;
        for &start in &starts[first..last] {
            let window = &self.recorded_audio[start..start + SYMBOL_TOTAL_SAMPLES];
            let voice: f64 = window[..SAMPLES_PER_SYMBOL].iter().map(|&s| (s as f64).powi(2)).sum();
            if voice == 0.0 { return None; }
            let guard_start = SAMPLES_PER_SYMBOL + crate::audio::CLOCK_RADIUS;
            let guard: f64 = window[guard_start..guard_start + GUARD_SAMPLES / 2]
                .iter().map(|&s| (s as f64).powi(2)).sum();
            let ratio = guard * SAMPLES_PER_SYMBOL as f64 / (voice * (GUARD_SAMPLES / 2) as f64);
            if !ratio.is_finite() { return None; }
            if ratio < 0.15 { clear_guards += 1; }
            score += ratio;
        }
        if clear_guards * 4 < (last - first) * 3 { return None; }
        let upper_period = (period + margin).min(nominal + radius);
        let predicted = starts[last] as f64 + (frame_symbols - last) as f64 * upper_period;
        let nominal_end = offset as f64 + frame_symbols as f64 * nominal;
        let delay = (((predicted - nominal_end) / nominal).ceil() as isize)
            .clamp(-(maximum_delay as isize), maximum_delay as isize);
        if (0..=(PREAMBLE_LEN + 2) as isize).contains(&delay) { return None; }

        let header_end = first + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let classes: Option<Vec<_>> = (0..header_end)
            .map(|i| spectra.classify_full(i, &self.centers)).collect();
        let classes = classes?;
        let symbols: Vec<_> = classes.iter().map(|class| class.vowel).collect();
        let confidences: Vec<_> = classes.iter().map(|class| class.confidence).collect();
        if self.locked_frame_len(&mut spectra, &[0], &symbols, &confidences) != Some(length) { return None; }
        Some((score / intervals, delay))
    }

    /// An intact preamble AND pilot can identify a late frame even when
    /// its weak length header is damaged. Unlike a length lock this only
    /// preserves a candidate and recording budget; CRC-checked decoding
    /// still has to establish the actual payload and length. Require the
    /// complete sync pattern, not the fuzzy acquisition thresholds.
    fn strong_sync_at(&self, spectra: &mut AlignedSpectra<'_>, pos: usize, symbols: &[usize]) -> bool {
        let pattern = PREAMBLE_START_VOWELS.into_iter().chain(PILOT);
        let sync_len = PREAMBLE_LEN + PILOT_LEN;
        let Some(observed) = symbols.get(pos..pos + sync_len) else { return false };
        if observed.iter().copied().eq(pattern.clone()) {
            return true;
        }
        let Some(eq) = spectra.pilot_eq(pos + PREAMBLE_LEN, self.profile) else {
            return false;
        };
        pattern.enumerate().all(|(k, vowel)| {
            spectra.classify_template(pos + k, &eq)
                .is_some_and(|class| class.vowel == vowel)
        })
    }

    /// One prioritized attempt using a phase the clock probe verified.
    /// Recheck the local header on the current grid, then use the unchanged
    /// CRC/validator-checked decoder. Failure does not consume any of the
    /// original nominal or clock-search budget, mutate hints, or publish an
    /// observation; all existing retries remain available in their old order.
    fn try_clock_hint(&mut self) -> Option<DecodedFrame> {
        if self.profile != EccProfile::Standard && self.validator.is_none() { return None; }
        let hint = self.frame_clock_hint?;
        let start = hint.offset / SYMBOL_TOTAL_SAMPLES;
        if self.frame_length_hints.get(&start) != Some(&Some(hint.length)) {
            return None;
        }
        // A later sender's clear clock cannot jump ahead of an earlier
        // retained frame, including one with a damaged/conflicting header.
        // Let the original candidate order resolve that case instead.
        if self.frame_start_hints.iter().any(|&earlier| earlier < start)
            || self.frame_length_hints.first_key_value().is_some_and(|(&earlier, _)| earlier < start)
        {
            return None;
        }
        #[cfg(test)]
        { self.priority_clock_grids += 1; }
        let data_start = PREAMBLE_LEN + PILOT_LEN;
        let frame_symbols = data_start + formant::symbols_for_bytes(self.profile.encoded_bytes(hint.length));
        let mut spectra = AlignedSpectra::clocked(&self.recorded_audio, hint.offset, frame_symbols);
        if spectra.windows.len() < frame_symbols { return None; }
        let classes: Option<Vec<_>> = (0..frame_symbols)
            .map(|i| spectra.classify_full(i, &self.centers)).collect();
        let classes = classes?;
        let symbols: Vec<_> = classes.iter().map(|class| class.vowel).collect();
        let confidences: Vec<_> = classes.iter().map(|class| class.confidence).collect();
        if self.locked_frame_len(&mut spectra, &[0], &symbols, &confidences) != Some(hint.length) { return None; }
        let seconds: Vec<_> = classes.iter().map(|class| class.second).collect();
        let payload = self.try_decode_symbols_locked(
            &symbols[data_start..], Some(&confidences[data_start..]), Some(&seconds[data_start..]),
            Some(hint.length), None, None,
        );
        let (frame, observed, confidence) = if let Some(payload) = payload {
            (spectra.decoded_frame(payload, data_start, self.profile),
                symbols[data_start..].to_vec(), confidences[data_start..].to_vec())
        } else {
            self.try_clock_frequency_compensation(&mut spectra, 0, Some(hint.length))?
        };
        self.last_attempt = Some((observed, confidence));
        Some(frame)
    }

    fn try_clock_frequency_compensation(
        &self, spectra: &mut AlignedSpectra<'_>, pos: usize, length: Option<usize>,
    ) -> Option<ClockRecovery> {
        if self.profile != EccProfile::Standard && self.validator.is_none() { return None; }
        let data_start = pos + PREAMBLE_LEN + PILOT_LEN;
        let end = length.map_or(spectra.windows.len(), |len|
            data_start + formant::symbols_for_bytes(self.profile.encoded_bytes(len)));
        let mut corrected = spectra.frequency_compensated(pos, end)?;
        let eq = corrected.pilot_eq(pos + PREAMBLE_LEN, self.profile)?;
        let header_end = data_start + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let classes: Option<Vec<_>> = (0..header_end).map(|i| corrected.classify_template(i, &eq)).collect();
        let mut classes = classes?;
        let symbols: Vec<_> = classes.iter().map(|class| class.vowel).collect();
        let confidences: Vec<_> = classes.iter().map(|class| class.confidence).collect();
        let local_length = self.locked_frame_len(&mut corrected, &[pos], &symbols, &confidences);
        if length.zip(local_length).is_some_and(|(a, b)| a != b) {
            return None;
        }
        if local_length.is_none() && !self.strong_sync_at(&mut corrected, pos, &symbols) { return None; }
        let locked = length.or(local_length);
        let data_end = locked.map_or(end, |len|
            data_start + formant::symbols_for_bytes(self.profile.encoded_bytes(len)));
        if data_end > end { return None; }
        for i in header_end..data_end { classes.push(corrected.classify_template(i, &eq)?); }
        let symbols: Vec<_> = classes.iter().map(|class| class.vowel).collect();
        let confidences: Vec<_> = classes.iter().map(|class| class.confidence).collect();
        let seconds: Vec<_> = classes.iter().map(|class| class.second).collect();
        let payload = self.try_decode_trial_symbols(&symbols[data_start..data_end], &confidences[data_start..data_end],
            &seconds[data_start..data_end], locked)?;
        let observed_end = data_start + formant::symbols_for_bytes(self.profile.encoded_bytes(payload.len()));
        if observed_end > data_end { return None; }
        Some((corrected.decoded_frame(payload, data_start, self.profile),
            symbols[data_start..observed_end].to_vec(), confidences[data_start..observed_end].to_vec()))
    }

    fn try_envelope_clock(&self) -> Option<ClockRecovery> {
        if self.profile != EccProfile::Standard && self.validator.is_none() { return None; }
        let period = envelope_clock_period(&self.recorded_audio)?;
        for phase in 0..MAX_ENVELOPE_CLOCK_PHASES {
            #[cfg(test)]
            self.envelope_clock_phases.set(self.envelope_clock_phases.get() + 1);
            let offset = phase * SYMBOL_TOTAL_SAMPLES / 16;
            let mut spectra = AlignedSpectra::uniform_clocked(&self.recorded_audio, offset, period);
            let length = self.frame_length_hints.get(&(offset / SYMBOL_TOTAL_SAMPLES)).copied().flatten();
            if let Some(frame) = self.try_clock_frequency_compensation(&mut spectra, 0, length) {
                return Some(frame);
            }
        }
        None
    }

    /// A speculative, integrity-checked analysis leaves the reception intact
    /// on failure. Final analysis also closes undecodable receptions.
    fn analyze(&mut self, finalize: bool) -> Result<Option<Vec<u8>>, crate::Error> {
        #[cfg(test)]
        {
            self.clock_grids = 0;
            self.clock_candidate_peak = 0;
            self.nominal_grids = 0;
            self.priority_clock_grids = 0;
            self.envelope_clock_phases.set(0);
        }
        let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if self.symbols_recorded < min_data_symbols {
            if finalize {
                self.state = DecoderState::Listening;
                self.recent_vowels.clear();
            }
            return Err(crate::Error::DecodeFailed);
        }
        if let Some(frame) = self.try_clock_hint() {
            return Ok(Some(self.finish_frame(frame)));
        }
        if let Some((frame, symbols, confidences)) = self.try_envelope_clock() {
            self.last_attempt = Some((symbols, confidences));
            return Ok(Some(self.finish_frame(frame)));
        }

        let steps_per_symbol: usize = 16;
        let step = SYMBOL_TOTAL_SAMPLES / steps_per_symbol;

        let mut best_payload: Option<DecodedFrame> = None;
        let mut best_confidence: f64 = 0.0;
        let mut best_attempt: Option<(f64, Vec<usize>, Vec<f64>)> = None;
        let mut nominal_locked_length = self.common_hinted_length(&self.frame_start_hints);
        // Classifier/alignment retries may produce the same failed codeword.
        // Share exact-input failures for this analysis only, never successes.
        let mut failures = FailedPayloadCache::default();

        // Try more offsets to handle misalignment from padding
        let max_offset_steps = steps_per_symbol * 2;
        let mut clock_candidates: std::collections::BTreeMap<usize, (f64, usize, Option<usize>)> =
            std::collections::BTreeMap::new();
        let mut clock_trials = 0;

        // Pair failed nominal alignments with a supported clock hypothesis.
        // Trying phases independently also avoids letting one low-energy
        // denoiser tail stand in for every phase of the same physical frame.
        for (offset_step, clocked) in (0..max_offset_steps)
            .flat_map(|offset| [(offset, false), (offset, true)])
        {
            let mut locked_length = nominal_locked_length;
            let sample_offset = if !clocked {
                #[cfg(test)]
                { self.nominal_grids += 1; }
                offset_step * step
            } else {
                // A successful nominal decode needs no clock hypotheses.
                if best_payload.is_some() || clock_trials >= MAX_CLOCK_CANDIDATES { continue; }
                let next = clock_candidates.iter().min_by(|a, b| a.1.0.total_cmp(&b.1.0))
                    .map(|(&key, &(_, offset, length))| (key, offset, length));
                let Some((key, offset, length)) = next else { continue; };
                clock_candidates.remove(&key);
                clock_trials += 1;
                // A prior frame can have a valid header but a failed CRC.
                // Carry only this physical frame's supported length into its
                // variable-rate grid, never another candidate's length lock.
                locked_length = length;
                #[cfg(test)]
                { self.clock_grids += 1; }
                offset
            };
            let mut spectra = if clocked {
                AlignedSpectra::clocked(&self.recorded_audio, sample_offset, usize::MAX)
            } else {
                AlignedSpectra::new(&self.recorded_audio, sample_offset, usize::MAX)
            };

            let mut symbols: Vec<usize> = Vec::new();
            let mut confidences: Vec<f64> = Vec::new();
            let mut seconds: Vec<usize> = Vec::new();
            let mut centroids: Vec<f64> = Vec::new();
            let mut confidence_sum = 0.0f64;
            let mut sym_idx = 0;

            let centers = self.centers;
            loop {
                let Some(class) = spectra.classify_full(sym_idx, &centers)
                else {
                    break;
                };
                symbols.push(class.vowel);
                confidences.push(class.confidence);
                seconds.push(class.second);
                centroids.push(class.centroid);
                confidence_sum += class.confidence;
                sym_idx += 1;
            }

            if symbols.len() < min_data_symbols {
                continue;
            }

            // Locate start preamble + pilot candidates: exact vowel matches
            // first, then structural centroid matches (channel tilt
            // survivors) ranked by pattern quality. Each candidate gets its
            // own pilot-derived calibrated centers and channel EQ, used
            // strictly as decode retries.
            let sync_len = PREAMBLE_LEN + PILOT_LEN;
            let scan_end = symbols.len().saturating_sub(min_data_symbols + sync_len);
            let cand_positions = if clocked {
                // This grid starts at a supported physical frame boundary.
                // Nominal-grid hints cannot index a variable-rate grid.
                if scan_end > 0 { vec![0] } else { Vec::new() }
            } else {
                self.candidate_positions(&symbols, &centroids, scan_end)
            };
            let anchored = if clocked { Some(0) } else { self.frame_start_hints.first().copied().or(self.preamble_pos_hint).or_else(|| {
                (0..scan_end.min(PREAMBLE_SCAN_WINDOWS))
                    .find(|&pos| symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_START_VOWELS)
            }) };
            let first_start = cand_positions.first().map_or(0, |&p| p + sync_len);

            // Candidate end positions: last end-preamble match, then
            // no end-preamble stripping as fallback.
            let mut end_candidates = Vec::new();
            for pos in (first_start + min_data_symbols
                ..=symbols.len().saturating_sub(PREAMBLE_LEN))
                .rev()
            {
                // An end-shaped payload fragment far from the end of the
                // recording is not a framing candidate. Keep enough margin
                // for the preamble, silence finalization, and alignment slop.
                if symbols.len() - pos > PREAMBLE_LEN + 10 {
                    break;
                }
                if pos + PREAMBLE_LEN > symbols.len() {
                    continue;
                }
                if symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_END_VOWELS {
                    end_candidates.push(pos);
                    break;
                }
            }
            end_candidates.push(symbols.len());

            // Track the highest-confidence aligned data region for soft
            // combining across repeated transmissions, even if RS fails.
            // Only preamble-anchored offsets qualify: without the anchor the
            // stream offset is unknown and combining would merge misaligned
            // symbol streams.
            let avg_conf = confidence_sum / symbols.len().max(1) as f64;
            if let Some(anchor) = anchored {
                let anchor_start = anchor + sync_len;
                if anchor_start < symbols.len()
                    && best_attempt.as_ref().map_or(true, |(c, _, _)| avg_conf > *c)
                {
                    best_attempt = Some((
                        avg_conf,
                        symbols[anchor_start..].to_vec(),
                        confidences[anchor_start..].to_vec(),
                    ));
                }
            }

            let record_success = |payload: Vec<u8>,
                                      observed: &[usize],
                                      data_start: usize,
                                      spectra: &AlignedSpectra<'_>,
                                      best_payload: &mut Option<DecodedFrame>,
                                      best_confidence: &mut f64| {
                let expected = encoded_payload_symbols(&payload, self.profile);
                let matches = expected.iter().zip(observed)
                    .filter(|(a, b)| a == b).count();
                let agreement = matches as f64 / expected.len() as f64;
                if best_payload.is_none() || agreement > *best_confidence {
                    *best_confidence = agreement;
                    *best_payload = Some(spectra.decoded_frame(payload, data_start, self.profile));
                }
            };

            if locked_length.is_none() {
                locked_length =
                    self.locked_frame_len(&mut spectra, &cand_positions, &symbols, &confidences);
                // A newly established length also invalidates speculative
                // successes collected at earlier, less useful alignments.
                if best_payload.as_ref().is_some_and(|payload| {
                    locked_length.is_some_and(|len| payload.payload.len() != len)
                }) {
                    best_payload = None;
                    best_confidence = 0.0;
                }
            }
            if !clocked { nominal_locked_length = locked_length; }
            let locked = locked_length;
            if clocked && (self.profile == EccProfile::Standard || self.validator.is_some()) {
                if let Some(length) = locked {
                    // A slow-clock completion probe can arrive after the
                    // corrected frame plus ordinary trailing-slot slop.
                    // Its supported header supplies an exact endpoint on
                    // this grid; CRC/validator checks still decide delivery.
                    // Never apply this prefix retry to unchecked Legacy.
                    let end = sync_len + formant::symbols_for_bytes(self.profile.encoded_bytes(length));
                    if end <= symbols.len() && !end_candidates.contains(&end) {
                        end_candidates.push(end);
                    }
                }
            }
            for &pos in &cand_positions {
                let locked = if clocked { locked } else {
                    self.frame_length_hints.get(&(sample_offset / SYMBOL_TOTAL_SAMPLES + pos))
                        .copied().unwrap_or(locked)
                };
                let data_start = pos + sync_len;

                // Melody-derived length hint (the solo's start position),
                // computed only on the expensive no-length-lock path and
                // only for checked profiles. Note rows are cached per
                // alignment, so repeated candidates share the work.
                let melody_hint = if locked.is_none()
                    && (self.profile == EccProfile::Standard || self.validator.is_some())
                {
                    melody_length_hint(
                        &mut spectra, data_start, symbols.len() - data_start, self.profile,
                    )
                } else {
                    None
                };

                // Per-candidate pilot-derived retries (computed lazily only
                // after the default decode fails at least once).
                let mut cal: Option<(Vec<usize>, Vec<f64>, Vec<usize>)> = None;
                let mut eq_alt: Option<(Vec<usize>, Vec<f64>, Vec<usize>)> = None;
                let mut tmpl_alt: Option<(Vec<usize>, Vec<f64>, Vec<usize>)> = None;
                let mut f0a_alt: Option<(Vec<usize>, Vec<f64>, Vec<usize>)> = None;
                let mut vote_alt: Option<(Vec<usize>, Vec<f64>, Vec<usize>)> = None;
                let mut derived = false;

                for &data_end in &end_candidates {
                    if data_end <= data_start
                        || data_end - data_start < min_data_symbols
                        || (!frame_geometry_allows(data_end - data_start, self.profile, locked)
                            && !(locked.is_none()
                                && hint_allows(data_end - data_start, self.profile, melody_hint)))
                    {
                        continue;
                    }
                    if let Some(payload) = self.try_decode_symbols_locked(
                        &symbols[data_start..data_end],
                        Some(&confidences[data_start..data_end]),
                        Some(&seconds[data_start..data_end]),
                        locked,
                        Some(&mut failures),
                        melody_hint,
                    ) {
                        record_success(
                            payload, &symbols[data_start..data_end],
                            data_start, &spectra,
                            &mut best_payload, &mut best_confidence,
                        );
                        if best_confidence == 1.0 {
                            break;
                        }
                    }
                    if !derived {
                        if let Some(l) = locked {
                            let need = formant::symbols_for_bytes(
                                self.profile.encoded_bytes(l),
                            );
                            if data_end.saturating_sub(data_start) < need {
                                continue;
                            }
                        }
                        derived = true;
                        if let Some(centers) =
                            spectra.pilot_centers(pos + PREAMBLE_LEN)
                        {
                            let mut cs = symbols.clone();
                            let mut cc = confidences.clone();
                            let mut c2 = seconds.clone();
                            for i in data_start..cs.len() {
                                if let Some(class) =
                                    spectra.classify_full(i, &centers)
                                {
                                    cs[i] = class.vowel;
                                    cc[i] = class.confidence;
                                    c2[i] = class.second;
                                }
                            }
                            cal = Some((cs, cc, c2));
                        }
                        if let Some(eq) = spectra.pilot_eq(pos + PREAMBLE_LEN, self.profile)
                        {
                            let mut cs = symbols.clone();
                            let mut cc = confidences.clone();
                            let mut c2 = seconds.clone();
                            for i in data_start..cs.len() {
                                if let Some(class) =
                                    spectra.classify_eq(i, &eq)
                                {
                                    cs[i] = class.vowel;
                                    cc[i] = class.confidence;
                                    c2[i] = class.second;
                                }
                            }
                            eq_alt = Some((cs, cc, c2));
                            let mut ts = symbols.clone();
                            let mut tc = confidences.clone();
                            let mut t2 = seconds.clone();
                            for i in data_start..ts.len() {
                                if let Some(class) =
                                    spectra.classify_template(i, &eq)
                                {
                                    ts[i] = class.vowel;
                                    tc[i] = class.confidence;
                                    t2[i] = class.second;
                                }
                            }
                            tmpl_alt = Some((ts, tc, t2));
                            let mut fs_ = symbols.clone();
                            let mut fc = confidences.clone();
                            let mut f2_ = seconds.clone();
                            for i in data_start..fs_.len() {
                                if let Some(class) =
                                    spectra.classify_f0a(i, &eq)
                                {
                                    fs_[i] = class.vowel;
                                    fc[i] = class.confidence;
                                    f2_[i] = class.second;
                                }
                            }
                            f0a_alt = Some((fs_, fc, f2_));
                        }
                        // Ensemble vote across all computed streams.
                        {
                            let mut voters: Vec<(&[usize], &[f64])> =
                                vec![(&symbols, &confidences)];
                            for alt in [&eq_alt, &tmpl_alt, &f0a_alt] {
                                if let Some((cs, cc, _)) = alt.as_ref() {
                                    voters.push((cs, cc));
                                }
                            }
                            if voters.len() >= 3 {
                                vote_alt = Some(ensemble_vote(&voters));
                            }
                        }
                    }
                    // Compare every classifier's successful codeword with
                    // its own observations. Raw confidence at this alignment
                    // says nothing about the quality of an EQ/template retry.
                    for (cs, cc, c2) in [&cal, &eq_alt, &tmpl_alt, &f0a_alt, &vote_alt]
                        .into_iter()
                        .flatten()
                    {
                        if let Some(payload) = self.try_decode_symbols_locked(
                            &cs[data_start..data_end],
                            Some(&cc[data_start..data_end]),
                            Some(&c2[data_start..data_end]),
                            locked,
                            Some(&mut failures),
                            melody_hint,
                        ) {
                            record_success(
                                payload, &cs[data_start..data_end],
                                data_start, &spectra,
                                &mut best_payload, &mut best_confidence,
                            );
                        }
                        if best_confidence == 1.0 {
                            break;
                        }
                    }
                    if best_confidence == 1.0 {
                        break;
                    }
                }
                if best_confidence == 1.0 {
                    break;
                }
            }

            // An exact header + payload + parity codeword needs no repairs.
            if best_confidence == 1.0 {
                break;
            }

            // Also try without any preamble stripping (misaligned history
            // may have clipped the preamble off entirely).
            for &data_end in &end_candidates {
                if data_end < min_data_symbols {
                    continue;
                }
                if let Some(payload) = self.try_decode_symbols_locked(
                    &symbols[..data_end],
                    Some(&confidences[..data_end]),
                    Some(&seconds[..data_end]),
                    locked,
                    Some(&mut failures),
                    None,
                ) {
                    record_success(
                        payload, &symbols[..data_end],
                        0, &spectra,
                        &mut best_payload, &mut best_confidence,
                    );
                }
            }

            if clocked && best_payload.is_none() {
                if let Some((frame, symbols, confidences)) = self.try_clock_frequency_compensation(&mut spectra, 0, locked_length) {
                    let confidence = confidences.iter().sum::<f64>() / confidences.len().max(1) as f64;
                    best_payload = Some(frame);
                    best_attempt = Some((confidence, symbols, confidences));
                }
            }

            // A verified clock recovery is a complete result, not another
            // length lock for a different nominal alignment or sender.
            if clocked && best_payload.is_some()
                && (self.profile == EccProfile::Standard || self.validator.is_some())
            {
                break;
            }
            if !clocked && best_payload.is_none() {
                for &pos in &cand_positions {
                    let length = self.locked_frame_len(&mut spectra, &[pos], &symbols, &confidences);
                    if length.is_none()
                        && !self.strong_sync_at(&mut spectra, pos, &symbols)
                    {
                        continue;
                    }
                    // A remembered length is not evidence that this phase
                    // is aligned. Keep the original local sync/header gate
                    // before reusing another phase's intact length.
                    let length = length.or_else(|| self.frame_length_hints
                        .get(&(sample_offset / SYMBOL_TOTAL_SAMPLES + pos)).copied().flatten());
                    let offset = sample_offset + pos * SYMBOL_TOTAL_SAMPLES;
                    let Some(sync) = self.recorded_audio.get(offset..offset + sync_len * SYMBOL_TOTAL_SAMPLES) else {
                        continue;
                    };
                    let score: f64 = sync.chunks_exact(SYMBOL_TOTAL_SAMPLES).map(|slot| {
                        let voice: f64 = slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| (s as f64).powi(2)).sum();
                        let guard: f64 = slot[SAMPLES_PER_SYMBOL..].iter().map(|&s| (s as f64).powi(2)).sum();
                        if voice > 0.0 { guard / voice } else { f64::INFINITY }
                    }).sum();
                    if !score.is_finite() { continue; }
                    let candidate = clock_candidates.entry(offset / SYMBOL_TOTAL_SAMPLES).or_insert((score, offset, length));
                    if score < candidate.0 { *candidate = (score, offset, length); }
                    if clock_candidates.len() > MAX_CLOCK_CANDIDATES {
                        let worst = *clock_candidates.iter().max_by(|a, b| a.1.0.total_cmp(&b.1.0)).unwrap().0;
                        clock_candidates.remove(&worst);
                    }
                    #[cfg(test)]
                    { self.clock_candidate_peak = self.clock_candidate_peak.max(clock_candidates.len()); }
                }
            }
        }

        if finalize || best_payload.is_some() {
            self.state = DecoderState::Listening;
            self.recent_vowels.clear();
        }
        if let Some((_, syms, confs)) = best_attempt {
            self.last_attempt = Some((syms, confs));
        }

        if let Some(frame) = best_payload {
            Ok(Some(self.finish_frame(frame)))
        } else {
            Err(crate::Error::DecodeFailed)
        }
    }

    /// Candidate starts: verified sync positions first, then the
    /// original Listening lock, pattern matches, and blind head positions.
    /// Sliding header hints can recover a real frame after noise triggered
    /// acquisition early. All candidates still require checked decoding.
    fn candidate_positions(
        &self,
        symbols: &[usize],
        centroids: &[f64],
        max_scan: usize,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        for &pos in &self.frame_start_hints {
            if pos < max_scan && !out.contains(&pos) {
                out.push(pos);
            }
        }
        if let Some(h) = self.preamble_pos_hint {
            if h < max_scan && !out.contains(&h) {
                out.push(h);
            }
        }
        for pos in preamble_candidates(symbols, centroids, max_scan) {
            if !out.contains(&pos) {
                out.push(pos);
            }
        }
        for pos in 0..HISTORY_WINDOWS.saturating_sub(PREAMBLE_LEN + 1).max(3) {
            if pos < max_scan && !out.contains(&pos) {
                out.push(pos);
            }
        }
        out
    }

    fn try_decode_symbols_locked(
        &self,
        symbols: &[usize],
        confidences: Option<&[f64]>,
        seconds: Option<&[usize]>,
        locked_len: Option<usize>,
        failures: Option<&mut FailedPayloadCache>,
        melody_hint: Option<usize>,
    ) -> Option<Vec<u8>> {
        let mut local_failures = FailedPayloadCache::default();
        let failures = failures.unwrap_or(&mut local_failures);
        #[cfg(test)]
        let before = failures.records;
        let result = decode_symbol_stream_cached(
            symbols, confidences, seconds, self.profile, locked_len, self.validator,
            Some(&mut *failures), melody_hint,
        );
        #[cfg(test)]
        self.trial_failure_records.set(self.trial_failure_records.get() + failures.records - before);
        result
    }

    /// An end-preamble-shaped payload fragment is not a complete frame.
    /// Legacy reception requires an independently established frame length
    /// or an external validator. Standard verifies its CRC on every attempt;
    /// TransferRx supplies the validator for Paranoid frames.
    fn try_decode_trial_symbols(
        &self,
        symbols: &[usize],
        confidences: &[f64],
        seconds: &[usize],
        locked: Option<usize>,
    ) -> Option<Vec<u8>> {
        self.try_decode_trial_symbols_cached(symbols, confidences, seconds, locked, None)
    }

    fn try_decode_trial_symbols_cached(
        &self, symbols: &[usize], confidences: &[f64], seconds: &[usize],
        locked: Option<usize>, failures: Option<&mut FailedPayloadCache>,
    ) -> Option<Vec<u8>> {
        #[cfg(test)]
        if locked.is_none() { self.trial_length_searches.set(self.trial_length_searches.get() + 1); }
        if self.profile == EccProfile::Legacy && locked.is_none() && self.validator.is_none() {
            return None;
        }
        self.try_decode_symbols_locked(symbols, Some(confidences), Some(seconds), locked, failures, None)
    }

    /// Preserve the existing trial-wide pristine-header rule across phases,
    /// but only when all remembered evidence among these candidates agrees.
    fn common_hinted_length(&self, candidates: &[usize]) -> Option<usize> {
        let mut common = None;
        for pos in candidates {
            if let Some(hint) = self.frame_length_hints.get(pos) {
                let length = (*hint)?;
                if common.is_some_and(|previous| previous != length) { return None; }
                common = Some(length);
            }
        }
        common
    }

    fn remember_frame_length(&mut self, start: usize, length: usize) {
        self.frame_length_hints.entry(start)
            .and_modify(|previous| {
                if *previous != Some(length) { *previous = None; }
            })
            .or_insert(Some(length));
    }

    /// Try to LOCK the payload length from the default-classified data
    /// region: require an exact codeword over high-confidence symbols.
    /// The caller must also verify the surrounding sync sequence; payload
    /// data can coincidentally contain a valid header codeword.
    fn locked_header_len(&self, symbols: &[usize], confidences: &[f64]) -> Option<usize> {
        let hdr_syms = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if symbols.len() < hdr_syms || confidences.len() < hdr_syms {
            return None;
        }
        let confs = &confidences[..hdr_syms];
        let min = confs.iter().cloned().fold(f64::MAX, f64::min);
        let mean = confs.iter().sum::<f64>() / hdr_syms as f64;
        if min < 0.4 || mean < 0.7 {
            return None;
        }
        let header = formant::symbols_to_bytes(&symbols[..hdr_syms], ENCODED_DATA_OFFSET);
        let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
        let len = rs_length.decode(&header).map(|d| d[0] as usize)?;
        if len == 0 || len > MAX_LENGTH_VARIABLE {
            return None;
        }
        // Zero-correction check: the received header must BE the codeword,
        // not merely near one — RS "fixing" a high-confidence byte is
        // itself evidence the region isn't what we think it is.
        (rs_length.encode(&[len as u8]) == header).then_some(len)
    }

    /// A tilted channel may corrupt every raw header classification even
    /// though equalization recovers it exactly. Establish the length before
    /// trying payload repairs, so a speculative decode at a false preamble
    /// cannot end the reception before the real codeword has arrived.
    fn locked_frame_len(
        &self,
        spectra: &mut AlignedSpectra<'_>,
        candidates: &[usize],
        symbols: &[usize],
        confidences: &[f64],
    ) -> Option<usize> {
        let sync_len = PREAMBLE_LEN + PILOT_LEN;
        for &pos in candidates {
            let ds = pos + sync_len;
            // Payload data can coincidentally contain a valid three-byte
            // header. A length lock also needs evidence of the frame's
            // known sync sequence, not just a plausible codeword.
            let preamble_matches = symbols[pos..pos + PREAMBLE_LEN].iter()
                .zip(PREAMBLE_START_VOWELS).filter(|(a, b)| **a == *b).count();
            let pilot_matches = symbols[pos + PREAMBLE_LEN..ds].iter()
                .zip(PILOT).filter(|(a, b)| **a == *b).count();
            if preamble_matches < PREAMBLE_LEN - 1 || pilot_matches < PILOT_LEN - 2 {
                continue;
            }
            if let Some(len) = self.locked_header_len(&symbols[ds..], &confidences[ds..]) {
                return Some(len);
            }
        }
        let header_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        for &pos in candidates {
            let ds = pos + sync_len;
            if ds + header_symbols > symbols.len() {
                continue;
            }
            let Some(eq) = spectra.pilot_eq(pos + PREAMBLE_LEN, self.profile) else {
                continue;
            };
            // A false preamble has arbitrary data in its "pilot". Demand
            // that the derived EQ explains the known pilot before letting
            // it establish a frame length.
            let pilot_matches = PILOT.iter().enumerate().filter(|&(k, &v)| {
                spectra.classify_template(pos + PREAMBLE_LEN + k, &eq)
                    .is_some_and(|class| class.vowel == v)
            }).count();
            if pilot_matches < PILOT_LEN - 2 {
                continue;
            }
            let preamble_matches = PREAMBLE_START_VOWELS.iter().enumerate().filter(|&(k, &v)| {
                spectra.classify_template(pos + k, &eq)
                    .is_some_and(|class| class.vowel == v)
            }).count();
            if preamble_matches < PREAMBLE_LEN - 1 {
                continue;
            }
            for template in [false, true] {
                let classes: Option<Vec<_>> = (ds..ds + header_symbols).map(|i| {
                    if template {
                        spectra.classify_template(i, &eq)
                    } else {
                        spectra.classify_eq(i, &eq)
                    }
                }).collect();
                let Some(classes) = classes else { continue };
                let syms: Vec<_> = classes.iter().map(|c| c.vowel).collect();
                let confs: Vec<_> = classes.iter().map(|c| c.confidence).collect();
                if let Some(len) = self.locked_header_len(&syms, &confs) {
                    return Some(len);
                }
            }
        }
        None
    }

    // Original full-FFT helpers, retained only as independent test references.
    /// Fully classify the window at symbol index `i` for the given sample
    /// offset, using explicit classification centers.
    #[cfg(test)]
    fn classify_full_at(
        &self,
        sample_offset: usize,
        sym_idx: usize,
        centers: &[f64; NUM_VOWELS],
    ) -> Option<formant::SymbolClass> {
        let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
        if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
            return None;
        }
        let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        Some(formant::classify_symbol_full(&spectrum, centers))
    }

    /// Classify the window at symbol index `i` through the channel EQ.
    #[cfg(test)]
    fn classify_eq_at(
        &self,
        sample_offset: usize,
        sym_idx: usize,
        eq: &[f64],
    ) -> Option<formant::SymbolClass> {
        let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
        if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
            return None;
        }
        let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        Some(formant::classify_symbol_eq(&spectrum, eq))
    }

    /// Classify the window at symbol index `i` with matched-filter
    /// templates (through the channel EQ).
    #[cfg(test)]
    fn classify_template_at(
        &self,
        sample_offset: usize,
        sym_idx: usize,
        eq: &[f64],
    ) -> Option<formant::SymbolClass> {
        let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
        if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
            return None;
        }
        let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        Some(formant::classify_symbol_template(&spectrum, eq))
    }

    /// Classify the window at symbol index `i` with the F0-adaptive
    /// centroid classifier (through the channel EQ).
    #[cfg(test)]
    fn classify_f0a_at(
        &self,
        sample_offset: usize,
        sym_idx: usize,
        eq: &[f64],
    ) -> Option<formant::SymbolClass> {
        let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
        if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
            return None;
        }
        let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        Some(formant::classify_symbol_f0_adaptive(&spectrum, eq))
    }

    /// Measure calibrated centers from the pilot windows starting at symbol
    /// index `pilot_start` for the given sample offset.
    #[cfg(test)]
    fn pilot_centers_at(
        &self,
        sample_offset: usize,
        pilot_start: usize,
    ) -> Option<[f64; NUM_VOWELS]> {
        let mut measured = Vec::with_capacity(PILOT_LEN);
        for (k, &vowel) in PILOT.iter().enumerate() {
            let start = sample_offset + (pilot_start + k) * SYMBOL_TOTAL_SAMPLES;
            if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
                return None;
            }
            let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
            let mut spectrum = vec![0.0f32; voiced.len()];
            fft::power_spectrum(voiced, &mut spectrum);
            let f0 = formant::detect_f0(&spectrum);
            measured.push((vowel, formant::detect_formants(&spectrum, f0)));
        }
        calibrate_centers(&measured)
    }

    /// Estimate a channel EQ from the pilot windows starting at symbol index
    /// `pilot_start` for the given sample offset.
    #[cfg(test)]
    fn pilot_eq_at(&self, sample_offset: usize, pilot_start: usize) -> Option<Vec<f64>> {
        let mut measured = Vec::with_capacity(PILOT_LEN);
        for k in 0..PILOT_LEN {
            let start = sample_offset + (pilot_start + k) * SYMBOL_TOTAL_SAMPLES;
            if start + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
                return None;
            }
            let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
            let mut spectrum = vec![0.0f32; voiced.len()];
            fft::power_spectrum(voiced, &mut spectrum);
            measured.push((k, spectrum));
        }
        formant::estimate_eq_with_profile(&measured, self.profile)
    }
}

/// Maximum symbols flipped to their runner-up decision during Chase retry.
/// 2^CHASE_MAX_FLIPS - 1 hypotheses are tried on the RS failure path.
const CHASE_MAX_FLIPS: usize = 6;

/// Only symbols below this confidence participate in Chase flipping.
const CHASE_CONF_LIMIT: f64 = 0.6;

struct FailedPayload {
    profile: EccProfile,
    bytes: Vec<u8>,
    erasures: Vec<usize>,
}

/// Analysis-local, bounded memoization of failures BEFORE header/agreement gates.
/// A header-only Chase flip leaves the payload codeword and its erasures
/// unchanged, so correcting it again cannot yield a different result. Keep
/// only one previous failure per payload length, comparing full inputs rather
/// than fingerprints. Standalone symbol-stream calls use their own local cache.
/// Successful payloads are never cached: their header, confidence-dependent,
/// and caller-supplied validation checks must be repeated for each trial.
#[derive(Default)]
struct FailedPayloadCache {
    by_length: Vec<Option<FailedPayload>>,
    #[cfg(test)]
    hits: usize,
    #[cfg(test)]
    records: usize,
}

impl FailedPayloadCache {
    fn contains(
        &mut self,
        profile: EccProfile,
        len: usize,
        bytes: &[u8],
        erasures: &mut Option<Vec<usize>>,
        compute_erasures: impl FnOnce() -> Vec<usize>,
    ) -> bool {
        let Some(Some(previous)) = self.by_length.get(len) else { return false };
        if previous.profile != profile || previous.bytes != bytes {
            return false;
        }
        // Don't sort confidence values on the clean hard-decision path.
        let erasures = erasures.get_or_insert_with(compute_erasures);
        let hit = previous.erasures == *erasures;
        #[cfg(test)]
        if hit {
            self.hits += 1;
        }
        hit
    }

    fn record(&mut self, profile: EccProfile, len: usize, bytes: &[u8], erasures: &[usize]) {
        // Internal callers use supported frame geometry. Keep the memory
        // bound explicit even if this helper gains another caller later.
        if !(1..=MAX_LENGTH_VARIABLE).contains(&len) || bytes.len() > 255 || erasures.len() > 255 {
            return;
        }
        if self.by_length.len() <= len {
            self.by_length.resize_with(len + 1, || None);
        }
        self.by_length[len] = Some(FailedPayload {
            profile,
            bytes: bytes.to_vec(),
            erasures: erasures.to_vec(),
        });
        #[cfg(test)]
        {
            self.records += 1;
        }
    }
}

/// Decode an aligned data-region symbol stream (post-preamble/pilot) into a
/// payload. Public so external soft-combining (e.g. `transfer::TransferRx`)
/// can re-decode merged symbol streams.
///
/// `seconds` (runner-up vowel decisions, parallel to `symbols`) enables
/// Chase-style retry: when straight and erasure decoding both fail, the
/// weakest symbols are flipped to their second-choice classification in
/// increasing hypothesis order and RS is retried. RS syndrome
/// self-verification gates every hypothesis, so a wrong flip cannot pass
/// silently any more than a wrong hard decision could.
pub fn decode_symbol_stream(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    seconds: Option<&[usize]>,
    profile: EccProfile,
) -> Option<Vec<u8>> {
    decode_symbol_stream_locked(symbols, confidences, seconds, profile, None)
}

/// Like [`decode_symbol_stream`], with an optional LENGTH LOCK: when the
/// caller already obtained a trustworthy payload length (hard-decision
/// header decode over high-confidence symbols), every interpretation at any
/// other length is noise — locking the length closes the whole
/// wrong-length miscorrection class and collapses the candidate search.
pub fn decode_symbol_stream_locked(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    seconds: Option<&[usize]>,
    profile: EccProfile,
    locked_len: Option<usize>,
) -> Option<Vec<u8>> {
    decode_symbol_stream_cached(
        symbols, confidences, seconds, profile, locked_len, None,
        Some(&mut FailedPayloadCache::default()), None,
    )
}

/// Like [`decode_symbol_stream_locked`], but require caller validation of
/// each candidate before accepting it. Rejected hard-decision results still
/// allow erasure recovery, alternate lengths, and Chase hypotheses. Use this
/// for checksums layered over raw profiles, including soft-combined streams;
/// filtering the final result would skip those recovery opportunities.
/// The predicate may be called repeatedly, including for the same bytes.
pub fn decode_symbol_stream_validated(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    seconds: Option<&[usize]>,
    profile: EccProfile,
    locked_len: Option<usize>,
    validator: fn(&[u8]) -> bool,
) -> Option<Vec<u8>> {
    decode_symbol_stream_cached(
        symbols, confidences, seconds, profile, locked_len, Some(validator),
        Some(&mut FailedPayloadCache::default()), None,
    )
}

fn decode_symbol_stream_cached(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    seconds: Option<&[usize]>,
    profile: EccProfile,
    locked_len: Option<usize>,
    validator: Option<fn(&[u8]) -> bool>,
    failures: Option<&mut FailedPayloadCache>,
    length_hint: Option<usize>,
) -> Option<Vec<u8>> {
    decode_symbol_stream_with_syndromes(symbols, confidences, seconds, profile,
        locked_len, validator, failures, true, length_hint)
}

fn decode_symbol_stream_with_syndromes(
    symbols: &[usize], confidences: Option<&[f64]>, seconds: Option<&[usize]>,
    profile: EccProfile, locked_len: Option<usize>, validator: Option<fn(&[u8]) -> bool>,
    mut failures: Option<&mut FailedPayloadCache>, reuse_syndromes: bool,
    length_hint: Option<usize>,
) -> Option<Vec<u8>> {
    let hinted = locked_len.is_none()
        && profile != EccProfile::Legacy
        && (profile == EccProfile::Standard || validator.is_some())
        && hint_allows(symbols.len(), profile, length_hint);
    if !frame_geometry_allows(symbols.len(), profile, locked_len) && !hinted {
        return None;
    }
    let length_hint = length_hint.filter(|_| hinted || frame_geometry_allows(symbols.len(), profile, locked_len));
    if let Some(payload) = decode_symbols_once(
        symbols, confidences, profile, false, locked_len, validator, failures.as_deref_mut(),
        length_hint,
    ) {
        return Some(payload);
    }

    // Chase retry: flip the weakest symbols to their runner-up decisions.
    let (confs, secs) = (confidences?, seconds?);
    if confs.len() != symbols.len() || secs.len() != symbols.len() {
        return None;
    }
    let mut order: Vec<usize> = (0..symbols.len())
        .filter(|&i| confs[i] < CHASE_CONF_LIMIT && secs[i] != symbols[i])
        .collect();
    order.sort_by(|&a, &b| confs[a].partial_cmp(&confs[b]).unwrap_or(std::cmp::Ordering::Equal));
    order.truncate(CHASE_MAX_FLIPS);
    if order.is_empty() {
        return None;
    }

    // Hypotheses in order of increasing flip count: cheapest (most likely)
    // corrections first.
    let k = order.len();
    let mut masks: Vec<u32> = (1..(1u32 << k)).collect();
    masks.sort_by_key(|m| m.count_ones());

    let mut trial = symbols.to_vec();
    let mut trial_conf = confs.to_vec();
    // No new syndrome cache or byte conversion on the clean first pass.
    // Snapshot only the largest possible body, excluding the length header.
    let mut retry_syndromes = reuse_syndromes.then(|| {
        let bytes = formant::symbols_to_bytes(symbols,
            (symbols.len() * BITS_PER_SYMBOL / 8).min(ENCODED_DATA_OFFSET + 255));
        RetrySyndromes::new(&bytes[ENCODED_DATA_OFFSET..])
    });
    for mask in masks {
        trial.copy_from_slice(symbols);
        trial_conf.copy_from_slice(confs);
        for (j, &pos) in order.iter().enumerate() {
            if mask & (1 << j) != 0 {
                trial[pos] = secs[pos];
                // The flip asserts this position; don't let erasure marking
                // immediately doubt it.
                trial_conf[pos] = 1.0;
            }
        }
        // Standard trials require a valid CRC; Legacy trials require the
        // agreement gate. Caller validation also gates each retry internally.
        if let Some(payload) =
            decode_symbols_once_with_syndromes(
                &trial, Some(&trial_conf), profile, true, locked_len, validator, failures.as_deref_mut(),
                retry_syndromes.as_mut(), length_hint,
            )
        {
            // A hypothesis is not an observation. The acceptance gate must
            // still agree with the original received symbols/confidences,
            // not just with the decisions this retry changed itself.
            if profile != EccProfile::Legacy
                || reencode_agreement(&payload, symbols, confidences, profile).is_some()
            {
                return Some(payload);
            }
        }
    }
    None
}

/// A classifier or Chase retry can change symbol values, not the number
/// of slots. Skip work only when no supported frame can fit this region,
/// applying exactly the existing byte-count/trailing-slop rules. This is
/// independent of signal level, pitch, vowel confidence, and channel EQ.
fn frame_geometry_allows(symbol_count: usize, profile: EccProfile, locked_len: Option<usize>) -> bool {
    let bytes = symbol_count * BITS_PER_SYMBOL / 8;
    let fits = |len| {
        let needed = profile.encoded_bytes(len);
        bytes >= needed && (profile == EccProfile::Paranoid || bytes - needed <= 3)
    };
    if let Some(len) = locked_len {
        (1..=MAX_LENGTH_VARIABLE).contains(&len) && fits(len)
    } else {
        (1..=MAX_LENGTH_VARIABLE).any(fits)
    }
}

/// Melody-derived payload length: the framing melody after the payload is
/// deterministic per profile (Coltrane's solo, then the held-E end
/// marker), so the solo's START position is structural length evidence —
/// readable through channels far beyond decodability (exact on all 70
/// hostile-channel lab cases; see examples/melody_parity_lab.rs, including
/// custom payload melodies, which leave the framing melody unchanged).
/// Scores every admissible length by the mean per-window template match
/// against the outro schedule that length implies and returns the argmax.
/// This is a HINT: it can never lock a length, outrank header evidence,
/// or deliver without the ordinary CRC/validator gates. Legacy frames
/// sing one arpeggio throughout — no transition — and are excluded.
fn melody_length_hint(
    spectra: &mut AlignedSpectra<'_>,
    data_start: usize,
    region_symbols: usize,
    profile: EccProfile,
) -> Option<usize> {
    if profile == EccProfile::Legacy {
        return None;
    }
    let header_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
    let body_symbols = region_symbols.checked_sub(header_symbols)?;
    let mut best: Option<(f64, usize)> = None;
    for len in 1..=MAX_LENGTH_VARIABLE {
        let body_bytes = profile.encoded_bytes(len) - ENCODED_DATA_OFFSET;
        if 4 * body_bytes > body_symbols {
            break; // encoded_bytes is monotone in len
        }
        let outro_bytes = body_bytes - len;
        if outro_bytes == 0 {
            continue;
        }
        let outro_symbols = 4 * outro_bytes + PREAMBLE_LEN;
        let mut score = 0.0f64;
        for s in 4 * len..4 * body_bytes {
            let row = spectra.note_scores(data_start + header_symbols + s)?;
            let hz = outro_f0(s - 4 * len, outro_symbols);
            score += row[formant::bank_note_index(hz)];
        }
        let mean = score / (4 * outro_bytes) as f64;
        if best.map_or(true, |(b, _)| mean > b) {
            best = Some((mean, len));
        }
    }
    best.map(|(_, len)| len)
}

/// The melody hint supplies structural end evidence, so its gated
/// candidate may decode as a prefix of a longer region (trailing voiced
/// background after a damaged header) that the ordinary slop rules would
/// reject. Integrity gates are unchanged.
fn hint_allows(symbol_count: usize, profile: EccProfile, hint: Option<usize>) -> bool {
    hint.is_some_and(|h| symbol_count * BITS_PER_SYMBOL / 8 >= profile.encoded_bytes(h))
}

/// One full decode attempt (hard-decision + erasure fallback) over a fixed
/// symbol stream.
///
/// `gate_all`: apply the Legacy anti-miscorrection agreement gate to every success
/// (used for Chase trials). Without it, only the risky paths are gated:
/// erasure-fallback decodes (their residual syndrome check is weak once
/// most parity is spent on erasures) and derived-length candidates. The
/// pre-existing trusted path — hard-decision decode at the header-approved
/// length — stays ungated. Standard always verifies CRC before accepting
/// either hard-decision or erasure results. Caller validation runs at the
/// same stage, so rejection never bypasses the erasure fallback.
fn decode_symbols_once(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    profile: EccProfile,
    gate_all: bool,
    locked_len: Option<usize>,
    validator: Option<fn(&[u8]) -> bool>,
    failures: Option<&mut FailedPayloadCache>,
    length_hint: Option<usize>,
) -> Option<Vec<u8>> {
    decode_symbols_once_with_syndromes(symbols, confidences, profile, gate_all,
        locked_len, validator, failures, None, length_hint)
}

fn decode_symbols_once_with_syndromes(
    symbols: &[usize], confidences: Option<&[f64]>, profile: EccProfile, gate_all: bool,
    locked_len: Option<usize>, validator: Option<fn(&[u8]) -> bool>,
    mut failures: Option<&mut FailedPayloadCache>, mut retry_syndromes: Option<&mut RetrySyndromes>,
    length_hint: Option<usize>,
) -> Option<Vec<u8>> {
    {
        if symbols.is_empty() {
            return None;
        }

        let min_len_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if symbols.len() < min_len_symbols {
            return None;
        }

        let max_bytes = (symbols.len() * BITS_PER_SYMBOL) / 8;
        let data_encoded = formant::symbols_to_bytes(symbols, max_bytes);

        if data_encoded.len() < ENCODED_DATA_OFFSET {
            return None;
        }

        // Decode length block: try hard-decision first, then erasure fallback
        let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
        let header_block = rs_length.prepare_decode(&data_encoded[..ENCODED_DATA_OFFSET]);
        let header_len = header_block
            .decode_with_erasures(&[])
            .or_else(|| {
                let confs = confidences?;
                let erasures = compute_byte_erasures(confs, 0, ENCODED_DATA_OFFSET, ENCODED_DATA_OFFSET - 1);
                if erasures.is_empty() {
                    return None;
                }
                header_block.decode_with_erasures(&erasures)
            })
            .map(|d| d[0] as usize)
            .filter(|&l| l > 0 && l <= MAX_LENGTH_VARIABLE);

        // The 3-byte length header only corrects 1 error, making it the
        // weakest link no matter how much payload parity there is. Under the
        // paranoid profile, fall back to brute-forcing candidate lengths —
        // RS syndrome self-verification (plus the transfer layer's CRC)
        // makes wrong candidates fail explicitly.
        let mut candidates: Vec<usize> = Vec::new();
        if let Some(l) = locked_len {
            // Length is locked: no other interpretation is admissible.
            candidates.push(l);
        } else if let Some(l) = header_len {
            candidates.push(l);
        }
        // Melody-derived length (the solo's start position): one
        // prioritized extra candidate right after real header evidence —
        // never a lock. Only checked profiles use it (Standard's CRC or a
        // caller validator); Legacy sings one arpeggio throughout and has
        // no melody transition to read. The candidate passes the same
        // decode/integrity gates as any inferred length.
        let melody_hint = length_hint.filter(|&h| {
            locked_len.is_none()
                && (1..=MAX_LENGTH_VARIABLE).contains(&h)
                && Some(h) != header_len
                && profile != EccProfile::Legacy
                && (profile == EccProfile::Standard || validator.is_some())
                && profile.encoded_bytes(h) <= data_encoded.len()
        });
        if let Some(h) = melody_hint {
            candidates.push(h);
        }
        if locked_len.is_some() {
            // No alternates under a length lock.
        } else if profile == EccProfile::Paranoid {
            for l in (1..=MAX_LENGTH_VARIABLE).rev() {
                if Some(l) != header_len
                    && Some(l) != melody_hint
                    && profile.encoded_bytes(l) <= data_encoded.len()
                {
                    candidates.push(l);
                }
            }
        } else {
            // Standard / Legacy: the 3-byte length header corrects only 1
            // error — the weakest link. The stream length itself implies the
            // payload length, since header + payload + parity must fill the
            // data region (up to a few bytes of end-detection slop). Add the
            // implied lengths as gated candidates even when the header
            // decoded (it may have decoded WRONG — it's that weak). Only
            // lengths with ≥4 parity bytes qualify — weaker codewords have a
            // real chance of syndrome-checking garbage, and this path also
            // runs against truncated streams from false end-preamble matches
            // mid-reception.
            for l in (1..=MAX_LENGTH_VARIABLE).rev() {
                let total = profile.encoded_bytes(l);
                if Some(l) != header_len
                    && Some(l) != melody_hint
                    && profile.ecc_bytes(l) >= 4
                    && total <= data_encoded.len()
                    && data_encoded.len() - total <= 3
                {
                    candidates.push(l);
                }
            }
        }

        // All candidate bodies start at the same symbol. Prepare their
        // common confidence ordering lazily, at most once for this fixed
        // hypothesis; a clean hard-decision result needs no erasure work.
        let (max_body_bytes, max_ecc) = candidates.iter().fold((0, 0), |(body, ecc), &len| {
            let total = profile.encoded_bytes(len);
            if total <= data_encoded.len() {
                (body.max(total - ENCODED_DATA_OFFSET), ecc.max(profile.ecc_bytes(len)))
            } else {
                (body, ecc)
            }
        });
        let body_erasures = PreparedByteErasures::new(
            confidences, formant::symbols_for_bytes(ENCODED_DATA_OFFSET), max_body_bytes,
        );
        let mut body_decoder = PrefixDecoder::new(
            &data_encoded[ENCODED_DATA_OFFSET..ENCODED_DATA_OFFSET + max_body_bytes], max_ecc,
        );
        let mut best_risky: Option<(f64, Vec<u8>)> = None;
        for len in candidates {
            let n_ecc = profile.ecc_bytes(len);
            let protected_len = profile.protected_bytes(len);
            let expected_total = profile.encoded_bytes(len);
            if data_encoded.len() < expected_total {
                continue;
            }
            // A short valid-looking header cannot explain a much longer
            // reception. Allow the end preamble and silence/alignment slop,
            // but do not decode an arbitrary prefix as a complete message.
            // The melody hint is the one exception: the solo's position is
            // structural end evidence, so its candidate may decode as a
            // prefix of trailing voiced background — CRC still gates.
            if profile != EccProfile::Paranoid
                && Some(len) != melody_hint
                && data_encoded.len() - expected_total > 3 {
                continue;
            }

            // Decode payload block: hard-decision first, then erasure fallback
            let payload_block = &data_encoded[ENCODED_DATA_OFFSET..expected_total];
            let mut erasures = None;
            let compute_erasures = || body_erasures.for_length(protected_len + n_ecc, n_ecc);
            if failures.as_deref_mut().is_some_and(|cache| {
                cache.contains(profile, len, payload_block, &mut erasures, compute_erasures)
            }) {
                continue;
            }
            let prepared = if let Some(retry) = retry_syndromes.as_deref_mut() {
                retry.prepare_decode(payload_block, protected_len, n_ecc)
            } else {
                body_decoder.prepare_decode(protected_len, n_ecc)
            };
            let mut used_erasures = false;
            // Reject miscorrections before deciding whether to try erasures.
            // A bad-CRC hard result must not hide a valid erasure decode.
            let mut had_profile_valid_result = false;
            let mut validate = |block| {
                let payload = profile.verify_payload(block)?;
                // Caller predicates may depend on external state. Never
                // memoize their rejections as deterministic RS/CRC failures.
                had_profile_valid_result = true;
                validator.map_or(true, |v| v(&payload)).then_some(payload)
            };
            let decoded = prepared
                .decode_with_erasures(&[])
                .and_then(&mut validate)
                .or_else(|| {
                    let erasures = erasures.get_or_insert_with(compute_erasures);
                    if erasures.is_empty() {
                        return None;
                    }
                    used_erasures = true;
                    prepared.decode_with_erasures(erasures)
                        .and_then(&mut validate)
                });
            if decoded.is_none() && !had_profile_valid_result {
                if let Some(cache) = failures.as_deref_mut() {
                    cache.record(
                        profile, len, payload_block,
                        erasures.get_or_insert_with(compute_erasures),
                    );
                }
            }
            if let Some(payload) = decoded {
                // Legacy anti-miscorrection gate for risky successes. Erasure
                // decoding with most parity spent on erasures retains only a
                // weak residual syndrome check (~1% garbage acceptance), and
                // derived lengths / Chase trials multiply attempt volume —
                // without a gate, misaligned or truncated streams eventually
                // "decode". Standard has already passed CRC validation;
                // Paranoid relies on the transfer layer's CRC-32 gate.
                let pristine_header = Some(len) == header_len
                    && rs_length.encode(&[len as u8]) == data_encoded[..ENCODED_DATA_OFFSET];
                let approved = pristine_header || Some(len) == locked_len;
                let risky = gate_all || used_erasures || !approved;
                if profile != EccProfile::Legacy || !risky {
                    return Some(payload);
                }
                // Risky successes compete: a corrupted header can decode to
                // a wrong length whose codeword still RS-"corrects", so the
                // candidate whose re-encoding best matches the received
                // stream wins instead of whichever was tried first.
                if let Some(score) = reencode_agreement(&payload, symbols, confidences, profile)
                {
                    if best_risky.as_ref().map_or(true, |(s, _)| score > *s) {
                        best_risky = Some((score, payload));
                    }
                }
            }
        }

        best_risky.map(|(_, p)| p)
    }
}

/// Reconstruct a candidate's header, payload, and parity observations.
fn encoded_payload_symbols(payload: &[u8], profile: EccProfile) -> Vec<usize> {
    let len = payload.len();
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let mut encoded = rs_length.encode(&[len as u8]);
    let protected = profile.protect_payload(payload);
    let rs_data = ReedSolomon::new(protected.len(), profile.ecc_bytes(len));
    encoded.extend_from_slice(&rs_data.encode(&protected));
    formant::bytes_to_symbols(&encoded)
}

/// Confidence-aware miscorrection filter. True corrections usually change
/// low-confidence symbols; disagreement with many high-confidence symbols
/// suggests a bad candidate. This is a heuristic, not an integrity checksum.
fn reencode_agreement(
    payload: &[u8],
    symbols: &[usize],
    confidences: Option<&[f64]>,
    profile: EccProfile,
) -> Option<f64> {
    let len = payload.len();
    if len == 0 || len > MAX_LENGTH_VARIABLE {
        return None;
    }
    let expect_syms = encoded_payload_symbols(payload, profile);
    let n = expect_syms.len().min(symbols.len());
    if n == 0 {
        return None;
    }

    let matches = (0..n).filter(|&i| expect_syms[i] == symbols[i]).count();
    let overall = matches as f64 / n as f64;

    let Some(confs) = confidences.filter(|c| c.len() >= n) else {
        // No confidence data: require near-total agreement (an RS decode
        // that "corrected" 3 clean bytes of a wrong interpretation already
        // reaches ~0.87).
        return (overall >= 0.92).then_some(overall);
    };

    const HIGH_CONF: f64 = 0.55;
    let mut high_total = 0usize;
    let mut high_bad = 0usize;
    for i in 0..n {
        if confs[i] >= HIGH_CONF {
            high_total += 1;
            if expect_syms[i] != symbols[i] {
                high_bad += 1;
            }
        }
    }
    if high_total < 20 {
        // Confidence signal too flat to locate the disagreements — fall
        // back to demanding near-total agreement.
        return (overall >= 0.92).then_some(overall);
    }
    // A decode RS could truly produce disagrees with the received stream in
    // at most the corrected bytes (≤ ~27% of bytes via the erasure path,
    // realistically 1-2 wrong symbols per corrupted byte) — overall
    // agreement below 0.80 means the interpretation is wrong, not noisy.
    let high_bad_frac = high_bad as f64 / high_total as f64;
    (overall >= 0.80 && high_bad_frac <= 0.06).then_some(overall - high_bad_frac)
}

/// Calibrate classification centers from measured pilot (vowel, centroid)
/// pairs. Returns None — keep defaults — unless the measurement passes the
/// sanity gate: all four vowels seen, centers strictly ascending, ≥ 120 Hz
/// apart, and within ±150 Hz of the static expected centers. Calibration can
/// only help, never hurt.
fn calibrate_centers(measured: &[(usize, f64)]) -> Option<[f64; NUM_VOWELS]> {
    let mut sums = [0.0f64; NUM_VOWELS];
    let mut counts = [0usize; NUM_VOWELS];
    for &(vowel, centroid) in measured {
        if vowel < NUM_VOWELS {
            sums[vowel] += centroid;
            counts[vowel] += 1;
        }
    }
    let mut centers = [0.0f64; NUM_VOWELS];
    for v in 0..NUM_VOWELS {
        if counts[v] == 0 {
            return None;
        }
        centers[v] = sums[v] / counts[v] as f64;
        if (centers[v] - formant::EXPECTED_F2_CENTROIDS[v]).abs() > 150.0 {
            return None;
        }
        if v > 0 && centers[v] - centers[v - 1] < 120.0 {
            return None;
        }
    }
    Some(centers)
}

/// Confidence threshold below which a symbol is considered unreliable.
/// Normalized confidence is 0..1; weaker symbols cluster below ~0.3.
/// Lowered to 0.25 to be more aggressive with erasure marking.
const ERASURE_CONFIDENCE_THRESHOLD: f64 = 0.25;

/// Convert per-symbol confidences to byte-level RS erasure positions.
///
/// With 2-bit symbols, each byte is exactly 4 symbols (no straddling).
/// Byte confidence = min of contributing symbol confidences.
/// Returns positions of the weakest bytes (below threshold), capped at
/// `2 * ecc_len / 3` to leave room for hard error correction.
fn compute_byte_erasures(
    symbol_confidences: &[f64],
    sym_offset: usize,
    num_bytes: usize,
    ecc_len: usize,
) -> Vec<usize> {
    // RS can correct: 2*errors + erasures <= ecc_len
    // Allow up to 2/3 of ECC capacity for erasures, leaving room for some errors
    let max_erasures = (ecc_len * 2) / 3;
    if max_erasures == 0 {
        return Vec::new();
    }

    // Compute which symbols contribute to each byte based on BITS_PER_SYMBOL
    let mut byte_confs: Vec<(usize, f64)> = (0..num_bytes)
        .map(|byte_idx| {
            // Byte i spans bits [i*8, i*8+7]
            let bit_start = byte_idx * 8;
            let bit_end = bit_start + 7;
            // Symbol j spans bits [j*BITS_PER_SYMBOL, j*BITS_PER_SYMBOL + BITS_PER_SYMBOL - 1]
            let first_sym = bit_start / BITS_PER_SYMBOL;
            let last_sym = bit_end / BITS_PER_SYMBOL;

            // Take min confidence of all contributing symbols
            let mut min_conf = 1.0f64;
            for sym_idx in first_sym..=last_sym {
                let conf = symbol_confidences
                    .get(sym_offset + sym_idx)
                    .copied()
                    .unwrap_or(0.0);
                min_conf = min_conf.min(conf);
            }
            (byte_idx, min_conf)
        })
        .collect();

    // Sort by confidence ascending (weakest first)
    byte_confs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    byte_confs
        .iter()
        .filter(|(_, conf)| *conf < ERASURE_CONFIDENCE_THRESHOLD)
        .take(max_erasures)
        .map(|(pos, _)| *pos)
        .collect()
}

/// Every candidate body is a prefix of the same fixed confidence stream.
/// Sort its weak bytes once, then project that ordering onto each prefix.
/// Stable tie ordering, missing-symbol confidence, and the erasure budget
/// exactly match `compute_byte_erasures`. A new Chase hypothesis gets a new
/// view, so promoted decisions cannot reuse stale confidence information.
struct PreparedByteErasures<'a> {
    confidences: Option<&'a [f64]>,
    sym_offset: usize,
    max_bytes: usize,
    order: std::cell::OnceCell<Vec<usize>>,
}

impl<'a> PreparedByteErasures<'a> {
    fn new(confidences: Option<&'a [f64]>, sym_offset: usize, max_bytes: usize) -> Self {
        Self {
            confidences,
            sym_offset,
            // No supported RS body exceeds 255 bytes, regardless of the
            // amount of trailing audio in a recording.
            max_bytes: max_bytes.min(255),
            order: std::cell::OnceCell::new(),
        }
    }

    fn for_length(&self, num_bytes: usize, ecc_len: usize) -> Vec<usize> {
        debug_assert!(num_bytes <= self.max_bytes);
        let max_erasures = ecc_len * 2 / 3;
        let Some(confs) = self.confidences else { return Vec::new() };
        if num_bytes == 0 || max_erasures == 0 {
            return Vec::new();
        }
        let order = self.order.get_or_init(|| {
            let mut weak: Vec<(usize, f64)> = (0..self.max_bytes).filter_map(|byte| {
                let first = byte * 8 / BITS_PER_SYMBOL;
                let last = (byte * 8 + 7) / BITS_PER_SYMBOL;
                let mut confidence = 1.0f64;
                for symbol in first..=last {
                    confidence = confidence.min(confs.get(self.sym_offset + symbol).copied().unwrap_or(0.0));
                }
                (confidence < ERASURE_CONFIDENCE_THRESHOLD).then_some((byte, confidence))
            }).collect();
            // `min` starts at 1.0 and ignores NaNs, so every retained value
            // is ordered. Filtering strong bytes before this stable sort
            // preserves the reference's ordering, including equal ties.
            weak.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            weak.into_iter().map(|(byte, _)| byte).collect()
        });
        order.iter().copied().filter(|&byte| byte < num_bytes).take(max_erasures).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn resample_test_clock(samples: &[f32], ratio: f64) -> Vec<f32> {
        resample_test_clock_variable(samples, |_| ratio)
    }

    fn echo_test_capture(
        profile: EccProfile, ratio: f64, delay_ms: usize, bad_crc: bool, contour: usize,
    ) -> (Vec<u8>, Vec<f32>, usize) {
        echo_test_capture_len(profile, ratio, delay_ms, bad_crc, contour, 64)
    }

    fn echo_test_capture_len(
        profile: EccProfile, ratio: f64, delay_ms: usize, bad_crc: bool, contour: usize, length: usize,
    ) -> (Vec<u8>, Vec<f32>, usize) {
        let mut state = (length * 7919 + contour * 379 + 17) as u32;
        let mut payload: Vec<_> = (0..length).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }).collect();
        if profile == EccProfile::Paranoid {
            payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        }
        let track: Vec<_> = if contour == 3 { default_byte_pitch(payload.len()) } else { (0..payload.len()).map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let step = if contour == 1 {
                if i % 2 == 0 { 19.0 } else { 0.0 }
            } else { ((state >> 24) % 20) as f64 + [-0.3, 0.0, 0.3][i % 3] };
            440.0 * 2.0f64.powf((47.0 + step - 69.0) / 12.0)
        }).collect() };
        let mut song = crate::encoder::encode_with_profile(&payload, 50, Some(&track), profile).unwrap();
        if bad_crc {
            let mut protected = profile.protect_payload(&payload);
            let crc_index = if profile == EccProfile::Standard { payload.len() } else { payload.len() - 4 };
            protected[crc_index] ^= 0x80;
            let mut bytes = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1).encode(&[payload.len() as u8]);
            bytes.extend(ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len())).encode(&protected));
            let data_symbols = formant::bytes_to_symbols(&bytes);
            let payload_start = PREAMBLE_LEN + PILOT_LEN + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
            let payload_end = payload_start + formant::symbols_for_bytes(payload.len());
            let total = song.len() / SYMBOL_TOTAL_SAMPLES;
            for (i, symbol) in data_symbols.into_iter().enumerate() {
                let pos = PREAMBLE_LEN + PILOT_LEN + i;
                let hz = if pos < payload_start { framing_lick_f0(pos, payload_start) }
                    else if pos < payload_end { track[(pos - payload_start) / 4] }
                    else { outro_f0(pos - payload_end, total - payload_end) };
                let slot = formant::synthesize_symbol(symbol, hz, 0.5 * melody_accent(pos), pos);
                for (sample, value) in song[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES]
                    .iter_mut().zip(slot) { *sample = value as f32; }
            }
        }
        let lead = 8 * SYMBOL_TOTAL_SAMPLES + SYMBOL_TOTAL_SAMPLES / 3 + 137;
        let mut input = vec![0.0; lead];
        input.extend_from_slice(&song);
        input.resize(input.len() + 12 * SYMBOL_TOTAL_SAMPLES, 0.0);
        let delay = delay_ms * SAMPLE_RATE as usize / 1000;
        let mut echo = input.clone();
        for i in delay..echo.len() { echo[i] += 0.5 * input[i - delay]; }
        (payload, resample_test_clock(&echo, ratio), (lead as f64 / ratio).round() as usize)
    }

    #[test]
    fn silent_tail_releases_short_acquisition_and_preserves_unread_audio() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            for initial_level in [0.0, 0.125] {
                let mut decoder = Decoder::with_profile(profile);
                if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                decoder.history = PREAMBLE_START_VOWELS.iter().map(|&vowel|
                    formant::synthesize_symbol(vowel, 185.0, 0.5, 0).into_iter().map(|s| s as f32).collect()).collect();
                decoder.start_receiving(true);
                decoder.rx_rms_ema = initial_level;
                decoder.last_attempt = Some((vec![2; 17], vec![0.25; 17]));
                let previous = decoder.last_attempt.clone();
                let quiet = vec![0.0; SYMBOL_TOTAL_SAMPLES];
                let needed = if profile == EccProfile::Paranoid { 4 } else { 6 };
                for _ in 0..PILOT_LEN {
                    assert!(decoder.decode(&quiet).unwrap().is_none());
                    assert!(decoder.is_receiving(), "preserve the existing pilot-collection period");
                }
                for _ in 0..needed - 1 {
                    assert!(decoder.decode(&quiet).unwrap().is_none());
                    assert!(decoder.is_receiving(), "retain the original silence-run threshold");
                }
                let mut payload = b"after quiet tail".to_vec();
                if profile == EccProfile::Paranoid { payload.extend(crate::checksum::crc32(&payload).to_le_bytes()); }
                let mut next = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
                next.resize(next.len() + 12 * SYMBOL_TOTAL_SAMPLES, 0.0);
                decoder.sample_buffer = next.clone();
                assert!(decoder.process_symbol_window(&quiet).unwrap().is_none());
                assert!(!decoder.is_receiving());
                assert_eq!(decoder.last_attempt, previous, "a short noise acquisition adds no failed-copy observation");
                assert_eq!(decoder.sample_buffer.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                    next.iter().map(|s| s.to_bits()).collect::<Vec<_>>());
                assert_eq!(decoder.decode(&[]).unwrap(), Some(payload));
                assert!(decoder.decode(&[]).unwrap().is_none());
            }
        }
    }

    #[test]
    fn short_echo_clock_estimates_preserve_gates_and_recover_frames() {
        for (contour, delay, ratio) in [(1, 60, 1.02), (2, 60, 1.02), (3, 65, 0.98)] {
            let (payload, audio, offset) = echo_test_capture_len(EccProfile::Standard, ratio, delay, false, contour, 4);
            let cut = offset - SYMBOL_TOTAL_SAMPLES;
            for gain in [0.0001f32, 1.0, 10000.0] {
                let mut decoder = Decoder::new();
                decoder.recorded_audio = audio[cut..].iter().map(|s| s * gain).collect();
                decoder.symbols_recorded = decoder.recorded_audio.len() / SYMBOL_TOTAL_SAMPLES;
                assert!(envelope_clock_period_with_lags(&decoder.recorded_audio, 8, 24).is_none(),
                    "this regression must exercise the short-recording path");
                let period = envelope_clock_period(&decoder.recorded_audio).expect("short echo clock");
                assert!((period - SYMBOL_TOTAL_SAMPLES as f64 / ratio).abs() < SYMBOL_TOTAL_SAMPLES as f64 * 0.0025);
                let (frame, symbols, confidences) = decoder.try_envelope_clock().expect("checked short recovery");
                assert_eq!(frame.payload, payload);
                assert_eq!(symbols.len(), formant::symbols_for_bytes(EccProfile::Standard.encoded_bytes(payload.len())));
                assert_eq!(symbols.len(), confidences.len());
                assert!(decoder.envelope_clock_phases.get() <= MAX_ENVELOPE_CLOCK_PHASES);
            }
            let (_, bad, bad_offset) = echo_test_capture_len(EccProfile::Standard, ratio, delay, true, contour, 4);
            let mut decoder = Decoder::new();
            decoder.recorded_audio = bad[bad_offset - SYMBOL_TOTAL_SAMPLES..].to_vec();
            decoder.symbols_recorded = decoder.recorded_audio.len() / SYMBOL_TOTAL_SAMPLES;
            assert!(decoder.try_envelope_clock().is_none(), "short spans do not waive integrity");
        }
        for slots in [32, 64, 72, 96, 128] {
            let count = slots * SYMBOL_TOTAL_SAMPLES;
            for value in [0.0, 0.125, -0.125] {
                assert!(envelope_clock_period(&vec![value; count]).is_none());
            }
            let mut state = 73u32;
            let noise: Vec<_> = (0..count).map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            }).collect();
            assert!(envelope_clock_period(&noise).is_none());
        }
        for ratio in [0.98, 1.0, 1.02] {
            let (_, long, _) = echo_test_capture(EccProfile::Standard, ratio, 25, false, 1);
            assert_eq!(envelope_clock_period(&long), envelope_clock_period_with_lags(&long, 8, 24),
                "long recordings retain the original estimator bit for bit");
        }
    }

    #[test]
    fn echo_clock_frequency_cache_preserves_samples_ffts_and_endpoints() {
        for ratio in [0.98, 1.02] {
            let (payload, audio, offset) = echo_test_capture(EccProfile::Standard, ratio, 60, false, 1);
            let count = PREAMBLE_LEN + PILOT_LEN
                + formant::symbols_for_bytes(EccProfile::Standard.encoded_bytes(payload.len()));
            let mut original = AlignedSpectra::clocked(&audio, offset, count);
            let raw: Vec<Vec<f32>> = (0..count).map(|i| original.get(i).unwrap().to_vec()).collect();
            let starts = original.clock_starts.clone();
            let mut cached = original.frequency_compensated(0, count).unwrap();
            let mut uncached = AlignedSpectra::clocked(&audio, offset, count);
            let mut lazy = uncached.frequency_compensated(0, count).unwrap();
            assert_eq!(uncached.transforms, 0, "constructing a lazy retry must not run FFTs");
            assert!((cached.frequency_ratio - ratio).abs() < 0.001);
            assert_eq!(cached.clock_starts, starts);
            for i in 0..count {
                assert_eq!(cached.get(i), lazy.get(i));
                assert_eq!(original.get(i), Some(raw[i].as_slice()), "raw spectra must remain unchanged");
            }
            assert_eq!(original.transforms, count);
            assert_eq!(cached.transforms, 0, "reuse every already-cached FFT");
            assert_eq!(lazy.transforms, count, "one FFT per new analysis window");
            assert!(cached.get(count).is_none());
            assert!(original.frequency_compensated(0, count + 1).is_none());
            assert!(cached.frequency_compensated(0, count).is_none(), "never normalize twice");
            let data_start = PREAMBLE_LEN + PILOT_LEN;
            assert_eq!(cached.decoded_frame(payload.clone(), data_start, EccProfile::Standard).audio_end,
                original.decoded_frame(payload, data_start, EccProfile::Standard).audio_end);
            assert_eq!(original.audio.as_ptr(), audio.as_ptr());
            assert_eq!(cached.audio.as_ptr(), audio.as_ptr());
            assert_eq!(original.clock_starts, starts);
        }
        let (_, audio, offset) = echo_test_capture(EccProfile::Standard, 1.0, 60, false, 1);
        let mut nominal = AlignedSpectra::uniform_clocked(&audio, offset, SYMBOL_TOTAL_SAMPLES as f64);
        let count = nominal.windows.len();
        assert!(nominal.frequency_compensated(0, count).is_none());
        for invalid in [0.0, f64::NAN, f64::INFINITY, -1.0,
            (SYMBOL_TOTAL_SAMPLES + crate::audio::CLOCK_RADIUS + 1) as f64] {
            assert!(AlignedSpectra::uniform_clocked(&audio, offset, invalid).windows.is_empty());
        }
    }

    #[test]
    fn trial_failure_cache_preserves_rungs_and_reuses_exact_failures() {
        let mut saved = 0;
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            for (ratio, delay, prefix) in [(1.0, 0, usize::MAX), (0.98, 60, 180), (1.02, 25, 260)] {
                let (_, audio, offset) = echo_test_capture(profile, ratio, delay, false, 1);
                let cut = offset - SYMBOL_TOTAL_SAMPLES;
                let count = prefix.min((audio.len() - cut) / SYMBOL_TOTAL_SAMPLES);
                let make = || {
                    let mut decoder = Decoder::with_profile(profile);
                    if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                    decoder.recorded_audio = audio[cut..].to_vec();
                    decoder.symbols_recorded = count;
                    decoder.preamble_pos_hint = Some(1);
                    decoder
                };
                let mut uncached = make();
                let original = uncached.trial_decode_cached(count, None);
                let mut cached = make();
                let mut failures = FailedPayloadCache::default();
                let actual = cached.trial_decode_cached(count, Some(&mut failures));
                assert_eq!(actual.as_ref().map(|f| (&f.payload, f.audio_end)),
                    original.as_ref().map(|f| (&f.payload, f.audio_end)), "{profile:?}, {ratio}, prefix {prefix}");
                assert_eq!(cached.last_attempt, uncached.last_attempt);
                assert_eq!(cached.trial_length_searches.get(), uncached.trial_length_searches.get());
                assert!(cached.trial_failure_records.get() <= uncached.trial_failure_records.get());
                saved += uncached.trial_failure_records.get() - cached.trial_failure_records.get();
                assert!(failures.by_length.len() <= MAX_LENGTH_VARIABLE + 1);
                assert!(failures.by_length.iter().flatten().all(|entry|
                    entry.bytes.len() <= 255 && entry.erasures.len() <= 255));
            }
        }
        assert!(saved > 0, "sharing the probe-local cache must eliminate repeated RS work");
    }

    #[test]
    fn echo_clock_envelope_requires_stable_bounded_rhythm() {
        for contour in [1, 2] {
            for ratio in [0.98, 1.0, 1.02] {
                let (_, audio, _) = echo_test_capture(EccProfile::Standard, ratio, 25, false, contour);
                let expected = SYMBOL_TOTAL_SAMPLES as f64 / ratio;
                for gain in [0.0001f32, 1.0, 10000.0] {
                    let scaled: Vec<_> = audio.iter().map(|sample| sample * gain).collect();
                    if ratio == 1.0 {
                        assert!(envelope_clock_period(&scaled).is_none(), "nominal rhythms need no new search");
                    } else {
                        let period = envelope_clock_period(&scaled).expect("stable echo clock");
                        assert!((period - expected).abs() < SYMBOL_TOTAL_SAMPLES as f64 * 0.001,
                            "contour {contour}, ratio {ratio}, gain {gain}, period {period}");
                        assert!((period - SYMBOL_TOTAL_SAMPLES as f64).abs() <= crate::audio::CLOCK_RADIUS as f64);
                    }
                }
            }
        }
        let samples = 128 * SYMBOL_TOTAL_SAMPLES;
        for value in [0.0, 0.125, -0.125] {
            assert!(envelope_clock_period(&vec![value; samples]).is_none());
        }
        let mut state = 73u32;
        let noise: Vec<_> = (0..samples).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }).collect();
        assert!(envelope_clock_period(&noise).is_none());
        let (_, slow, _) = echo_test_capture(EccProfile::Standard, 0.98, 25, false, 1);
        let (_, fast, _) = echo_test_capture(EccProfile::Standard, 1.02, 25, false, 1);
        let changing: Vec<_> = slow.into_iter().chain(fast).collect();
        assert!(envelope_clock_period(&changing).is_none(), "do not flatten changing clocks into one rate");
        assert!(envelope_clock_period(&changing[..32 * SYMBOL_TOTAL_SAMPLES]).is_none());
    }

    #[test]
    fn echo_clock_retries_require_sync_length_and_integrity() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for bad_crc in [false, true] {
                let (payload, audio, offset) = echo_test_capture(profile, 0.98, 60, bad_crc, 1);
                let cut = offset - SYMBOL_TOTAL_SAMPLES;
                let mut decoder = Decoder::with_profile(profile);
                if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                decoder.state = DecoderState::Receiving;
                decoder.recorded_audio = audio[cut..].to_vec();
                decoder.symbols_recorded = decoder.recorded_audio.len() / SYMBOL_TOTAL_SAMPLES;
                decoder.sample_buffer = vec![0.125, -0.0, 0.25];
                decoder.last_attempt = Some((vec![2; 17], vec![0.25; 17]));
                let recording = decoder.recorded_audio.clone();
                let unread = decoder.sample_buffer.clone();
                let previous = decoder.last_attempt.clone();
                let result = decoder.try_envelope_clock();
                assert_eq!(result.as_ref().map(|r| &r.0.payload), (!bad_crc).then_some(&payload));
                assert!(decoder.envelope_clock_phases.get() <= MAX_ENVELOPE_CLOCK_PHASES);
                assert_eq!(decoder.recorded_audio, recording);
                assert_eq!(decoder.sample_buffer, unread);
                assert_eq!(decoder.last_attempt, previous, "failed or speculative attempts cannot publish observations");
                assert!(matches!(decoder.state, DecoderState::Receiving));
                if let Some((frame, symbols, confidences)) = result {
                    assert_eq!(symbols.len(), confidences.len());
                    assert_eq!(symbols.len(), formant::symbols_for_bytes(profile.encoded_bytes(payload.len())));
                    let expected: Vec<_> = recording[frame.audio_end..].iter().chain(&unread)
                        .map(|sample| sample.to_bits()).collect();
                    assert_eq!(decoder.finish_frame(frame), payload);
                    assert_eq!(decoder.sample_buffer.iter().map(|s| s.to_bits()).collect::<Vec<_>>(), expected);
                }
                decoder.recorded_audio = recording;
                decoder.set_payload_validator(|_| false);
                assert!(decoder.try_envelope_clock().is_none(), "caller rejection must remain final");
                let mut grid = AlignedSpectra::clocked(&audio, offset, usize::MAX);
                assert!(decoder.try_clock_frequency_compensation(&mut grid, 0, Some(payload.len())).is_none());
            }
        }
        let (payload, audio, offset) = echo_test_capture(EccProfile::Standard, 0.98, 60, false, 1);
        for profile in [EccProfile::Legacy, EccProfile::Paranoid] {
            let mut decoder = Decoder::with_profile(profile);
            decoder.recorded_audio = audio.clone();
            let mut grid = AlignedSpectra::clocked(&audio, offset, usize::MAX);
            assert!(decoder.try_clock_frequency_compensation(&mut grid, 0, Some(payload.len())).is_none());
            assert!(decoder.try_envelope_clock().is_none());
            assert_eq!(decoder.envelope_clock_phases.get(), 0);
        }
        let decoder = Decoder::new();
        let mut grid = AlignedSpectra::clocked(&audio, offset, usize::MAX);
        assert!(decoder.try_clock_frequency_compensation(&mut grid, 0, Some(1)).is_none());
        let mut shifted = AlignedSpectra::clocked(&audio, offset + 2 * SYMBOL_TOTAL_SAMPLES, usize::MAX);
        assert!(decoder.try_clock_frequency_compensation(&mut shifted, 0, Some(payload.len())).is_none(),
            "a clock estimate cannot replace frame synchronization");
    }

    #[test]
    fn echo_clock_recovery_preserves_the_next_melody() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for (length, contour, delay) in [(64, 2, 25), (4, 1, 60), (4, 2, 60), (4, 3, 65), (10, 3, 25), (16, 2, 25)] {
                for ratio in [0.98, 1.02] {
                    let (first, mut capture, _) = echo_test_capture_len(profile, ratio, delay, false, contour, length);
                    let mut second = b"next melody".to_vec();
                    if profile == EccProfile::Paranoid { second.extend(crate::checksum::crc32(&second).to_le_bytes()); }
                    let track = default_byte_pitch(second.len());
                    let next = crate::encoder::encode_with_profile(&second, 50, Some(&track), profile).unwrap();
                    capture.extend(resample_test_clock(&next, 2.0 - ratio));
                    // A short-frame outro can resemble a fresh preamble. Allow
                    // its unchanged eight-slot pilot collection plus the six-slot
                    // silence threshold and one fractional-window margin. Keep
                    // the original 64-byte regression's tail unchanged.
                    let tail_slots = if length == 64 { 12 } else { PILOT_LEN + 6 + 1 };
                    capture.resize(capture.len() + tail_slots * SYMBOL_TOTAL_SAMPLES, 0.0);
                    for chunk_size in [511, capture.len()] {
                        let mut decoder = Decoder::with_profile(profile);
                        if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                        let mut received = Vec::new();
                        for chunk in capture.chunks(chunk_size) {
                            if let Some(bytes) = decoder.decode(chunk).unwrap() { received.push(bytes); }
                            while let Some(bytes) = decoder.decode(&[]).unwrap() { received.push(bytes); }
                        }
                        assert_eq!(received, vec![first.clone(), second.clone()],
                            "{profile:?}, length {length}, contour {contour}, delay {delay}, ratio {ratio}, chunk size {chunk_size}");
                        assert!(!decoder.is_receiving(), "{profile:?}, length {length}, contour {contour}, delay {delay}, ratio {ratio}, chunk size {chunk_size}, recorded {}, silence {}, level {}",
                            decoder.symbols_recorded, decoder.rx_silence_run, decoder.rx_rms_ema);
                    }
                }
            }
        }
    }

    fn resample_test_clock_variable(samples: &[f32], ratio: impl Fn(f64) -> f64) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len());
        let mut position = 0.0;
        while position < (samples.len() - 1) as f64 {
            let index = position as usize;
            let fraction = (position - index as f64) as f32;
            out.push(samples[index] * (1.0 - fraction) + samples[index + 1] * fraction);
            position += ratio(position);
        }
        out
    }

    fn probed_test_decoder(profile: EccProfile, payload: &[u8]) -> (Decoder, Vec<f32>) {
        let frame = crate::encoder::encode_with_profile(payload, 50, None, profile).unwrap();
        let audio = resample_test_clock(&frame, 0.98);
        let mut decoder = Decoder::with_profile(profile);
        decoder.set_payload_validator(has_test_crc);
        decoder.state = DecoderState::Receiving;
        decoder.recorded_audio = audio[..HEADER_PROBE_WINDOWS * SYMBOL_TOTAL_SAMPLES].to_vec();
        decoder.symbols_recorded = HEADER_PROBE_WINDOWS;
        for (start, _, length) in decoder.predict_frame_ends(HEADER_PROBE_WINDOWS) {
            if let Some(length) = length { decoder.remember_frame_length(start, length); }
        }
        assert!(decoder.frame_clock_hint.is_some(), "{profile:?}: the actual prefix must support the hint");
        (decoder, audio)
    }

    #[test]
    fn block_clock_drift_does_not_discard_long_melodies() {
        for len in [140usize, 64, 17, 1] {
            let payload: Vec<_> = (0..len).map(|i| (i * 73 + 19) as u8).collect();
            let song = crate::encode(&payload, 50).unwrap();
            for ratio in [0.999, 1.001, 0.995, 1.005, 0.98, 1.02] {
                let mut capture = vec![0.0; SYMBOL_TOTAL_SAMPLES / 3];
                capture.extend(resample_test_clock(&song, ratio));
                capture.resize(capture.len() + 12 * SYMBOL_TOTAL_SAMPLES, 0.0);
                let mut rx = Decoder::new();
                let mut received = Vec::new();
                for chunk in capture.chunks(2048) {
                    if let Ok(Some(bytes)) = rx.decode(chunk) { received.push(bytes); }
                    assert!(rx.clock_grids <= MAX_CLOCK_CANDIDATES);
                    assert!(rx.clock_candidate_peak <= MAX_CLOCK_CANDIDATES);
                }
                assert_eq!(received, vec![payload.clone()], "length {len}, ratio {ratio}");
            }
        }
    }

    #[test]
    fn clocked_spectra_preserve_samples_and_bound_cached_work() {
        let payload: Vec<_> = (0..64).map(|i| (i * 73 + 19) as u8).collect();
        let song = crate::encode(&payload, 50).unwrap();
        let eq = vec![1.0; FFT_SIZE];
        for ratio in [0.98, 1.02] {
            let audio = resample_test_clock(&song, ratio);
            for offset in [0, SYMBOL_TOTAL_SAMPLES / 3, audio.len(), usize::MAX] {
                for limit in [0, 5, usize::MAX] {
                    let mut spectra = AlignedSpectra::clocked(&audio, offset, limit);
                    let starts = spectra.clock_starts.as_ref().unwrap().clone();
                    assert_eq!(starts.len(), spectra.windows.len());
                    assert!(starts.len() <= limit);
                    let bound = audio.len().checked_sub(offset).and_then(|n| n.checked_sub(SAMPLES_PER_SYMBOL))
                        .map_or(0, |n| n / (SYMBOL_TOTAL_SAMPLES - crate::audio::CLOCK_RADIUS) + 1);
                    assert!(starts.len() <= bound);
                    for pair in starts.windows(2) {
                        assert!(pair[1] > pair[0]);
                        assert!((pair[1] - pair[0]).abs_diff(SYMBOL_TOTAL_SAMPLES) <= crate::audio::CLOCK_RADIUS);
                        assert!(pair[1] >= pair[0] + SAMPLES_PER_SYMBOL);
                    }
                    for (index, &start) in starts.iter().enumerate() {
                        let mut expected = vec![0.0; SAMPLES_PER_SYMBOL];
                        fft::power_spectrum(&audio[start..start + SAMPLES_PER_SYMBOL], &mut expected);
                        assert_eq!(spectra.get(index).unwrap(), &expected[..RETAINED_SPECTRUM_BINS]);
                        assert_same_spectral_class(spectra.classify_full(index, &formant::EXPECTED_F2_CENTROIDS),
                            Some(formant::classify_symbol_full(&expected, &formant::EXPECTED_F2_CENTROIDS)));
                        assert_same_spectral_class(spectra.classify_eq(index, &eq),
                            Some(formant::classify_symbol_eq(&expected, &eq)));
                        assert_same_spectral_class(spectra.classify_template(index, &eq),
                            Some(formant::classify_symbol_template(&expected, &eq)));
                        assert_same_spectral_class(spectra.classify_f0a(index, &eq),
                            Some(formant::classify_symbol_f0_adaptive(&expected, &eq)));
                        assert_eq!(spectra.transforms, index + 1, "every classifier shares one FFT");
                    }
                    assert!(spectra.get(starts.len()).is_none());
                }
            }
        }
    }

    #[test]
    fn block_clock_holds_its_measured_rate_when_guards_are_obscured() {
        let payload: Vec<_> = (0..64).map(|i| (i * 73 + 19) as u8).collect();
        let song = crate::encode(&payload, 50).unwrap();
        let obscured = vec![0.25; SYMBOL_TOTAL_SAMPLES];
        assert_eq!(crate::audio::guard_clock_evidence(&obscured), None);
        assert_eq!(crate::audio::guard_clock_step(&obscured), SYMBOL_TOTAL_SAMPLES,
            "live streaming retains its original fallback");
        for ratio in [0.98, 1.02] {
            let mut audio = resample_test_clock(&song[..120 * SYMBOL_TOTAL_SAMPLES], ratio);
            audio.extend(obscured.repeat(40));
            let spectra = AlignedSpectra::clocked(&audio, 0, usize::MAX);
            let starts = spectra.clock_starts.unwrap();
            let tail = &starts[starts.len() - 30..];
            let held_step = tail[1] - tail[0];
            assert!(held_step.abs_diff(SYMBOL_TOTAL_SAMPLES) > crate::audio::CLOCK_RADIUS / 4,
                "missing guard evidence must not reset an established clock: {ratio}, step {held_step}");
            assert!(held_step.abs_diff(SYMBOL_TOTAL_SAMPLES) <= crate::audio::CLOCK_RADIUS);
            assert!(tail.windows(2).all(|pair| pair[1] - pair[0] == held_step),
                "an obscured guard cannot update the measured period");
        }
    }

    #[test]
    fn phased_header_hints_bound_incomplete_frame_trials() {
        let payload: Vec<_> = (0..crate::transfer::MAX_TRANSFER_PAYLOAD)
            .map(|i| (i * 73 + crate::transfer::MAX_TRANSFER_PAYLOAD) as u8).collect();
        let frame = crate::transfer::encode(&payload, 50).unwrap();
        for ratio in [0.98, 1.02] {
            let warped = resample_test_clock(&frame, ratio);
            let mut input = warped[..SYMBOL_TOTAL_SAMPLES].repeat(100);
            input.resize(input.len() + SYMBOL_TOTAL_SAMPLES / 3, 0.0);
            input.extend_from_slice(&warped[..64 * SYMBOL_TOTAL_SAMPLES]);
            let mut decoder = Decoder::with_profile(EccProfile::Paranoid);
            decoder.set_payload_validator(has_test_crc);
            for chunk in input.chunks(511) {
                assert!(decoder.decode(chunk).unwrap().is_none());
            }
            assert!(decoder.frame_length_hints.values().any(|&length| length == Some(payload.len() + 4)),
                "the phased scan must preserve its intact header, ratio {ratio}");
            assert!(decoder.frame_length_hints.len() <= decoder.symbols_recorded);
            assert!(decoder.trial_decode(decoder.symbols_recorded - PREAMBLE_LEN).is_none());
            assert_eq!(decoder.trial_length_searches.get(), 0,
                "an incomplete frame with a supported length cannot require speculative length searches");
        }
    }

    #[test]
    fn clock_hint_rechecks_evidence_integrity_and_lifecycle() {
        let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let (mut decoder, audio) = probed_test_decoder(profile, &payload);
            let hint = decoder.frame_clock_hint.unwrap();
            let hints = decoder.frame_length_hints.clone();
            let old_attempt = (vec![2; 9], vec![0.5; 9]);
            decoder.last_attempt = Some(old_attempt.clone());
            decoder.sample_buffer = vec![0.125, -0.0];
            let prefix = decoder.recorded_audio.clone();
            assert!(decoder.try_clock_hint().is_none(), "an incomplete body cannot deliver");
            assert_eq!(decoder.recorded_audio, prefix);
            assert_eq!(decoder.last_attempt, Some(old_attempt.clone()));
            decoder.recorded_audio = audio;
            decoder.symbols_recorded = decoder.recorded_audio.len() / SYMBOL_TOTAL_SAMPLES;
            let checked = decoder.try_clock_hint().expect("the intact clocked body should decode");
            assert_eq!(checked.payload, payload);
            assert!(checked.audio_end < decoder.recorded_audio.len());

            decoder.last_attempt = Some(old_attempt.clone());
            decoder.set_payload_validator(|_| false);
            let before: Vec<_> = decoder.recorded_audio.iter().map(|sample| sample.to_bits()).collect();
            assert!(decoder.try_clock_hint().is_none(), "a hint cannot bypass the caller validator");
            assert_eq!(decoder.recorded_audio.iter().map(|sample| sample.to_bits()).collect::<Vec<_>>(), before);
            assert_eq!(decoder.last_attempt, Some(old_attempt.clone()));
            assert_eq!(decoder.sample_buffer.iter().map(|sample| sample.to_bits()).collect::<Vec<_>>(),
                vec![0.125f32.to_bits(), (-0.0f32).to_bits()]);
            assert_eq!(decoder.frame_length_hints, hints);
            assert_eq!(decoder.state, DecoderState::Receiving);

            decoder.set_payload_validator(has_test_crc);
            let start = hint.offset / SYMBOL_TOTAL_SAMPLES;
            decoder.remember_frame_length(start, hint.length - 1);
            assert!(decoder.try_clock_hint().is_none(), "conflicting lengths cannot supply a priority hint");
            decoder.frame_length_hints = hints;
            let old_offset = decoder.frame_clock_hint.as_ref().unwrap().offset;
            decoder.frame_clock_hint.as_mut().unwrap().offset += 2 * SYMBOL_TOTAL_SAMPLES;
            decoder.remember_frame_length(start + 2, hint.length);
            let grids_before = decoder.priority_clock_grids;
            assert!(decoder.try_clock_hint().is_none(), "a later hint cannot jump ahead of another retained frame");
            assert_eq!(decoder.priority_clock_grids, grids_before, "ordering must be checked before any clock work");
            decoder.frame_clock_hint.as_mut().unwrap().offset = old_offset;
            decoder.frame_length_hints.remove(&(start + 2));
            if profile != EccProfile::Standard {
                decoder.validator = None;
                assert!(decoder.try_clock_hint().is_none(), "unchecked profiles cannot use the prefix shortcut");
                decoder.set_payload_validator(has_test_crc);
            }
            let slot: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.5, 0)
                .into_iter().map(|sample| sample as f32).collect();
            decoder.recorded_audio[..HEADER_PROBE_WINDOWS * SYMBOL_TOTAL_SAMPLES]
                .copy_from_slice(&slot.repeat(HEADER_PROBE_WINDOWS));
            assert!(decoder.try_clock_hint().is_none(), "the phase-local header must still verify");
            assert_eq!(decoder.last_attempt, Some(old_attempt));
            decoder.reset();
            assert!(decoder.frame_clock_hint.is_none());
            decoder.frame_clock_hint = Some(hint);
            decoder.start_receiving(false);
            assert!(decoder.frame_clock_hint.is_none(), "reacquisition must not inherit an old phase");
        }
    }

    #[test]
    fn failed_clock_hint_preserves_the_original_analysis_search() {
        let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut observations = Vec::new();
            for use_hint in [false, true] {
                let (mut decoder, audio) = probed_test_decoder(profile, &payload);
                if !use_hint { decoder.frame_clock_hint = None; }
                decoder.recorded_audio = audio.clone();
                decoder.symbols_recorded = audio.len() / SYMBOL_TOTAL_SAMPLES;
                decoder.set_payload_validator(|_| false);
                assert!(decoder.analyze(false).is_err());
                assert_eq!(decoder.priority_clock_grids, usize::from(use_hint));
                assert_eq!(decoder.nominal_grids, 32, "no nominal hypotheses may be removed");
                assert!(decoder.clock_grids > 0 && decoder.clock_grids <= MAX_CLOCK_CANDIDATES);
                assert_eq!(decoder.state, DecoderState::Receiving);
                assert_eq!(decoder.recorded_audio, audio);
                observations.push((decoder.clock_grids, decoder.clock_candidate_peak, decoder.last_attempt.clone()));
                decoder.set_payload_validator(has_test_crc);
                assert_eq!(decoder.analyze(false).unwrap(), Some(payload.clone()),
                    "rejecting a priority attempt cannot hide a later permitted recovery");
            }
            assert_eq!(observations[0], observations[1],
                "a rejected priority pass must retain the same fallback grids and combining observations");
        }
    }

    #[test]
    fn supported_slow_clock_completes_before_nominal_alignment_search() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let length = if profile == EccProfile::Standard { MAX_LENGTH_VARIABLE - 4 } else { 120 };
            let mut payload: Vec<_> = (0..length).map(|i| (i * 73 + length) as u8).collect();
            payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
            let track = (profile == EccProfile::Standard).then(|| default_byte_pitch(payload.len()));
            let frame = crate::encoder::encode_with_profile(&payload, 50, track.as_deref(), profile).unwrap();
            let warped = resample_test_clock(&frame, 0.98);
            let mut capture = warped[..SYMBOL_TOTAL_SAMPLES].repeat(100);
            capture.resize(capture.len() + SYMBOL_TOTAL_SAMPLES / 3, 0.0);
            capture.extend(warped);
            capture.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            let mut decoder = Decoder::with_profile(profile);
            if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
            let mut received = Vec::new();
            for chunk in capture.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    assert_eq!(decoder.nominal_grids, 0,
                        "{profile:?}: a supported drifting clock should not need the nominal search first");
                    assert_eq!(decoder.priority_clock_grids, 1);
                    assert_eq!(decoder.clock_grids, 0);
                    received.push(bytes);
                }
            }
            assert_eq!(received, vec![payload]);
        }
    }

    #[test]
    fn clock_completion_hints_wait_for_the_encoded_body() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let length = if profile == EccProfile::Standard { MAX_LENGTH_VARIABLE } else { 124 };
            let payload: Vec<_> = (0..length).map(|i| (i * 73 + length) as u8).collect();
            let frame = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
            for ratio in [1.0, 0.98, 1.02] {
                let warped = resample_test_clock(&frame, ratio);
                let lead = 100 * SYMBOL_TOTAL_SAMPLES + SYMBOL_TOTAL_SAMPLES / 3;
                let mut input = warped[..SYMBOL_TOTAL_SAMPLES].repeat(100);
                input.resize(lead, 0.0);
                input.extend_from_slice(&warped[..64 * SYMBOL_TOTAL_SAMPLES]);
                let mut decoder = Decoder::with_profile(profile);
                if profile == EccProfile::Paranoid { decoder.set_payload_validator(|_| false); }
                for chunk in input.chunks(511) {
                    assert!(decoder.decode(chunk).unwrap().is_none());
                }
                let frame_slots = PREAMBLE_LEN + PILOT_LEN
                    + formant::symbols_for_bytes(profile.encoded_bytes(length));
                let body_arrival = lead + (frame_slots as f64 * SYMBOL_TOTAL_SAMPLES as f64 / ratio).ceil() as usize;
                let hints = decoder.frame_end_hints.as_ref().unwrap();
                let first = *hints.iter().min().unwrap() * SYMBOL_TOTAL_SAMPLES;
                assert!(first >= body_arrival,
                    "{profile:?}, ratio {ratio}: probe at {first}, encoded body arrives at {body_arrival}");
                let latest = lead + (frame_slots + PREAMBLE_LEN + 3
                    + crate::audio::clock_slack_windows(frame_slots)) * SYMBOL_TOTAL_SAMPLES;
                assert!(hints.iter().all(|&end| end * SYMBOL_TOTAL_SAMPLES <= latest));
                if ratio == 1.0 {
                    assert!(first <= body_arrival + (PREAMBLE_LEN + 3) * SYMBOL_TOTAL_SAMPLES,
                        "an ordinary frame must retain its timely probe");
                }
                if ratio > 1.0 {
                    let nominal = lead / SYMBOL_TOTAL_SAMPLES + frame_slots + PREAMBLE_LEN + 2;
                    assert!(first < nominal * SYMBOL_TOTAL_SAMPLES, "a measured fast clock gains an earlier probe");
                    assert!(hints.contains(&nominal) || hints.contains(&(nominal + 1)),
                        "the original nominal deadline remains available if the clock changes");
                    let fallback = nominal + crate::audio::clock_slack_windows(frame_slots);
                    assert!(hints.contains(&fallback) || hints.contains(&(fallback + 1)),
                        "the original slow-clock fallback remains available too");
                }
            }
        }
    }

    #[test]
    fn clock_probe_estimates_require_support_and_stay_bounded() {
        let length = MAX_LENGTH_VARIABLE;
        let payload: Vec<_> = (0..length).map(|i| (i * 73 + length) as u8).collect();
        let frame = crate::encode(&payload, 50).unwrap();
        let frame_symbols = PREAMBLE_LEN + PILOT_LEN
            + formant::symbols_for_bytes(EccProfile::Standard.encoded_bytes(length));
        for ratio in [1.0, 0.98, 1.02] {
            let warped = resample_test_clock(&frame, ratio);
            for gain in [0.001, 1.0, 1000.0] {
                let mut decoder = Decoder::new();
                decoder.recorded_audio = warped[..HEADER_PROBE_WINDOWS * SYMBOL_TOTAL_SAMPLES]
                    .iter().map(|sample| sample * gain).collect();
                let mut estimates = 0;
                for phase in 0..16 {
                    let offset = phase * SYMBOL_TOTAL_SAMPLES / 16;
                    if let Some((score, delay)) = decoder.clock_probe_shift(
                        offset, HEADER_PROBE_WINDOWS, frame_symbols, length,
                    ) {
                        estimates += 1;
                        assert!(score.is_finite());
                        if ratio < 1.0 { assert!(delay > (PREAMBLE_LEN + 2) as isize); }
                        else { assert!(delay < 0); }
                        assert!(delay.unsigned_abs() <= crate::audio::clock_slack_windows(frame_symbols));
                        assert!(decoder.clock_probe_shift(
                            offset, HEADER_PROBE_WINDOWS, frame_symbols, length - 1,
                        ).is_none(), "timing evidence cannot establish a different header length");
                    }
                    assert!(decoder.clock_probe_shift(offset, 24, frame_symbols, length).is_none(),
                        "require enough post-acquisition clock evidence");
                }
                if ratio == 1.0 { assert_eq!(estimates, 0, "do not postpone an ordinary clock"); }
                else { assert!(estimates > 0, "clear clock-drift evidence survives gain {gain}"); }
                assert!(decoder.clock_probe_shift(usize::MAX, HEADER_PROBE_WINDOWS, frame_symbols, length).is_none());
            }
        }
    }

    #[test]
    fn sync_only_three_slot_silence_completes_before_the_next_sender() {
        let mut payload: Vec<_> = (0..13).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        let mut next = vec![0x53];
        next.extend(crate::checksum::crc32(&next).to_le_bytes());
        let next_track = default_byte_pitch(next.len());
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut song = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
            let header_start = (PREAMBLE_LEN + PILOT_LEN) * SYMBOL_TOTAL_SAMPLES;
            let header_slots = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
            let vowel: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.5, 0)
                .into_iter().map(|sample| sample as f32).collect();
            song[header_start..header_start + header_slots * SYMBOL_TOTAL_SAMPLES]
                .copy_from_slice(&vowel.repeat(header_slots));
            song.truncate(song.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES);
            let mut decoder = Decoder::with_profile(profile);
            decoder.set_payload_validator(has_test_crc);
            for chunk in song.chunks(511) {
                assert!(decoder.decode(chunk).unwrap().is_none());
            }
            assert!(!decoder.frame_start_hints.is_empty(), "require the actual verified sync");
            assert!(decoder.frame_length_hints.is_empty(), "do not invent a length from the damaged header");
            let quiet = vec![0.0; SYMBOL_TOTAL_SAMPLES];
            for _ in 0..2 {
                assert!(decoder.decode(&quiet).unwrap().is_none());
            }
            assert_eq!(decoder.rx_silence_run, 2, "exercise strict silence, not periodic residuals");
            assert_eq!(decoder.quiet_completion_probes, 0);
            assert_eq!(decoder.decode(&quiet).unwrap(), Some(payload.clone()),
                "{profile:?}: verified completion must fit the three-slot gap");
            let mut following = crate::encoder::encode_with_profile(&next, 50, Some(&next_track), profile).unwrap();
            following.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            let mut received = Vec::new();
            for chunk in following.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) { received.push(bytes); }
            }
            assert_eq!(received, vec![next.clone()], "{profile:?}: the next sender must still arrive once");
        }
    }

    #[test]
    fn failed_strict_silence_probe_preserves_paranoid_preview_and_finalization() {
        let mut payload: Vec<_> = (0..13).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        for trusted_header in [false, true] {
            let mut song = crate::encoder::encode_with_profile(&payload, 50, None, EccProfile::Paranoid).unwrap();
            if !trusted_header {
                let header_start = (PREAMBLE_LEN + PILOT_LEN) * SYMBOL_TOTAL_SAMPLES;
                let header_slots = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                let vowel: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.5, 0)
                    .into_iter().map(|sample| sample as f32).collect();
                song[header_start..header_start + header_slots * SYMBOL_TOTAL_SAMPLES]
                    .copy_from_slice(&vowel.repeat(header_slots));
            }
            song.truncate(song.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES);
            let mut decoder = Decoder::with_profile(EccProfile::Paranoid);
            decoder.set_payload_validator(|_| false);
            for chunk in song.chunks(511) {
                assert!(decoder.decode(chunk).unwrap().is_none());
            }
            assert!(!decoder.frame_start_hints.is_empty());
            assert_eq!(!decoder.frame_length_hints.is_empty(), trusted_header);
            let preview = decoder.preview_symbols.clone();
            let recent = decoder.recent_symbols.clone();
            let centroids = decoder.recent_rx_centroids.clone();
            let quiet = vec![0.0; SYMBOL_TOTAL_SAMPLES];
            for count in 1..=3 {
                assert!(decoder.decode(&quiet).unwrap().is_none(), "the caller's rejection must be honored");
                assert!(decoder.is_receiving());
                assert_eq!(decoder.rx_silence_run, count);
                assert_eq!(decoder.preview_symbols, preview, "silence cannot pollute preview");
                assert_eq!(decoder.recent_symbols, recent);
                assert_eq!(decoder.recent_rx_centroids, centroids);
                assert_eq!(decoder.quiet_completion_probes, usize::from(count == 3 && !trusted_header));
            }
            let mut expected = song.clone();
            expected.extend(quiet.repeat(3));
            assert_eq!(decoder.recorded_audio, expected, "a failed probe retains the entire copy");
            assert!(decoder.decode(&quiet).is_err(), "fourth-slot failed-copy finalization is unchanged");
            assert!(!decoder.is_receiving());
            assert_eq!(decoder.quiet_completion_probes, usize::from(!trusted_header));
        }
    }

    #[test]
    fn sync_only_quiet_probes_never_supply_or_override_a_length() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let mut decoder = Decoder::with_profile(profile);
            let start = 7;
            let minimum = PREAMBLE_LEN + PILOT_LEN
                + formant::symbols_for_bytes(profile.encoded_bytes(1));
            let earliest = start + minimum - crate::audio::clock_slack_windows(minimum);
            decoder.symbols_recorded = earliest;
            decoder.preamble_pos_hint = Some(start);
            assert!(!decoder.can_probe_sync_only_tail(), "a Listening lock is not verified sync");
            decoder.frame_start_hints.push(start);
            assert_eq!(decoder.can_probe_sync_only_tail(), profile == EccProfile::Standard);
            decoder.set_payload_validator(has_test_crc);
            decoder.symbols_recorded = earliest - 1;
            assert!(!decoder.can_probe_sync_only_tail());
            decoder.symbols_recorded = earliest;
            assert!(decoder.can_probe_sync_only_tail());
            assert!(!decoder.has_possible_complete_frame(), "unknown length remains unknown");
            assert!(decoder.frame_length_hints.is_empty());
            decoder.quiet_completion_probes = MAX_QUIET_COMPLETION_PROBES;
            assert!(!decoder.can_probe_sync_only_tail(), "share the existing per-recording budget");
            decoder.quiet_completion_probes = 0;
            decoder.remember_frame_length(start, 64);
            assert!(!decoder.can_probe_sync_only_tail(), "a known longer frame cannot use the unknown-length fallback");
            assert!(!decoder.has_possible_complete_frame());
            decoder.remember_frame_length(start, 63);
            assert!(!decoder.can_probe_sync_only_tail(), "conflicting evidence cannot become sync-only evidence");
        }
    }

    #[test]
    fn sync_only_quiet_probe_preserves_an_incomplete_melody_and_its_unknown_length() {
        let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut song = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
            let header_start = (PREAMBLE_LEN + PILOT_LEN) * SYMBOL_TOTAL_SAMPLES;
            let header_slots = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
            let vowel: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.5, 0)
                .into_iter().map(|sample| sample as f32).collect();
            song[header_start..header_start + header_slots * SYMBOL_TOTAL_SAMPLES]
                .copy_from_slice(&vowel.repeat(header_slots));
            for sample in &mut song[100 * SYMBOL_TOTAL_SAMPLES..110 * SYMBOL_TOTAL_SAMPLES] {
                *sample *= 0.01;
            }
            let mut decoder = Decoder::with_profile(profile);
            decoder.set_payload_validator(has_test_crc);
            for chunk in song[..110 * SYMBOL_TOTAL_SAMPLES].chunks(511) {
                assert!(decoder.decode(chunk).unwrap().is_none());
            }
            assert!(!decoder.frame_start_hints.is_empty(), "the real sync must qualify");
            assert!(decoder.frame_length_hints.is_empty(), "the damaged header cannot supply a length");
            assert!(decoder.quiet_completion_probes > 0, "{profile:?}: exercise the sync-only probe");
            assert!(decoder.quiet_completion_probes <= MAX_QUIET_COMPLETION_PROBES);
            assert!(decoder.is_receiving());
            assert_eq!(decoder.recorded_audio, song[..110 * SYMBOL_TOTAL_SAMPLES]);
            let mut tail = song[110 * SYMBOL_TOTAL_SAMPLES..].to_vec();
            tail.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            let mut received = Vec::new();
            for chunk in tail.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) { received.push(bytes); }
            }
            assert_eq!(received, vec![payload.clone()], "{profile:?}: recover the complete melody after the rejected early probe");
        }
    }

    #[test]
    fn quiet_completion_requires_supported_length_integrity_and_a_bounded_budget() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let mut decoder = Decoder::with_profile(profile);
            let length = 64;
            let start = 7;
            let symbols = PREAMBLE_LEN + PILOT_LEN
                + formant::symbols_for_bytes(profile.encoded_bytes(length));
            let earliest = start + symbols - crate::audio::clock_slack_windows(symbols);
            decoder.symbols_recorded = earliest;
            decoder.frame_start_hints.push(start);
            assert!(!decoder.has_possible_complete_frame(), "sync alone cannot invent a length");
            decoder.remember_frame_length(start, length);
            assert_eq!(decoder.has_possible_complete_frame(), profile == EccProfile::Standard,
                "unchecked profiles cannot gain an early completion path");
            decoder.set_payload_validator(has_test_crc);
            decoder.symbols_recorded = earliest - 1;
            assert!(!decoder.has_possible_complete_frame(), "the bounded body must be able to fit");
            decoder.symbols_recorded += 1;
            for used in 0..MAX_QUIET_COMPLETION_PROBES {
                decoder.quiet_completion_probes = used;
                assert!(decoder.has_possible_complete_frame());
            }
            decoder.quiet_completion_probes = MAX_QUIET_COMPLETION_PROBES;
            assert!(!decoder.has_possible_complete_frame());
            decoder.quiet_completion_probes = 0;
            decoder.remember_frame_length(start, length - 1);
            assert!(!decoder.has_possible_complete_frame(), "conflicting lengths stay unsupported");
            decoder.rx_quiet_probe_run = 4;
            decoder.quiet_completion_probes = MAX_QUIET_COMPLETION_PROBES;
            decoder.start_receiving(false);
            assert_eq!(decoder.rx_quiet_probe_run, 0);
            assert_eq!(decoder.quiet_completion_probes, 0);
            assert!(!decoder.has_possible_complete_frame());
            decoder.rx_quiet_probe_run = 4;
            decoder.quiet_completion_probes = MAX_QUIET_COMPLETION_PROBES;
            decoder.reset();
            assert_eq!(decoder.rx_quiet_probe_run, 0);
            assert_eq!(decoder.quiet_completion_probes, 0);
        }
    }

    #[test]
    fn quiet_completion_failure_preserves_a_late_musical_phrase() {
        let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut song = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
            let body = PREAMBLE_LEN + PILOT_LEN
                + formant::symbols_for_bytes(profile.encoded_bytes(payload.len()));
            let quiet_start = body - crate::audio::clock_slack_windows(body) + 2;
            let stop = quiet_start + 10;
            assert!(stop < body);
            for sample in &mut song[quiet_start * SYMBOL_TOTAL_SAMPLES..stop * SYMBOL_TOTAL_SAMPLES] {
                *sample *= 0.01;
            }
            let mut decoder = Decoder::with_profile(profile);
            decoder.set_payload_validator(has_test_crc);
            for chunk in song[..stop * SYMBOL_TOTAL_SAMPLES].chunks(511) {
                assert!(decoder.decode(chunk).unwrap().is_none(), "an incomplete frame cannot deliver");
            }
            assert!(decoder.quiet_completion_probes > 0, "{profile:?}: the quiet phrase must exercise the probe");
            assert!(decoder.quiet_completion_probes <= MAX_QUIET_COMPLETION_PROBES);
            assert!(decoder.is_receiving(), "{profile:?}: a rejected probe must keep receiving");
            assert_eq!(decoder.symbols_recorded, stop);
            assert_eq!(decoder.recorded_audio, song[..stop * SYMBOL_TOTAL_SAMPLES],
                "the quiet phrase and the preceding notes must remain intact");
            let mut tail = song[stop * SYMBOL_TOTAL_SAMPLES..].to_vec();
            tail.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            let mut received = Vec::new();
            for chunk in tail.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) { received.push(bytes); }
            }
            assert_eq!(received, vec![payload.clone()], "{profile:?}: the whole quiet melody still decodes");
        }
    }

    #[test]
    fn delivery_context_is_bounded_exact_and_consumed_once() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            for validated in [false, true] {
                let mut decoder = Decoder::with_profile(profile);
                if validated { decoder.set_payload_validator(has_test_crc); }
                let end = 5 * SYMBOL_TOTAL_SAMPLES + 127;
                let mut audio: Vec<f32> = (0..10 * SYMBOL_TOTAL_SAMPLES).map(|i| i as f32).collect();
                audio[end - 1] = -0.0;
                audio[end - 2] = f32::from_bits(1);
                let supported = profile == EccProfile::Standard || validated;
                let context_start = if supported { end - DELIVERY_CONTEXT_WINDOWS * SYMBOL_TOTAL_SAMPLES } else { end };
                let expected_context: Vec<_> = audio[context_start..end].iter().map(|sample| sample.to_bits()).collect();
                let expected_history = audio[end..end + 4 * SYMBOL_TOTAL_SAMPLES].to_vec();
                decoder.recorded_audio = audio.clone();
                decoder.finish_frame(DecodedFrame { payload: vec![0x53], audio_end: end });
                assert_eq!(decoder.delivery_context.iter().map(|sample| sample.to_bits()).collect::<Vec<_>>(), expected_context);
                assert!(decoder.delivery_context.len() <= DELIVERY_CONTEXT_WINDOWS * SYMBOL_TOTAL_SAMPLES);
                assert_eq!(decoder.sample_buffer, audio[end..], "lookbehind is separate from unread lookahead");
                assert!(decoder.history.is_empty());
                assert!(decoder.recent_vowels.is_empty(), "delivered context is not a Listening observation");

                decoder.history = expected_history.chunks_exact(SYMBOL_TOTAL_SAMPLES).map(<[f32]>::to_vec).collect();
                decoder.sample_buffer.drain(..expected_history.len());
                let unread = decoder.sample_buffer.clone();
                decoder.start_receiving(true);
                let mut expected = expected_context.clone();
                expected.extend(expected_history.iter().map(|sample| sample.to_bits()));
                assert_eq!(decoder.recorded_audio.iter().map(|sample| sample.to_bits()).collect::<Vec<_>>(), expected);
                let context_windows = if supported { DELIVERY_CONTEXT_WINDOWS } else { 0 };
                assert_eq!(decoder.symbols_recorded, context_windows + 4);
                assert_eq!(decoder.preamble_pos_hint, Some(context_windows));
                assert_eq!(decoder.sample_buffer, unread);
                assert!(decoder.delivery_context.is_empty(), "context is consumed once at acquisition");
                decoder.start_receiving(false);
                assert!(decoder.recorded_audio.is_empty(), "an unrelated reacquisition cannot reuse old context");
                decoder.delivery_context = vec![0.25; SYMBOL_TOTAL_SAMPLES];
                decoder.reset();
                assert!(decoder.delivery_context.is_empty());
            }
        }
    }

    #[test]
    fn delivery_context_expires_when_listening_history_is_discontinuous() {
        let quiet = vec![0.0; SYMBOL_TOTAL_SAMPLES];
        let mut decoder = Decoder::new();
        decoder.delivery_context = vec![0.25; DELIVERY_CONTEXT_WINDOWS * SYMBOL_TOTAL_SAMPLES];
        for _ in 0..3 {
            assert!(decoder.decode(&quiet).unwrap().is_none());
            assert!(!decoder.delivery_context.is_empty());
        }
        assert!(decoder.decode(&quiet).unwrap().is_none());
        assert!(decoder.delivery_context.is_empty(), "cleared history cannot be joined to old context");

        decoder.reset();
        decoder.delivery_context = vec![0.25; DELIVERY_CONTEXT_WINDOWS * SYMBOL_TOTAL_SAMPLES];
        decoder.history = vec![quiet.clone(); HISTORY_WINDOWS];
        assert!(decoder.decode(&quiet).unwrap().is_none());
        assert_eq!(decoder.history.len(), HISTORY_WINDOWS);
        assert!(decoder.delivery_context.is_empty(), "slid history is no longer adjacent to delivery");
    }

    #[test]
    fn successful_frame_preserves_exact_lookahead_before_unread_audio() {
        let mut decoder = Decoder::new();
        decoder.recorded_audio = (0..8 * SYMBOL_TOTAL_SAMPLES).map(|i| i as f32).collect();
        let end = 3 * SYMBOL_TOTAL_SAMPLES + 127;
        decoder.recorded_audio[end] = -0.0;
        decoder.recorded_audio[end + 1] = f32::from_bits(1);
        decoder.sample_buffer = vec![0.125, -0.25, 2.0];
        let expected: Vec<_> = decoder.recorded_audio[end..].iter().chain(&decoder.sample_buffer)
            .map(|sample| sample.to_bits()).collect();
        decoder.state = DecoderState::Receiving;
        decoder.active_count = 12;
        decoder.silence_count = 3;
        decoder.history.push(vec![42.0; SYMBOL_TOTAL_SAMPLES]);
        decoder.recent_vowels.extend(PREAMBLE_START_VOWELS);
        decoder.frame_start_hints.push(0);
        decoder.remember_frame_length(0, 1);
        decoder.frame_end_hints = Some(vec![42]);
        decoder.frame_clock_hint = Some(ClockHint { score: 0.0, offset: 0, length: 1 });
        decoder.rx_quiet_probe_run = 4;
        decoder.quiet_completion_probes = MAX_QUIET_COMPLETION_PROBES;
        let payload = decoder.finish_frame(DecodedFrame { payload: vec![0x53], audio_end: end });
        assert_eq!(payload, [0x53]);
        assert_eq!(decoder.sample_buffer.iter().map(|sample| sample.to_bits()).collect::<Vec<_>>(), expected);
        assert!(decoder.recorded_audio.is_empty());
        assert!(decoder.history.is_empty());
        assert!(decoder.recent_vowels.is_empty());
        assert!(decoder.frame_start_hints.is_empty());
        assert!(decoder.frame_length_hints.is_empty());
        assert!(decoder.frame_end_hints.is_none());
        assert!(decoder.frame_clock_hint.is_none());
        assert_eq!(decoder.rx_quiet_probe_run, 0);
        assert_eq!(decoder.quiet_completion_probes, 0);
        assert_eq!(decoder.symbols_recorded, 0);
        assert_eq!(decoder.active_count, 0);
        assert_eq!(decoder.silence_count, 0);
        assert!(!decoder.is_receiving());
    }

    #[test]
    fn successful_frame_endpoint_uses_the_winning_alignment() {
        let payload = b"different clocks";
        let data_start = PREAMBLE_LEN + PILOT_LEN;
        let offset = SYMBOL_TOTAL_SAMPLES / 3;
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let frame = crate::encoder::encode_with_profile(payload, 50, None, profile).unwrap();
            let last = data_start + formant::symbols_for_bytes(profile.encoded_bytes(payload.len())) - 1;
            let mut nominal_audio = vec![0.0; offset];
            nominal_audio.extend_from_slice(&frame);
            let nominal = AlignedSpectra::new(&nominal_audio, offset, usize::MAX);
            let decoded = nominal.decoded_frame(payload.to_vec(), data_start, profile);
            assert_eq!(decoded.audio_end, offset + last * SYMBOL_TOTAL_SAMPLES + SAMPLES_PER_SYMBOL);
            for ratio in [0.98, 1.02] {
                let mut audio = vec![0.0; offset];
                audio.extend(resample_test_clock(&frame, ratio));
                let clocked = AlignedSpectra::clocked(&audio, offset, usize::MAX);
                let decoded = clocked.decoded_frame(payload.to_vec(), data_start, profile);
                let starts = clocked.clock_starts.as_ref().unwrap();
                assert_eq!(decoded.audio_end, starts[last] + SAMPLES_PER_SYMBOL);
                assert!(decoded.audio_end <= starts[last + 1]);
                assert_ne!(decoded.audio_end, offset + last * SYMBOL_TOTAL_SAMPLES + SAMPLES_PER_SYMBOL);
                assert_eq!(clocked.transforms, 0, "retaining lookahead adds no FFT work");
            }
        }
    }

    #[test]
    fn consecutive_clocked_frames_preserve_the_next_song() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let length = if profile == EccProfile::Standard { MAX_LENGTH_VARIABLE - 4 } else { 120 };
            let mut first: Vec<_> = (0..length).map(|i| (i * 73 + length) as u8).collect();
            first.extend(crate::checksum::crc32(&first).to_le_bytes());
            let mut second = vec![0x53];
            second.extend(crate::checksum::crc32(&second).to_le_bytes());
            let first_track = (profile == EccProfile::Standard).then(|| default_byte_pitch(first.len()));
            let first_frame = crate::encoder::encode_with_profile(&first, 50, first_track.as_deref(), profile).unwrap();
            let second_track = vec![220.0; second.len()];
            let second_frame = crate::encoder::encode_with_profile(&second, 50, Some(&second_track), profile).unwrap();
            for (keep_end, later_ratio, chunk_size) in [(true, 0.98, 511), (false, 0.98, 511),
                (false, 1.02, 2048), (false, 1.02, usize::MAX)] {
                let first_end = first_frame.len() - if keep_end { 0 } else { PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES };
                let mut capture = vec![0.0; SYMBOL_TOTAL_SAMPLES / 3];
                capture.extend(resample_test_clock_variable(&first_frame[..first_end], |position| {
                    if position < (64 * SYMBOL_TOTAL_SAMPLES) as f64 { 0.98 } else { later_ratio }
                }));
                capture.extend(resample_test_clock(&second_frame, 1.02));
                capture.extend_from_slice(&second_frame);
                capture.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
                let mut decoder = Decoder::with_profile(profile);
                if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                let mut received = Vec::new();
                for chunk in capture.chunks(chunk_size) {
                    if let Ok(Some(bytes)) = decoder.decode(chunk) { received.push(bytes); }
                    while decoder.sample_buffer.len() >= SYMBOL_TOTAL_SAMPLES {
                        if let Ok(Some(bytes)) = decoder.decode(&[]) { received.push(bytes); }
                    }
                }
                assert_eq!(received, vec![first.clone(), second.clone(), second.clone()],
                    "{profile:?}, {chunk_size}-sample callbacks, end marker {keep_end}, later clock {later_ratio}");
                assert!(!decoder.is_receiving());
            }
        }
    }

    #[test]
    fn clock_timed_completion_tolerates_rate_changes_after_the_header() {
        let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
        let track: Vec<_> = (0..payload.len()).map(|i| {
            440.0 * 2.0f64.powf((47.0 + ((i * 13) % 20) as f64 - 69.0) / 12.0)
        }).collect();
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut frame = crate::encoder::encode_with_profile(&payload, 50, Some(&track), profile).unwrap();
            let frame_slots = frame.len() / SYMBOL_TOTAL_SAMPLES;
            frame.truncate(frame.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES);
            for (first_ratio, later_ratio) in [(0.98, 1.0), (0.98, 1.02), (1.02, 1.0), (1.02, 0.98)] {
                let lead = SYMBOL_TOTAL_SAMPLES / 3;
                let mut capture = vec![0.0; lead];
                let mut position = 0.0;
                while position < (frame.len() - 1) as f64 {
                    let index = position as usize;
                    let fraction = (position - index as f64) as f32;
                    capture.push(frame[index] * (1.0 - fraction) + frame[index + 1] * fraction);
                    position += if position < (64 * SYMBOL_TOTAL_SAMPLES) as f64 { first_ratio } else { later_ratio };
                }
                let background: Vec<f32> = formant::synthesize_symbol(2, 207.65, 0.5, 0)
                    .into_iter().map(|sample| sample as f32).collect();
                capture.extend(background.repeat(64));
                let mut decoder = Decoder::with_profile(profile);
                if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                let mut delivered_at = None;
                for (index, chunk) in capture.chunks(511).enumerate() {
                    if let Ok(Some(bytes)) = decoder.decode(chunk) {
                        assert_eq!(bytes, payload);
                        assert!(delivered_at.is_none(), "duplicate delivery");
                        delivered_at = Some((index + 1) * 511);
                    }
                }
                let deadline = lead + (frame_slots + crate::audio::clock_slack_windows(frame_slots) + 4)
                    * SYMBOL_TOTAL_SAMPLES;
                assert!(delivered_at.is_some_and(|time| time <= deadline),
                    "{profile:?}, clock {first_ratio} → {later_ratio}: delivery {delivered_at:?}, deadline {deadline}");
            }
        }
    }

    #[test]
    fn remembered_length_does_not_qualify_an_unaligned_clock_grid() {
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|sample| sample as f32).collect();
        let mut decoder = Decoder::new();
        decoder.recorded_audio = background.repeat(100);
        decoder.symbols_recorded = 100;
        decoder.frame_start_hints.push(0);
        decoder.remember_frame_length(0, 10);
        assert!(decoder.analyze(true).is_err());
        assert_eq!(decoder.clock_grids, 0,
            "a remembered length cannot replace phase-local sync/header evidence");
        assert_eq!(decoder.clock_candidate_peak, 0);
    }

    #[test]
    fn phased_header_hints_are_frame_local_and_conflicts_stay_unlocked() {
        let mut decoder = Decoder::new();
        decoder.remember_frame_length(11, 64);
        decoder.remember_frame_length(11, 64);
        decoder.remember_frame_length(35, 14);
        assert_eq!(decoder.common_hinted_length(&[0, 11]), Some(64));
        assert_eq!(decoder.common_hinted_length(&[35]), Some(14));
        assert_eq!(decoder.common_hinted_length(&[11, 35]), None);
        assert_eq!(decoder.common_hinted_length(&[0]), None);
        decoder.remember_frame_length(11, 65);
        decoder.remember_frame_length(11, 64);
        assert_eq!(decoder.frame_length_hints[&11], None,
            "later votes must not erase an observed conflict");
        assert_eq!(decoder.common_hinted_length(&[11, 35]), None);
        assert_eq!(decoder.common_hinted_length(&[35]), Some(14));
        decoder.start_receiving(false);
        assert!(decoder.frame_length_hints.is_empty(), "a new recording cannot inherit old lengths");
    }

    #[test]
    fn block_clock_retries_preserve_profiles_contours_and_quiet_phrases() {
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            let mut payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
            payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
            for melody in 0..4 {
                let track: Vec<_> = (0..payload.len()).map(|i| {
                    let midi = match melody {
                        0 => 47 + i % 20,
                        1 => 66 - i % 20,
                        2 => if i % 2 == 0 { 47 } else { 66 },
                        _ => 47 + (i * 73 + 11) % 20,
                    };
                    let detune = if melody == 3 { [-0.3, 0.0, 0.3][i % 3] } else { 0.0 };
                    440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0)
                }).collect();
                let mut song = crate::encoder::encode_with_profile(&payload, 50, Some(&track), profile).unwrap();
                if melody == 3 {
                    for sample in &mut song[64 * SYMBOL_TOTAL_SAMPLES..96 * SYMBOL_TOTAL_SAMPLES] {
                        *sample *= 0.01;
                    }
                }
                for ratio in [0.98, 1.02] {
                    let mut capture = vec![0.0; SYMBOL_TOTAL_SAMPLES / 3];
                    capture.extend(resample_test_clock(&song, ratio));
                    capture.resize(capture.len() + 12 * SYMBOL_TOTAL_SAMPLES, 0.0);
                    let mut rx = Decoder::with_profile(profile);
                    if profile == EccProfile::Paranoid { rx.set_payload_validator(has_test_crc); }
                    let mut received = Vec::new();
                    for chunk in capture.chunks(511) {
                        if let Ok(Some(bytes)) = rx.decode(chunk) { received.push(bytes); }
                    }
                    assert_eq!(received, vec![payload.clone()], "{profile:?}, melody {melody}, ratio {ratio}");
                }
            }
        }
    }

    #[test]
    fn block_clock_retries_keep_the_fast_path_and_honor_validation() {
        let payload = b"clock integrity";
        let song = crate::encode(payload, 50).unwrap();
        let mut clean = Decoder::new();
        clean.recorded_audio = song.clone();
        clean.symbols_recorded = song.len() / SYMBOL_TOTAL_SAMPLES;
        assert_eq!(clean.analyze(true).unwrap().as_deref(), Some(payload.as_slice()));
        assert_eq!(clean.clock_grids, 0, "a successful ordinary decode needs no clock grids");
        assert_eq!(clean.priority_clock_grids, 0, "ordinary timing cannot invent a slow-clock hint");
        let mut rejected = Decoder::new();
        rejected.recorded_audio = resample_test_clock(&song, 0.98);
        rejected.symbols_recorded = rejected.recorded_audio.len() / SYMBOL_TOTAL_SAMPLES;
        rejected.set_payload_validator(|_| false);
        assert!(rejected.analyze(true).is_err());
        assert!(rejected.clock_grids > 0, "the validator must also reject clock-tracked hypotheses");
        assert!(rejected.clock_grids <= MAX_CLOCK_CANDIDATES);
        assert!(rejected.clock_candidate_peak <= MAX_CLOCK_CANDIDATES);
    }

    #[test]
    fn block_clock_drift_finishes_without_end_marker_or_silence() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for length in [64, MAX_LENGTH_VARIABLE] {
                let mut payload: Vec<_> = (0..length - 4).map(|i| (i * 73 + 19) as u8).collect();
                payload.extend(crate::checksum::crc32(&payload).to_le_bytes());
                let mut song = crate::encoder::encode_with_profile(&payload, 50, None, profile).unwrap();
                let frame_slots = song.len() / SYMBOL_TOTAL_SAMPLES;
                song.truncate(song.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES);
                for ratio in [0.98, 1.02] {
                    let lead = SYMBOL_TOTAL_SAMPLES / 3;
                    let mut capture = vec![0.0; lead];
                    capture.extend(resample_test_clock(&song, ratio));
                    // No end marker and no trailing silence: only validated
                    // length probes can finish before the recording deadline.
                    let background: Vec<f32> = formant::synthesize_symbol(2, 207.65, 0.5, 0)
                        .into_iter().map(|s| s as f32).collect();
                    capture.extend(background.repeat(60));
                    let mut decoder = Decoder::with_profile(profile);
                    if profile == EccProfile::Paranoid { decoder.set_payload_validator(has_test_crc); }
                    let mut delivered_at = None;
                    for (index, chunk) in capture.chunks(511).enumerate() {
                        if let Ok(Some(bytes)) = decoder.decode(chunk) {
                            assert_eq!(bytes, payload);
                            assert!(delivered_at.is_none(), "duplicate delivery");
                            delivered_at = Some((index + 1) * 511);
                        }
                    }
                    let deadline = lead + (frame_slots + crate::audio::clock_slack_windows(frame_slots) + 4)
                        * SYMBOL_TOTAL_SAMPLES;
                    assert!(delivered_at.is_some_and(|at| at <= deadline),
                        "{profile:?}, {length} bytes, ratio {ratio}: delivery {delivered_at:?}, limit {deadline}");
                }
            }
        }
    }

    #[test]
    fn block_clock_retries_reject_bad_crc_and_recover_a_later_sender() {
        let payload: Vec<_> = (0..140).map(|i| (i * 73 + 19) as u8).collect();
        let profile = EccProfile::Standard;
        let track = default_byte_pitch(payload.len());
        let mut bad = crate::encode(&payload, 50).unwrap();
        let mut protected = profile.protect_payload(&payload);
        protected[0] ^= 0x40;
        let body = ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len())).encode(&protected);
        let first = PREAMBLE_LEN + PILOT_LEN + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let outro = first + formant::symbols_for_bytes(payload.len());
        let slots = bad.len() / SYMBOL_TOTAL_SAMPLES;
        for (i, vowel) in formant::bytes_to_symbols(&body).into_iter().enumerate() {
            let pos = first + i;
            let hz = if pos < outro { track[i / 4] } else { outro_f0(pos - outro, slots - outro) };
            let slot: Vec<f32> = formant::synthesize_symbol(vowel, hz, 0.5 * melody_accent(pos), pos)
                .into_iter().map(|s| s as f32).collect();
            bad[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES].copy_from_slice(&slot);
        }
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|s| s as f32).collect();
        let next = b"the next clock";
        for (first_ratio, next_ratio, phase) in [(1.02, 0.98, 0), (0.98, 1.02, 0),
            (0.98, 1.02, SYMBOL_TOTAL_SAMPLES / 3)] {
            let mut capture = background.repeat(100);
            capture.resize(capture.len() + phase, 0.0);
            capture.extend(resample_test_clock(&bad, first_ratio));
            capture.extend(resample_test_clock(&crate::encode(next, 50).unwrap(), next_ratio));
            capture.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            let mut decoder = Decoder::new();
            let mut received = Vec::new();
            let mut used_clock = false;
            let mut used_priority = false;
            for chunk in capture.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) { received.push(bytes); }
                used_clock |= decoder.clock_grids > 0;
                used_priority |= decoder.priority_clock_grids > 0;
                assert!(decoder.clock_grids <= MAX_CLOCK_CANDIDATES);
                assert!(decoder.priority_clock_grids <= 1);
                assert!(decoder.clock_candidate_peak <= MAX_CLOCK_CANDIDATES);
            }
            assert!(used_clock);
            if phase > 0 { assert!(used_priority, "the phased bad-CRC slow frame must exercise the priority gate"); }
            assert_eq!(received, vec![next.to_vec()],
                "only a validated frame may be delivered; clock {first_ratio}, phase {phase}, state {:?}, starts {:?}, recorded {}, deadline {}, clock grids {}",
                decoder.state, decoder.frame_start_hints, decoder.symbols_recorded, decoder.max_record_symbols, decoder.clock_grids);
        }
    }

    #[test]
    fn prepared_byte_erasures_match_reference_across_prefixes_and_budgets() {
        let near_threshold = f64::from_bits(ERASURE_CONFIDENCE_THRESHOLD.to_bits() - 1);
        let special = [
            f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5, -0.0, 0.0,
            0.1, near_threshold, ERASURE_CONFIDENCE_THRESHOLD, 0.4, 1.0, 2.0,
        ];
        let mut state = 0x1234abcdu32;
        let varied: Vec<f64> = (0..1100).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            special[(state >> 24) as usize % special.len()]
        }).collect();
        let cases = [
            Vec::new(), vec![0.1], vec![f64::NAN; 1020], vec![0.0; 1020],
            vec![1.0; 1020], varied[..17].to_vec(), varied[..509].to_vec(), varied,
        ];
        for confs in &cases {
            for offset in [0, 1, 12, 43, 1200] {
                let prepared = PreparedByteErasures::new(Some(confs), offset, 255);
                for bytes in 0..=255 {
                    // Every prefix, including partially observed bytes and
                    // all-missing tails, at every RS parity budget.
                    for ecc in 0..=255 {
                        assert_eq!(prepared.for_length(bytes, ecc),
                            compute_byte_erasures(confs, offset, bytes, ecc),
                            "{} confidences, offset {offset}, bytes {bytes}, ecc {ecc}", confs.len());
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_byte_erasures_are_lazy_bounded_and_hypothesis_local() {
        let mut confs = vec![0.1; 4 * 255];
        for max_bytes in [0, 1, 3, 32, 255, usize::MAX] {
            let prepared = PreparedByteErasures::new(Some(&confs), 0, max_bytes);
            assert_eq!(prepared.max_bytes, max_bytes.min(255));
            assert!(prepared.order.get().is_none());
            assert!(prepared.for_length(prepared.max_bytes, 1).is_empty());
            assert!(prepared.for_length(0, 255).is_empty());
            assert!(prepared.order.get().is_none());
            if prepared.max_bytes == 0 {
                continue;
            }
            prepared.for_length(prepared.max_bytes, 255);
            let order = prepared.order.get().unwrap();
            assert_eq!(order.len(), prepared.max_bytes);
            let storage = order.as_ptr();
            for bytes in (1..=prepared.max_bytes).rev() {
                assert_eq!(prepared.for_length(bytes, 12), (0..bytes.min(8)).collect::<Vec<_>>());
                assert_eq!(prepared.order.get().unwrap().as_ptr(), storage);
            }
        }
        let absent = PreparedByteErasures::new(None, 0, 255);
        assert!(absent.for_length(255, 255).is_empty());
        assert!(absent.order.get().is_none());

        let original = PreparedByteErasures::new(Some(&confs), 0, 255);
        assert_eq!(original.for_length(255, 6), vec![0, 1, 2, 3]);
        drop(original);
        // A Chase flip asserts one symbol, but only removes its byte from
        // erasures if no other weak symbol still contributes to that byte.
        confs[..5].fill(1.0);
        let changed = PreparedByteErasures::new(Some(&confs), 0, 255);
        assert_eq!(changed.for_length(255, 6), vec![1, 2, 3, 4]);
        assert_eq!(changed.for_length(255, 6), compute_byte_erasures(&confs, 0, 255, 6));
    }

    fn assert_same_spectral_class(
        actual: Option<formant::SymbolClass>, expected: Option<formant::SymbolClass>,
    ) {
        match (actual, expected) {
            (Some(a), Some(b)) => {
                assert_eq!(a.vowel, b.vowel);
                assert_eq!(a.second, b.second);
                assert_eq!(a.confidence.to_bits(), b.confidence.to_bits());
                assert_eq!(a.centroid.to_bits(), b.centroid.to_bits());
            }
            (None, None) => {}
            _ => panic!("cached and reference window availability differ"),
        }
    }

    #[test]
    fn aligned_spectra_preserve_all_classification_fields_bit_for_bit() {
        let mut decoder = Decoder::new();
        for midi in 47..=66 {
            for detune in [-0.3, 0.0, 0.3] {
                let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                for vowel in 0..NUM_VOWELS {
                    let volume = [0.001, 0.05, 0.5][(midi as usize + vowel) % 3];
                    decoder.recorded_audio.extend(formant::synthesize_symbol(vowel, hz, volume, vowel)
                        .into_iter().map(|sample| sample as f32));
                }
            }
        }
        decoder.recorded_audio.extend(vec![0.0; SYMBOL_TOTAL_SAMPLES]);
        decoder.recorded_audio.extend(vec![0.07; SYMBOL_TOTAL_SAMPLES]);
        let mut state = 713u32;
        decoder.recorded_audio.extend((0..SYMBOL_TOTAL_SAMPLES).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 16) as f32 / 65535.0 - 0.5) * 0.2
        }));
        let eqs: Vec<Vec<f64>> = [-40.0, 0.0, 50.0].into_iter().map(|tilt| {
            (0..FFT_SIZE).map(|bin| {
                let x = ((bin as f64 * HZ_PER_BIN - F2_LO) / (F2_HI - F2_LO)).clamp(0.0, 1.0);
                10.0f64.powf(tilt * x / 10.0)
            }).collect()
        }).collect();
        for offset in [0, 1, GUARD_SAMPLES / 2, SYMBOL_TOTAL_SAMPLES / 2, SYMBOL_TOTAL_SAMPLES + 37] {
            let mut spectra = AlignedSpectra::new(&decoder.recorded_audio, offset, usize::MAX);
            for i in 0..spectra.windows.len() {
                let start = offset + i * SYMBOL_TOTAL_SAMPLES;
                let mut full = vec![0.0; SAMPLES_PER_SYMBOL];
                fft::power_spectrum(&decoder.recorded_audio[start..start + SAMPLES_PER_SYMBOL], &mut full);
                let compact = spectra.get(i).unwrap();
                assert!(compact.iter().zip(&full).all(|(a, b)| a.to_bits() == b.to_bits()));
                let f0 = formant::detect_f0(&full);
                assert_eq!(formant::detect_f0(compact).to_bits(), f0.to_bits());
                assert_eq!(formant::detect_formants_f1f2(compact, f0), formant::detect_formants_f1f2(&full, f0));
                for shift in [-100.0, 0.0, 100.0] {
                    let centers = formant::EXPECTED_F2_CENTROIDS.map(|center| center + shift);
                    assert_same_spectral_class(spectra.classify_full(i, &centers),
                        decoder.classify_full_at(offset, i, &centers));
                }
                for eq in &eqs {
                    assert_same_spectral_class(spectra.classify_eq(i, eq), decoder.classify_eq_at(offset, i, eq));
                    assert_same_spectral_class(spectra.classify_template(i, eq), decoder.classify_template_at(offset, i, eq));
                    assert_same_spectral_class(spectra.classify_f0a(i, eq), decoder.classify_f0a_at(offset, i, eq));
                }
            }
            assert_eq!(spectra.transforms, spectra.windows.len(), "one FFT per window, not per classifier");
        }
    }

    #[test]
    fn aligned_spectra_preserve_pilot_eq_and_calibration_bit_for_bit() {
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            let mut decoder = Decoder::with_profile(profile);
            decoder.recorded_audio = crate::encoder::encode_with_profile(
                b"spectral reuse", 50, None, profile,
            ).unwrap();
            for offset in [0, 1, SYMBOL_TOTAL_SAMPLES / 2, SYMBOL_TOTAL_SAMPLES + 37] {
                let mut spectra = AlignedSpectra::new(&decoder.recorded_audio, offset, usize::MAX);
                let n = spectra.windows.len();
                for pos in [0, PREAMBLE_LEN, 5, 8, 17, n - PILOT_LEN, n - PILOT_LEN + 1] {
                    assert_eq!(spectra.pilot_centers(pos).map(|centers| centers.map(f64::to_bits)),
                        decoder.pilot_centers_at(offset, pos).map(|centers| centers.map(f64::to_bits)));
                    let actual = spectra.pilot_eq(pos, profile);
                    let expected = decoder.pilot_eq_at(offset, pos);
                    assert_eq!(actual.map(|eq| eq.into_iter().map(f64::to_bits).collect::<Vec<_>>()),
                        expected.map(|eq| eq.into_iter().map(f64::to_bits).collect::<Vec<_>>()));
                    let transforms = spectra.transforms;
                    spectra.pilot_centers(pos);
                    spectra.pilot_eq(pos, profile);
                    assert_eq!(spectra.transforms, transforms);
                }
            }
        }
    }

    #[test]
    fn aligned_spectra_preserve_window_boundaries_and_bound_storage() {
        for len in [0, 1, SAMPLES_PER_SYMBOL - 1, SAMPLES_PER_SYMBOL,
            SYMBOL_TOTAL_SAMPLES - 1, SYMBOL_TOTAL_SAMPLES,
            SYMBOL_TOTAL_SAMPLES + SAMPLES_PER_SYMBOL - 1,
            SYMBOL_TOTAL_SAMPLES + SAMPLES_PER_SYMBOL, 3 * SYMBOL_TOTAL_SAMPLES] {
            let audio: Vec<f32> = (0..len).map(|i| ((i * 73 + 41) % 251) as f32 / 251.0).collect();
            for offset in [0, 1, SAMPLES_PER_SYMBOL - 1, SYMBOL_TOTAL_SAMPLES - 1, usize::MAX] {
                for limit in [0, 1, 5, usize::MAX] {
                    let mut spectra = AlignedSpectra::new(&audio, offset, limit);
                    assert!(spectra.windows.len() <= limit);
                    assert!(spectra.windows.len() <= len / SYMBOL_TOTAL_SAMPLES + 1);
                    for i in 0..5 {
                        let voiced = offset.checked_add(i * SYMBOL_TOTAL_SAMPLES)
                            .and_then(|start| start.checked_add(SAMPLES_PER_SYMBOL).map(|end| (start, end)))
                            .and_then(|(start, end)| audio.get(start..end))
                            .filter(|_| i < limit);
                        let expected = voiced.map(|samples| {
                            let mut full = vec![0.0; SAMPLES_PER_SYMBOL];
                            fft::power_spectrum(samples, &mut full);
                            full[..RETAINED_SPECTRUM_BINS].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        });
                        assert_eq!(spectra.get(i).map(|bins| bins.iter().map(|x| x.to_bits()).collect::<Vec<_>>()), expected);
                    }
                    let transforms = spectra.transforms;
                    for i in (0..5).rev() {
                        spectra.get(i);
                    }
                    assert!(spectra.get(usize::MAX).is_none());
                    assert_eq!(spectra.transforms, transforms);
                    assert_eq!(transforms, spectra.windows.len());
                }
            }
        }
        // Compact storage works at either sample rate, including the
        // harmonic detector's neighboring bins, without retaining a full FFT.
        assert!(RETAINED_SPECTRUM_BINS < SAMPLES_PER_SYMBOL / 2);
        assert!(RETAINED_SPECTRUM_BINS > (HARMONIC_CEILING / HZ_PER_BIN).round() as usize + 2);
    }

    #[test]
    fn failure_cache_keys_are_exact_and_storage_is_bounded() {
        let mut cache = FailedPayloadCache::default();
        let profile = EccProfile::Standard;
        let bytes = [31, 41, 59, 26];
        cache.record(profile, 11, &bytes, &[2, 1]);
        let mut erasures = None;
        assert!(cache.contains(profile, 11, &bytes, &mut erasures, || vec![2, 1]));
        assert_eq!(erasures, Some(vec![2, 1]));
        assert!(!cache.contains(profile, 11, &bytes, &mut None, || vec![1, 2]));
        assert!(!cache.contains(profile, 11, &bytes, &mut None, Vec::new));
        for (other_profile, len, other_bytes) in [
            (EccProfile::Legacy, 11, bytes),
            (EccProfile::Paranoid, 11, bytes),
            (profile, 12, bytes),
            (profile, 11, [31, 41, 59, 27]),
        ] {
            assert!(!cache.contains(other_profile, len, &other_bytes, &mut None,
                || panic!("unmatched bodies must not compute erasures")));
        }
        // Replacing a failure at the same length cannot grow the cache.
        for len in 1..=MAX_LENGTH_VARIABLE {
            for generation in 0..3 {
                cache.record(profile, len, &[generation; 255], &[0, 1]);
            }
        }
        assert_eq!(cache.by_length.len(), MAX_LENGTH_VARIABLE + 1);
        assert_eq!(cache.by_length.iter().flatten().count(), MAX_LENGTH_VARIABLE);
        assert!(cache.by_length.iter().flatten().all(|entry| entry.bytes == [2; 255]));
        let records = cache.records;
        cache.record(profile, 0, &bytes, &[]);
        cache.record(profile, MAX_LENGTH_VARIABLE + 1, &bytes, &[]);
        cache.record(profile, usize::MAX, &bytes, &[]);
        cache.record(profile, 11, &[0; 256], &[]);
        cache.record(profile, 11, &bytes, &[0; 256]);
        assert_eq!(cache.records, records);
        assert_eq!(cache.by_length.len(), MAX_LENGTH_VARIABLE + 1);
    }

    #[test]
    fn chase_syndrome_reuse_preserves_results_and_validator_order() {
        thread_local! {
            static TRACE: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        fn validate(payload: &[u8]) -> bool {
            TRACE.with(|trace| {
                let mut trace = trace.borrow_mut();
                trace.push(payload.to_vec());
                trace.len() % 3 == 0
            })
        }
        let mut repeated_validation = false;
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in [4, 17, 64, 120, 140] {
                let payload: Vec<_> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
                let original = encoded_payload_symbols(&payload, profile);
                for damage in 0..3 {
                    let mut symbols = original.clone();
                    let mut confidences = vec![0.9; symbols.len()];
                    if damage > 0 {
                        for byte in 0..profile.ecc_bytes(len) / 2 + 1 {
                            let pos = formant::symbols_for_bytes(ENCODED_DATA_OFFSET + byte) + 3;
                            symbols[pos] ^= 1;
                            confidences[pos] = 0.4;
                        }
                    }
                    if damage == 2 {
                        for pos in [0, 4] {
                            symbols[pos] ^= 1;
                            confidences[pos] = 0.1;
                        }
                    }
                    for locked in [None, Some(len)] {
                        let run = |reuse| {
                            TRACE.with(|trace| trace.borrow_mut().clear());
                            let result = decode_symbol_stream_with_syndromes(
                                &symbols, Some(&confidences), Some(&original), profile, locked,
                                Some(validate), Some(&mut FailedPayloadCache::default()), reuse,
                            None);
                            let trace = TRACE.with(|trace| std::mem::take(&mut *trace.borrow_mut()));
                            (result, trace)
                        };
                        let expected = run(false);
                        let actual = run(true);
                        repeated_validation |= actual.1.len() > 1;
                        assert_eq!(actual, expected, "{profile:?}, length {len}, damage {damage}, lock {locked:?}");
                    }
                }
            }
        }
        assert!(repeated_validation, "exercise stateful rejection followed by later candidates");
    }

    #[test]
    fn failure_cache_reuses_all_header_only_chase_failures() {
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in [40, MAX_LENGTH_VARIABLE] {
                let mut state = 713u32;
                let mut bytes = ReedSolomon::new(1, 2).encode(&[len as u8]);
                bytes.extend((0..profile.encoded_bytes(len) - ENCODED_DATA_OFFSET).map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    (state >> 24) as u8
                }));
                let symbols = formant::bytes_to_symbols(&bytes);
                let mut confs = vec![1.0; symbols.len()];
                let mut seconds = symbols.clone();
                for i in 0..CHASE_MAX_FLIPS {
                    confs[i] = 0.1;
                    seconds[i] ^= 1;
                }
                let mut cache = FailedPayloadCache::default();
                assert!(decode_symbol_stream_cached(&symbols, Some(&confs), Some(&seconds),
                    profile, Some(len), None, None, None).is_none());
                assert!(decode_symbol_stream_cached(&symbols, Some(&confs), Some(&seconds),
                    profile, Some(len), None, Some(&mut cache), None).is_none());
                assert_eq!(cache.records, 1, "{profile:?}, {len} bytes");
                assert_eq!(cache.hits, (1 << CHASE_MAX_FLIPS) - 1);
            }
        }
    }

    #[test]
    fn failure_cache_does_not_hide_changed_payloads_or_erasures() {
        let profile = EccProfile::Standard;
        let payload = b"retry notes";
        assert_eq!(profile.ecc_bytes(payload.len()), 6);
        let original = encoded_payload_symbols(payload, profile);
        let mut damaged = original.clone();
        let mut confs = vec![1.0; original.len()];
        let body_start = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let positions: Vec<usize> = (0..4).map(|i| body_start + 4 * i + 3).collect();
        for &pos in &positions {
            damaged[pos] ^= 1;
            confs[pos] = 0.4; // Eligible for Chase, but not an erasure.
        }
        let mut cache = FailedPayloadCache::default();
        assert!(decode_symbols_once(&damaged, Some(&confs), profile, false,
            Some(payload.len()), None, Some(&mut cache), None).is_none());
        assert_eq!(cache.records, 1);

        // Header-only retries hit the cache; the first payload correction
        // changes the bytes and must still get a fresh attempt.
        let mut seconds = original.clone();
        for i in 0..3 {
            confs[i] = 0.1;
            seconds[i] ^= 1;
        }
        assert_eq!(decode_symbol_stream_cached(&damaged, Some(&confs), Some(&seconds),
            profile, Some(payload.len()), None, Some(&mut cache), None).as_deref(), Some(payload.as_slice()));
        assert!(cache.hits >= 4);

        // Same bad bytes, newly useful erasure information: retry, don't
        // reuse the earlier failure with the empty erasure list.
        for &pos in &positions {
            confs[pos] = 0.0;
        }
        let hits = cache.hits;
        assert_eq!(decode_symbols_once(&damaged, Some(&confs), profile, false,
            Some(payload.len()), None, Some(&mut cache), None).as_deref(), Some(payload.as_slice()));
        assert_eq!(cache.hits, hits);
        assert_eq!(decode_symbols_once(&original, None, profile, false,
            Some(payload.len()), None, Some(&mut cache), None).as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn failure_cache_does_not_cache_header_or_agreement_rejection() {
        let profile = EccProfile::Legacy;
        let payload = b"twenty melodic bytes";
        assert_eq!(payload.len(), 20);
        let original = encoded_payload_symbols(payload, profile);
        let mut bytes = formant::symbols_to_bytes(&original, profile.encoded_bytes(payload.len()));
        for byte in &mut bytes[ENCODED_DATA_OFFSET..ENCODED_DATA_OFFSET + 4] {
            *byte ^= 255;
        }
        let symbols = formant::bytes_to_symbols(&bytes);
        let mut cache = FailedPayloadCache::default();
        // Hard RS corrects four bytes, but its match rate fails the risky
        // Legacy gate with no confidence evidence. That isn't a body failure.
        assert!(decode_symbols_once(&symbols, None, profile, true,
            Some(payload.len()), None, Some(&mut cache), None).is_none());
        assert_eq!(cache.records, 0);
        assert_eq!(decode_symbols_once(&symbols, None, profile, false,
            Some(payload.len()), None, Some(&mut cache), None).as_deref(), Some(payload.as_slice()));
        assert!(cache.by_length.is_empty());
    }

    #[test]
    fn failure_cache_matches_uncached_decoding_across_profiles_and_lengths() {
        let mut cache = FailedPayloadCache::default();
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                let payload: Vec<u8> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
                let original = encoded_payload_symbols(&payload, profile);
                for damage in 0..3 {
                    let mut symbols = original.clone();
                    if damage == 1 {
                        symbols[4 * ENCODED_DATA_OFFSET] ^= 1;
                    } else if damage == 2 {
                        symbols[0] ^= 1;
                        symbols[4] ^= 1;
                    }
                    let confs = vec![0.9; symbols.len()];
                    let expected = decode_symbol_stream_cached(&symbols, Some(&confs), Some(&original),
                        profile, None, None, None, None);
                    let got = decode_symbol_stream_cached(&symbols, Some(&confs), Some(&original),
                        profile, None, None, Some(&mut cache), None);
                    assert_eq!(got, expected, "{profile:?}, length {len}, damage {damage}");
                    if damage < 2 {
                        assert_eq!(got.as_deref(), Some(payload.as_slice()));
                    }
                }
            }
            // Exhaust the retry ladder on varied low-confidence streams,
            // both with and without locks. Use small unlocked cases to keep
            // the deliberately uncached reference inexpensive.
            for len in [1, 4, 11, 40, 80, MAX_LENGTH_VARIABLE] {
                for seed in 0..4 {
                    let mut state = (len * 17 + seed) as u32;
                    let symbols: Vec<usize> = (0..4 * profile.encoded_bytes(len)).map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        (state >> 30) as usize
                    }).collect();
                    let confs: Vec<f64> = (0..symbols.len()).map(|i| [0.0, 0.1, 0.4, 0.9][(i + seed) % 4]).collect();
                    let seconds: Vec<usize> = symbols.iter().enumerate().map(|(i, s)| (s + 1 + i % 3) % 4).collect();
                    let locks = if len <= 11 { vec![None, Some(len)] } else { vec![Some(len)] };
                    for lock in locks {
                        let expected = decode_symbol_stream_cached(&symbols, Some(&confs), Some(&seconds), profile, lock, None, None, None);
                        assert_eq!(decode_symbol_stream_cached(&symbols, Some(&confs), Some(&seconds),
                            profile, lock, None, Some(&mut cache), None), expected);
                    }
                }
            }
        }
        assert!(cache.by_length.len() <= MAX_LENGTH_VARIABLE + 1);
    }

    #[test]
    fn geometry_preflight_preserves_every_supported_frame_and_slop() {
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                let total = profile.encoded_bytes(len);
                for slop in 0..=3 {
                    for partial in 0..4 {
                        let symbols = 4 * (total + slop) + partial;
                        assert!(frame_geometry_allows(symbols, profile, Some(len)));
                        assert!(frame_geometry_allows(symbols, profile, None));
                    }
                }
                assert!(!frame_geometry_allows(4 * total - 1, profile, Some(len)));
                assert_eq!(frame_geometry_allows(4 * (total + 4), profile, Some(len)),
                    profile == EccProfile::Paranoid);
            }
            let minimum = 4 * profile.encoded_bytes(1);
            assert!(!frame_geometry_allows(minimum - 1, profile, None));
            let beyond_maximum = 4 * (profile.encoded_bytes(MAX_LENGTH_VARIABLE) + 4);
            assert_eq!(frame_geometry_allows(beyond_maximum, profile, None),
                profile == EccProfile::Paranoid);
        }
    }

    #[test]
    fn checksum_rejects_an_rs_valid_wrong_payload() {
        let payload = b"verified melody";
        let profile = EccProfile::Standard;
        let mut protected = profile.protect_payload(payload);
        protected[2] ^= 0x40; // Keep the original checksum, but rebuild valid RS parity.
        let rs = ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len()));
        let mut codeword = ReedSolomon::new(1, 2).encode(&[payload.len() as u8]);
        codeword.extend(rs.encode(&protected));
        assert!(rs.decode(&codeword[ENCODED_DATA_OFFSET..]).is_some());
        let symbols = formant::bytes_to_symbols(&codeword);
        assert!(decode_symbol_stream(&symbols, None, None, profile).is_none());
    }

    #[test]
    fn checksum_and_payload_errors_remain_correctable() {
        let payload = b"verified melody";
        let profile = EccProfile::Standard;
        let original = encoded_payload_symbols(payload, profile);
        let bytes = formant::symbols_to_bytes(&original, profile.encoded_bytes(payload.len()));
        for pos in ENCODED_DATA_OFFSET..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[pos] ^= 0x5a;
            let symbols = formant::bytes_to_symbols(&damaged);
            assert_eq!(
                decode_symbol_stream(&symbols, None, None, profile).as_deref(),
                Some(payload.as_slice()),
                "byte {pos}"
            );
        }
    }

    #[test]
    fn checksum_rejection_still_allows_erasure_recovery() {
        let payload = b"retry notes";
        let profile = EccProfile::Standard;
        let protected = profile.protect_payload(payload);
        let n_ecc = profile.ecc_bytes(payload.len());
        assert_eq!(n_ecc, 6);
        let rs = ReedSolomon::new(protected.len(), n_ecc);
        let original = rs.encode(&protected);

        // Two codewords at minimum distance 7. Corrupt four bytes toward
        // the wrong one: hard RS sees only three errors and miscorrects.
        // The original is recoverable if those four positions are erased.
        let mut wrong_payload = protected.clone();
        wrong_payload[0] ^= 0x40; // Deliberately retain the original CRC.
        let wrong = rs.encode(&wrong_payload);
        let differences: Vec<usize> = original.iter().zip(&wrong)
            .enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i))
            .collect();
        assert_eq!(differences.len(), n_ecc + 1);
        let mut damaged = original.clone();
        for &pos in &differences[..4] {
            damaged[pos] = wrong[pos];
        }
        assert_eq!(rs.decode(&damaged).as_deref(), Some(wrong_payload.as_slice()));
        assert!(profile.verify_payload(wrong_payload).is_none());

        let mut frame = ReedSolomon::new(1, 2).encode(&[payload.len() as u8]);
        frame.extend(damaged);
        let symbols = formant::bytes_to_symbols(&frame);
        let mut confidence = vec![1.0; symbols.len()];
        for &pos in &differences[..4] {
            let start = formant::symbols_for_bytes(ENCODED_DATA_OFFSET + pos);
            confidence[start..start + formant::symbols_for_bytes(1)].fill(0.0);
        }
        assert_eq!(
            decode_symbol_stream(&symbols, Some(&confidence), None, profile).as_deref(),
            Some(payload.as_slice())
        );
    }

    // Hard RS selects a wrong CRC-bearing message; erasing the seven
    // damaged bytes, or restoring just the first symbol, recovers the truth.
    fn external_validator_miscorrection_fixture() -> (Vec<u8>, Vec<usize>, Vec<f64>) {
        let profile = EccProfile::Paranoid;
        let mut payload = b"melodies".to_vec();
        payload.extend_from_slice(&crate::checksum::crc32(&payload).to_le_bytes());
        let n_ecc = profile.ecc_bytes(payload.len());
        assert_eq!(n_ecc, 12);
        let rs = ReedSolomon::new(payload.len(), n_ecc);
        let original = rs.encode(&payload);
        let mut wrong_payload = payload.clone();
        wrong_payload[0] ^= 0x40; // Keep the original CRC.
        let wrong = rs.encode(&wrong_payload);
        let differences: Vec<usize> = original.iter().zip(&wrong).enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i)).collect();
        assert_eq!(differences.len(), n_ecc + 1);
        let erased = &differences[..7];
        let mut damaged = original.clone();
        for &pos in erased {
            damaged[pos] = wrong[pos];
        }
        assert_eq!(rs.decode(&damaged).as_deref(), Some(wrong_payload.as_slice()));
        assert_eq!(rs.decode_with_erasures(&damaged, erased).as_deref(), Some(payload.as_slice()));

        let mut frame = ReedSolomon::new(1, 2).encode(&[payload.len() as u8]);
        frame.extend(damaged);
        let symbols = formant::bytes_to_symbols(&frame);
        let mut confidences = vec![1.0; symbols.len()];
        for &pos in erased {
            let start = formant::symbols_for_bytes(ENCODED_DATA_OFFSET + pos);
            confidences[start..start + 4].fill(0.0);
        }
        (payload, symbols, confidences)
    }

    fn has_test_crc(bytes: &[u8]) -> bool {
        EccProfile::Standard.verify_payload(bytes.to_vec()).is_some()
    }

    #[test]
    fn external_validator_rejection_still_allows_erasure_recovery() {
        let (payload, symbols, confidences) = external_validator_miscorrection_fixture();
        let mut decoder = Decoder::with_profile(EccProfile::Paranoid);
        decoder.set_payload_validator(has_test_crc);
        for lock in [None, Some(payload.len())] {
            assert_eq!(decoder.try_decode_symbols_locked(
                &symbols, Some(&confidences), None, lock, None,
            None).as_deref(), Some(payload.as_slice()));
        }
    }

    #[test]
    fn external_validator_rejection_still_allows_chase_recovery() {
        let (payload, symbols, _) = external_validator_miscorrection_fixture();
        let profile = EccProfile::Paranoid;
        let original = encoded_payload_symbols(&payload, profile);
        let mut confidences = vec![1.0; symbols.len()];
        let first = symbols.iter().zip(&original).position(|(a, b)| a != b).unwrap();
        confidences[first] = 0.4; // Chase-eligible, but not an erasure.
        let mut seconds = symbols.clone();
        seconds[first] = original[first];
        let mut decoder = Decoder::with_profile(profile);
        decoder.set_payload_validator(has_test_crc);
        assert!(decoder.try_decode_symbols_locked(
            &symbols, Some(&confidences), None, Some(payload.len()), None,
        None).is_none(), "without the runner-up there is no correct hypothesis");
        for lock in [None, Some(payload.len())] {
            assert_eq!(decoder.try_decode_symbols_locked(
                &symbols, Some(&confidences), Some(&seconds), lock, None,
            None).as_deref(), Some(payload.as_slice()));
        }
    }

    #[test]
    fn external_validator_rejection_still_allows_alternate_lengths() {
        let profile = EccProfile::Paranoid;
        let mut payload = vec![0; 32];
        payload.extend_from_slice(&crate::checksum::crc32(&payload).to_le_bytes());
        let mut symbols = encoded_payload_symbols(&payload, profile);
        let short_header = formant::bytes_to_symbols(&ReedSolomon::new(1, 2).encode(&[1]));
        symbols[..short_header.len()].copy_from_slice(&short_header);
        // The all-zero prefix is an intact RS codeword at the wrong length.
        assert_eq!(decode_symbol_stream(&symbols, None, None, profile), Some(vec![0]));
        assert_eq!(decode_symbol_stream_validated(
            &symbols, None, None, profile, None, has_test_crc,
        ).as_deref(), Some(payload.as_slice()));
        // Validation cannot relax an independently established length lock.
        assert!(decode_symbol_stream_validated(
            &symbols, None, None, profile, Some(1), has_test_crc,
        ).is_none());
    }

    #[test]
    fn failure_cache_does_not_cache_external_validator_rejection() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn accept_after_first_call(_: &[u8]) -> bool {
            CALLS.fetch_add(1, Ordering::SeqCst) > 0
        }
        let payload = b"accept on retry";
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            let symbols = encoded_payload_symbols(payload, profile);
            let mut cache = FailedPayloadCache::default();
            let mut decoder = Decoder::with_profile(profile);
            decoder.set_payload_validator(accept_after_first_call);
            CALLS.store(0, Ordering::SeqCst);
            assert!(decoder.try_decode_symbols_locked(
                &symbols, None, None, Some(payload.len()), Some(&mut cache),
            None).is_none());
            assert_eq!(cache.records, 0);
            assert_eq!(decoder.try_decode_symbols_locked(
                &symbols, None, None, Some(payload.len()), Some(&mut cache),
            None).as_deref(), Some(payload.as_slice()));
            assert_eq!(CALLS.load(Ordering::SeqCst), 2);
            assert_eq!(cache.hits, 0);
        }
    }

    #[test]
    fn external_validator_cannot_bypass_standard_crc() {
        let profile = EccProfile::Standard;
        let payload = b"verified melody";
        let mut protected = profile.protect_payload(payload);
        protected[2] ^= 0x40;
        let rs = ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len()));
        let mut codeword = ReedSolomon::new(1, 2).encode(&[payload.len() as u8]);
        codeword.extend(rs.encode(&protected));
        let symbols = formant::bytes_to_symbols(&codeword);
        let mut decoder = Decoder::with_profile(profile);
        decoder.set_payload_validator(|_| {
            panic!("bad internal CRC must never reach the caller's validator")
        });
        assert!(decoder.try_decode_symbols_locked(
            &symbols, None, None, Some(payload.len()), None,
        None).is_none());
    }

    #[test]
    fn checksum_frames_recover_a_damaged_header_at_every_length() {
        let profile = EccProfile::Standard;
        for len in 1..=MAX_LENGTH_VARIABLE {
            let payload: Vec<u8> = (0..len).map(|i| (i * 37 + len) as u8).collect();
            let original = encoded_payload_symbols(&payload, profile);
            let bytes = formant::symbols_to_bytes(&original, profile.encoded_bytes(len));
            for untouched in 0..ENCODED_DATA_OFFSET {
                let mut damaged = bytes.clone();
                for pos in 0..ENCODED_DATA_OFFSET {
                    if pos != untouched {
                        damaged[pos] ^= 0x5a;
                    }
                }
                // Two header errors exceed its hard-decision radius. The
                // frame size implies the right length, and CRC verifies it.
                let symbols = formant::bytes_to_symbols(&damaged);
                assert_eq!(
                    decode_symbol_stream(&symbols, None, None, profile).as_deref(),
                    Some(payload.as_slice()),
                    "length {len}, intact header byte {untouched}"
                );
            }
        }
    }

    #[test]
    fn legacy_frame_layout_and_explicit_decode() {
        let payload = b"legacy song";
        let audio = crate::encoder::encode_with_profile(payload, 50, None, EccProfile::Legacy).unwrap();
        let legacy_bytes = ENCODED_DATA_OFFSET + payload.len() + ecc_bytes_for_length(payload.len());
        let expected_samples = (2 * PREAMBLE_LEN + PILOT_LEN + 4 * legacy_bytes) * SYMBOL_TOTAL_SAMPLES;
        assert_eq!(audio.len(), expected_samples);
        let mut legacy = Decoder::with_profile(EccProfile::Legacy);
        assert_eq!(legacy.decode(&audio).unwrap().as_deref(), Some(payload.as_slice()));
        // Default decoding never falls back to accepting an unverified frame.
        let mut checked = Decoder::new();
        assert!(!matches!(checked.decode(&audio), Ok(Some(_))));
        assert!(!matches!(checked.decode(&vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]), Ok(Some(_))));
    }

    #[test]
    fn checksum_framing_preserves_the_payload_melody() {
        let payload = b"same notes, same rhythm";
        let track = default_byte_pitch(payload.len());
        let checked = crate::encoder::encode_with_profile(payload, 50, Some(&track), EccProfile::Standard).unwrap();
        let legacy = crate::encoder::encode_with_profile(payload, 50, Some(&track), EccProfile::Legacy).unwrap();
        let melody_start = (PREAMBLE_LEN + PILOT_LEN + 4 * ENCODED_DATA_OFFSET) * SYMBOL_TOTAL_SAMPLES;
        let melody_end = melody_start + 4 * payload.len() * SYMBOL_TOTAL_SAMPLES;
        assert_eq!(&checked[melody_start..melody_end], &legacy[melody_start..melody_end]);
    }

    #[test]
    fn frame_length_finishes_without_end_preamble_or_silence() {
        for len in [1, 24, MAX_LENGTH_VARIABLE] {
            let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
            let mut audio = crate::encode(&payload, 50).unwrap();
            let original_end = audio.len();
            audio.truncate(original_end - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES);
            // Replace the entire end preamble and following background with
            // a loud periodic vowel. There is no silence to trigger analysis.
            for pos in 0..12 {
                audio.extend(formant::synthesize_symbol(2, 207.65, 0.5, pos)
                    .into_iter().map(|sample| sample as f32));
            }
            let mut decoder = Decoder::new();
            let mut delivered_at = None;
            for (chunk_index, chunk) in audio.chunks(511).enumerate() {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    assert_eq!(bytes, payload);
                    assert!(delivered_at.is_none(), "duplicate delivery");
                    delivered_at = Some((chunk_index + 1) * 511);
                }
            }
            let delivered_at = delivered_at.expect("voiced background must not stall the frame");
            assert!(delivered_at <= original_end + 3 * SYMBOL_TOTAL_SAMPLES);
        }
    }

    #[test]
    fn unsuccessful_length_probe_does_not_cut_off_a_message() {
        let payload: Vec<u8> = (0..40).map(|i| (i * 73 + 41) as u8).collect();
        let audio = crate::encode(&payload, 50).unwrap();
        let prefix = 50 * SYMBOL_TOTAL_SAMPLES;
        let mut decoder = Decoder::new();
        assert!(decoder.decode(&audio[..prefix]).unwrap().is_none());
        assert!(decoder.is_receiving());
        // A timing hint may be wrong. Force an early probe, while most of
        // the actual frame is still in flight, then keep feeding the tune.
        decoder.frame_end_hints = Some(vec![decoder.symbols_recorded + 1]);
        let checkpoint = prefix + SYMBOL_TOTAL_SAMPLES;
        assert!(decoder.decode(&audio[prefix..checkpoint]).unwrap().is_none());
        assert!(decoder.is_receiving());
        assert_eq!(decoder.recorded_audio.len(), checkpoint);
        assert_eq!(decoder.decode(&audio[checkpoint..]).unwrap().as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn unanchored_trials_do_not_publish_or_overwrite_combining_observations() {
        let slot: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.2, 0)
            .into_iter().map(|sample| sample as f32).collect();
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let mut decoder = Decoder::with_profile(profile);
            decoder.recorded_audio = slot.repeat(40);
            assert!(decoder.trial_decode(40).is_none());
            assert!(decoder.last_attempt().is_none(), "{profile:?}: no alignment, no observation");

            // A later unanchored trial must not replace a previously anchored
            // observation from this reception with an unknown-offset stream.
            let previous = (vec![3, 1, 2, 0], vec![0.9, 0.8, 0.7, 0.6]);
            decoder.last_attempt = Some(previous.clone());
            assert!(decoder.trial_decode(40).is_none());
            assert_eq!(decoder.last_attempt(), Some(&previous));

            // A known start may lie outside the prefix used for a trial.
            // Wait for data, rather than slicing beyond the classified region.
            decoder.frame_start_hints = vec![35];
            assert!(decoder.trial_decode(40).is_none());
            assert_eq!(decoder.last_attempt(), Some(&previous));
        }
    }

    #[test]
    fn unanchored_trials_still_try_integrity_checked_blind_candidates() {
        let payload = b"x";
        let sync_len = PREAMBLE_LEN + PILOT_LEN;
        for profile in [EccProfile::Standard, EccProfile::Paranoid, EccProfile::Legacy] {
            let mut audio = crate::encoder::encode_with_profile(payload, 50, None, profile).unwrap();
            // Replace the entire sync/pilot with unrelated audio. The blind
            // candidate at zero still leads to an intact checked codeword.
            let slot: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.2, 0)
                .into_iter().map(|sample| sample as f32).collect();
            audio[..sync_len * SYMBOL_TOTAL_SAMPLES].copy_from_slice(&slot.repeat(sync_len));
            let n_data = audio.len() / SYMBOL_TOTAL_SAMPLES - PREAMBLE_LEN;
            let mut decoder = Decoder::with_profile(profile);
            decoder.set_payload_validator(|bytes| bytes == b"x");
            decoder.recorded_audio = audio;
            assert_eq!(decoder.trial_decode(n_data).map(|frame| frame.payload).as_deref(), Some(payload.as_slice()), "{profile:?}");
            assert!(decoder.last_attempt().is_none(), "blind success does not invent alignment");
        }
    }

    #[test]
    fn late_maximum_frame_survives_the_recording_deadline() {
        let payload: Vec<u8> = (0..MAX_LENGTH_VARIABLE)
            .map(|i| (i * 73 + 41) as u8).collect();
        let track: Vec<f64> = (0..payload.len()).map(|i| {
            let midi = 47.0 + ((i * 13 + 7) % 20) as f64;
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        }).collect();
        let frame = crate::encode_with_pitch(&payload, 50, Some(&track)).unwrap();
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|sample| sample as f32).collect();
        for before_deadline in [32, 15] {
            let mut decoder = Decoder::new();
            for _ in 0..HISTORY_WINDOWS {
                assert!(decoder.decode(&background).unwrap().is_none());
            }
            assert!(decoder.is_receiving(), "background must exercise early acquisition");
            let initial_limit = decoder.max_record_symbols;
            let prefix_slots = initial_limit - before_deadline;
            for _ in HISTORY_WINDOWS..prefix_slots {
                assert!(decoder.decode(&background).unwrap().is_none());
            }
            let mut received = frame.clone();
            for _ in 0..12 {
                received.extend(&background);
            }
            let mut deliveries = Vec::new();
            for chunk in received.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    deliveries.push(bytes);
                }
            }
            assert_eq!(deliveries, vec![payload.clone()],
                "song starting {before_deadline} windows before the original deadline");
        }
    }

    #[test]
    fn damaged_header_after_idle_preserves_the_complete_melody() {
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|sample| sample as f32).collect();
        for len in [40, MAX_LENGTH_VARIABLE] {
            let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
            let track: Vec<f64> = (0..len).map(|i| {
                let midi = 47.0 + ((i * 13 + 7) % 20) as f64;
                440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
            }).collect();
            let mut frame = crate::encode_with_pitch(&payload, 50, Some(&track)).unwrap();
            let mut header = ReedSolomon::new(1, 2).encode(&[len as u8]);
            // Two wrong header bytes exceed its correction radius. Keep
            // every pitch, the sync/pilot, and the protected body intact.
            header[0] ^= 0x5a;
            header[1] ^= 0x5a;
            for (i, vowel) in formant::bytes_to_symbols(&header).into_iter().enumerate() {
                let pos = PREAMBLE_LEN + PILOT_LEN + i;
                let slot: Vec<f32> = formant::synthesize_symbol(
                    vowel, FRAMING_LICK[pos], 0.5 * melody_accent(pos), pos,
                ).into_iter().map(|sample| sample as f32).collect();
                frame[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES]
                    .copy_from_slice(&slot);
            }
            let silence = vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES];
            let mut direct = Decoder::new();
            let direct_result = direct.decode(&frame).unwrap()
                .or_else(|| direct.decode(&silence).ok().flatten());
            assert_eq!(direct_result.as_deref(), Some(payload.as_slice()),
                "the damaged frame itself must remain recoverable, {len} bytes");

            let mut decoder = Decoder::new();
            for _ in 0..100 {
                assert!(decoder.decode(&background).unwrap().is_none());
            }
            assert!(decoder.is_receiving());
            let mut audio = frame;
            audio.extend(&silence);
            let mut deliveries = Vec::new();
            for chunk in audio.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    deliveries.push(bytes);
                }
            }
            assert_eq!(deliveries, vec![payload],
                "idle acquisition must not lose the damaged-header melody, {len} bytes");
        }
    }

    #[test]
    fn sync_only_candidate_neither_delivers_bad_crc_nor_hides_the_next_frame() {
        let payload: Vec<u8> = (0..40).map(|i| (i * 73 + 41) as u8).collect();
        let track = default_byte_pitch(payload.len());
        let profile = EccProfile::Standard;
        let mut frame = crate::encode(&payload, 50).unwrap();
        let mut protected = profile.protect_payload(&payload);
        protected[0] ^= 0x40; // Valid RS codeword below, but deliberately stale CRC.
        let mut encoded = ReedSolomon::new(1, 2).encode(&[payload.len() as u8]);
        encoded[0] ^= 0x5a;
        encoded[1] ^= 0x5a;
        encoded.extend(ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len())).encode(&protected));
        let payload_start = PREAMBLE_LEN + PILOT_LEN + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let payload_end = payload_start + formant::symbols_for_bytes(payload.len());
        let frame_slots = frame.len() / SYMBOL_TOTAL_SAMPLES;
        for (i, vowel) in formant::bytes_to_symbols(&encoded).into_iter().enumerate() {
            let pos = PREAMBLE_LEN + PILOT_LEN + i;
            let hz = if pos < payload_start { FRAMING_LICK[pos] }
                else if pos < payload_end { track[(pos - payload_start) / 4] }
                else { outro_f0(pos - payload_end, frame_slots - payload_end) };
            let slot: Vec<f32> = formant::synthesize_symbol(vowel, hz, 0.5 * melody_accent(pos), pos)
                .into_iter().map(|sample| sample as f32).collect();
            frame[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES].copy_from_slice(&slot);
        }
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|sample| sample as f32).collect();
        let mut decoder = Decoder::new();
        for _ in 0..100 {
            assert!(decoder.decode(&background).unwrap().is_none());
        }
        let original_limit = decoder.max_record_symbols;
        let prefix = 50 * SYMBOL_TOTAL_SAMPLES;
        assert!(decoder.decode(&frame[..prefix]).unwrap().is_none());
        assert!(decoder.frame_start_hints.contains(&100));
        assert!(decoder.frame_end_hints.as_ref().is_none_or(Vec::is_empty),
            "a sync-only candidate must not fabricate a length lock");
        assert!(decoder.frame_length_hints.is_empty());
        assert!(decoder.max_record_symbols > original_limit);

        let next = b"next valid melody";
        let mut remainder = frame[prefix..].to_vec();
        remainder.extend(crate::encode(next, 50).unwrap());
        remainder.extend(vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]);
        let mut deliveries = Vec::new();
        for chunk in remainder.chunks(511) {
            if let Ok(Some(bytes)) = decoder.decode(chunk) {
                deliveries.push(bytes);
            }
        }
        assert_eq!(deliveries, vec![next.to_vec()],
            "reject the first frame's bad CRC and recover the next frame without a reset");
    }

    #[test]
    fn failed_recording_rollover_is_bounded_and_silence_releases_it() {
        let background: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
            .into_iter().map(|sample| sample as f32).collect();
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut decoder = Decoder::with_profile(profile);
            if profile == EccProfile::Paranoid {
                decoder.set_payload_validator(|bytes| {
                    EccProfile::Standard.verify_payload(bytes.to_vec()).is_some()
                });
            }
            for _ in 0..HISTORY_WINDOWS {
                assert!(decoder.decode(&background).unwrap().is_none());
            }
            for _ in 0..3 {
                // Advance the same deadline branch without synthesizing
                // minutes of idle audio for a bounded-state unit test.
                decoder.max_record_symbols = decoder.symbols_recorded + HEADER_PROBE_WINDOWS;
                for _ in 1..HEADER_PROBE_WINDOWS {
                    assert!(decoder.decode(&background).unwrap().is_none());
                }
                assert!(decoder.decode(&background).is_err());
                assert_eq!(decoder.symbols_recorded, HEADER_PROBE_WINDOWS);
                assert_eq!(decoder.recorded_audio.len(), HEADER_PROBE_WINDOWS * SYMBOL_TOTAL_SAMPLES);
                assert!(decoder.history.is_empty());
                assert!(decoder.sample_buffer.is_empty());
                assert!(decoder.frame_start_hints.is_empty());
                assert!(decoder.frame_length_hints.is_empty());
                assert!(decoder.frame_end_hints.is_none());
                assert!(!decoder.rescan_after_failed_probe);
                let frame = formant::symbols_for_bytes(profile.encoded_bytes(MAX_LENGTH_VARIABLE))
                    + PREAMBLE_LEN + PILOT_LEN;
                let slack = (frame * (GUARD_SAMPLES / 4)).div_ceil(SYMBOL_TOTAL_SAMPLES);
                let maximum = frame + slack + HEADER_PROBE_WINDOWS + 10;
                assert_eq!(decoder.max_record_symbols, maximum);
            }
            // Rollover preserves the level estimate, so quiet following
            // audio releases the lock instead of starting a full timeout.
            for _ in 0..10 {
                assert!(!matches!(decoder.decode(&vec![0.0; SYMBOL_TOTAL_SAMPLES]), Ok(Some(_))));
            }
            assert!(!decoder.is_receiving(), "silence must release {profile:?} after rollover");
        }
    }

    #[test]
    fn nonfinite_input_does_not_poison_receiving_level_or_recording() {
        let payload = b"keep listening after an invalid sample";
        let audio = crate::encode(payload, 50).unwrap();
        let prefix = 32 * SYMBOL_TOTAL_SAMPLES;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut decoder = Decoder::new();
            assert!(decoder.decode(&audio[..prefix]).unwrap().is_none());
            assert!(decoder.is_receiving());
            assert!(decoder.rx_rms_ema > 0.0 && decoder.rx_rms_ema.is_finite());
            let mut window = audio[prefix..prefix + SYMBOL_TOTAL_SAMPLES].to_vec();
            window[SAMPLES_PER_SYMBOL / 2] = invalid;
            assert!(decoder.decode(&window).unwrap().is_none());
            assert!(decoder.rx_rms_ema.is_finite(), "level poisoned by {invalid}");
            assert!(decoder.recorded_audio.iter().all(|sample| sample.is_finite()));
            let result = decoder.decode(&audio[prefix + SYMBOL_TOTAL_SAMPLES..]).unwrap()
                .or_else(|| decoder.decode(&vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]).unwrap());
            assert_eq!(result.as_deref(), Some(payload.as_slice()));
        }
    }

    #[test]
    fn quiet_notes_remain_voiced_across_register() {
        for midi in 47..=66 {
            let hz = 440.0 * 2.0f64.powf((midi as f64 - 69.0) / 12.0);
            for vowel in 0..NUM_VOWELS {
                let audio = formant::synthesize_symbol(vowel, hz, 0.001, midi as usize);
                let voiced: Vec<_> = audio[..SAMPLES_PER_SYMBOL].iter().map(|&x| x as f32).collect();
                assert!(has_voiced_periodicity(&voiced), "MIDI {midi}, vowel {vowel}");
            }
        }
    }

    #[test]
    fn silence_dc_and_noise_are_not_voiced() {
        for dc in [0.0, 0.01, -0.003] {
            assert!(!has_voiced_periodicity(&vec![dc; SAMPLES_PER_SYMBOL]));
        }
        let mut seed = 123u32;
        let noise: Vec<_> = (0..SAMPLES_PER_SYMBOL).map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            0.01 + 0.001 * (seed as f32 / u32::MAX as f32 - 0.5)
        }).collect();
        assert!(!has_voiced_periodicity(&noise));
    }

    #[test]
    fn long_quiet_phrase_does_not_end_reception() {
        let payload: Vec<_> = (0..32).map(|i| (i * 73 + 19) as u8).collect();
        let track = default_byte_pitch(payload.len());
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut audio = crate::encoder::encode_with_profile(&payload, 50, Some(&track), profile).unwrap();
            // Two seconds of a phrase drop 40 dB below preceding audio.
            let start = (PREAMBLE_LEN + PILOT_LEN + 4 * (ENCODED_DATA_OFFSET + 8)) * SYMBOL_TOTAL_SAMPLES;
            for sample in &mut audio[start..start + 32 * SYMBOL_TOTAL_SAMPLES] {
                *sample *= 0.01;
            }
            let mut decoder = Decoder::with_profile(profile);
            let mut result = None;
            for chunk in audio.chunks(437) {
                if let Some(bytes) = decoder.decode(chunk).unwrap() {
                    result = Some(bytes);
                    break;
                }
            }
            if result.is_none() {
                result = decoder.decode(&vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]).unwrap();
            }
            assert_eq!(result.as_deref(), Some(payload.as_slice()), "{profile:?}");
        }
    }

    #[test]
    fn detuned_melody_roundtrips_across_register() {
        let payload: Vec<u8> = (0..80).map(|i| (i * 73 + 41) as u8).collect();
        let track: Vec<_> = (0..payload.len()).map(|i| {
            // Include pitches between the template bank's semitones.
            let midi = 47.0 + (i * 13 % 20) as f64 + if i % 2 == 0 { 0.3 } else { -0.3 };
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        }).collect();
        let audio = crate::encode_with_pitch(&payload, 50, Some(&track)).unwrap();
        assert_eq!(Decoder::new().decode(&audio).unwrap().as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn missing_end_preamble_still_finalizes_on_silence() {
        let payload = b"a quiet ending still completes the song";
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            let mut audio = crate::encoder::encode_with_profile(payload, 50, None, profile).unwrap();
            let end = audio.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES;
            audio[end..].fill(0.0);
            audio.extend(vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]);
            let mut decoder = Decoder::with_profile(profile);
            assert_eq!(decoder.decode(&audio).unwrap().as_deref(), Some(payload.as_slice()));
            assert!(!decoder.is_receiving());
        }
    }

    /// Deterministic Box-Muller noise for melody-hint tests.
    fn hint_noise(samples: &[f32], snr_db: f64, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut lcg = move || {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (state >> 16) as f32 / 65535.0
        };
        let power: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
            / samples.len() as f64;
        let rms = (power.sqrt() / 10.0f64.powf(snr_db / 20.0)) as f32;
        samples.iter().map(|&s| {
            let u1 = lcg().max(1e-10);
            let u2 = lcg();
            s + (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * rms
        }).collect()
    }

    #[test]
    fn melody_length_hint_is_exact_for_default_and_custom_melodies() {
        let sync = PREAMBLE_LEN + PILOT_LEN;
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for len in [1usize, 17, 64, if profile == EccProfile::Paranoid { 120 } else { 140 }] {
                let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
                let custom: Vec<f64> = (0..len)
                    .map(|i| 440.0 * 2.0f64.powf((47 + (i * 7 + 3) % 20) as f64 - 69.0) / 12.0)
                    .map(|_| 0.0)
                    .collect();
                let _ = custom;
                let tracks: [Option<Vec<f64>>; 2] = [
                    None,
                    Some((0..len).map(|i| {
                        440.0 * 2.0f64.powf(((47 + (i * 7 + 3) % 20) as f64 - 69.0) / 12.0)
                    }).collect()),
                ];
                for track in &tracks {
                    let audio = crate::encoder::encode_with_profile(
                        &payload, 50, track.as_deref(), profile,
                    ).unwrap();
                    for noisy in [false, true] {
                        let audio = if noisy { hint_noise(&audio, 3.0, 9) } else { audio.clone() };
                        let mut spectra = AlignedSpectra::new(&audio, 0, usize::MAX);
                        let windows = audio.len() / SYMBOL_TOTAL_SAMPLES;
                        let hint = melody_length_hint(&mut spectra, sync, windows - sync, profile);
                        assert_eq!(hint, Some(len), "{profile:?} len {len} noisy {noisy}");
                    }
                }
            }
        }
    }

    #[test]
    fn melody_length_hint_excludes_legacy() {
        let payload = [7u8; 24];
        let audio = crate::encoder::encode_with_profile(
            &payload, 50, None, EccProfile::Legacy,
        ).unwrap();
        let mut spectra = AlignedSpectra::new(&audio, 0, usize::MAX);
        let windows = audio.len() / SYMBOL_TOTAL_SAMPLES;
        let sync = PREAMBLE_LEN + PILOT_LEN;
        assert_eq!(
            melody_length_hint(&mut spectra, sync, windows - sync, EccProfile::Legacy),
            None,
        );
    }

    #[test]
    fn melody_hint_rescues_a_damaged_header_before_trailing_symbols() {
        // Symbol-domain: a destroyed header plus 60 junk symbols beyond
        // the frame defeats both the header path and the length-slop
        // candidates; the melody hint's gated prefix candidate recovers.
        let profile = EccProfile::Standard;
        let payload: Vec<u8> = (0..33).map(|i| (i * 29 + 5) as u8).collect();
        let protected = profile.protect_payload(&payload);
        let header = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1).encode(&[payload.len() as u8]);
        let body = ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len()))
            .encode(&protected);
        let mut bytes = header;
        bytes[0] ^= 0x5a;
        bytes[1] ^= 0xa5;
        bytes.extend_from_slice(&body);
        let mut symbols = formant::bytes_to_symbols(&bytes);
        symbols.extend(std::iter::repeat(2usize).take(60));

        assert_eq!(
            decode_symbol_stream_cached(&symbols, None, None, profile, None, None,
                Some(&mut FailedPayloadCache::default()), None),
            None,
            "without the melody hint this region must stay undecodable",
        );
        assert_eq!(
            decode_symbol_stream_cached(&symbols, None, None, profile, None, None,
                Some(&mut FailedPayloadCache::default()), Some(payload.len())).as_deref(),
            Some(payload.as_slice()),
        );
        // A wrong hint is harmless noise: some other gated candidate, or
        // nothing, but never a wrong payload.
        for wrong in [1usize, 5, 100, 140] {
            let out = decode_symbol_stream_cached(&symbols, None, None, profile, None, None,
                Some(&mut FailedPayloadCache::default()), Some(wrong));
            assert!(out.is_none() || out.as_deref() == Some(payload.as_slice()));
        }
        // Unchecked Legacy must ignore the hint entirely.
        assert_eq!(
            decode_symbol_stream_cached(&symbols, None, None, EccProfile::Legacy, None, None,
                Some(&mut FailedPayloadCache::default()), Some(payload.len())),
            None,
        );
    }

    #[test]
    fn melody_hint_does_not_change_intact_frame_decoding() {
        // With a healthy header, hint values right or wrong must leave the
        // result untouched (priority ordering only).
        let profile = EccProfile::Standard;
        let payload: Vec<u8> = (0..40).map(|i| (i * 31 + 3) as u8).collect();
        let protected = profile.protect_payload(&payload);
        let mut bytes = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1).encode(&[payload.len() as u8]);
        bytes.extend(ReedSolomon::new(protected.len(), profile.ecc_bytes(payload.len()))
            .encode(&protected));
        let symbols = formant::bytes_to_symbols(&bytes);
        for hint in [None, Some(payload.len()), Some(7), Some(140)] {
            assert_eq!(
                decode_symbol_stream_cached(&symbols, None, None, profile, None, None,
                    Some(&mut FailedPayloadCache::default()), hint).as_deref(),
                Some(payload.as_slice()),
                "hint {hint:?}",
            );
        }
    }

    #[test]
    fn full_decoder_recovers_damaged_header_with_trailing_voiced_background() {
        // End-to-end: real audio, destroyed length header, 30 slots of
        // voiced non-modem background after the frame, then silence. The
        // solo's position is the only length evidence left.
        let payload: Vec<u8> = (0..33).map(|i| (i * 29 + 5) as u8).collect();
        let mut audio = crate::encoder::encode_with_profile(
            &payload, 50, None, EccProfile::Standard,
        ).unwrap();
        let mut header = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1)
            .encode(&[payload.len() as u8]);
        header[0] ^= 0x5a;
        header[1] ^= 0xa5;
        for (i, vowel) in formant::bytes_to_symbols(&header).into_iter().enumerate() {
            let pos = PREAMBLE_LEN + PILOT_LEN + i;
            let slot: Vec<f32> = formant::synthesize_symbol(
                vowel, FRAMING_LICK[pos], 0.5 * melody_accent(pos), pos,
            ).into_iter().map(|s| s as f32).collect();
            audio[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES]
                .copy_from_slice(&slot);
        }
        let base = audio.len() / SYMBOL_TOTAL_SAMPLES;
        for j in 0..30 {
            let slot: Vec<f32> = formant::synthesize_symbol(2, 200.0, 0.4, base + j)
                .into_iter().map(|s| s as f32).collect();
            audio.extend_from_slice(&slot);
        }
        audio.extend(std::iter::repeat(0.0f32).take(SAMPLE_RATE as usize));

        let mut decoder = Decoder::new();
        let mut delivered = None;
        for chunk in audio.chunks(1024) {
            if let Ok(Some(bytes)) = decoder.decode(chunk) {
                delivered = Some(bytes);
            }
        }
        assert_eq!(delivered.as_deref(), Some(payload.as_slice()));
    }
}
