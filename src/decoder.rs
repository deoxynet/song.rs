//! Decoder: f32 audio samples → payload (vowel modem, streaming).
//!
//! Symbols are vowel indices; pitch carries no data and is ignored entirely
//! by the primary (F2-centroid) detection path.

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

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

/// How deep into the recording the start preamble can plausibly sit: the
/// recording begins with at most HISTORY_WINDOWS of history plus the
/// fallback-lock margin, so the preamble always lives near the head.
/// Scanning further only surfaces preamble-shaped DATA patterns (any
/// 00 11 00 11 bit run), which under channel tilt out-score the genuine —
/// attenuated — preamble.
const PREAMBLE_SCAN_WINDOWS: usize = 24;

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
            state: DecoderState::Listening,
            sample_buffer: Vec::new(),
            recent_vowels: Vec::new(),
            recent_symbols: Vec::new(),
            recorded_audio: Vec::new(),
            symbols_recorded: 0,
            max_record_symbols: 0,
            history: Vec::new(),
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
    /// fail it are treated as failures: mid-reception trial decodes keep
    /// receiving instead of terminating on plausible garbage, and analysis
    /// never returns a payload the validator rejects.
    pub fn set_payload_validator(&mut self, f: fn(&[u8]) -> bool) {
        self.validator = Some(f);
    }

    /// True while the decoder has locked onto a transmission (i.e. it is
    /// receiving or analyzing, not idle-listening for a preamble).
    pub fn is_receiving(&self) -> bool {
        self.state != DecoderState::Listening
    }

    /// Best-aligned (symbols, confidences) from the most recent decode
    /// attempt, for soft combining across repeated transmissions.
    pub fn last_attempt(&self) -> Option<&(Vec<usize>, Vec<f64>)> {
        self.last_attempt.as_ref()
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

    pub fn decode(&mut self, samples: &[f32]) -> Result<Option<Vec<u8>>, crate::Error> {
        self.sample_buffer.extend_from_slice(samples);

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
                    // After ~4 windows of silence (~60ms at any sample rate),
                    // clear history and recent_vowels to reset alignment.
                    if self.silence_count >= 4 {
                        self.history.clear();
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
                // 12 windows ≈ 720ms is safely above typical message lengths.
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
                        self.eq = formant::estimate_eq(&spectra);
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
                // Profile-dependent run length: the transfer flow (Paranoid)
                // needs corrupted receptions to finalize quickly between
                // repeats (4 windows ≈ 240 ms). The standard decoder demands
                // a longer run (6 windows ≈ 360 ms) because denoiser gain
                // dips and deep fading legitimately silence ~240 ms
                // mid-message — but a reception whose end preamble was
                // missed must STILL finalize on trailing silence, or it
                // sits in Receiving until the recording buffer fills.
                let is_silent = self.rx_rms_ema > 0.0 && rms < 0.3 * self.rx_rms_ema;
                if is_silent {
                    self.rx_silence_run += 1;
                    let needed = if self.profile == EccProfile::Paranoid { 4 } else { 6 };
                    let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                    if self.rx_silence_run >= needed
                        && self.symbols_recorded
                            > min_data_symbols + PREAMBLE_LEN + PILOT_LEN
                    {
                        self.state = DecoderState::Analyzing;
                        return self.analyze();
                    }
                    if self.profile == EccProfile::Paranoid {
                        // Don't classify near-silent windows (they'd pollute
                        // the preview/end-detection streams); the recording
                        // keeps them for analyze().
                        return Ok(None);
                    }
                } else {
                    self.rx_silence_run = 0;
                    self.rx_rms_ema = if self.rx_rms_ema == 0.0 {
                        rms
                    } else {
                        0.9 * self.rx_rms_ema + 0.1 * rms
                    };
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
                        if let Some(payload) = self.trial_decode(trial_n) {
                            self.state = DecoderState::Listening;
                            self.recent_vowels.clear();
                            self.recent_symbols.clear();
                            return Ok(Some(payload));
                        }
                        // Trial decode failed — likely misaligned due to history
                        // prepend, or a false positive vowel/symbol match from data.
                        // Don't transition to Analyzing on first false positive;
                        // continue receiving to find the real end preamble.
                    }
                }

                if self.symbols_recorded >= self.max_record_symbols {
                    self.state = DecoderState::Analyzing;
                    return self.analyze();
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

        // Prepend history windows to recording for alignment recovery
        self.recorded_audio.clear();
        let n_history = self.history.len();
        for w in &self.history {
            self.recorded_audio.extend_from_slice(w);
        }
        self.symbols_recorded = n_history;
        self.history.clear();
        // The lock fired on the window that completed the preamble pattern,
        // so the preamble occupies the last PREAMBLE_LEN history windows.
        self.preamble_pos_hint =
            saw_preamble.then(|| n_history.saturating_sub(PREAMBLE_LEN));

        let max_total_bytes = ENCODED_DATA_OFFSET
            + MAX_LENGTH_VARIABLE
            + self.profile.ecc_bytes(MAX_LENGTH_VARIABLE);
        let max_data_syms = formant::symbols_for_bytes(max_total_bytes);
        self.max_record_symbols =
            max_data_syms + PREAMBLE_LEN + PILOT_LEN + n_history + 10;

        self.recorded_audio
            .reserve(self.max_record_symbols * SYMBOL_TOTAL_SAMPLES);
    }

    fn trial_decode(&mut self, n_data_symbols: usize) -> Option<Vec<u8>> {
        let mut symbols = Vec::with_capacity(n_data_symbols);
        let mut confidences = Vec::with_capacity(n_data_symbols);
        let mut seconds = Vec::with_capacity(n_data_symbols);
        let mut centroids = Vec::with_capacity(n_data_symbols);

        let centers = self.centers;
        for i in 0..n_data_symbols {
            let class = self.classify_full_at(0, i, &centers)?;
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
        let anchored = self.preamble_pos_hint.or_else(|| {
            (0..scan_end.min(PREAMBLE_SCAN_WINDOWS))
                .find(|&pos| symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_START_VOWELS)
        });
        let anchor_start = anchored.map_or(0, |p| p + sync_len);
        self.last_attempt = Some((
            symbols[anchor_start..].to_vec(),
            confidences[anchor_start..].to_vec(),
        ));

        // No preamble located: single attempt from the start of the region.
        if cand_positions.is_empty() {
            return self.try_decode_symbols(&symbols, Some(&confidences), Some(&seconds));
        }

        // One pristine (zero-correction, high-confidence) header decode
        // among the candidates locks the payload length for the whole
        // trial: any interpretation at another length — from any candidate
        // position or classification rung — is inadmissible noise.
        let locked = cand_positions.iter().find_map(|&pos| {
            let ds = pos + sync_len;
            self.locked_header_len(&symbols[ds..], &confidences[ds..])
        });
        for &pos in &cand_positions {
            let data_start = pos + sync_len;
            if let Some(payload) = self.try_decode_symbols_locked(
                &symbols[data_start..],
                Some(&confidences[data_start..]),
                Some(&seconds[data_start..]),
                locked,
            ) {
                return Some(payload);
            }
            // With a locked length, a region too short to hold the frame
            // cannot decode at ANY rung — skip the expensive
            // re-classification sweeps (this is the common case for false
            // end-preamble fires mid-reception).
            if let Some(l) = locked {
                let need =
                    formant::symbols_for_bytes(ENCODED_DATA_OFFSET + l + self.profile.ecc_bytes(l));
                if symbols.len().saturating_sub(data_start) < need {
                    continue;
                }
            }
            // Retry with pilot-calibrated centers (absorbs mild channel tilt).
            let cal_centers = self
                .calibrated
                .or_else(|| self.pilot_centers_at(0, pos + PREAMBLE_LEN));
            if let Some(centers) = cal_centers {
                let mut cal_syms = symbols[data_start..].to_vec();
                let mut cal_confs = confidences[data_start..].to_vec();
                let mut cal_seconds = seconds[data_start..].to_vec();
                for (k, i) in (data_start..symbols.len()).enumerate() {
                    if let Some(class) = self.classify_full_at(0, i, &centers) {
                        cal_syms[k] = class.vowel;
                        cal_confs[k] = class.confidence;
                        cal_seconds[k] = class.second;
                    }
                }
                if let Some(payload) = self.try_decode_symbols_locked(
                    &cal_syms,
                    Some(&cal_confs),
                    Some(&cal_seconds),
                    locked,
                ) {
                    return Some(payload);
                }
            }
            // Retry through the pilot-derived channel EQ (absorbs heavy tilt).
            let eq = self
                .pilot_eq_at(0, pos + PREAMBLE_LEN)
                .or_else(|| self.eq.clone());
            if let Some(eq) = eq {
                let n = symbols.len() - data_start;
                let classify_stream =
                    |f: &dyn Fn(&Self, usize) -> Option<formant::SymbolClass>| {
                        let mut cs = symbols[data_start..].to_vec();
                        let mut cc = confidences[data_start..].to_vec();
                        let mut c2 = seconds[data_start..].to_vec();
                        for k in 0..n {
                            if let Some(class) = f(self, data_start + k) {
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
                let eq_s = classify_stream(&|d, i| d.classify_eq_at(0, i, &eq));
                let tm_s = classify_stream(&|d, i| d.classify_template_at(0, i, &eq));
                let fa_s = classify_stream(&|d, i| d.classify_f0a_at(0, i, &eq));
                for (cs, cc, c2) in [&eq_s, &tm_s, &fa_s] {
                    if let Some(payload) =
                        self.try_decode_symbols_locked(cs, Some(cc), Some(c2), locked)
                    {
                        return Some(payload);
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
                    self.try_decode_symbols_locked(&vs, Some(&vc), Some(&v2), locked)
                {
                    return Some(payload);
                }
            }
        }
        None
    }

    fn analyze(&mut self) -> Result<Option<Vec<u8>>, crate::Error> {
        let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if self.symbols_recorded < min_data_symbols {
            self.state = DecoderState::Listening;
            self.recent_vowels.clear();
            return Err(crate::Error::DecodeFailed);
        }

        let total_audio_len = self.recorded_audio.len();
        let steps_per_symbol: usize = 16;
        let step = SYMBOL_TOTAL_SAMPLES / steps_per_symbol;

        let mut best_payload: Option<Vec<u8>> = None;
        let mut best_confidence: f64 = 0.0;
        let mut best_attempt: Option<(f64, Vec<usize>, Vec<f64>)> = None;

        // Try more offsets to handle misalignment from padding
        let max_offset_steps = steps_per_symbol * 2;

        for offset_step in 0..max_offset_steps {
            let sample_offset = offset_step * step;

            let mut symbols: Vec<usize> = Vec::new();
            let mut confidences: Vec<f64> = Vec::new();
            let mut seconds: Vec<usize> = Vec::new();
            let mut centroids: Vec<f64> = Vec::new();
            let mut confidence_sum = 0.0f64;
            let mut sym_idx = 0;

            let centers = self.centers;
            loop {
                let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
                if start + SAMPLES_PER_SYMBOL > total_audio_len {
                    break;
                }

                let Some(class) = self.classify_full_at(sample_offset, sym_idx, &centers)
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
            let cand_positions = self.candidate_positions(&symbols, &centroids, scan_end);
            let anchored = self.preamble_pos_hint.or_else(|| {
                (0..scan_end.min(PREAMBLE_SCAN_WINDOWS))
                    .find(|&pos| symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_START_VOWELS)
            });
            let first_start = cand_positions.first().map_or(0, |&p| p + sync_len);

            // Candidate end positions: last end-preamble match, then
            // no end-preamble stripping as fallback.
            let mut end_candidates = Vec::new();
            for pos in (first_start + min_data_symbols
                ..=symbols.len().saturating_sub(PREAMBLE_LEN))
                .rev()
            {
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
                                      best_payload: &mut Option<Vec<u8>>,
                                      best_confidence: &mut f64| {
                if best_payload.is_none() || avg_conf > *best_confidence {
                    *best_confidence = avg_conf;
                    *best_payload = Some(payload);
                }
            };

            let locked = cand_positions.iter().find_map(|&pos| {
                let ds = pos + sync_len;
                self.locked_header_len(&symbols[ds..], &confidences[ds..])
            });
            for &pos in &cand_positions {
                let data_start = pos + sync_len;

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
                    {
                        continue;
                    }
                    if let Some(payload) = self.try_decode_symbols_locked(
                        &symbols[data_start..data_end],
                        Some(&confidences[data_start..data_end]),
                        Some(&seconds[data_start..data_end]),
                        locked,
                    ) {
                        record_success(payload, &mut best_payload, &mut best_confidence);
                        continue;
                    }
                    if !derived {
                        if let Some(l) = locked {
                            let need = formant::symbols_for_bytes(
                                ENCODED_DATA_OFFSET + l + self.profile.ecc_bytes(l),
                            );
                            if data_end.saturating_sub(data_start) < need {
                                continue;
                            }
                        }
                        derived = true;
                        if let Some(centers) =
                            self.pilot_centers_at(sample_offset, pos + PREAMBLE_LEN)
                        {
                            let mut cs = symbols.clone();
                            let mut cc = confidences.clone();
                            let mut c2 = seconds.clone();
                            for i in data_start..cs.len() {
                                if let Some(class) =
                                    self.classify_full_at(sample_offset, i, &centers)
                                {
                                    cs[i] = class.vowel;
                                    cc[i] = class.confidence;
                                    c2[i] = class.second;
                                }
                            }
                            cal = Some((cs, cc, c2));
                        }
                        if let Some(eq) = self.pilot_eq_at(sample_offset, pos + PREAMBLE_LEN)
                        {
                            let mut cs = symbols.clone();
                            let mut cc = confidences.clone();
                            let mut c2 = seconds.clone();
                            for i in data_start..cs.len() {
                                if let Some(class) =
                                    self.classify_eq_at(sample_offset, i, &eq)
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
                                    self.classify_template_at(sample_offset, i, &eq)
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
                                    self.classify_f0a_at(sample_offset, i, &eq)
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
                    let retry = cal
                        .as_ref()
                        .and_then(|(cs, cc, c2)| {
                            self.try_decode_symbols_locked(
                                &cs[data_start..data_end],
                                Some(&cc[data_start..data_end]),
                                Some(&c2[data_start..data_end]),
                                locked,
                            )
                        })
                        .or_else(|| {
                            let (cs, cc, c2) = eq_alt.as_ref()?;
                            self.try_decode_symbols_locked(
                                &cs[data_start..data_end],
                                Some(&cc[data_start..data_end]),
                                Some(&c2[data_start..data_end]),
                                locked,
                            )
                        })
                        .or_else(|| {
                            let (cs, cc, c2) = tmpl_alt.as_ref()?;
                            self.try_decode_symbols_locked(
                                &cs[data_start..data_end],
                                Some(&cc[data_start..data_end]),
                                Some(&c2[data_start..data_end]),
                                locked,
                            )
                        })
                        .or_else(|| {
                            let (cs, cc, c2) = f0a_alt.as_ref()?;
                            self.try_decode_symbols_locked(
                                &cs[data_start..data_end],
                                Some(&cc[data_start..data_end]),
                                Some(&c2[data_start..data_end]),
                                locked,
                            )
                        })
                        .or_else(|| {
                            let (cs, cc, c2) = vote_alt.as_ref()?;
                            self.try_decode_symbols_locked(
                                &cs[data_start..data_end],
                                Some(&cc[data_start..data_end]),
                                Some(&c2[data_start..data_end]),
                                locked,
                            )
                        });
                    if let Some(payload) = retry {
                        record_success(payload, &mut best_payload, &mut best_confidence);
                    }
                }
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
                ) {
                    record_success(payload, &mut best_payload, &mut best_confidence);
                }
            }
        }

        self.state = DecoderState::Listening;
        self.recent_vowels.clear();
        if let Some((_, syms, confs)) = best_attempt {
            self.last_attempt = Some((syms, confs));
        }

        if let Some(payload) = best_payload {
            Ok(Some(payload))
        } else {
            Err(crate::Error::DecodeFailed)
        }
    }

    /// Candidate start-preamble positions: the recorded lock position hint
    /// first (exact by construction when the Listening lock saw the
    /// preamble), then pattern-matched candidates, then blind head positions
    /// (the preamble always sits within the prepended history). All are
    /// verified by RS-gated decode attempts.
    fn candidate_positions(
        &self,
        symbols: &[usize],
        centroids: &[f64],
        max_scan: usize,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(h) = self.preamble_pos_hint {
            if h < max_scan {
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

    fn try_decode_symbols(
        &self,
        symbols: &[usize],
        confidences: Option<&[f64]>,
        seconds: Option<&[usize]>,
    ) -> Option<Vec<u8>> {
        self.try_decode_symbols_locked(symbols, confidences, seconds, None)
    }

    fn try_decode_symbols_locked(
        &self,
        symbols: &[usize],
        confidences: Option<&[f64]>,
        seconds: Option<&[usize]>,
        locked_len: Option<usize>,
    ) -> Option<Vec<u8>> {
        decode_symbol_stream_locked(symbols, confidences, seconds, self.profile, locked_len)
            .filter(|p| self.validator.map_or(true, |v| v(p)))
    }

    /// Try to LOCK the payload length from the default-classified data
    /// region: a hard-decision header decode over uniformly high-confidence
    /// symbols cannot be wrong about the length, and every wrong-length
    /// interpretation later rungs might hallucinate is then inadmissible.
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

    /// Fully classify the window at symbol index `i` for the given sample
    /// offset, using explicit classification centers.
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
        formant::estimate_eq(&measured)
    }
}

/// Maximum symbols flipped to their runner-up decision during Chase retry.
/// 2^CHASE_MAX_FLIPS - 1 hypotheses are tried on the RS failure path.
const CHASE_MAX_FLIPS: usize = 6;

/// Only symbols below this confidence participate in Chase flipping.
const CHASE_CONF_LIMIT: f64 = 0.6;

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
    if let Some(payload) = decode_symbols_once(symbols, confidences, profile, false, locked_len) {
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
        // Chase trials are always gated: flipping symbols and re-running RS
        // hundreds of times would otherwise buy garbage a lottery ticket.
        if let Some(payload) =
            decode_symbols_once(&trial, Some(&trial_conf), profile, true, locked_len)
        {
            return Some(payload);
        }
    }
    None
}

/// One full decode attempt (hard-decision + erasure fallback) over a fixed
/// symbol stream.
///
/// `gate_all`: apply the anti-miscorrection agreement gate to every success
/// (used for Chase trials). Without it, only the risky paths are gated:
/// erasure-fallback decodes (their residual syndrome check is weak once
/// most parity is spent on erasures) and derived-length candidates. The
/// pre-existing trusted path — hard-decision decode at the header-approved
/// length — stays ungated.
fn decode_symbols_once(
    symbols: &[usize],
    confidences: Option<&[f64]>,
    profile: EccProfile,
    gate_all: bool,
    locked_len: Option<usize>,
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
        let header_len = rs_length
            .decode(&data_encoded[..ENCODED_DATA_OFFSET])
            .or_else(|| {
                let confs = confidences?;
                let erasures = compute_byte_erasures(confs, 0, ENCODED_DATA_OFFSET, ENCODED_DATA_OFFSET - 1);
                if erasures.is_empty() {
                    return None;
                }
                rs_length.decode_with_erasures(&data_encoded[..ENCODED_DATA_OFFSET], &erasures)
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
        if locked_len.is_some() {
            // No alternates under a length lock.
        } else if profile == EccProfile::Paranoid {
            for l in (1..=MAX_LENGTH_VARIABLE).rev() {
                if Some(l) != header_len
                    && ENCODED_DATA_OFFSET + l + profile.ecc_bytes(l) <= data_encoded.len()
                {
                    candidates.push(l);
                }
            }
        } else {
            // Standard profile: the 3-byte length header corrects only 1
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
                let total = ENCODED_DATA_OFFSET + l + profile.ecc_bytes(l);
                if Some(l) != header_len
                    && profile.ecc_bytes(l) >= 4
                    && total <= data_encoded.len()
                    && data_encoded.len() - total <= 3
                {
                    candidates.push(l);
                }
            }
        }

        let mut best_risky: Option<(f64, Vec<u8>)> = None;
        for len in candidates {
            let n_ecc = profile.ecc_bytes(len);
            let expected_total = ENCODED_DATA_OFFSET + len + n_ecc;
            if data_encoded.len() < expected_total {
                continue;
            }

            // Decode payload block: hard-decision first, then erasure fallback
            let rs_data = ReedSolomon::new(len, n_ecc);
            let payload_block =
                &data_encoded[ENCODED_DATA_OFFSET..ENCODED_DATA_OFFSET + len + n_ecc];
            let mut used_erasures = false;
            let decoded = rs_data.decode(payload_block).or_else(|| {
                let confs = confidences?;
                let sym_offset = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                let erasures = compute_byte_erasures(confs, sym_offset, len + n_ecc, n_ecc);
                if erasures.is_empty() {
                    return None;
                }
                used_erasures = true;
                rs_data.decode_with_erasures(payload_block, &erasures)
            });
            if let Some(payload) = decoded {
                // Anti-miscorrection gate for risky successes. Erasure
                // decoding with most parity spent on erasures retains only a
                // weak residual syndrome check (~1% garbage acceptance), and
                // derived lengths / Chase trials multiply attempt volume —
                // without a gate, misaligned or truncated streams eventually
                // "decode". The Paranoid profile skips it: the transfer
                // layer's CRC-32 is a stronger end gate.
                let approved = Some(len) == header_len || Some(len) == locked_len;
                let risky = gate_all || used_erasures || !approved;
                if profile == EccProfile::Paranoid || !risky {
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

/// Confidence-aware anti-miscorrection check: re-encode `payload` and
/// compare symbol-by-symbol against what was received. Returns the
/// agreement score (higher = better) when the decode passes the gate,
/// None when it looks like a miscorrection.
///
/// A TRUE decode disagrees with the received stream only where the channel
/// corrupted symbols — and those classify with LOW confidence. A
/// miscorrection "corrects" bytes the channel delivered cleanly, producing
/// high-confidence disagreements that a real channel error pattern cannot.
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
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let mut encoded = rs_length.encode(&[len as u8]);
    let n_ecc = profile.ecc_bytes(len);
    let rs_data = ReedSolomon::new(len, n_ecc);
    encoded.extend_from_slice(&rs_data.encode(payload));
    let expect_syms = formant::bytes_to_symbols(&encoded);
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
/// `ecc_len / 2` to leave room for hard error correction.
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
