/// Sample rate in Hz.
#[cfg(not(feature = "8khz"))]
pub const SAMPLE_RATE: f64 = 48000.0;
#[cfg(feature = "8khz")]
pub const SAMPLE_RATE: f64 = 8000.0;

/// Melody: F0 sequence indexed by symbol position within a transmission.
///
/// Pitch carries no data — the decoder is F0-agnostic (F2-centroid detection) —
/// so the encoder is free to play a melody. D lydian arpeggio, up-down:
/// D3, G#3, C#4, G#3 (root, #4, maj7, #4). Notes chosen so every vowel's
/// F2-band centroid stays well-anchored at every note, including under
/// reverb combs: each note keeps ≥3 harmonics in vowel 0's Gaussian core so
/// no single comb null can flip it. A3 (H5 = 1100 Hz has only one in-band
/// neighbor) and E3/G3/B3 were measured to misclassify vowel 0 under reverb
/// and are excluded.
pub const MELODY: [f64; 4] = [146.83, 207.65, 277.18, 207.65];

/// F0 for the symbol at `pos` (position within the transmission, preamble included).
#[inline]
pub fn melody_f0(pos: usize) -> f64 {
    MELODY[pos % MELODY.len()]
}

/// The default payload melody: the 16-bar head of John Coltrane's
/// "Giant Steps" (1960), one note per payload byte (= one quarter note at
/// ~250 bpm), transposed down an octave plus a major third into the
/// modem's B2..F#4 compass — the true contour fits with no octave folds.
///
/// Pitch carries no data, so `encode` is free to sing this over the
/// payload; sync (preamble + pilot), header, and RS parity keep the
/// margin-verified arpeggio (`MELODY`), and the transfer profile uses the
/// arpeggio throughout — the worst measured channels (heavy tilt plus an
/// aggressive denoiser) still favor the armored notes, and secrets don't
/// get to trade margin for style. Payloads longer than one chorus loop
/// the form, which is what the form is for.
pub const GIANT_STEPS: [f64; 64] = [
    // 1: F#5 D5              2: B4 G4
    293.66, 293.66, 233.08, 233.08, 196.00, 196.00, 155.56, 155.56,
    // 3: Bb4 (held)          4: B4 A4
    185.00, 185.00, 185.00, 185.00, 196.00, 196.00, 174.61, 174.61,
    // 5: D5 Bb4              6: G4 Eb4
    233.08, 233.08, 185.00, 185.00, 155.56, 155.56, 123.47, 123.47,
    // 7: F#4 (held)          8: G4 F4
    146.83, 146.83, 146.83, 146.83, 155.56, 155.56, 138.59, 138.59,
    // 9: Bb4 (held)          10: B4 A4
    185.00, 185.00, 185.00, 185.00, 196.00, 196.00, 174.61, 174.61,
    // 11: D5 (held)          12: Eb5 C#5
    233.08, 233.08, 233.08, 233.08, 246.94, 246.94, 220.00, 220.00,
    // 13: F#5 (held)         14: G5 F5
    293.66, 293.66, 293.66, 293.66, 311.13, 311.13, 277.18, 277.18,
    // 15: Bb5 (held)         16: F#5 F#5 (turnaround)
    369.99, 369.99, 369.99, 369.99, 293.66, 293.66, 293.66, 293.66,
];

/// The default per-payload-byte pitch track: the Giant Steps head, cycled
/// to cover `len` bytes.
pub fn default_byte_pitch(len: usize) -> Vec<f64> {
    (0..len).map(|i| GIANT_STEPS[i % GIANT_STEPS.len()]).collect()
}

/// Accent (loudness) pattern aligned with the melody: a gentle musical pulse.
/// The decoder is amplitude-invariant (survives 0.1× scaling), so dynamics
/// are aesthetically free.
pub const ACCENT: [f64; 4] = [1.0, 0.85, 0.92, 0.85];

/// Accent multiplier for the symbol at `pos`.
#[inline]
pub fn melody_accent(pos: usize) -> f64 {
    ACCENT[pos % ACCENT.len()]
}

/// Vibrato rate (Hz) — sung, not synthesized.
pub const VIBRATO_RATE: f64 = 5.0;

/// Vibrato depth (fractional F0 deviation). ±0.6% ≈ ±13 Hz at the highest
/// harmonic — negligible against the 300 Hz detection Gaussian.
pub const VIBRATO_DEPTH: f64 = 0.006;

/// Maximum harmonic frequency (Hz) for synthesis and detection.
/// Keeps all harmonics below Nyquist in 8 kHz mode; HF rolloff makes
/// anything above ~3 kHz negligible anyway.
pub const HARMONIC_CEILING: f64 = 3800.0;

/// Max harmonics in the synthesis model: enough to reach HARMONIC_CEILING
/// from the lowest melody note (D3 = 146.83 Hz → 25 harmonics).
pub const NUM_HARMONICS: usize = 26;

/// Samples per symbol (50 ms voiced duration).
#[cfg(not(feature = "8khz"))]
pub const SAMPLES_PER_SYMBOL: usize = 2400; // 50 ms at 48 kHz
#[cfg(feature = "8khz")]
pub const SAMPLES_PER_SYMBOL: usize = 400; // 50 ms at 8 kHz

/// Guard silence after each symbol (10 ms).
#[cfg(not(feature = "8khz"))]
pub const GUARD_SAMPLES: usize = 480; // 10 ms at 48 kHz
#[cfg(feature = "8khz")]
pub const GUARD_SAMPLES: usize = 80; // 10 ms at 8 kHz

/// Total samples per symbol slot (symbol + guard).
/// 2880 at 48 kHz, 480 at 8 kHz.
pub const SYMBOL_TOTAL_SAMPLES: usize = SAMPLES_PER_SYMBOL + GUARD_SAMPLES;

/// Number of vowel shapes in the alphabet.
/// Reduced to 4 for VoIP compatibility.
pub const NUM_VOWELS: usize = 4;

/// Total symbol alphabet size. Symbols are vowels — pitch carries no data.
pub const NUM_SYMBOLS: usize = NUM_VOWELS;

/// Bits per symbol (log2(4) = 2).
pub const BITS_PER_SYMBOL: usize = 2;

/// Preamble length in symbols.
pub const PREAMBLE_LEN: usize = 4;

/// Calibration pilot: known symbols transmitted after the start preamble.
/// Each vowel appears twice with varied transitions. The receiver measures
/// each vowel's F2 centroid through the live channel and classifies data
/// against measured centers — absorbing bandpass tilt, codec coloration,
/// and AGC settle.
pub const PILOT: [usize; 8] = [0, 1, 2, 3, 2, 3, 0, 1];

/// Pilot length in symbols.
pub const PILOT_LEN: usize = PILOT.len();

/// FFT size for spectral analysis (same as symbol length).
pub const FFT_SIZE: usize = SAMPLES_PER_SYMBOL;

/// Hz per FFT bin = SAMPLE_RATE / FFT_SIZE.
pub const HZ_PER_BIN: f64 = SAMPLE_RATE / FFT_SIZE as f64;

/// Formant bandwidth for F1 (Hz) — Gaussian sigma.
/// (Unused in F2-only mode, kept for compatibility)
pub const BW1: f64 = 80.0;

/// Formant bandwidth for F2 (Hz) — Gaussian sigma.
/// Very wide bandwidth (300 Hz) helps with F0-independent detection
/// by spreading energy across multiple harmonics.
pub const BW2: f64 = 300.0;

/// F1 band lower bound (Hz).
/// (Unused in F2-only mode — VoIP crushes this band)
pub const F1_LO: f64 = 300.0;

/// F1 band upper bound (Hz).
/// (Unused in F2-only mode)
pub const F1_HI: f64 = 850.0;

/// F2 band lower bound (Hz) — above VoIP processing.
pub const F2_LO: f64 = 1000.0;

/// F2 band upper bound (Hz).
pub const F2_HI: f64 = 2500.0;

/// Candidate F0 values for F0 detection in the decoder (the melody notes).
/// Only used by the secondary F1+F2 detection path; the primary F2-centroid
/// path is F0-agnostic.
pub const F0_CANDIDATES: [f64; 3] = [146.83, 207.65, 277.18];

/// Encoded data offset (1 byte length + 2 bytes ECC for length).
pub const ENCODED_DATA_OFFSET: usize = 3;

/// Maximum variable-length payload size.
pub const MAX_LENGTH_VARIABLE: usize = 140;

/// Maximum total encoded data size.
pub const MAX_DATA_SIZE: usize = 256;

/// Compute ECC byte count for a given payload length.
///
/// Matches C++: `len < 4 ? 2 : max(4, 2*(len/5))`
pub fn ecc_bytes_for_length(len: usize) -> usize {
    if len < 4 {
        2
    } else {
        4usize.max(2 * (len / 5))
    }
}

/// ECC policy. Sender and receiver must agree out-of-band (no wire signaling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EccProfile {
    /// Adaptive ~40% parity (`ecc_bytes_for_length`) — general messaging.
    #[default]
    Standard,
    /// ~100% parity, capped at the RS(255) codeword limit — for short
    /// zero-error-tolerance payloads (API keys). Corrects up to 50% byte
    /// corruption in a single codeword; strictly stronger than interleaving
    /// at these payload sizes.
    Paranoid,
}

impl EccProfile {
    pub fn ecc_bytes(self, len: usize) -> usize {
        match self {
            Self::Standard => ecc_bytes_for_length(len),
            Self::Paranoid => len.clamp(4, 255 - len),
        }
    }
}
