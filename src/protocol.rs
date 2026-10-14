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

/// The Lick, in E minor: E–F#–G–A–F#–D–E. This relative-minor
/// voicing fits the default Giant Steps head's opening Gmaj7 harmony:
/// G, D, and F# are chord tones; A and E supply the ninth and thirteenth.
/// Eighth notes occupy two symbol slots (120 ms), the second F# a quarter,
/// and the final E is tied across the cadence. One phrase exactly fills
/// sync + pilot + length header.
/// Rhythm: https://pianowithjonny.com/piano-lessons/the-lick-jazz-meme-the-complete-guide/
pub const FRAMING_LICK: [f64; 24] = [
    164.81, 164.81, // E3
    185.00, 185.00, // F#3
    196.00, 196.00, // G3
    220.00, 220.00, // A3
    185.00, 185.00, 185.00, 185.00, // F#3, quarter
    146.83, 146.83, // D3
    164.81, 164.81, 164.81, 164.81, 164.81,
    164.81, 164.81, 164.81, 164.81, 164.81, // E3, tied resolution
];

/// The Lick over the E♭maj7 bar of the Coltrane cycle: C minor, the
/// relative minor of E♭, voiced up a minor sixth from home — down a major
/// third would put its B♭ below the B2 compass floor.
pub const FRAMING_LICK_EB: [f64; 24] = [
    261.63, 261.63, // C4
    293.66, 293.66, // D4
    311.13, 311.13, // Eb4
    349.23, 349.23, // F4
    293.66, 293.66, 293.66, 293.66, // D4, quarter
    233.08, 233.08, // Bb3
    261.63, 261.63, 261.63, 261.63, 261.63,
    261.63, 261.63, 261.63, 261.63, 261.63, // C4, tied resolution
];

/// The Lick over the Bmaj7 bar of the Coltrane cycle: G# minor, the
/// relative minor of B, voiced up a major third from home.
pub const FRAMING_LICK_B: [f64; 24] = [
    207.65, 207.65, // G#3
    233.08, 233.08, // A#3
    246.94, 246.94, // B3
    277.18, 277.18, // C#4
    233.08, 233.08, 233.08, 233.08, // A#3, quarter
    185.00, 185.00, // F#3
    207.65, 207.65, 207.65, 207.65, 207.65,
    207.65, 207.65, 207.65, 207.65, 207.65, // G#3, tied resolution
];

/// One Lick per key center, in the head's order of the Coltrane cycle
/// (G → E♭ → B): even a jingle's outro plays the changes.
pub const FRAMING_LICK_CYCLE: [&[f64; 24]; 3] =
    [&FRAMING_LICK, &FRAMING_LICK_EB, &FRAMING_LICK_B];

/// Frame-relative melody for the intro region (sync + pilot + header) and
/// for short jingle outros. The intro is exactly one phrase, so sync,
/// pilot, and header always keep the home E-minor voicing and the pilot's
/// calibration reference never changes. Later phrases (short outros only)
/// transpose The Lick through the Coltrane cycle (G → E♭ → B) — the
/// 24-slot rhythm stays identical per phrase, which is what the
/// short-echo clock ladder correlates on; only the key moves. Start a
/// phrase only when all seven notes fit; otherwise hold home E instead of
/// cutting the next transposition off mid-lick.
pub fn framing_lick_f0(pos: usize, total_symbols: usize) -> f64 {
    framing_lick_cycle_f0(pos, total_symbols, 0)
}

/// The shared cycle engine. `first_phrase` picks where in the Coltrane
/// cycle the first phrase lands: the intro passes 0 (home Em over G); a
/// short jingle outro passes 1 (Cm over E♭), so intro → outro performs a
/// real Giant Steps modulation even when no phrase past the first ever
/// fits — Standard short outros are 36 slots, exactly one phrase plus the
/// held-E tail, and would otherwise never leave home.
fn framing_lick_cycle_f0(pos: usize, total_symbols: usize, first_phrase: usize) -> f64 {
    let phrase = pos / FRAMING_LICK.len();
    let resolution_start = 14;
    if phrase * FRAMING_LICK.len() + resolution_start < total_symbols {
        FRAMING_LICK_CYCLE[(phrase + first_phrase) % FRAMING_LICK_CYCLE.len()]
            [pos % FRAMING_LICK.len()]
    } else {
        FRAMING_LICK[0]
    }
}

/// Outros shorter than this many slots repeat the framing Lick instead of
/// starting the solo. Measured, minimal bound: at 8 kHz a 4-byte frame's
/// varied solo slots defeat every short-echo clock fallback rung
/// (8→16/4→12/4→8 correlations) under a 65 ms echo at a 0.98 clock
/// (`short_echo_clock_estimates_preserve_gates_and_recover_frames`) — the
/// tiniest frames leave too few repeated slots per half once the solo
/// varies them, while the periodic Lick passes. Larger frames (outros of
/// 11+ bytes) pass the full echo matrix at both sample rates with the
/// solo and keep it; a wider 76-slot threshold was tried and broke the
/// tuned 48 kHz next-melody echo tails. Musically: the band doesn't hand
/// out a solo on a jingle.
pub const SOLO_MIN_OUTRO_SLOTS: usize = 44;

/// Checksum/parity/outro melody: Coltrane takes a solo — on frames long
/// enough to pay for one (see `SOLO_MIN_OUTRO_SLOTS`). One eighth note
/// per two symbol slots (120 ms ≈ 250 bpm — his tempo), wrapping modulo
/// the table for outros longer than two choruses. Pitch carries no data,
/// so the cut lands wherever the parity budget ends.
pub fn outro_f0(pos: usize, total_symbols: usize) -> f64 {
    if total_symbols < SOLO_MIN_OUTRO_SLOTS {
        // Jingle outros start the cycle at E♭ (phrase index 1): the intro
        // already played home, so the frame itself moves G → E♭.
        return framing_lick_cycle_f0(pos, total_symbols, 1);
    }
    if pos + PREAMBLE_LEN >= total_symbols {
        // The end preamble is a sync marker, not part of the solo: the
        // band cuts Coltrane off and holds the home E. Keeping this pitch
        // fixed also preserves end-detection margin on hostile channels.
        FRAMING_LICK[0]
    } else {
        GIANT_STEPS_SOLO[pos % GIANT_STEPS_SOLO.len()]
    }
}

/// The pilot is part of The Lick in current frames. Legacy recordings
/// retain the original arpeggio, including its calibration reference.
pub fn pilot_f0(index: usize, profile: EccProfile) -> f64 {
    let pos = PREAMBLE_LEN + index;
    if profile == EccProfile::Legacy {
        melody_f0(pos)
    } else {
        FRAMING_LICK[pos]
    }
}

/// The default payload melody: the 16-bar head of John Coltrane's
/// "Giant Steps" (1960), one note per payload byte (= one quarter note at
/// ~250 bpm), transposed down an octave plus a major third into the
/// modem's B2..F#4 compass — the true contour fits with no octave folds.
///
/// Pitch carries no data, so `encode` is free to sing this over the
/// payload; sync (preamble + pilot), header, checksum, and RS parity sing
/// The Lick (`FRAMING_LICK`) in the relative minor of the opening Gmaj7.
/// The transfer profile uses the armored arpeggio (`MELODY`) for its
/// payload. Payloads longer than one chorus loop the form.
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

/// Coltrane's solo, first two choruses, as the checksum/parity outro
/// melody: quantized from a MIDI transcription of the 1959 recording to
/// its own eighth-note grid (103 ms median IOI), voiced key-true with the
/// head arrangement (-4 semitones ≡ the head's -16 mod 12) and
/// octave-folded into the B2..F#4 compass (225/256 eighths needed no
/// fold). Each eighth note holds two 60 ms symbol slots, so the outro
/// solos at ~250 bpm. Rests sustain the previous note: the modem cannot
/// breathe. Regenerate with the quantization above if the transcription
/// or compass changes.
pub const GIANT_STEPS_SOLO: [f64; 512] = [
    // bar 1: A3 C4 E4 G3 F#4 F#4 F#4 E4
    220.00, 220.00, 261.63, 261.63, 329.63, 329.63, 196.00, 196.00,
    369.99, 369.99, 369.99, 369.99, 369.99, 369.99, 329.63, 329.63,
    // bar 2: D4 B3 G3 G3 Bb3 C4 D4 F4
    293.66, 293.66, 246.94, 246.94, 196.00, 196.00, 196.00, 196.00,
    233.08, 233.08, 261.63, 261.63, 293.66, 293.66, 349.23, 349.23,
    // bar 3: Eb4 Bb3 G3 G3 G3 E3 Eb3 C#3
    311.13, 311.13, 233.08, 233.08, 196.00, 196.00, 196.00, 196.00,
    196.00, 196.00, 164.81, 164.81, 155.56, 155.56, 138.59, 138.59,
    // bar 4: B2 C#3 Eb3 E3 F#3 G#3 Bb3 C#4
    123.47, 123.47, 138.59, 138.59, 155.56, 155.56, 164.81, 164.81,
    185.00, 185.00, 207.65, 207.65, 233.08, 233.08, 277.18, 277.18,
    // bar 5: C4 G#3 E3 Eb3 D3 B3 Bb3 G#3
    261.63, 261.63, 207.65, 207.65, 164.81, 164.81, 155.56, 155.56,
    146.83, 146.83, 246.94, 246.94, 233.08, 233.08, 207.65, 207.65,
    // bar 6: G3 Bb3 Eb4 G3 Bb3 C#4 E4 G#3
    196.00, 196.00, 233.08, 233.08, 311.13, 311.13, 196.00, 196.00,
    233.08, 233.08, 277.18, 277.18, 329.63, 329.63, 207.65, 207.65,
    // bar 7: B3 C#4 Eb4 F#4 C4 E4 F#4 A3
    246.94, 246.94, 277.18, 277.18, 311.13, 311.13, 369.99, 369.99,
    261.63, 261.63, 329.63, 329.63, 369.99, 369.99, 220.00, 220.00,
    // bar 8: F#4 E4 E4 D4 D4 D4 D4 F#4
    369.99, 369.99, 329.63, 329.63, 329.63, 329.63, 293.66, 293.66,
    293.66, 293.66, 293.66, 293.66, 293.66, 293.66, 369.99, 369.99,
    // bar 9: F4 E4 E4 Eb4 B3 Bb3 G#3 F#3
    349.23, 349.23, 329.63, 329.63, 329.63, 329.63, 311.13, 311.13,
    246.94, 246.94, 233.08, 233.08, 207.65, 207.65, 185.00, 185.00,
    // bar 10: F#3 E4 Bb3 C#4 B3 Bb3 C#4 C4
    185.00, 185.00, 329.63, 329.63, 233.08, 233.08, 277.18, 277.18,
    246.94, 246.94, 233.08, 233.08, 277.18, 277.18, 261.63, 261.63,
    // bar 11: G#3 F3 F3 Eb3 C3 D3 B3 Bb3
    207.65, 207.65, 174.61, 174.61, 174.61, 174.61, 155.56, 155.56,
    130.81, 130.81, 146.83, 146.83, 246.94, 246.94, 233.08, 233.08,
    // bar 12: G#3 G3 F3 Eb3 F3 G3 Bb3 B3
    207.65, 207.65, 196.00, 196.00, 174.61, 174.61, 155.56, 155.56,
    174.61, 174.61, 196.00, 196.00, 233.08, 233.08, 246.94, 246.94,
    // bar 13: A3 G3 F#3 F#3 D3 E3 F#3 G3
    220.00, 220.00, 196.00, 196.00, 185.00, 185.00, 185.00, 185.00,
    146.83, 146.83, 164.81, 164.81, 185.00, 185.00, 196.00, 196.00,
    // bar 14: A3 B3 B3 F#4 F4 E4 Eb4 C#4
    220.00, 220.00, 246.94, 246.94, 246.94, 246.94, 369.99, 369.99,
    349.23, 349.23, 329.63, 329.63, 311.13, 311.13, 277.18, 277.18,
    // bar 15: C#4 C#4 C#4 Eb4 D4 C#4 Bb3 B3
    277.18, 277.18, 277.18, 277.18, 277.18, 277.18, 311.13, 311.13,
    293.66, 293.66, 277.18, 277.18, 233.08, 233.08, 246.94, 246.94,
    // bar 16: C#4 C#4 F#4 B3 B3 F#4 F#4 F#4
    277.18, 277.18, 277.18, 277.18, 369.99, 369.99, 246.94, 246.94,
    246.94, 246.94, 369.99, 369.99, 369.99, 369.99, 369.99, 369.99,
    // bar 17: F#4 F#4 G3 F#4 F#4 F#4 D4 F#4
    369.99, 369.99, 369.99, 369.99, 196.00, 196.00, 369.99, 369.99,
    369.99, 369.99, 369.99, 369.99, 293.66, 293.66, 369.99, 369.99,
    // bar 18: Bb3 Bb3 Bb3 Bb3 Bb3 Bb3 Bb3 G#3
    233.08, 233.08, 233.08, 233.08, 233.08, 233.08, 233.08, 233.08,
    233.08, 233.08, 233.08, 233.08, 233.08, 233.08, 207.65, 207.65,
    // bar 19: G#3 Eb4 G3 F#4 F4 E4 G#3 F#4
    207.65, 207.65, 311.13, 311.13, 196.00, 196.00, 369.99, 369.99,
    349.23, 349.23, 329.63, 329.63, 207.65, 207.65, 369.99, 369.99,
    // bar 20: Eb4 C#4 B3 Eb4 B3 B3 B3 F#4
    311.13, 311.13, 277.18, 277.18, 246.94, 246.94, 311.13, 311.13,
    246.94, 246.94, 246.94, 246.94, 246.94, 246.94, 369.99, 369.99,
    // bar 21: Bb3 Bb3 F#4 F#4 D4 Bb3 Bb3 Bb3
    233.08, 233.08, 233.08, 233.08, 369.99, 369.99, 369.99, 369.99,
    293.66, 293.66, 233.08, 233.08, 233.08, 233.08, 233.08, 233.08,
    // bar 22: Bb3 Eb4 G3 Bb3 C#4 E4 G#3 B3
    233.08, 233.08, 311.13, 311.13, 196.00, 196.00, 233.08, 233.08,
    277.18, 277.18, 329.63, 329.63, 207.65, 207.65, 246.94, 246.94,
    // bar 23: C#4 F#4 C4 C4 E4 F#4 D4 D4
    277.18, 277.18, 369.99, 369.99, 261.63, 261.63, 261.63, 261.63,
    329.63, 329.63, 369.99, 369.99, 293.66, 293.66, 293.66, 293.66,
    // bar 24: D4 D4 B3 A3 G#3 G3 G#3 G#3
    293.66, 293.66, 293.66, 293.66, 246.94, 246.94, 220.00, 220.00,
    207.65, 207.65, 196.00, 196.00, 207.65, 207.65, 207.65, 207.65,
    // bar 25: B3 B3 G#3 Eb4 Eb4 G#3 F#3 F#3
    246.94, 246.94, 246.94, 246.94, 207.65, 207.65, 311.13, 311.13,
    311.13, 311.13, 207.65, 207.65, 185.00, 185.00, 185.00, 185.00,
    // bar 26: F#3 F#3 E4 B3 B3 B3 B3 F3
    185.00, 185.00, 185.00, 185.00, 329.63, 329.63, 246.94, 246.94,
    246.94, 246.94, 246.94, 246.94, 246.94, 246.94, 174.61, 174.61,
    // bar 27: F3 Eb4 G3 F4 D4 C4 Bb3 C4
    174.61, 174.61, 311.13, 311.13, 196.00, 196.00, 349.23, 349.23,
    293.66, 293.66, 261.63, 261.63, 233.08, 233.08, 261.63, 261.63,
    // bar 28: D4 F4 Eb4 F4 G3 F#4 F#4 G3
    293.66, 293.66, 349.23, 349.23, 311.13, 311.13, 349.23, 349.23,
    196.00, 196.00, 369.99, 369.99, 369.99, 369.99, 196.00, 196.00,
    // bar 29: G3 G3 G3 G3 G3 G3 F#4 D4
    196.00, 196.00, 196.00, 196.00, 196.00, 196.00, 196.00, 196.00,
    196.00, 196.00, 196.00, 196.00, 369.99, 369.99, 293.66, 293.66,
    // bar 30: E4 F#4 D4 B3 A3 G3 G#3 Bb3
    329.63, 329.63, 369.99, 369.99, 293.66, 293.66, 246.94, 246.94,
    220.00, 220.00, 196.00, 196.00, 207.65, 207.65, 233.08, 233.08,
    // bar 31: B3 G#3 Eb4 Bb3 Bb3 G#3 F#3 F#3
    246.94, 246.94, 207.65, 207.65, 311.13, 311.13, 233.08, 233.08,
    233.08, 233.08, 207.65, 207.65, 185.00, 185.00, 185.00, 185.00,
    // bar 32: Eb4 Eb4 Eb4 Eb4 Eb4 Eb4 Eb4 C4
    311.13, 311.13, 311.13, 311.13, 311.13, 311.13, 311.13, 311.13,
    311.13, 311.13, 311.13, 311.13, 311.13, 311.13, 261.63, 261.63,
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
/// harmonic — small compared with the formant bandwidth. A ±0.4% depth
/// and a steeper HF rolloff were tried against a perceived upper-harmonic
/// chirp and reverted: both passed the channel matrix but made no audible
/// difference, so the original voicing stands.
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
/// A 180 Hz sigma keeps adjacent vowels distinct after spectral tilt and
/// denoising across the B2–F#4 register. Pitch, vibrato, and timing are
/// independent of this timbre setting.
pub const BW2: f64 = 180.0;

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

/// Compute adaptive ECC for a protected block length (excluding parity).
///
/// Matches C++: `len < 4 ? 2 : max(4, 2*(len/5))`
/// For message payload lengths, use `EccProfile::ecc_bytes` so Standard's
/// checksum is included in the protected block size.
pub fn ecc_bytes_for_length(len: usize) -> usize {
    if len < 4 {
        2
    } else {
        4usize.max(2 * (len / 5))
    }
}

/// Frame protection policy. Sender and receiver must agree out-of-band.
/// Standard frames include a CRC; pre-CRC recordings require `Legacy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EccProfile {
    /// CRC-32 plus adaptive ~40% parity — the default message format.
    #[default]
    Standard,
    /// Original message format without a checksum. For older recordings
    /// and peers only; RS decoding can miscorrect beyond its error budget.
    Legacy,
    /// ~100% parity, capped at the RS(255) codeword limit. This raw profile
    /// adds no checksum itself; the transfer layer supplies CRC validation.
    /// Up to 50% erasures or 25% unknown byte errors per codeword at 1:1 parity.
    Paranoid,
}

impl EccProfile {
    /// Integrity bytes carried inside the RS-protected payload block.
    pub fn checksum_bytes(self) -> usize {
        if self == Self::Standard {
            4
        } else {
            0
        }
    }

    /// Payload and integrity bytes, before Reed–Solomon parity.
    pub fn protected_bytes(self, len: usize) -> usize {
        len + self.checksum_bytes()
    }

    pub fn ecc_bytes(self, len: usize) -> usize {
        match self {
            Self::Standard => ecc_bytes_for_length(self.protected_bytes(len)),
            Self::Legacy => ecc_bytes_for_length(len),
            Self::Paranoid => len.clamp(4, 255 - len),
        }
    }

    /// Header + payload + checksum + parity, excluding acoustic sync.
    pub fn encoded_bytes(self, len: usize) -> usize {
        ENCODED_DATA_OFFSET + self.protected_bytes(len) + self.ecc_bytes(len)
    }

    pub(crate) fn protect_payload(self, payload: &[u8]) -> Vec<u8> {
        let mut protected = payload.to_vec();
        if self == Self::Standard {
            protected.extend_from_slice(&crate::checksum::crc32(payload).to_le_bytes());
        }
        protected
    }

    pub(crate) fn verify_payload(self, mut protected: Vec<u8>) -> Option<Vec<u8>> {
        if self == Self::Standard {
            let len = protected.len().checked_sub(self.checksum_bytes())?;
            if len == 0 {
                return None;
            }
            let checksum = u32::from_le_bytes(protected[len..].try_into().ok()?);
            if crate::checksum::crc32(&protected[..len]) != checksum {
                return None;
            }
            protected.truncate(len);
        }
        Some(protected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_lick_has_the_original_intervals_and_rhythm_in_e_minor() {
        let notes = [164.81, 185.00, 196.00, 220.00, 185.00, 146.83, 164.81];
        let durations = [2, 2, 2, 2, 4, 2, 10];
        let expected: Vec<f64> = notes.iter().zip(durations)
            .flat_map(|(&hz, slots)| std::iter::repeat(hz).take(slots))
            .collect();
        assert_eq!(FRAMING_LICK.as_slice(), expected);
        assert_eq!(FRAMING_LICK.len(), PREAMBLE_LEN + PILOT_LEN + 4 * ENCODED_DATA_OFFSET);
        assert_eq!(GIANT_STEPS[0], 293.66); // D4, fifth of the opening Gmaj7.
        // Gmaj7 chord tones and its 9/13 extensions: no F-natural or C-natural.
        for hz in FRAMING_LICK {
            let midi = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i32;
            assert!([7, 11, 2, 6, 9, 4].contains(&(midi % 12)));
        }
    }

    #[test]
    fn solo_outro_stays_in_compass_at_eighth_note_granularity() {
        // Every slot is a chromatic-register note the melody tests cover,
        // rounded to the same cent-level table as the other constants.
        for hz in GIANT_STEPS_SOLO {
            assert!((123.47..=369.99).contains(&hz), "B2..F#4 compass: {hz}");
            let midi = 69.0 + 12.0 * (hz / 440.0).log2();
            assert!((midi - midi.round()).abs() < 0.001, "off-grid pitch {hz}");
        }
        // One eighth note = two symbol slots: pitches change only on even
        // slot boundaries.
        for pair in GIANT_STEPS_SOLO.chunks(2) {
            assert_eq!(pair[0], pair[1]);
        }
        assert_eq!(GIANT_STEPS_SOLO.len() % FRAMING_LICK.len(), 8); // 512 slots, 2 choruses
    }

    #[test]
    fn solo_outro_covers_every_profile_length_and_pins_the_end_marker() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                let slots = 4 * (profile.checksum_bytes() + profile.ecc_bytes(len)) + PREAMBLE_LEN;
                for pos in 0..slots {
                    let expected = if slots < SOLO_MIN_OUTRO_SLOTS {
                        // No solo on a jingle; the outro starts the cycle
                        // at E♭ so intro → outro plays the first change.
                        framing_lick_cycle_f0(pos, slots, 1)
                    } else if pos + PREAMBLE_LEN >= slots {
                        FRAMING_LICK[0] // end sync marker holds home E
                    } else {
                        GIANT_STEPS_SOLO[pos % GIANT_STEPS_SOLO.len()]
                    };
                    assert_eq!(outro_f0(pos, slots), expected);
                }
            }
        }
        // Wraps modulo the two-chorus table for the longest outros.
        let long = 3 * GIANT_STEPS_SOLO.len();
        assert_eq!(outro_f0(GIANT_STEPS_SOLO.len() + 3, long), GIANT_STEPS_SOLO[3]);
    }

    #[test]
    fn framing_lick_cycle_transposes_exactly_through_the_changes_in_compass() {
        // Same rhythm; exact +8/+4 semitone transpositions of every slot,
        // on the same cent-rounded table as the other note constants.
        for (shift, lick) in [(8, FRAMING_LICK_EB), (4, FRAMING_LICK_B)] {
            for (home, hz) in FRAMING_LICK.iter().zip(lick) {
                let semis = 12.0 * (hz / home).log2();
                assert!((semis - shift as f64).abs() < 0.01, "{home} -> {hz}");
                assert!((123.47..=369.99).contains(&hz), "B2..F#4 compass: {hz}");
            }
        }
        // Chord tones plus the 9 and the 7/13 of each phrase's bar:
        // C minor over E♭maj7, G# minor over Bmaj7.
        for (lick, pcs) in [
            (FRAMING_LICK_EB, [3, 7, 10, 2, 5, 0]),
            (FRAMING_LICK_B, [11, 3, 6, 10, 1, 8]),
        ] {
            for hz in lick {
                let midi = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i32;
                assert!(pcs.contains(&(midi % 12)), "{hz}");
            }
        }
        // Short outros cycle phrases G → E♭ → B; incomplete tails hold home E.
        let slots = 3 * FRAMING_LICK.len() + 10;
        for (phrase, lick) in FRAMING_LICK_CYCLE.iter().enumerate() {
            for offset in 0..FRAMING_LICK.len() {
                assert_eq!(framing_lick_f0(phrase * FRAMING_LICK.len() + offset, slots), lick[offset]);
            }
        }
        for offset in 0..10 {
            assert_eq!(framing_lick_f0(3 * FRAMING_LICK.len() + offset, slots), 164.81);
        }
        // Jingle outros start at E♭ (Cm) and still hold home E on tails:
        // a Standard short outro (36 slots) is one Cm phrase + held E.
        for offset in 0..FRAMING_LICK.len() {
            assert_eq!(framing_lick_cycle_f0(offset, 36, 1), FRAMING_LICK_EB[offset]);
        }
        for offset in 24..36 {
            assert_eq!(framing_lick_cycle_f0(offset, 36, 1), 164.81);
        }
        // A 40-slot Paranoid one-byte outro reaches G#m before the hold.
        for offset in 0..16 {
            assert_eq!(framing_lick_cycle_f0(24 + offset, 40, 1), FRAMING_LICK_B[offset]);
        }
    }

    #[test]
    fn framing_intro_region_always_keeps_the_home_voicing() {
        // The intro (sync + pilot + header) is exactly one Lick phrase; the
        // solo outro can never reach the pilot's calibration reference.
        for pos in 0..FRAMING_LICK.len() {
            assert_eq!(framing_lick_f0(pos, FRAMING_LICK.len()), FRAMING_LICK[pos]);
        }
    }

    #[test]
    fn pilot_pitch_matches_the_framing_profile() {
        for i in 0..PILOT_LEN {
            for profile in [EccProfile::Standard, EccProfile::Paranoid] {
                assert_eq!(pilot_f0(i, profile), FRAMING_LICK[PREAMBLE_LEN + i]);
            }
            assert_eq!(pilot_f0(i, EccProfile::Legacy), melody_f0(PREAMBLE_LEN + i));
        }
    }

    #[test]
    fn profile_geometry_covers_all_payload_lengths() {
        for len in 1..=MAX_LENGTH_VARIABLE {
            let payload: Vec<u8> = (0..len).map(|i| (i * 37 + len) as u8).collect();
            for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
                let protected = profile.protect_payload(&payload);
                assert_eq!(protected.len(), profile.protected_bytes(len));
                let codeword_len = protected.len() + profile.ecc_bytes(len);
                assert!(codeword_len <= 255);
                assert_eq!(profile.encoded_bytes(len), ENCODED_DATA_OFFSET + codeword_len);
                assert_eq!(profile.verify_payload(protected).as_deref(), Some(payload.as_slice()));
            }
            let added_bytes = EccProfile::Standard.encoded_bytes(len)
                - EccProfile::Legacy.encoded_bytes(len);
            assert!(matches!(added_bytes, 4 | 6));
        }
    }
}
