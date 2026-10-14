//! Streaming FEC codec: continuous RS-coded symbol stream with block interleaving.
//!
//! Eliminates per-frame preamble overhead and ACK round-trips. Data flows as a
//! continuous RS(15,k) coded stream over GF(2^4), where each GF(2^4) symbol maps
//! to exactly two 2-bit modem symbols (vowels). No bit straddling anywhere:
//! byte = 2 nibbles, nibble = 2 vowels.
//!
//! Pitch carries no data — the transmitter plays the melody (protocol::MELODY)
//! across the stream, and the receiver's vowel detection is F0-agnostic.
//! Default framing protects the length and payload with CRC-32 inside RS.
//! Both peers must use this framing; old checksum-free recordings require
//! an explicit [`StreamProfile::Legacy`] receiver (no automatic downgrade).
//!
//! ```text
//! Send: length + bytes + CRC -> nibbles -> RS encode -> interleave -> vowels -> audio
//! Recv: audio -> preamble -> classify -> pair -> deinterleave -> RS -> CRC check -> bytes
//! ```

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::rs4::ReedSolomon4;
use std::collections::BTreeSet;

/// Codeword length for GF(2^4): n = 2^4 - 1 = 15.
const CODEWORD_LEN: usize = 15;

/// Wire length is one byte. Retain only enough words for the largest frame.
const MAX_STREAM_PAYLOAD: usize = 255;

/// Failure-only CRC search: bound both complete candidates and queued states.
const MAX_STREAM_RETRY_TRIALS: usize = 512;
const MAX_STREAM_RETRY_FRONTIER: usize = 512;

/// A preamble plus two slots of timing lookahead; payload samples are replayed.
const STREAM_LISTEN_WINDOWS: usize = PREAMBLE_LEN + 2;
const STREAM_PHASE_STEP: usize = SYMBOL_TOTAL_SAMPLES / 4;
const STREAM_PHASE_REFINE_STEP: usize = GUARD_SAMPLES / 2;
type WindowMeasurement = (usize, f64, f64); // vowel, centroid, RMS

/// Track the guard's central half, moving the next boundary by at most
/// 2.5 ms per slot in 0.5 ms steps. No extra transmitted timing markers.
#[cfg(test)]
const STREAM_CLOCK_RADIUS: usize = crate::audio::CLOCK_RADIUS;

/// Stream start preamble: max-distance vowel pair (vowels 0 and 3).
const STREAM_PREAMBLE_START: [usize; PREAMBLE_LEN] = [0, 3, 0, 3];

/// Stream end preamble: inverted pattern. Transmitted as a musical cadence,
/// but the receiver does NOT rely on it — with 2-bit symbols, any 4-symbol
/// pattern aliases random data once per ~256 positions. End-of-stream is
/// detected by trailing silence instead, which data cannot imitate.
const STREAM_PREAMBLE_END: [usize; PREAMBLE_LEN] = [3, 0, 3, 0];

/// Trailing silence appended by the transmitter after the end preamble,
/// in symbol slots. Must exceed END_SILENCE_WINDOWS.
const TRAILING_SILENCE_SLOTS: usize = 5;

/// Consecutive near-silent, unvoiced windows that end the stream on the receiver.
/// Four windows on the 60 ms grid (~230–240 ms including guards): longer
/// than any tolerated mid-stream dropout, shorter than the trailing silence.
const END_SILENCE_WINDOWS: usize = 4;

/// A candidate silent window falls below this fraction of the running RMS
/// average (~ -10 dB), between the noise floor at 20 dB SNR (0.1×) and voiced
/// level (1×). Voiced periodicity vetoes silence for quiet musical phrases.
const SILENCE_RMS_RATIO: f64 = 0.3;

/// Confidence threshold for erasure marking: symbols below this are erasure
/// candidates (weakest-first, capped at the parity budget) when hard-decision
/// RS decoding fails.
const ERASURE_THRESHOLD: f64 = 1.0;

/// Wire framing for the streaming codec, independent of its RS parity setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamProfile {
    /// Length + payload + four little-endian IEEE CRC-32 bytes, all inside RS.
    /// The checksum covers the length byte and payload, not zero padding.
    /// Detects accidental corruption; it is not cryptographic authentication.
    #[default]
    Standard,
    /// Original length + payload framing, with no checksum. Select explicitly
    /// only for compatibility with older peers/recordings; RS can miscorrect.
    Legacy,
}

impl StreamProfile {
    fn checksum_bytes(self) -> usize {
        match self {
            Self::Standard => 4,
            Self::Legacy => 0,
        }
    }
}

/// Configuration for the streaming codec.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Parity symbols per RS codeword, 0..=14 (default: 4 → corrects 2 symbol errors).
    /// Data symbols = 15 - parity.
    pub rs_parity: u8,
    /// Block interleaver depth, 1..=255 (default: 2). 1 = no interleaving.
    pub interleave_depth: u8,
    /// Audio volume 0..=100 (default: 50).
    pub volume: u8,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            rs_parity: 4,
            interleave_depth: 2,
            volume: 50,
        }
    }
}

impl StreamConfig {
    fn data_per_codeword(&self) -> usize {
        assert!(self.rs_parity < CODEWORD_LEN as u8, "stream parity must be 0..=14");
        assert!(self.interleave_depth > 0, "stream interleave depth must be at least 1");
        CODEWORD_LEN - self.rs_parity as usize
    }
}

// --- Block Interleaver ---
//
// Writes D codewords row-by-row into a D×N matrix, reads column-by-column.
// A burst of B consecutive bad symbols gets spread across ceil(B/D) codewords,
// each seeing at most ceil(B/D) errors. Latency: D * N * symbol_time.

/// Block interleaver: buffers D codewords, emits them column-interleaved.
struct Interleaver {
    depth: usize,
    /// Accumulates D * CODEWORD_LEN symbols before emitting.
    buf: Vec<u8>,
    /// Interleaved output queue.
    out: Vec<u8>,
}

impl Interleaver {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Push a symbol. If a full block (D codewords) is accumulated, interleave it.
    fn push(&mut self, sym: u8) {
        self.buf.push(sym);
        let block_size = self.depth * CODEWORD_LEN;
        if self.buf.len() == block_size {
            // Read column-by-column from D×N matrix (rows = codewords, cols = symbol positions)
            for col in 0..CODEWORD_LEN {
                for row in 0..self.depth {
                    self.out.push(self.buf[row * CODEWORD_LEN + col]);
                }
            }
            self.buf.clear();
        }
    }

    /// Flush any partial block by zero-padding to a full block, then interleaving.
    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let block_size = self.depth * CODEWORD_LEN;
        while self.buf.len() < block_size {
            self.buf.push(0);
        }
        for col in 0..CODEWORD_LEN {
            for row in 0..self.depth {
                self.out.push(self.buf[row * CODEWORD_LEN + col]);
            }
        }
        self.buf.clear();
    }

    /// Pop the next interleaved symbol, if available.
    fn pop(&mut self) -> Option<u8> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0))
        }
    }
}

/// Block deinterleaver: inverse of interleaver. Accumulates D*N symbols,
/// then reads them back in row order (undoing the column interleaving).
struct Deinterleaver {
    depth: usize,
    buf: Vec<(u8, f64)>,
    out: Vec<(u8, f64)>,
}

impl Deinterleaver {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Push a (symbol, confidence) pair. If a full block is accumulated, deinterleave it.
    fn push(&mut self, sym: u8, conf: f64) {
        self.buf.push((sym, conf));
        let block_size = self.depth * CODEWORD_LEN;
        if self.buf.len() == block_size {
            // Write was column-by-column (D symbols per column, N columns).
            // To undo: read row-by-row from N×D layout (N cols, D per col).
            // Input order: col0_row0, col0_row1, ..., col0_rowD-1, col1_row0, ...
            // Output order: row0_col0, row0_col1, ..., row0_colN-1, row1_col0, ...
            for row in 0..self.depth {
                for col in 0..CODEWORD_LEN {
                    self.out.push(self.buf[col * self.depth + row]);
                }
            }
            self.buf.clear();
        }
    }

    fn pop(&mut self) -> Option<(u8, f64)> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0))
        }
    }

    /// Flush partial block by padding and deinterleaving.
    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let block_size = self.depth * CODEWORD_LEN;
        while self.buf.len() < block_size {
            self.buf.push((0, 0.0));
        }
        for row in 0..self.depth {
            for col in 0..CODEWORD_LEN {
                self.out.push(self.buf[col * self.depth + row]);
            }
        }
        self.buf.clear();
    }
}

// --- StreamTx (encoder) ---

/// Streaming transmitter: feeds bytes, emits continuous audio.
///
/// Usage: call `feed()` once with a payload (length and CRC added automatically),
/// call `finish()` when done, then call `emit()` repeatedly to get audio samples.
pub struct StreamTx {
    config: StreamConfig,
    profile: StreamProfile,
    rs: ReedSolomon4,
    interleaver: Interleaver,
    /// Buffer of 4-bit symbols (nibbles) awaiting RS encoding.
    nibble_buf: Vec<u8>,
    /// Modem (vowel) symbol queue ready for synthesis.
    vowel_queue: Vec<usize>,
    /// Melody position: counts every modem symbol synthesized (preamble included).
    melody_pos: usize,
    sample_pos: usize,
    current_symbol_audio: Vec<f64>,
    /// Trailing silence samples still to emit after the end preamble.
    silence_samples_left: usize,
    preamble_sent: bool,
    flushing: bool,
    end_queued: bool,
    done: bool,
    send_volume: f64,
}

impl StreamTx {
    /// Create a transmitter using the CRC-protected streaming wire format.
    pub fn new(config: StreamConfig) -> Self {
        Self::with_profile(config, StreamProfile::Standard)
    }

    /// Select wire framing explicitly; Legacy is for older checksum-free peers.
    /// Panics if parity is above 14 or interleaver depth is zero.
    pub fn with_profile(config: StreamConfig, profile: StreamProfile) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon4::new(data_len, config.rs_parity as usize);
        let interleaver = Interleaver::new(config.interleave_depth as usize);
        let send_volume = config.volume as f64 / 100.0;
        Self {
            config,
            profile,
            rs,
            interleaver,
            nibble_buf: Vec::new(),
            vowel_queue: Vec::new(),
            melody_pos: 0,
            sample_pos: 0,
            current_symbol_audio: Vec::new(),
            silence_samples_left: 0,
            preamble_sent: false,
            flushing: false,
            end_queued: false,
            done: false,
            send_volume,
        }
    }

    /// Queue raw bytes for transmission (caller handles framing).
    /// Standard expects `[length, payload..., CRC32(length + payload)...]`,
    /// with four little-endian checksum bytes. Legacy omits the checksum.
    /// Raw chunks may split that frame anywhere; no prefix/checksum is added here.
    pub fn feed_raw(&mut self, bytes: &[u8]) {
        // Each byte is exactly two GF(2^4) symbols (nibbles), MSB first.
        for &b in bytes {
            self.nibble_buf.push(b >> 4);
            self.nibble_buf.push(b & 0x0F);
        }
    }

    /// Queue one payload with an automatic length prefix and profile checksum.
    /// The length byte always describes caller-visible payload bytes (max 255).
    pub fn feed(&mut self, payload: &[u8]) {
        assert!(payload.len() <= MAX_STREAM_PAYLOAD, "stream payload max 255 bytes");
        let mut frame = Vec::with_capacity(1 + payload.len() + self.profile.checksum_bytes());
        frame.push(payload.len() as u8);
        frame.extend_from_slice(payload);
        if self.profile == StreamProfile::Standard {
            frame.extend_from_slice(&crate::checksum::crc32(&frame).to_le_bytes());
        }
        self.feed_raw(&frame);
    }

    /// Signal that no more data will be fed. Flushes remaining nibbles with zero-padding.
    pub fn finish(&mut self) {
        self.flushing = true;
    }

    /// Fill the output buffer with audio samples. Returns number of samples written.
    pub fn emit(&mut self, out: &mut [f32]) -> usize {
        if self.done {
            return 0;
        }

        let mut written = 0;

        while written < out.len() {
            // Ensure start preamble
            if !self.preamble_sent {
                for &sym in &STREAM_PREAMBLE_START {
                    self.vowel_queue.push(sym);
                }
                self.preamble_sent = true;
            }

            // Encode nibbles into RS codewords
            self.encode_codewords();

            // Handle flushing
            if self.flushing && !self.end_queued {
                self.flush_final();
            }

            // If we have no symbols and we're done, stop
            if self.vowel_queue.is_empty()
                && self.current_symbol_audio.is_empty()
                && self.silence_samples_left == 0
            {
                if self.end_queued {
                    self.done = true;
                }
                break;
            }

            // Synthesize from symbol queue
            written += self.synthesize_into(&mut out[written..]);
        }

        written
    }

    /// Returns true when all queued data has been fully emitted as audio.
    pub fn is_idle(&self) -> bool {
        self.done
    }

    /// Drain interleaver output into the vowel queue, two vowels per GF(2^4) symbol.
    fn drain_interleaver(&mut self) {
        while let Some(sym) = self.interleaver.pop() {
            self.vowel_queue.push(((sym >> 2) & 0x3) as usize);
            self.vowel_queue.push((sym & 0x3) as usize);
        }
    }

    fn encode_codewords(&mut self) {
        let k = self.config.data_per_codeword();
        while self.nibble_buf.len() >= k {
            let data: Vec<u8> = self.nibble_buf.drain(..k).collect();
            let encoded = self.rs.encode(&data);
            for &sym in &encoded {
                self.interleaver.push(sym);
            }
        }
        self.drain_interleaver();
    }

    fn flush_final(&mut self) {
        let k = self.config.data_per_codeword();

        // Pad remaining nibbles to fill a codeword
        if !self.nibble_buf.is_empty() {
            while self.nibble_buf.len() < k {
                self.nibble_buf.push(0);
            }
            let data: Vec<u8> = self.nibble_buf.drain(..k).collect();
            let encoded = self.rs.encode(&data);
            for &sym in &encoded {
                self.interleaver.push(sym);
            }
        }

        // Flush interleaver (pads partial block)
        self.interleaver.flush();
        self.drain_interleaver();

        // End preamble, then trailing silence so the receiver's
        // silence-based end detection has something to detect.
        for &sym in &STREAM_PREAMBLE_END {
            self.vowel_queue.push(sym);
        }
        self.silence_samples_left = TRAILING_SILENCE_SLOTS * SYMBOL_TOTAL_SAMPLES;

        self.end_queued = true;
    }

    fn synthesize_into(&mut self, out: &mut [f32]) -> usize {
        let mut written = 0;

        while written < out.len() {
            if !self.current_symbol_audio.is_empty() {
                let remaining = self.current_symbol_audio.len() - self.sample_pos;
                let to_write = remaining.min(out.len() - written);
                for i in 0..to_write {
                    out[written + i] = self.current_symbol_audio[self.sample_pos + i] as f32;
                }
                written += to_write;
                self.sample_pos += to_write;
                if self.sample_pos >= self.current_symbol_audio.len() {
                    self.current_symbol_audio.clear();
                    self.sample_pos = 0;
                }
                continue;
            }

            if self.vowel_queue.is_empty() {
                // Trailing silence after the last symbol
                if self.silence_samples_left > 0 {
                    let to_write = self.silence_samples_left.min(out.len() - written);
                    for i in 0..to_write {
                        out[written + i] = 0.0;
                    }
                    written += to_write;
                    self.silence_samples_left -= to_write;
                    continue;
                }
                break;
            }

            // Full slot: voiced window + release tail through the guard
            let sym = self.vowel_queue.remove(0);
            let volume = self.send_volume * melody_accent(self.melody_pos);
            self.current_symbol_audio =
                formant::synthesize_symbol(sym, melody_f0(self.melody_pos), volume, self.melody_pos);
            self.melody_pos += 1;
            self.sample_pos = 0;
        }

        written
    }
}

// --- StreamRx (decoder) ---

struct ReceivedWord {
    symbols: [u8; CODEWORD_LEN],
    confidences: [f64; CODEWORD_LEN],
}

struct WordCandidate {
    data: Vec<u8>,
    penalty: u64,
}

/// Integer confidence costs make ordering deterministic, including ties.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RetryChoice {
    penalty: u64,
    ranks: Vec<u8>,
    /// Change only this word or a later one: each rank combination has one path.
    next_change: usize,
}

#[cfg(test)]
#[derive(Default)]
struct RetryStats {
    trials: usize,
    peak_frontier: usize,
    candidates: usize,
}

/// Streaming receiver: ingests audio samples, outputs decoded bytes.
pub struct StreamRx {
    config: StreamConfig,
    profile: StreamProfile,
    rs: ReedSolomon4,
    deinterleaver: Deinterleaver,
    sample_buf: Vec<f32>,
    /// Bounded preamble history, independent of the caller's capture phase.
    listening_audio: Vec<f32>,
    /// Cached quarter-slot measurements, shifted with the retained audio.
    listening_measurements: Vec<Option<WindowMeasurement>>,
    /// Slow-rising noise floor estimate (Listening state).
    noise_floor: f64,
    /// Listening windows observed (the floor is only meaningful after a
    /// few of them; before that a cold start would make the preamble
    /// itself the floor).
    listen_windows: usize,
    state: RxState,
    /// First vowel of an incomplete GF(2^4) symbol pair.
    pending_half: Option<(usize, f64)>,
    codeword_buf: Vec<u8>,
    codeword_conf: Vec<f64>,
    /// Completed payload bytes, retained until read even as later streams arrive.
    output_buf: Vec<u8>,
    /// Decoded GF(2^4) symbols, retaining unknown positions from failed words.
    /// Removing a failed word would shift later data into the header/payload.
    decoded_nibbles: Vec<Option<u8>>,
    /// Standard-only observations for CRC-guided retries; bounded by wire length.
    received_words: Vec<ReceivedWord>,
    #[cfg(test)]
    retry_stats: RetryStats,
    stream_symbol_count: usize,
    /// Smoothed slot advance, held through unvoiced dropouts/trailing silence.
    clock_period: f64,
    /// Running RMS average of voiced windows (for relative silence detection).
    rms_ema: f64,
    /// Consecutive near-silent windows seen (candidate end-of-stream).
    silence_run: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RxState {
    Listening,
    Streaming,
}

impl StreamRx {
    /// Create a receiver that requires the CRC-protected streaming wire format.
    pub fn new(config: StreamConfig) -> Self {
        Self::with_profile(config, StreamProfile::Standard)
    }

    /// Select wire framing explicitly. Standard never falls back to Legacy.
    /// Panics if parity is above 14 or interleaver depth is zero.
    pub fn with_profile(config: StreamConfig, profile: StreamProfile) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon4::new(data_len, config.rs_parity as usize);
        let deinterleaver = Deinterleaver::new(config.interleave_depth as usize);
        Self {
            config,
            profile,
            rs,
            deinterleaver,
            sample_buf: Vec::new(),
            listening_audio: Vec::new(),
            listening_measurements: Vec::new(),
            noise_floor: 0.0,
            listen_windows: 0,
            state: RxState::Listening,
            pending_half: None,
            codeword_buf: Vec::new(),
            codeword_conf: Vec::new(),
            output_buf: Vec::new(),
            decoded_nibbles: Vec::new(),
            received_words: Vec::new(),
            #[cfg(test)]
            retry_stats: RetryStats::default(),
            stream_symbol_count: 0,
            clock_period: SYMBOL_TOTAL_SAMPLES as f64,
            rms_ema: 0.0,
            silence_run: 0,
        }
    }

    /// Feed audio samples into the receiver. NaN and infinities are treated
    /// as zero-valued dropouts, preserving timing and every finite sample.
    pub fn ingest(&mut self, samples: &[f32]) {
        crate::audio::append_finite_samples(&mut self.sample_buf, samples);
        self.process_samples();
    }

    /// Read queued decoded payload bytes in reception order, possibly spanning
    /// several completed streams. Returns number of bytes written to `out`.
    pub fn read(&mut self, out: &mut [u8]) -> usize {
        let n = self.output_buf.len().min(out.len());
        out[..n].copy_from_slice(&self.output_buf[..n]);
        self.output_buf.drain(..n);
        n
    }

    /// Consume all queued decoded payload bytes in reception order.
    pub fn read_all(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output_buf)
    }

    /// Returns true if the receiver is currently in the streaming state.
    /// Acquisition uses up to two slots of timing lookahead after the preamble;
    /// buffered payload samples are replayed, not discarded.
    pub fn is_active(&self) -> bool {
        self.state == RxState::Streaming
    }

    fn process_samples(&mut self) {
        while self.sample_buf.len() >= SYMBOL_TOTAL_SAMPLES {
            let was_streaming = self.state == RxState::Streaming;
            let step = if was_streaming {
                let voiced = &self.sample_buf[..SAMPLES_PER_SYMBOL];
                let rms = (voiced.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / voiced.len() as f64).sqrt();
                if self.is_silent_window(voiced, rms) {
                    self.clock_period.round() as usize
                } else {
                    self.stream_clock_step(&self.sample_buf[..SYMBOL_TOTAL_SAMPLES])
                }
            } else {
                SYMBOL_TOTAL_SAMPLES
            };
            // A positive correction needs a little more guard/lookahead.
            // Waiting makes the decision independent of callback boundaries.
            if self.sample_buf.len() < step { break; }
            let window: Vec<f32> = self.sample_buf[..SYMBOL_TOTAL_SAMPLES].to_vec();
            self.sample_buf.drain(..step);
            self.process_window(&window);
            if was_streaming && self.state == RxState::Streaming && self.silence_run == 0 {
                // Guard corrections average to the sender's slot duration.
                // Do not learn a nominal rate from windows without a voice.
                self.clock_period += 0.05 * (step as f64 - self.clock_period);
            }
        }
        // Only the voiced portion is evidence for silence. A capture can end
        // inside the final guard after phase recovery; do not require unseen
        // guard samples once the fourth complete analysis window is silent.
        // Never consume a partial data slot: a quiet voice must wait for its
        // guard and go through the ordinary symbol path above.
        if self.state == RxState::Streaming && self.silence_run == END_SILENCE_WINDOWS - 1
            && self.sample_buf.len() >= SAMPLES_PER_SYMBOL
        {
            let voiced = &self.sample_buf[..SAMPLES_PER_SYMBOL];
            let rms = (voiced.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / voiced.len() as f64).sqrt();
            if self.is_silent_window(voiced, rms) {
                self.sample_buf.drain(..SAMPLES_PER_SYMBOL);
                self.finish_stream();
            }
        }
    }

    fn stream_clock_step(&self, window: &[f32]) -> usize {
        crate::audio::guard_clock_step(window)
    }

    fn is_silent_window(&self, voiced: &[f32], rms: f64) -> bool {
        self.rms_ema > 0.0 && rms < SILENCE_RMS_RATIO * self.rms_ema
            && !crate::decoder::has_voiced_periodicity(voiced)
    }

    fn finish_stream(&mut self) {
        self.deinterleaver.flush();
        self.drain_deinterleaver();
        self.state = RxState::Listening;
        self.pending_half = None;
        self.silence_run = 0;
        self.finalize_output();
    }

    fn process_window(&mut self, window: &[f32]) {
        match self.state {
            RxState::Listening => {
                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
                let rms = (voiced.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / voiced.len() as f64).sqrt();
                self.listen_windows += 1;
                // Slow-rising, fast-falling ambient floor: quiet windows pull
                // it down immediately, loud ones lift it only gradually, so
                // a transmission's own preamble can't inflate it.
                if self.listen_windows == 1 || rms < self.noise_floor {
                    self.noise_floor = rms;
                } else {
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms;
                }
                self.listening_audio.extend_from_slice(window);
                let excess = self.listening_audio.len().saturating_sub(STREAM_LISTEN_WINDOWS * SYMBOL_TOTAL_SAMPLES);
                if excess > 0 {
                    self.listening_audio.drain(..excess);
                    let old = (excess / STREAM_PHASE_STEP).min(self.listening_measurements.len());
                    self.listening_measurements.drain(..old);
                }
                let count = self.listening_audio.len().saturating_sub(SYMBOL_TOTAL_SAMPLES) / STREAM_PHASE_STEP + 1;
                self.listening_measurements.resize(count, None);
                if let Some(start) = self.find_stream_preamble() {
                    // Lookahead already contains payload samples. Replay all
                    // of them before unprocessed callback audio on the new grid.
                    let data_start = start + PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES;
                    self.sample_buf.splice(..0, self.listening_audio[data_start..].iter().copied());
                    self.listening_audio.clear();
                    self.listening_measurements.clear();
                    self.state = RxState::Streaming;
                    self.pending_half = None;
                    self.codeword_buf.clear();
                    self.codeword_conf.clear();
                    self.decoded_nibbles.clear();
                    self.received_words.clear();
                    #[cfg(test)]
                    { self.retry_stats = RetryStats::default(); }
                    self.stream_symbol_count = 0;
                    self.clock_period = SYMBOL_TOTAL_SAMPLES as f64;
                    self.rms_ema = 0.0;
                    self.silence_run = 0;
                    self.deinterleaver =
                        Deinterleaver::new(self.config.interleave_depth as usize);
                }
            }
            RxState::Streaming => {
                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];

                // End-of-stream detection: a run of near-silent, unvoiced
                // windows. Relative level alone can mistake a quiet phrase
                // or a denoiser's gain dip for the end of the transmission.
                let rms = {
                    let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
                    (sum_sq / voiced.len() as f64).sqrt()
                };
                if self.is_silent_window(voiced, rms) {
                    // Don't feed this window yet: if voice resumes it becomes an
                    // erasure (keeping stream positions aligned); if the run
                    // completes it was the trailing silence and is dropped.
                    self.silence_run += 1;
                    if self.silence_run >= END_SILENCE_WINDOWS {
                        self.finish_stream();
                    }
                    return;
                }

                // Voice resumed after a short gap: the held-back silent windows
                // were a dropout, not the end. Feed them as zero-confidence
                // erasures so downstream symbol positions stay aligned.
                for _ in 0..self.silence_run {
                    self.push_stream_symbol(0, 0.0);
                }
                self.silence_run = 0;

                self.rms_ema = if self.rms_ema == 0.0 {
                    rms
                } else {
                    0.9 * self.rms_ema + 0.1 * rms
                };

                let (sym, conf) = classify_voiced_segment(voiced);
                self.push_stream_symbol(sym, conf);

                // Drain deinterleaved symbols into codeword buffer
                self.drain_deinterleaver();
            }
        }
    }

    fn listening_window(&mut self, start: usize) -> WindowMeasurement {
        let end = start + SYMBOL_TOTAL_SAMPLES;
        if start % STREAM_PHASE_STEP == 0 {
            let index = start / STREAM_PHASE_STEP;
            if let Some(measured) = self.listening_measurements[index] {
                return measured;
            }
            let measured = measure_window(&self.listening_audio[start..end]);
            self.listening_measurements[index] = Some(measured);
            measured
        } else {
            measure_window(&self.listening_audio[start..end])
        }
    }

    fn preamble_matches(&mut self, start: usize) -> bool {
        let measured: [_; PREAMBLE_LEN] = std::array::from_fn(|i| {
            self.listening_window(start + i * SYMBOL_TOTAL_SAMPLES)
        });
        let matches = measured.iter().zip(STREAM_PREAMBLE_START)
            .filter(|(m, expected)| m.0 == *expected).count();
        if matches == PREAMBLE_LEN { return true; }
        let floor_known = self.listen_windows >= PREAMBLE_LEN + 8;
        let clear_of_floor = measured.iter().all(|m| {
            m.2 > if floor_known { (3.0 * self.noise_floor).max(1e-6) } else { 1e-4 }
        });
        let centroids = measured.map(|m| m.1);
        clear_of_floor && (matches >= PREAMBLE_LEN - 1
            || crate::decoder::structural_start_match(&centroids))
    }

    fn preamble_guard_score(&self, start: usize) -> f64 {
        (0..PREAMBLE_LEN).map(|i| {
            let slot = &self.listening_audio[start + i * SYMBOL_TOTAL_SAMPLES
                ..start + (i + 1) * SYMBOL_TOTAL_SAMPLES];
            let voice: f64 = slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| (s as f64).powi(2)).sum();
            let guard: f64 = slot[SAMPLES_PER_SYMBOL..].iter().map(|&s| (s as f64).powi(2)).sum();
            if voice > 0.0 {
                guard / voice * SAMPLES_PER_SYMBOL as f64 / GUARD_SAMPLES as f64
            } else {
                f64::INFINITY
            }
        }).sum()
    }

    fn find_stream_preamble(&mut self) -> Option<usize> {
        if self.listening_audio.len() < STREAM_LISTEN_WINDOWS * SYMBOL_TOTAL_SAMPLES {
            return None;
        }
        let last = self.listening_audio.len() - PREAMBLE_LEN * SYMBOL_TOTAL_SAMPLES;
        for coarse in (0..=last).step_by(STREAM_PHASE_STEP) {
            if !self.preamble_matches(coarse) { continue; }
            // Refine only around the earliest plausible preamble. Comparing
            // unrelated later matches could mistake payload vowels for sync.
            let lo = coarse.saturating_sub(STREAM_PHASE_STEP);
            let hi = (coarse + 2 * STREAM_PHASE_STEP).min(last);
            let mut best = coarse;
            let mut score = self.preamble_guard_score(coarse);
            for start in (lo..=hi).step_by(STREAM_PHASE_REFINE_STEP) {
                let candidate = self.preamble_guard_score(start);
                if candidate < score {
                    score = candidate;
                    best = start;
                }
            }
            if !self.preamble_matches(best) { best = coarse; }
            // A trailing-edge candidate may improve with more audio. Keep
            // one full slot of lookahead before committing to its phase.
            return (best + SYMBOL_TOTAL_SAMPLES <= last).then_some(best);
        }
        None
    }

    /// Feed one modem (vowel) symbol into the pairing + deinterleave pipeline.
    ///
    /// Pairs consecutive vowels into a GF(2^4) symbol: hi 2 bits, lo 2 bits.
    /// Pair confidence = min of the two halves.
    fn push_stream_symbol(&mut self, sym: usize, conf: f64) {
        match self.pending_half.take() {
            None => self.pending_half = Some((sym, conf)),
            Some((hi, hi_conf)) => {
                let nibble = ((hi << 2) | sym) as u8;
                self.deinterleaver.push(nibble, hi_conf.min(conf));
            }
        }
        self.stream_symbol_count += 1;
    }

    fn drain_deinterleaver(&mut self) {
        while let Some((sym, conf)) = self.deinterleaver.pop() {
            self.codeword_buf.push(sym);
            self.codeword_conf.push(conf);

            if self.codeword_buf.len() == CODEWORD_LEN {
                self.decode_codeword();
            }
        }
    }

    fn decode_codeword(&mut self) {
        let encoded: Vec<u8> = self.codeword_buf.drain(..).collect();
        let confs: Vec<f64> = self.codeword_conf.drain(..).collect();

        // Long background acquisitions cannot grow frame history indefinitely.
        // Keep the last required word whole so nibble/codeword positions stay
        // aligned, even if a later CRC retry recovers a different length header.
        let max_nibbles = 2 * (1 + MAX_STREAM_PAYLOAD + self.profile.checksum_bytes());
        if self.decoded_nibbles.len() >= max_nibbles {
            return;
        }
        if self.profile == StreamProfile::Standard {
            self.received_words.push(ReceivedWord {
                symbols: encoded.as_slice().try_into().unwrap(),
                confidences: confs.as_slice().try_into().unwrap(),
            });
        }

        // Try hard-decision decode first, then erasure fallback with the
        // weakest symbols (lowest confidence first), capped at parity budget.
        let decoded = self.rs.decode(&encoded).or_else(|| {
            let mut candidates: Vec<(usize, f64)> = confs
                .iter()
                .enumerate()
                .filter(|(_, &c)| c < ERASURE_THRESHOLD)
                .map(|(i, &c)| (i, c))
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let max_erasures = self.config.rs_parity as usize;
            let erasures: Vec<usize> = candidates
                .iter()
                .take(max_erasures)
                .map(|&(i, _)| i)
                .collect();
            if erasures.is_empty() {
                return None;
            }
            self.rs.decode_with_erasures(&encoded, &erasures)
        });

        if let Some(data) = decoded {
            self.decoded_nibbles.extend(data.into_iter().map(Some));
        } else {
            self.decoded_nibbles.resize(
                self.decoded_nibbles.len() + self.config.data_per_codeword(), None,
            );
        }
    }

    /// Reassemble decoded nibbles into bytes using the length prefix.
    fn finalize_output(&mut self) {
        let payload = self.payload_from_nibbles(&self.decoded_nibbles).or_else(|| {
            // Legacy has no integrity oracle with which to select alternatives.
            (self.profile == StreamProfile::Standard).then(|| self.retry_payload()).flatten()
        });
        if let Some(payload) = payload {
            self.output_buf.extend_from_slice(&payload);
        }
    }

    fn payload_from_nibbles(&self, nibbles: &[Option<u8>]) -> Option<Vec<u8>> {
        // Two nibbles per byte, MSB first — exact, no bit accumulation needed.
        let mut bytes = nibbles
            .chunks_exact(2)
            .map(|pair| Some((pair[0]? << 4) | (pair[1]? & 0x0F)));

        // The length prefix, payload, and checksum nibbles must all be known.
        // Unknown padding/cadence after that range must not cost good data.
        let len_byte = bytes.next()??;
        let len_byte = len_byte as usize;
        let required = len_byte + self.profile.checksum_bytes();
        if len_byte == 0 || required > bytes.len() {
            return None;
        }
        let mut frame = Vec::with_capacity(1 + required);
        frame.push(len_byte as u8);
        for byte in bytes.take(required) {
            frame.push(byte?);
        }
        let payload_end = 1 + len_byte;
        if self.profile == StreamProfile::Standard {
            let actual = crate::checksum::crc32(&frame[..payload_end]).to_le_bytes();
            if frame[payload_end..] != actual {
                return None;
            }
        }

        Some(frame[1..payload_end].to_vec())
    }

    /// Generate distinct hard/weakest-prefix erasure results, only after the
    /// original complete frame failed CRC. Do not let a hard miscorrection or
    /// an over-wide erasure set hide a different, integrity-valid result.
    fn word_candidates(&self, received: &ReceivedWord) -> Vec<WordCandidate> {
        let mut weakest: Vec<_> = received.confidences.iter().enumerate()
            .filter(|(_, &c)| c < ERASURE_THRESHOLD)
            .map(|(i, &c)| (i, c)).collect();
        weakest.sort_by(|a, b| a.1.total_cmp(&b.1));
        let erasures: Vec<_> = weakest.iter().take(self.config.rs_parity as usize)
            .map(|&(i, _)| i).collect();
        let mut candidates: Vec<WordCandidate> = Vec::new();
        for count in 0..=erasures.len() {
            let Some(data) = self.rs.decode_with_erasures(&received.symbols, &erasures[..count]) else {
                continue;
            };
            if candidates.iter().any(|candidate| candidate.data == data) {
                continue;
            }
            let encoded = self.rs.encode(&data);
            let penalty = encoded.iter().zip(&received.symbols).zip(&received.confidences)
                .filter(|((a, b), _)| a != b)
                .map(|(_, &confidence)| {
                    let confidence = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 };
                    // Penalize changed high-confidence symbols most; a small
                    // per-edit cost also distinguishes zero-confidence ties.
                    10_000 + (confidence * 1_000_000.0).round() as u64
                }).sum();
            candidates.push(WordCandidate { data, penalty });
        }
        // Stable sort keeps hard decoding first when confidence costs tie.
        candidates.sort_by_key(|candidate| candidate.penalty);
        candidates
    }

    fn retry_payload(&mut self) -> Option<Vec<u8>> {
        let candidates: Vec<_> = self.received_words.iter()
            .map(|word| self.word_candidates(word)).collect();
        #[cfg(test)]
        { self.retry_stats.candidates = candidates.iter().map(Vec::len).sum(); }
        let k = self.config.data_per_codeword();
        let header_words = 2usize.div_ceil(k);
        if candidates.len() < header_words || candidates[..header_words].iter().any(Vec::is_empty) {
            return None;
        }

        // Each plausible decoded header gets its own search. Only words inside
        // that claimed frame participate: uncertain padding words must neither kill
        // a short valid message nor consume its candidate-search budget.
        let mut frontier = BTreeSet::new();
        let mut add_header = |header: &[u8]| {
            let first = &candidates[0][header[0] as usize].data;
            let low = if k >= 2 { first[1] } else { candidates[1][header[1] as usize].data[0] };
            let len = ((first[0] << 4) | low) as usize;
            let words = (2 * (1 + len + self.profile.checksum_bytes())).div_ceil(k);
            if len == 0 || words > candidates.len() || candidates[..words].iter().any(Vec::is_empty) {
                return;
            }
            let mut ranks = vec![0; words];
            ranks[..header_words].copy_from_slice(header);
            let penalty = ranks.iter().enumerate()
                .map(|(word, &rank)| candidates[word][rank as usize].penalty).sum();
            frontier.insert(RetryChoice { penalty, ranks, next_change: header_words });
        };
        for first in 0..candidates[0].len() {
            if header_words == 1 {
                add_header(&[first as u8]);
            } else {
                for second in 0..candidates[1].len() {
                    add_header(&[first as u8, second as u8]);
                }
            }
        }
        while frontier.len() > MAX_STREAM_RETRY_FRONTIER {
            frontier.pop_last();
        }

        for _ in 0..MAX_STREAM_RETRY_TRIALS {
            #[cfg(test)]
            { self.retry_stats.peak_frontier = self.retry_stats.peak_frontier.max(frontier.len()); }
            let Some(choice) = frontier.pop_first() else { break; };
            #[cfg(test)]
            { self.retry_stats.trials += 1; }
            let nibbles: Vec<_> = choice.ranks.iter().enumerate()
                .flat_map(|(word, &rank)| candidates[word][rank as usize].data.iter().copied().map(Some))
                .collect();
            if let Some(payload) = self.payload_from_nibbles(&nibbles) {
                return Some(payload);
            }
            for word in choice.next_change..choice.ranks.len() {
                let rank = choice.ranks[word] as usize;
                if rank + 1 >= candidates[word].len() { continue; }
                let mut next = choice.clone();
                next.ranks[word] += 1;
                next.next_change = word;
                next.penalty = choice.penalty - candidates[word][rank].penalty
                    + candidates[word][rank + 1].penalty;
                frontier.insert(next);
                if frontier.len() > MAX_STREAM_RETRY_FRONTIER {
                    frontier.pop_last();
                }
            }
        }
        None
    }
}

/// Measure a Listening window: (vowel index or NUM_VOWELS for silence,
/// F2 centroid, RMS).
fn measure_window(window: &[f32]) -> (usize, f64, f64) {
    let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
    let rms = {
        let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / voiced.len() as f64).sqrt()
    };
    if rms < 1e-6 {
        return (NUM_VOWELS, (F2_LO + F2_HI) / 2.0, rms);
    }
    // Hann-windowed spectrum, same as the streaming classification path.
    let mut spectrum = vec![0.0f32; voiced.len()];
    fft::power_spectrum(voiced, &mut spectrum);
    let f0 = formant::detect_f0(&spectrum);
    let f2 = formant::detect_formants(&spectrum, f0);
    let (vowel_idx, _) = formant::classify_vowel_f2(f2, &formant::EXPECTED_F2_CENTROIDS);
    (vowel_idx, f2, rms)
}

/// Classify a voiced segment into a symbol (vowel) with confidence.
fn classify_voiced_segment(voiced: &[f32]) -> (usize, f64) {
    let mut spectrum = vec![0.0f32; voiced.len()];
    fft::power_spectrum(voiced, &mut spectrum);
    let hybrid = formant::classify_symbol_full(&spectrum, &formant::EXPECTED_F2_CENTROIDS);
    let (f2, confidence) = formant::classify_vowel_f2(hybrid.centroid, &formant::EXPECTED_F2_CENTROIDS);
    if hybrid.vowel != f2 {
        // Sparse high-note harmonics can make F1 override a correct F2
        // decision. Ask the full F2 spectral shape to arbitrate only these
        // disagreements; never restrict or transpose the sender's pitch.
        let template = formant::classify_symbol_template(&spectrum, &[1.0; FFT_SIZE]);
        if template.vowel == f2 {
            return (f2, (confidence / (confidence + 1.0)).min(template.confidence));
        }
    }
    (hybrid.vowel, hybrid.confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_clock_corrections_are_bounded_directional_and_scale_independent() {
        let rx = StreamRx::new(StreamConfig::default());
        for midi in 47..=66 {
            let hz = 440.0 * 2.0f64.powf((midi as f64 - 69.0) / 12.0);
            for vowel in 0..NUM_VOWELS {
                let song: Vec<f32> = (0..3).flat_map(|pos| {
                    formant::synthesize_symbol(vowel, hz, 0.5, pos)
                        .into_iter().map(|s| s as f32)
                }).collect();
                for phase in [-(GUARD_SAMPLES as isize / 2), 0, GUARD_SAMPLES as isize / 2] {
                    let start = (SYMBOL_TOTAL_SAMPLES as isize + phase) as usize;
                    let original = &song[start..start + SYMBOL_TOTAL_SAMPLES];
                    let expected = rx.stream_clock_step(original);
                    assert!(expected.abs_diff(SYMBOL_TOTAL_SAMPLES) <= STREAM_CLOCK_RADIUS);
                    assert!(expected >= SAMPLES_PER_SYMBOL, "classification must not read a later slot");
                    match phase.cmp(&0) {
                        std::cmp::Ordering::Less => assert!(expected > SYMBOL_TOTAL_SAMPLES),
                        std::cmp::Ordering::Equal => assert_eq!(expected, SYMBOL_TOTAL_SAMPLES),
                        std::cmp::Ordering::Greater => assert!(expected < SYMBOL_TOTAL_SAMPLES),
                    }
                    for gain in [0.001, 0.1, 10.0, 1e10] {
                        let scaled: Vec<_> = original.iter().map(|&s| s * gain).collect();
                        assert_eq!(rx.stream_clock_step(&scaled), expected,
                            "MIDI {midi}, vowel {vowel}, phase {phase}, gain {gain}");
                    }
                }
            }
        }
        let mut seed = 71u32;
        for kind in 0..3 {
            for _ in 0..128 {
                let slot: Vec<_> = (0..SYMBOL_TOTAL_SAMPLES).map(|_| match kind {
                    0 => 0.0,
                    1 => 0.001,
                    _ => {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        (seed as f32 / u32::MAX as f32 - 0.5) * 0.002
                    }
                }).collect();
                assert_eq!(rx.stream_clock_step(&slot), SYMBOL_TOTAL_SAMPLES,
                    "stationary noise/DC/silence must not steer the clock");
            }
        }
    }

    #[test]
    fn stream_clock_correction_preserves_buffered_samples_and_waits_for_lookahead() {
        let song: Vec<f32> = (0..3).flat_map(|pos| {
            formant::synthesize_symbol(1, 220.0, 0.5, pos).into_iter().map(|s| s as f32)
        }).collect();
        for phase in [-(GUARD_SAMPLES as isize / 2), GUARD_SAMPLES as isize / 2] {
            let start = (SYMBOL_TOTAL_SAMPLES as isize + phase) as usize;
            let input = &song[start..start + SYMBOL_TOTAL_SAMPLES + STREAM_CLOCK_RADIUS];
            let mut rx = StreamRx::new(StreamConfig::default());
            rx.state = RxState::Streaming;
            rx.rms_ema = 0.1;
            let step = rx.stream_clock_step(&input[..SYMBOL_TOTAL_SAMPLES]);
            rx.ingest(&input[..SYMBOL_TOTAL_SAMPLES]);
            if step > SYMBOL_TOTAL_SAMPLES {
                assert_eq!(rx.stream_symbol_count, 0, "wait for actual lookahead, never invent samples");
                assert_eq!(rx.sample_buf, input[..SYMBOL_TOTAL_SAMPLES]);
            } else {
                assert_eq!(rx.stream_symbol_count, 1);
                assert_eq!(rx.sample_buf, input[step..SYMBOL_TOTAL_SAMPLES]);
            }
            rx.ingest(&input[SYMBOL_TOTAL_SAMPLES..]);
            assert_eq!(rx.stream_symbol_count, 1, "one correction must not duplicate a symbol");
            assert_eq!(rx.sample_buf, input[step..], "retain the exact unconsumed suffix");
        }
    }

    #[test]
    fn stream_clock_does_not_chase_vanishing_filter_tails() {
        let rx = StreamRx::new(StreamConfig::default());
        for vowel in 0..NUM_VOWELS {
            let mut slot: Vec<f32> = formant::synthesize_symbol(vowel, 220.0, 0.5, 0)
                .into_iter().map(|s| s as f32).collect();
            for i in 0..GUARD_SAMPLES {
                slot[SAMPLES_PER_SYMBOL + i] = 0.000001 * (-8.0 * i as f32 / GUARD_SAMPLES as f32).exp();
            }
            for gain in [0.001, 1.0, 1000.0] {
                let scaled: Vec<_> = slot.iter().map(|&s| s * gain).collect();
                assert_eq!(rx.stream_clock_step(&scaled), SYMBOL_TOTAL_SAMPLES,
                    "an already-clear guard must not chase its filter's asymptotic decay");
            }
        }
    }

    #[test]
    fn stream_clock_rate_is_held_through_silence_and_reset_by_new_sync() {
        for ratio in [0.98, 1.02] {
            let mut rx = StreamRx::new(StreamConfig::default());
            rx.state = RxState::Streaming;
            rx.rms_ema = 0.1;
            rx.clock_period = SYMBOL_TOTAL_SAMPLES as f64 * ratio;
            let previous = rx.clock_period.to_bits();
            let step = rx.clock_period.round() as usize;
            let partial = SAMPLES_PER_SYMBOL / 2;
            rx.ingest(&vec![0.0; 3 * step + partial]);
            assert!(rx.is_active());
            assert_eq!(rx.silence_run, 3);
            assert_eq!(rx.sample_buf.len(), partial, "silence follows the learned clock");
            assert_eq!(rx.clock_period.to_bits(), previous, "do not learn timing from silence");
            rx.ingest(&vec![0.0; SAMPLES_PER_SYMBOL - partial]);
            assert!(!rx.is_active());
            assert!(rx.sample_buf.is_empty());
            let sync: Vec<f32> = STREAM_PREAMBLE_START.into_iter().chain([1, 2])
                .enumerate().flat_map(|(pos, vowel)| {
                    formant::synthesize_symbol(vowel, melody_f0(pos), 0.5, pos)
                        .into_iter().map(|s| s as f32)
                }).collect();
            rx.ingest(&sync);
            assert!(rx.is_active());
            assert_eq!(rx.stream_symbol_count, 2);
            assert_eq!(rx.clock_period, SYMBOL_TOTAL_SAMPLES as f64,
                "a new sender must not inherit the preceding sender's clock");
            assert!(rx.read_all().is_empty());
        }
    }

    #[test]
    fn stream_clock_tracking_preserves_the_chromatic_register_and_contours() {
        let payload: Vec<_> = (0..17).map(|i| (i * 73 + 19) as u8).collect();
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            let config = StreamConfig::default();
            let mut tx = StreamTx::with_profile(config.clone(), profile);
            tx.feed(&payload);
            tx.encode_codewords();
            tx.flush_final();
            let vowels: Vec<_> = STREAM_PREAMBLE_START.into_iter().chain(tx.vowel_queue).collect();
            // Twenty sustained pitches, then ascending, descending, wide
            // leaps, and a shuffled/detuned contour unknown to the receiver.
            for melody in 0..24 {
                let mut song: Vec<f32> = vowels.iter().enumerate().flat_map(|(pos, &vowel)| {
                    let midi = match melody {
                        0..=19 => 47 + melody,
                        20 => 47 + (pos / 4) % 20,
                        21 => 66 - (pos / 4) % 20,
                        22 => if pos / 4 % 2 == 0 { 47 } else { 66 },
                        _ => 47 + (pos / 4 * 73 + 11) % 20,
                    };
                    let detune = if melody == 23 { [-0.3, 0.0, 0.3][pos / 4 % 3] } else { 0.0 };
                    let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                    formant::synthesize_symbol(vowel, hz, 0.5 * melody_accent(pos), pos)
                        .into_iter().map(|s| s as f32)
                }).collect();
                song.resize(song.len() + TRAILING_SILENCE_SLOTS * SYMBOL_TOTAL_SAMPLES, 0.0);
                for ratio in [0.98, 1.02] {
                    let mut warped = vec![0.0; SYMBOL_TOTAL_SAMPLES / 3];
                    let mut position = 0.0;
                    while position < (song.len() - 1) as f64 {
                        let i = position as usize;
                        let f = (position - i as f64) as f32;
                        warped.push(song[i] * (1.0 - f) + song[i + 1] * f);
                        position += ratio;
                    }
                    let mut rx = StreamRx::with_profile(config.clone(), profile);
                    for chunk in warped.chunks(511) {
                        rx.ingest(chunk);
                        assert!((rx.clock_period - SYMBOL_TOTAL_SAMPLES as f64).abs()
                            <= STREAM_CLOCK_RADIUS as f64, "clock holdover must remain bounded");
                    }
                    assert_eq!(rx.read_all(), payload, "{profile:?}, melody {melody}, ratio {ratio}");
                    assert!(!rx.is_active());
                }
            }
        }
    }

    #[test]
    fn streaming_classification_recognizes_every_vowel_across_the_register() {
        for midi in 47..=66 {
            for detune in [-0.3, 0.0, 0.3] {
                let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                for vowel in 0..NUM_VOWELS {
                    for pos in [0, 3, 7] {
                        let slot: Vec<_> = formant::synthesize_symbol(vowel, hz, 0.5 * melody_accent(pos), pos)
                            .into_iter().map(|s| s as f32).collect();
                        let (actual, confidence) = classify_voiced_segment(&slot[..SAMPLES_PER_SYMBOL]);
                        assert_eq!(actual, vowel, "MIDI {midi}, detune {detune}, position {pos}");
                        assert!((0.0..=1.0).contains(&confidence));
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_phase_acquisition_is_independent_of_pitch_and_profile() {
        let payload = b"pitch";
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            let config = StreamConfig::default();
            let mut tx = StreamTx::with_profile(config.clone(), profile);
            tx.feed(payload);
            tx.encode_codewords();
            tx.flush_final();
            let vowels: Vec<_> = STREAM_PREAMBLE_START.into_iter().chain(tx.vowel_queue).collect();
            for midi in 47..=66 {
                for detune in [-0.3, 0.0, 0.3] {
                    let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                    let mut song: Vec<f32> = vowels.iter().enumerate().flat_map(|(pos, &vowel)| {
                        formant::synthesize_symbol(vowel, hz, 0.5 * melody_accent(pos), pos)
                            .into_iter().map(|s| s as f32)
                    }).collect();
                    song.resize(song.len() + TRAILING_SILENCE_SLOTS * SYMBOL_TOTAL_SAMPLES, 0.0);
                    for phase in [0, 1, SYMBOL_TOTAL_SAMPLES / 2 - 1, 3 * SYMBOL_TOTAL_SAMPLES / 4] {
                        let mut audio = vec![0.0; phase];
                        audio.extend_from_slice(&song);
                        let mut rx = StreamRx::with_profile(config.clone(), profile);
                        for chunk in audio.chunks(511) { rx.ingest(chunk); }
                        let received = rx.read_all();
                        assert_eq!(received, payload,
                            "{profile:?}, MIDI {midi}, detune {detune}, phase {phase}");
                        assert!(!rx.is_active());
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_phase_measurements_remain_aligned_and_bounded_in_idle_audio() {
        let mut rx = StreamRx::new(StreamConfig::default());
        let mut seed = 139u32;
        for i in 0..512 {
            let slot: Vec<_> = (0..SYMBOL_TOTAL_SAMPLES).map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.002
            }).collect();
            rx.ingest(&slot);
            assert!(!rx.is_active(), "idle noise must not acquire a stream");
            assert!(rx.read_all().is_empty());
            assert!(rx.listening_audio.len() <= STREAM_LISTEN_WINDOWS * SYMBOL_TOTAL_SAMPLES);
            assert!(rx.listening_measurements.len() <= (STREAM_LISTEN_WINDOWS - 1) * 4 + 1);
            if i % 31 == 0 {
                for (index, cached) in rx.listening_measurements.iter().enumerate() {
                    if let Some(cached) = cached {
                        let start = index * STREAM_PHASE_STEP;
                        let expected = measure_window(&rx.listening_audio[start..start + SYMBOL_TOTAL_SAMPLES]);
                        assert_eq!(cached.0, expected.0);
                        assert_eq!(cached.1.to_bits(), expected.1.to_bits());
                        assert_eq!(cached.2.to_bits(), expected.2.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn partial_final_silence_does_not_wait_for_a_guard_or_consume_quiet_voice() {
        for vowel in 0..=NUM_VOWELS {
            let mut rx = StreamRx::new(StreamConfig::default());
            rx.state = RxState::Streaming;
            rx.rms_ema = 0.1;
            rx.ingest(&vec![0.0; (END_SILENCE_WINDOWS - 1) * SYMBOL_TOTAL_SAMPLES]);
            assert!(rx.is_active());
            assert_eq!(rx.silence_run, END_SILENCE_WINDOWS - 1);
            let slot: Vec<f32> = if vowel == NUM_VOWELS {
                vec![0.0; SYMBOL_TOTAL_SAMPLES]
            } else {
                formant::synthesize_symbol(vowel, 220.0, 0.00001, 7)
                    .into_iter().map(|s| s as f32).collect()
            };
            rx.ingest(&slot[..SAMPLES_PER_SYMBOL - 1]);
            assert!(rx.is_active(), "an incomplete analysis window must not end the stream");
            assert_eq!(rx.sample_buf.len(), SAMPLES_PER_SYMBOL - 1);
            rx.ingest(&slot[SAMPLES_PER_SYMBOL - 1..SAMPLES_PER_SYMBOL]);
            if vowel == NUM_VOWELS {
                assert!(!rx.is_active(), "the unused final guard must not delay silence completion");
                assert!(rx.sample_buf.is_empty());
                assert_eq!(rx.stream_symbol_count, 0);
            } else {
                assert!(rx.is_active(), "quiet vowel {vowel} must remain data");
                assert_eq!(rx.sample_buf.len(), SAMPLES_PER_SYMBOL);
                assert_eq!(rx.silence_run, END_SILENCE_WINDOWS - 1);
                rx.ingest(&slot[SAMPLES_PER_SYMBOL..]);
                assert!(rx.is_active());
                assert!(rx.sample_buf.is_empty());
                assert_eq!(rx.silence_run, 0);
                assert_eq!(rx.stream_symbol_count, END_SILENCE_WINDOWS,
                    "each held dropout and the returning voice must be consumed exactly once");
            }
            assert!(rx.read_all().is_empty());
        }
    }

    /// Use the real transmitter's symbol queue, including interleaver padding
    /// and the cadence. Explicitly pad the cadence's partial receive block just
    /// as Deinterleaver::flush does, so even trailing failed codewords are tested.
    fn framed_symbols(payload: &[u8], config: &StreamConfig) -> Vec<(usize, f64)> {
        framed_symbols_with_profile(payload, config, StreamProfile::Standard)
    }

    fn framed_symbols_with_profile(
        payload: &[u8], config: &StreamConfig, profile: StreamProfile,
    ) -> Vec<(usize, f64)> {
        let mut tx = StreamTx::with_profile(config.clone(), profile);
        tx.feed(payload);
        tx.encode_codewords();
        tx.flush_final();
        let mut symbols: Vec<_> = tx.vowel_queue.into_iter().map(|s| (s, 1.0)).collect();
        let block_slots = config.interleave_depth as usize * CODEWORD_LEN * 2;
        symbols.resize(symbols.len().div_ceil(block_slots) * block_slots, (0, 0.0));
        symbols
    }

    fn raw_framed_symbols(frame: &[u8], config: &StreamConfig) -> Vec<(usize, f64)> {
        let mut tx = StreamTx::new(config.clone());
        tx.feed_raw(frame);
        tx.encode_codewords();
        tx.flush_final();
        let mut symbols: Vec<_> = tx.vowel_queue.into_iter().map(|s| (s, 1.0)).collect();
        let block_slots = config.interleave_depth as usize * CODEWORD_LEN * 2;
        symbols.resize(symbols.len().div_ceil(block_slots) * block_slots, (0, 0.0));
        symbols
    }

    fn checked_frame(payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![payload.len() as u8];
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&crate::checksum::crc32(&frame).to_le_bytes());
        frame
    }

    /// Move just beyond the hard-decision radius toward another valid word.
    /// RS therefore returns the wrong data, although confidence-ranked erasures
    /// can recover the original. This constructs the failure without relying
    /// on a random beyond-budget pattern happening to miscorrect.
    fn miscorrect_word(
        symbols: &mut [(usize, f64)], config: &StreamConfig, index: usize,
        changed_data_nibble: usize, weak_confidence: f64,
    ) {
        let depth = config.interleave_depth as usize;
        let slots: Vec<_> = (0..CODEWORD_LEN).map(|column| {
            2 * ((index / depth) * depth * CODEWORD_LEN + column * depth + index % depth)
        }).collect();
        let original: Vec<_> = slots.iter().map(|&slot| {
            ((symbols[slot].0 << 2) | symbols[slot + 1].0) as u8
        }).collect();
        let rs = ReedSolomon4::new(config.data_per_codeword(), config.rs_parity as usize);
        let data = rs.decode(&original).unwrap();
        let mut other_data = data.clone();
        other_data[changed_data_nibble] ^= 1;
        let other = rs.encode(&other_data);
        let differences: Vec<_> = original.iter().zip(&other).enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i)).collect();
        assert_eq!(differences.len(), config.rs_parity as usize + 1);
        let errors = &differences[..differences.len() - config.rs_parity as usize / 2];
        let mut received = original;
        for &i in errors { received[i] = other[i]; }
        assert_eq!(rs.decode(&received), Some(other_data), "hard RS must actually miscorrect");
        assert_eq!(rs.decode_with_erasures(&received, errors), Some(data));
        replace_codeword(symbols, depth, index, &received, false);
        for &i in errors {
            symbols[slots[i]].1 = weak_confidence;
            symbols[slots[i] + 1].1 = weak_confidence;
        }
    }

    #[test]
    fn stream_crc_rejection_still_allows_erasure_recovery() {
        for parity in 2..=14 {
            for depth in [1, 2, 4] {
                let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                for len in [1usize, 17, 64, 255] {
                    let payload: Vec<_> = (0..len).map(|i| (i * 73 + 19) as u8).collect();
                    let clean = framed_symbols(&payload, &config);
                    let words = (2 * (len + 5)).div_ceil(config.data_per_codeword());
                    for index in [0, 1.min(words - 1), words - 1] {
                        for confidence in [0.01, 0.8, 0.99] {
                            let mut symbols = clean.clone();
                            miscorrect_word(&mut symbols, &config, index, 0, confidence);
                            assert_eq!(receive_symbols(&symbols, &config), payload,
                                "parity {parity}, depth {depth}, length {len}, word {index}, confidence {confidence}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stream_crc_recovery_combines_words_without_searching_padding() {
        let payload: Vec<_> = (0..40).map(|i| (i * 73 + 19) as u8).collect();
        for depth in [1, 2, 4, 16] {
            let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
            for confidence in [0.01, 0.8, 0.99] {
                let mut symbols = framed_symbols(&payload, &config);
                for word in 0..8 {
                    miscorrect_word(&mut symbols, &config, word, 2, confidence);
                }
                let mut rx = receive_symbols_state(&symbols, &config, StreamProfile::Standard);
                assert_eq!(rx.read_all(), payload, "depth {depth}, confidence {confidence}");
                assert!(rx.retry_stats.trials > 0);
                if confidence >= 0.8 {
                    assert!(rx.retry_stats.trials > 1, "must search past the cheapest complete candidate");
                }
                let trials_without_padding = rx.retry_stats.trials;
                let data_words = (2 * (payload.len() + 5)).div_ceil(config.data_per_codeword());
                let transmitted_words = data_words.div_ceil(depth as usize) * depth as usize;
                for word in data_words..transmitted_words {
                    miscorrect_word(&mut symbols, &config, word, 0, 0.01);
                }
                let mut rx = receive_symbols_state(&symbols, &config, StreamProfile::Standard);
                assert_eq!(rx.read_all(), payload);
                assert_eq!(rx.retry_stats.trials, trials_without_padding,
                    "uncertain padding must not consume CRC trials");
                assert!(rx.retry_stats.peak_frontier <= MAX_STREAM_RETRY_FRONTIER);
            }
        }
    }

    #[test]
    fn stream_crc_retries_narrower_erasure_sets_after_an_overwide_miscorrection() {
        for depth in [1usize, 2, 4] {
            let config = StreamConfig { interleave_depth: depth as u8, ..StreamConfig::default() };
            let payload: Vec<_> = (0..40).map(|i| (i * 73 + 19) as u8).collect();
            let mut symbols = framed_symbols(&payload, &config);
            let slots: Vec<_> = (0..CODEWORD_LEN).map(|col| 2 * ((1 / depth) * depth * CODEWORD_LEN + col * depth + 1 % depth)).collect();
            let original: Vec<_> = slots.iter().map(|&slot| ((symbols[slot].0 << 2) | symbols[slot + 1].0) as u8).collect();
            let rs = ReedSolomon4::new(11, 4);
            let data = rs.decode(&original).unwrap();
            let mut seed = 731u32;
            let mut fixture = None;
            for _ in 0..128 {
                let mut received = original.clone();
                for byte in &mut received[..3] {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    *byte ^= ((seed >> 16) % 15 + 1) as u8;
                }
                if rs.decode(&received).is_none()
                    && rs.decode_with_erasures(&received, &[0, 1]) == Some(data.clone())
                    && rs.decode_with_erasures(&received, &[0, 1, 3, 4]).is_some_and(|wrong| wrong != data)
                {
                    fixture = Some(received);
                    break;
                }
            }
            let received = fixture.expect("hard rejection followed by an over-wide miscorrection");
            replace_codeword(&mut symbols, depth, 1, &received, false);
            for (rank, column) in [0, 1, 3, 4].into_iter().enumerate() {
                symbols[slots[column]].1 = 0.01 * (rank + 1) as f64;
                symbols[slots[column] + 1].1 = 0.01 * (rank + 1) as f64;
            }
            assert_eq!(receive_symbols(&symbols, &config), payload);
        }
    }

    #[test]
    fn stream_crc_search_is_bounded_and_never_accepts_unchecked_candidates() {
        let config = StreamConfig::default();
        let payload: Vec<_> = (0..120).map(|i| (i * 73 + 19) as u8).collect();
        let mut frame = checked_frame(&payload);
        frame[1] ^= 1; // fixed bad CRC: no candidate for word zero repairs this
        let mut symbols = raw_framed_symbols(&frame, &config);
        for word in 1..21 { miscorrect_word(&mut symbols, &config, word, 0, 0.99); }
        let mut rx = receive_symbols_state(&symbols, &config, StreamProfile::Standard);
        assert!(rx.read_all().is_empty());
        assert_eq!(rx.retry_stats.trials, MAX_STREAM_RETRY_TRIALS);
        assert_eq!(rx.retry_stats.peak_frontier, MAX_STREAM_RETRY_FRONTIER);
        assert!(rx.retry_stats.candidates <= rx.received_words.len() * (config.rs_parity as usize + 1));

        let clean = framed_symbols(&payload, &config);
        let mut rx = receive_symbols_state(&clean, &config, StreamProfile::Standard);
        assert_eq!(rx.read_all(), payload);
        assert_eq!(rx.retry_stats.trials, 0, "CRC-valid fast path does not search");
        assert_eq!(rx.retry_stats.candidates, 0, "candidate generation is failure-only");
        let mut legacy = receive_symbols_state(&symbols, &config, StreamProfile::Legacy);
        assert!(!legacy.read_all().is_empty(), "Legacy retains its unchecked behavior");
        assert_eq!(legacy.retry_stats.trials, 0);
        assert!(legacy.received_words.is_empty(), "Legacy does not retain retry observations");
    }

    #[test]
    fn stream_frame_history_is_bounded_during_unending_input() {
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            for parity in [0, 1, 4, 14] {
                let config = StreamConfig { rs_parity: parity, ..StreamConfig::default() };
                let k = config.data_per_codeword();
                let max_words = (2 * (1 + MAX_STREAM_PAYLOAD + profile.checksum_bytes())).div_ceil(k);
                let mut rx = StreamRx::with_profile(config, profile);
                for pass in 0..2 {
                    let capacities = (rx.decoded_nibbles.capacity(), rx.received_words.capacity());
                    for _ in 0..4096 {
                        rx.codeword_buf.extend_from_slice(&[0; CODEWORD_LEN]);
                        rx.codeword_conf.extend_from_slice(&[1.0; CODEWORD_LEN]);
                        rx.decode_codeword();
                    }
                    assert_eq!(rx.decoded_nibbles.len(), max_words * k);
                    assert_eq!(rx.received_words.len(), if profile == StreamProfile::Standard { max_words } else { 0 });
                    if pass == 1 {
                        assert_eq!((rx.decoded_nibbles.capacity(), rx.received_words.capacity()), capacities);
                    }
                }
                rx.finalize_output();
                assert!(rx.read_all().is_empty());
            }
        }
    }

    #[test]
    fn invalid_stream_geometry_is_rejected_at_construction() {
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            for (parity, depth) in [(15, 1), (16, 2), (255, 1), (4, 0)] {
                let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                assert!(std::panic::catch_unwind(|| StreamTx::with_profile(config.clone(), profile)).is_err());
                assert!(std::panic::catch_unwind(|| StreamRx::with_profile(config.clone(), profile)).is_err());
            }
            for (parity, depth) in [(0, 1), (14, 1), (4, 255)] {
                let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                StreamTx::with_profile(config.clone(), profile);
                StreamRx::with_profile(config, profile);
            }
        }
    }

    #[test]
    fn checksum_rejects_rs_valid_wrong_stream_data() {
        for parity in [2, 4, 6, 8] {
            for depth in [1, 2, 4] {
                let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                for len in [1usize, 5, 17, 64, 255] {
                    let payload: Vec<_> = (0..len).map(|i| (i * 73 + 19) as u8).collect();
                    let frame = checked_frame(&payload);
                    assert_eq!(receive_symbols(&raw_framed_symbols(&frame, &config), &config), payload);
                    // Every damaged frame is encoded into valid RS words:
                    // parity success alone cannot detect this corruption.
                    for offset in [0, 1, 1 + len / 2, len, len + 1, len + 4] {
                        let mut wrong = frame.clone();
                        wrong[offset] ^= 1;
                        assert!(receive_symbols(&raw_framed_symbols(&wrong, &config), &config).is_empty(),
                            "parity {parity}, depth {depth}, length {len}, corrupt byte {offset}");
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_profiles_keep_legacy_explicit_at_every_payload_length() {
        for depth in [1, 2, 4] {
            let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
            for len in 1..=255 {
                let payload: Vec<_> = (0..len).map(|i| (i * 73 + len) as u8).collect();
                let standard = framed_symbols(&payload, &config);
                assert_eq!(standard, raw_framed_symbols(&checked_frame(&payload), &config));
                assert_eq!(receive_symbols(&standard, &config), payload);

                let mut old_frame = vec![len as u8];
                old_frame.extend_from_slice(&payload);
                let legacy = framed_symbols_with_profile(&payload, &config, StreamProfile::Legacy);
                assert_eq!(legacy, raw_framed_symbols(&old_frame, &config),
                    "Legacy must retain exactly the old length/payload/padding layout");
                assert!(standard.len() >= legacy.len());
                assert!(standard.len() - legacy.len() <= 2 * CODEWORD_LEN * depth as usize);
                assert_eq!(receive_symbols_with_profile(&legacy, &config, StreamProfile::Legacy), payload);
                assert!(receive_symbols(&legacy, &config).is_empty(),
                    "default must not downgrade to Legacy: depth {depth}, length {len}");
            }
        }
    }

    #[test]
    fn streaming_checksum_bytes_remain_rs_correctable() {
        for parity in [2, 4, 6, 8] {
            for depth in [1, 2, 4] {
                let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                let k = config.data_per_codeword();
                for len in [1usize, 5, 16, 64, 255] {
                    let payload: Vec<_> = (0..len).map(|i| (i * 73 + len) as u8).collect();
                    let clean = framed_symbols(&payload, &config);
                    // Corrupt the length, payload, and each checksum nibble.
                    let offsets = [0, 1, 2, 2 * len].into_iter()
                        .chain(2 * (len + 1)..2 * (len + 5));
                    for offset in offsets {
                        let word = offset / k;
                        let column = offset % k;
                        let slot = 2 * ((word / depth as usize) * depth as usize * CODEWORD_LEN
                            + column * depth as usize + word % depth as usize);
                        let mut damaged = clean.clone();
                        damaged[slot].0 ^= 1;
                        assert_eq!(receive_symbols(&damaged, &config), payload,
                            "parity {parity}, depth {depth}, length {len}, corrupt nibble {offset}");
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_profiles_preserve_slot_synthesis_and_bound_checksum_airtime() {
        for depth in [1usize, 2, 4] {
            let config = StreamConfig { interleave_depth: depth as u8, ..StreamConfig::default() };
            for len in [1usize, 5, 16, 64, 255] {
                let payload: Vec<_> = (0..len).map(|i| (i * 73 + 19) as u8).collect();
                let k = config.data_per_codeword();
                let mut lengths = Vec::new();
                for profile in [StreamProfile::Legacy, StreamProfile::Standard] {
                    let mut tx = StreamTx::with_profile(config.clone(), profile);
                    tx.feed(&payload);
                    tx.encode_codewords();
                    tx.flush_final();
                    let mut vowels = STREAM_PREAMBLE_START.to_vec();
                    vowels.extend_from_slice(&tx.vowel_queue);
                    let words = (2 * (1 + len + profile.checksum_bytes())).div_ceil(k);
                    let expected_slots = 2 * PREAMBLE_LEN + 2 * CODEWORD_LEN * words.div_ceil(depth) * depth;
                    assert_eq!(vowels.len(), expected_slots);
                    // Check the actual audio path against the existing melody,
                    // accents, vibrato, and slot synthesis, including checksum
                    // and cadence slots. No changed pitch schedule is hidden
                    // in the new frame layout.
                    tx.vowel_queue = vowels.clone();
                    let mut slot = vec![0.0; SYMBOL_TOTAL_SAMPLES];
                    for (pos, &vowel) in vowels.iter().enumerate() {
                        assert_eq!(tx.synthesize_into(&mut slot), SYMBOL_TOTAL_SAMPLES);
                        let expected: Vec<_> = formant::synthesize_symbol(
                            vowel, melody_f0(pos), 0.5 * melody_accent(pos), pos,
                        ).into_iter().map(|sample| sample as f32).collect();
                        assert_eq!(slot, expected, "{profile:?}, depth {depth}, length {len}, slot {pos}");
                    }
                    lengths.push(expected_slots);
                }
                // Default k=11: eight CRC nibbles add at most one interleaver
                // block, and sometimes fit in previously transmitted padding.
                assert!(lengths[1] >= lengths[0]);
                assert!(lengths[1] - lengths[0] <= 2 * CODEWORD_LEN * depth);
            }
        }
    }

    /// A deterministic received word that neither hard decoding nor the chosen
    /// erasure fallback can repair. Do not assume every beyond-budget pattern
    /// fails: some patterns are valid RS codewords or miscorrect to one.
    fn failed_word(config: &StreamConfig, weak: bool) -> Vec<u8> {
        let rs = ReedSolomon4::new(config.data_per_codeword(), config.rs_parity as usize);
        let mut seed = 731u32;
        for _ in 0..4096 {
            let word: Vec<_> = (0..CODEWORD_LEN).map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 28) as u8
            }).collect();
            if rs.decode(&word).is_none()
                && (!weak || rs.decode_with_erasures(&word, &[0]).is_none())
            {
                return word;
            }
        }
        panic!("no failed-word fixture for parity {}", config.rs_parity);
    }

    fn replace_codeword(
        symbols: &mut [(usize, f64)], depth: usize, index: usize, word: &[u8], weak: bool,
    ) {
        for (column, &nibble) in word.iter().enumerate() {
            let offset = 2 * ((index / depth) * depth * CODEWORD_LEN + column * depth + index % depth);
            let conf = if weak && column == 0 { 0.0 } else { 1.0 };
            symbols[offset] = ((nibble >> 2) as usize, conf);
            symbols[offset + 1] = ((nibble & 3) as usize, conf);
        }
    }

    fn receive_symbols(symbols: &[(usize, f64)], config: &StreamConfig) -> Vec<u8> {
        receive_symbols_with_profile(symbols, config, StreamProfile::Standard)
    }

    fn receive_symbols_with_profile(
        symbols: &[(usize, f64)], config: &StreamConfig, profile: StreamProfile,
    ) -> Vec<u8> {
        receive_symbols_state(symbols, config, profile).read_all()
    }

    fn receive_symbols_state(
        symbols: &[(usize, f64)], config: &StreamConfig, profile: StreamProfile,
    ) -> StreamRx {
        let mut rx = StreamRx::with_profile(config.clone(), profile);
        for &(vowel, conf) in symbols {
            rx.push_stream_symbol(vowel, conf);
            rx.drain_deinterleaver();
        }
        rx.finalize_output();
        rx
    }

    #[test]
    fn failed_codewords_cannot_shift_the_header_or_payload() {
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            for parity in 1..=14 {
                for depth in [1, 2, 4] {
                    let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                    for len in [1usize, 5, 16, 40, 255] {
                        let payload = vec![0x11; len];
                        let symbols = framed_symbols_with_profile(&payload, &config, profile);
                        assert_eq!(receive_symbols_with_profile(&symbols, &config, profile), payload);
                        let nibbles = 2 * (len + 1 + profile.checksum_bytes());
                        let data_words = nibbles.div_ceil(config.data_per_codeword());
                        let mut damaged_words = vec![0, 1.min(data_words - 1), data_words - 1];
                        damaged_words.sort_unstable();
                        damaged_words.dedup();
                        for index in damaged_words {
                            for weak in [false, true] {
                                // A single parity symbol plus one erasure leaves
                                // no redundancy with which to reject a word.
                                if weak && parity == 1 { continue; }
                                let mut damaged = symbols.clone();
                                let bad = failed_word(&config, weak);
                                replace_codeword(&mut damaged, depth as usize, index, &bad, weak);
                                assert!(receive_symbols_with_profile(&damaged, &config, profile).is_empty(),
                                    "{profile:?}, parity {parity}, depth {depth}, length {len}, failed word {index}, weak {weak}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn failed_padding_codewords_do_not_discard_complete_payload() {
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            for parity in 1..=14 {
                for depth in [1, 2, 4] {
                    let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
                    let bad = failed_word(&config, false);
                    for len in [1usize, 5, 16, 40, 255] {
                        let payload: Vec<_> = (0..len).map(|i| (i * 73 + len) as u8).collect();
                        let symbols = framed_symbols_with_profile(&payload, &config, profile);
                        let nibbles = 2 * (len + 1 + profile.checksum_bytes());
                        let data_words = nibbles.div_ceil(config.data_per_codeword());
                        let total_words = symbols.len() / (2 * CODEWORD_LEN);
                        let mut all_bad_padding = symbols.clone();
                        for index in data_words..total_words {
                            let mut damaged = symbols.clone();
                            replace_codeword(&mut damaged, depth as usize, index, &bad, false);
                            assert_eq!(receive_symbols_with_profile(&damaged, &config, profile), payload,
                                "{profile:?}, parity {parity}, depth {depth}, length {len}, failed padding {index}");
                            replace_codeword(&mut all_bad_padding, depth as usize, index, &bad, false);
                        }
                        assert_eq!(receive_symbols_with_profile(&all_bad_padding, &config, profile), payload,
                            "{profile:?}, parity {parity}, depth {depth}, length {len}, all padding failed");
                    }
                }
            }
        }
    }

    #[test]
    fn quiet_voiced_windows_stay_active_across_the_singing_register() {
        for midi in 47..=66 {
            for detune in [-0.3, 0.0, 0.3] {
                let hz = 440.0 * 2.0f64.powf((midi as f64 + detune - 69.0) / 12.0);
                for vowel in 0..NUM_VOWELS {
                    let slot: Vec<f32> = formant::synthesize_symbol(vowel, hz, 0.001, midi as usize)
                        .into_iter().map(|sample| sample as f32).collect();
                    let mut rx = StreamRx::new(StreamConfig::default());
                    rx.state = RxState::Streaming;
                    rx.rms_ema = 0.1;
                    for _ in 0..END_SILENCE_WINDOWS + 1 { rx.process_window(&slot); }
                    assert!(rx.is_active(), "MIDI {midi}, detune {detune}, vowel {vowel}");
                    assert_eq!(rx.silence_run, 0);
                    assert_eq!(rx.stream_symbol_count, END_SILENCE_WINDOWS + 1,
                        "quiet notes remain data, not delayed erasures");
                }
            }
        }
    }

    #[test]
    fn unvoiced_silence_dc_and_noise_still_end_streams() {
        for kind in 0..3 {
            let mut rx = StreamRx::new(StreamConfig::default());
            rx.state = RxState::Streaming;
            rx.rms_ema = 0.1;
            let mut seed = 137u32;
            for window in 0..END_SILENCE_WINDOWS {
                let slot: Vec<f32> = (0..SYMBOL_TOTAL_SAMPLES).map(|_| match kind {
                    0 => 0.0,
                    1 => 0.001,
                    _ => {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        0.001 + 0.002 * (seed as f32 / u32::MAX as f32 - 0.5)
                    }
                }).collect();
                rx.process_window(&slot);
                assert_eq!(rx.is_active(), window + 1 < END_SILENCE_WINDOWS);
                assert!(rx.read_all().is_empty());
            }
        }
    }

    #[test]
    fn nonfinite_input_does_not_poison_listening_noise_floor() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut rx = StreamRx::new(StreamConfig::default());
            let mut idle = vec![0.0; SYMBOL_TOTAL_SAMPLES];
            idle[SAMPLES_PER_SYMBOL / 2] = invalid;
            rx.ingest(&idle);
            assert_eq!(rx.noise_floor, 0.0, "noise floor poisoned by {invalid}");
            assert!(!rx.is_active());
            assert!(rx.read_all().is_empty());
            rx.ingest(&vec![0.0; 8 * SYMBOL_TOTAL_SAMPLES]);
            assert_eq!(rx.noise_floor, 0.0);
        }
    }

    #[test]
    fn test_interleaver_depth1_passthrough() {
        let mut il = Interleaver::new(1);
        // Depth 1: each block is 1*15 = 15 symbols, passthrough
        let data: Vec<u8> = (0..15).collect();
        for &sym in &data {
            il.push(sym);
        }
        let mut out = Vec::new();
        while let Some(s) = il.pop() {
            out.push(s);
        }
        assert_eq!(out, data);
    }

    #[test]
    fn test_interleaver_deinterleaver_roundtrip() {
        for depth in [1, 2, 3] {
            let mut il = Interleaver::new(depth);
            let mut dil = Deinterleaver::new(depth);

            // N codewords worth of data (exactly fills interleaver blocks)
            let n_codewords = depth * 2; // 2 full blocks
            let data: Vec<u8> = (0..(n_codewords * CODEWORD_LEN))
                .map(|i| (i % 16) as u8) // GF(2^4) symbols are 0..15
                .collect();

            // Interleave
            for &sym in &data {
                il.push(sym);
            }
            let mut interleaved = Vec::new();
            while let Some(s) = il.pop() {
                interleaved.push(s);
            }
            assert_eq!(
                interleaved.len(),
                data.len(),
                "depth={depth}: interleaved length mismatch"
            );

            // Deinterleave
            for &sym in &interleaved {
                dil.push(sym, 1.0);
            }
            let mut deinterleaved = Vec::new();
            while let Some((s, _)) = dil.pop() {
                deinterleaved.push(s);
            }
            assert_eq!(
                deinterleaved, data,
                "depth={depth}: roundtrip failed"
            );
        }
    }

    #[test]
    fn test_interleaver_burst_spread() {
        // With depth=2, a burst of errors in consecutive positions in the
        // interleaved stream should be spread across 2 codewords.
        let depth = 2;
        let mut il = Interleaver::new(depth);
        let mut dil = Deinterleaver::new(depth);

        // 2 codewords * 15 symbols = 30 symbols
        let data: Vec<u8> = (0..30).map(|i| (i % 16) as u8).collect();
        for &sym in &data {
            il.push(sym);
        }
        let mut interleaved = Vec::new();
        while let Some(s) = il.pop() {
            interleaved.push(s);
        }

        // Corrupt 2 consecutive symbols in the interleaved stream
        for i in 2..4 {
            interleaved[i] = 15; // arbitrary corruption
        }

        for &sym in &interleaved {
            dil.push(sym, 1.0);
        }
        let mut deinterleaved = Vec::new();
        while let Some((s, _)) = dil.pop() {
            deinterleaved.push(s);
        }

        // After deinterleaving, the 2 corrupted symbols should be spread
        // across 2 codewords (1 error each), not concentrated in one
        let cw1 = &deinterleaved[0..15];
        let cw2 = &deinterleaved[15..30];
        let orig_cw1 = &data[0..15];
        let orig_cw2 = &data[15..30];
        let diff1: usize = cw1.iter().zip(orig_cw1).filter(|(a, b)| a != b).count();
        let diff2: usize = cw2.iter().zip(orig_cw2).filter(|(a, b)| a != b).count();
        assert_eq!(diff1 + diff2, 2, "total errors should be 2");
        assert!(diff1 <= 1 && diff2 <= 1, "errors should be spread: cw1={diff1}, cw2={diff2}");
    }

    #[test]
    fn test_stream_config_default() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.data_per_codeword(), 11); // 15 - 4 parity = 11
        assert_eq!(cfg.rs_parity, 4);
        assert_eq!(cfg.interleave_depth, 2);
    }
}
