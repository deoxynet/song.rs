//! Streaming FEC codec: continuous RS-coded symbol stream with block interleaving.
//!
//! Eliminates per-frame preamble overhead and ACK round-trips. Data flows as a
//! continuous RS(15,k) coded stream over GF(2^4), where each GF(2^4) symbol maps
//! to exactly two 2-bit modem symbols (vowels). No bit straddling anywhere:
//! byte = 2 nibbles, nibble = 2 vowels.
//!
//! Pitch carries no data — the transmitter plays the melody (protocol::MELODY)
//! across the stream, and the receiver's vowel detection is F0-agnostic.
//!
//! ```text
//! Send: bytes -> nibbles -> RS(15,k) encode -> interleave -> 2 vowels/nibble -> audio
//! Recv: audio -> preamble detect -> classify vowels -> pair -> deinterleave -> RS decode -> bytes
//! ```

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::rs4::ReedSolomon4;

/// Codeword length for GF(2^4): n = 2^4 - 1 = 15.
const CODEWORD_LEN: usize = 15;

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

/// Consecutive near-silent windows that end the stream on the receiver.
/// 4 windows = 240 ms: longer than any tolerated mid-stream dropout,
/// shorter than the transmitted trailing silence.
const END_SILENCE_WINDOWS: usize = 4;

/// A window is "silent" if its RMS falls below this fraction of the running
/// signal RMS average (~ -10 dB). Sits between the noise floor at 20 dB SNR
/// (0.1×) and voiced level (1×).
const SILENCE_RMS_RATIO: f64 = 0.3;

/// Confidence threshold for erasure marking: symbols below this are erasure
/// candidates (weakest-first, capped at the parity budget) when hard-decision
/// RS decoding fails.
const ERASURE_THRESHOLD: f64 = 1.0;

/// Configuration for the streaming codec.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Parity symbols per RS codeword (default: 4 → corrects 2 symbol errors).
    /// Data symbols = 15 - parity.
    pub rs_parity: u8,
    /// Block interleaver depth (default: 2). 1 = no interleaving.
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
/// Usage: call `feed()` with payload bytes (length prefix is added automatically),
/// call `finish()` when done, then call `emit()` repeatedly to get audio samples.
pub struct StreamTx {
    config: StreamConfig,
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
    pub fn new(config: StreamConfig) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon4::new(data_len, config.rs_parity as usize);
        let interleaver = Interleaver::new(config.interleave_depth as usize);
        let send_volume = config.volume as f64 / 100.0;
        Self {
            config,
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
    pub fn feed_raw(&mut self, bytes: &[u8]) {
        // Each byte is exactly two GF(2^4) symbols (nibbles), MSB first.
        for &b in bytes {
            self.nibble_buf.push(b >> 4);
            self.nibble_buf.push(b & 0x0F);
        }
    }

    /// Queue data bytes with an automatic length prefix.
    /// The length byte is prepended so the receiver knows how many payload bytes to expect.
    pub fn feed(&mut self, payload: &[u8]) {
        assert!(payload.len() <= 255, "stream payload max 255 bytes");
        // Length prefix: 1 byte
        let len = payload.len() as u8;
        self.feed_raw(&[len]);
        // Payload
        self.feed_raw(payload);
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

/// Streaming receiver: ingests audio samples, outputs decoded bytes.
pub struct StreamRx {
    config: StreamConfig,
    rs: ReedSolomon4,
    deinterleaver: Deinterleaver,
    sample_buf: Vec<f32>,
    recent_vowels: Vec<usize>,
    /// Centroids parallel to recent_vowels, for structural preamble
    /// matching under channel tilt.
    recent_centroids: Vec<f64>,
    /// Window RMS parallel to recent_vowels, for gating fuzzy matches.
    recent_rms: Vec<f64>,
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
    output_buf: Vec<u8>,
    /// Decoded GF(2^4) symbols (nibbles) from RS decoding.
    decoded_nibbles: Vec<u8>,
    stream_symbol_count: usize,
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
    pub fn new(config: StreamConfig) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon4::new(data_len, config.rs_parity as usize);
        let deinterleaver = Deinterleaver::new(config.interleave_depth as usize);
        Self {
            config,
            rs,
            deinterleaver,
            sample_buf: Vec::new(),
            recent_vowels: Vec::new(),
            recent_centroids: Vec::new(),
            recent_rms: Vec::new(),
            noise_floor: 0.0,
            listen_windows: 0,
            state: RxState::Listening,
            pending_half: None,
            codeword_buf: Vec::new(),
            codeword_conf: Vec::new(),
            output_buf: Vec::new(),
            decoded_nibbles: Vec::new(),
            stream_symbol_count: 0,
            rms_ema: 0.0,
            silence_run: 0,
        }
    }

    /// Feed audio samples into the receiver.
    pub fn ingest(&mut self, samples: &[f32]) {
        self.sample_buf.extend_from_slice(samples);
        self.process_samples();
    }

    /// Read decoded bytes. Returns number of bytes written to `out`.
    pub fn read(&mut self, out: &mut [u8]) -> usize {
        let n = self.output_buf.len().min(out.len());
        out[..n].copy_from_slice(&self.output_buf[..n]);
        self.output_buf.drain(..n);
        n
    }

    /// Read all available decoded bytes.
    pub fn read_all(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output_buf)
    }

    /// Returns true if the receiver is currently in the streaming state.
    pub fn is_active(&self) -> bool {
        self.state == RxState::Streaming
    }

    fn process_samples(&mut self) {
        while self.sample_buf.len() >= SYMBOL_TOTAL_SAMPLES {
            let window: Vec<f32> = self.sample_buf[..SYMBOL_TOTAL_SAMPLES].to_vec();
            self.sample_buf.drain(..SYMBOL_TOTAL_SAMPLES);
            self.process_window(&window);
        }
    }

    fn process_window(&mut self, window: &[f32]) {
        match self.state {
            RxState::Listening => {
                let (vowel, centroid, rms) = measure_window(window);
                self.listen_windows += 1;
                // Slow-rising, fast-falling ambient floor: quiet windows pull
                // it down immediately, loud ones lift it only gradually, so
                // a transmission's own preamble can't inflate it.
                if self.noise_floor == 0.0 || rms < self.noise_floor {
                    self.noise_floor = rms;
                } else {
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms;
                }
                self.recent_vowels.push(vowel);
                self.recent_centroids.push(centroid);
                self.recent_rms.push(rms);
                if self.recent_vowels.len() > PREAMBLE_LEN {
                    self.recent_vowels
                        .drain(..self.recent_vowels.len() - PREAMBLE_LEN);
                    self.recent_centroids
                        .drain(..self.recent_centroids.len() - PREAMBLE_LEN);
                    self.recent_rms
                        .drain(..self.recent_rms.len() - PREAMBLE_LEN);
                }

                // Exact preamble match, or a gated fuzzy match: a single
                // corrupted preamble symbol must not cost the entire stream.
                // Fuzzy paths (3-of-4 vowels, or the structural lo-hi-lo-hi
                // centroid pattern under channel tilt) additionally require
                // every window to stand clear of the ambient floor so idle
                // noise cannot false-lock the receiver.
                let full = self.recent_vowels.len() == PREAMBLE_LEN;
                let exact = full && self.recent_vowels[..] == STREAM_PREAMBLE_START[..];
                let fuzzy = full && !exact && {
                    // The ambient floor is only trustworthy once some
                    // pre-stream audio established it; on a cold start the
                    // preamble itself would have set it.
                    let floor_known = self.listen_windows >= PREAMBLE_LEN + 8;
                    let clear_of_floor = self.recent_rms.iter().all(|&r| {
                        if floor_known {
                            r > (3.0 * self.noise_floor).max(1e-6)
                        } else {
                            r > 1e-4
                        }
                    });
                    let near = self
                        .recent_vowels
                        .iter()
                        .zip(&STREAM_PREAMBLE_START)
                        .filter(|(a, b)| a == b)
                        .count()
                        >= PREAMBLE_LEN - 1;
                    let structural =
                        crate::decoder::structural_start_match(&self.recent_centroids);
                    clear_of_floor && (near || structural)
                };
                if exact || fuzzy {
                    self.state = RxState::Streaming;
                    self.recent_vowels.clear();
                    self.recent_centroids.clear();
                    self.recent_rms.clear();
                    self.pending_half = None;
                    self.codeword_buf.clear();
                    self.codeword_conf.clear();
                    self.decoded_nibbles.clear();
                    self.output_buf.clear();
                    self.stream_symbol_count = 0;
                    self.rms_ema = 0.0;
                    self.silence_run = 0;
                    self.deinterleaver =
                        Deinterleaver::new(self.config.interleave_depth as usize);
                }
            }
            RxState::Streaming => {
                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];

                // End-of-stream detection: a run of near-silent windows.
                // (Data can alias any vowel pattern, but not silence.)
                let rms = {
                    let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
                    (sum_sq / voiced.len() as f64).sqrt()
                };
                if self.rms_ema > 0.0 && rms < SILENCE_RMS_RATIO * self.rms_ema {
                    // Don't feed this window yet: if voice resumes it becomes an
                    // erasure (keeping stream positions aligned); if the run
                    // completes it was the trailing silence and is dropped.
                    self.silence_run += 1;
                    if self.silence_run >= END_SILENCE_WINDOWS {
                        self.deinterleaver.flush();
                        self.drain_deinterleaver();
                        self.state = RxState::Listening;
                        self.recent_vowels.clear();
                        self.pending_half = None;
                        self.silence_run = 0;
                        self.finalize_output();
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
            self.decoded_nibbles.extend_from_slice(&data);
        }
    }

    /// Reassemble decoded nibbles into bytes using the length prefix.
    fn finalize_output(&mut self) {
        // Two nibbles per byte, MSB first — exact, no bit accumulation needed.
        let bytes: Vec<u8> = self
            .decoded_nibbles
            .chunks_exact(2)
            .map(|pair| (pair[0] << 4) | (pair[1] & 0x0F))
            .collect();

        if bytes.is_empty() {
            return;
        }

        // First byte is length prefix
        let len_byte = bytes[0] as usize;
        if len_byte == 0 || len_byte > bytes.len() - 1 {
            return;
        }

        self.output_buf.clear();
        self.output_buf.extend_from_slice(&bytes[1..1 + len_byte]);
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
    formant::classify_symbol(&spectrum, &formant::EXPECTED_F2_CENTROIDS)
}

#[cfg(test)]
mod tests {
    use super::*;

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
