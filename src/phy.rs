//! PHY Layer: Stop-and-Wait ARQ with fragmentation over the vocal modem.
//!
//! Provides reliable, acknowledged delivery of arbitrarily-sized messages
//! (up to 2,224 bytes) over the acoustic channel. Pure state machine —
//! no threads, no timers, no I/O. WASM-compatible by design.

use std::collections::VecDeque;

use crate::decoder::Decoder;
use crate::encoder;

// --- Header byte constants ---

const TYPE_MASK: u8 = 0xC0;
const TYPE_DATA: u8 = 0x00;
const TYPE_ACK: u8 = 0x40;
const TYPE_NAK: u8 = 0x80;

const MSG_SEQ_BIT: u8 = 0x20;
const MORE_BIT: u8 = 0x10;
const FRAG_MASK: u8 = 0x0F;

/// Maximum payload per fragment (140-byte codec max minus 1 header byte).
const MAX_FRAG_PAYLOAD: usize = 139;

/// Maximum number of fragments (4-bit index).
const MAX_FRAGMENTS: usize = 16;

/// Maximum message size: 16 * 139 = 2,224 bytes.
pub const MAX_MESSAGE_SIZE: usize = MAX_FRAGMENTS * MAX_FRAG_PAYLOAD;

/// Samples per millisecond.
#[cfg(not(feature = "8khz"))]
const SAMPLES_PER_MS: u64 = 48;
#[cfg(feature = "8khz")]
const SAMPLES_PER_MS: u64 = 8;

// --- Public types ---

/// Events produced by the PHY layer.
#[derive(Debug, Clone)]
pub enum PhyEvent {
    /// Complete reassembled message received.
    Received(Vec<u8>),
    /// All fragments of our message were ACKed.
    SendComplete,
    /// Send failed after max retries on a fragment.
    SendFailed,
}

/// Errors from PHY send operations.
#[derive(Debug, Clone)]
pub enum PhyError {
    /// Already transmitting a message.
    Busy,
    /// Message too large (> 2224 bytes).
    MessageTooLarge,
}

impl std::fmt::Display for PhyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhyError::Busy => write!(f, "already transmitting a message"),
            PhyError::MessageTooLarge => {
                write!(f, "message too large (max {} bytes)", MAX_MESSAGE_SIZE)
            }
        }
    }
}

impl std::error::Error for PhyError {}

/// PHY layer configuration.
pub struct PhyConfig {
    /// ACK timeout after DATA transmission, in milliseconds (default: 6000).
    /// Includes the ACK's 3.84 s airtime; leave room for round-trip channel latency.
    pub timeout_ms: u32,
    /// Maximum retransmission attempts per fragment (default: 5).
    pub max_retries: u8,
    /// Audio volume 0..=100 (default: 50).
    pub volume: u8,
}

impl Default for PhyConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 6000,
            max_retries: 5,
            volume: 50,
        }
    }
}

// --- State machine ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhyState {
    /// Listening for incoming frames; ready to send.
    Idle,
    /// Transmitting a DATA frame (decoder muted).
    TxData,
    /// Waiting for ACK/NAK after sending DATA.
    WaitAck,
    /// Transmitting an ACK or NAK frame (decoder muted).
    TxAck,
}

/// Reliable PHY layer over the vocal modem.
///
/// Wraps `Encoder` + `Decoder` with stop-and-wait ARQ and fragmentation.
/// Pure state machine: no threads, no timers, no I/O.
pub struct Phy {
    state: PhyState,
    config: PhyConfig,
    decoder: Decoder,
    // TX
    tx_buf: Vec<f32>,
    tx_pos: usize,
    tx_msg_seq: u8,
    tx_fragments: Vec<Vec<u8>>,
    tx_frag_index: u8,
    retries: u8,
    // RX reassembly
    rx_buf: Vec<u8>,
    rx_msg_seq: Option<u8>,
    rx_expected_frag: Option<u8>,
    last_rx_id: Option<(u8, u8)>,
    // Timing (sample-based clock, advanced in ingest only)
    sample_clock: u64,
    timeout_at: u64,
    // Event queue
    event_queue: VecDeque<PhyEvent>,
    // ACK queuing for when we receive DATA during WaitAck
    pending_ack: Option<Vec<f32>>,
    return_to_wait_ack: bool,
}

impl Phy {
    /// Create a new PHY instance with the given configuration.
    pub fn new(config: PhyConfig) -> Self {
        Self {
            state: PhyState::Idle,
            config,
            decoder: Decoder::new(),
            tx_buf: Vec::new(),
            tx_pos: 0,
            tx_msg_seq: 0,
            tx_fragments: Vec::new(),
            tx_frag_index: 0,
            retries: 0,
            rx_buf: Vec::new(),
            rx_msg_seq: None,
            rx_expected_frag: None,
            last_rx_id: None,
            sample_clock: 0,
            timeout_at: 0,
            event_queue: VecDeque::new(),
            pending_ack: None,
            return_to_wait_ack: false,
        }
    }

    /// Queue a message for transmission. Fragments automatically.
    pub fn send(&mut self, data: &[u8]) -> Result<(), PhyError> {
        if self.state != PhyState::Idle {
            return Err(PhyError::Busy);
        }
        if data.len() > MAX_MESSAGE_SIZE {
            return Err(PhyError::MessageTooLarge);
        }

        // Fragment the message
        let num_frags = if data.is_empty() {
            1
        } else {
            (data.len() + MAX_FRAG_PAYLOAD - 1) / MAX_FRAG_PAYLOAD
        };

        self.tx_fragments.clear();
        for i in 0..num_frags {
            let start = i * MAX_FRAG_PAYLOAD;
            let end = (start + MAX_FRAG_PAYLOAD).min(data.len());
            self.tx_fragments.push(data[start..end].to_vec());
        }

        // Toggle message sequence
        self.tx_msg_seq ^= 1;
        self.tx_frag_index = 0;
        self.retries = 0;

        self.encode_current_fragment();
        self.state = PhyState::TxData;
        Ok(())
    }

    /// Feed incoming mic samples to the decoder.
    pub fn ingest(&mut self, samples: &[f32]) {
        self.sample_clock += samples.len() as u64;

        match self.state {
            PhyState::TxData | PhyState::TxAck => {
                // Muted: discard to avoid self-decode
            }
            PhyState::Idle | PhyState::WaitAck => {
                if let Ok(Some(payload)) = self.decoder.decode(samples) {
                    self.handle_rx_frame(&payload);
                }
                if self.state == PhyState::WaitAck && self.sample_clock >= self.timeout_at {
                    self.handle_timeout();
                }
            }
        }
    }

    /// Fill outgoing speaker buffer. Returns number of samples written.
    pub fn emit(&mut self, out: &mut [f32]) -> usize {
        match self.state {
            PhyState::TxData | PhyState::TxAck => {
                let remaining = self.tx_buf.len() - self.tx_pos;
                let n = remaining.min(out.len());
                out[..n].copy_from_slice(&self.tx_buf[self.tx_pos..self.tx_pos + n]);
                self.tx_pos += n;

                // Fill remainder with silence
                for s in &mut out[n..] {
                    *s = 0.0;
                }

                if self.tx_pos >= self.tx_buf.len() {
                    self.finish_tx();
                }

                out.len()
            }
            PhyState::Idle | PhyState::WaitAck => {
                for s in out.iter_mut() {
                    *s = 0.0;
                }
                out.len()
            }
        }
    }

    /// Drain next event.
    pub fn poll(&mut self) -> Option<PhyEvent> {
        self.event_queue.pop_front()
    }

    /// Best-effort preview of the frame currently being received.
    /// Uncorrected bytes — may contain errors. Returns None if idle.
    pub fn preview(&self) -> Option<Vec<u8>> {
        let raw = self.decoder.preview_bytes()?;
        // Need RS length header (3 bytes) + PHY header (1 byte)
        if raw.len() < 4 {
            return None;
        }
        let header = raw[3];
        if header & TYPE_MASK != TYPE_DATA {
            return None;
        }
        let preview_data = &raw[4..];
        // For multi-fragment messages, prepend previously-received fragments
        if !self.rx_buf.is_empty() {
            let mut full = self.rx_buf.clone();
            full.extend_from_slice(preview_data);
            Some(full)
        } else {
            Some(preview_data.to_vec())
        }
    }

    /// True if send() would succeed (not currently transmitting).
    pub fn is_idle(&self) -> bool {
        self.state == PhyState::Idle
    }

    // --- Internal helpers ---

    fn finish_tx(&mut self) {
        match self.state {
            PhyState::TxData => {
                self.state = PhyState::WaitAck;
                self.timeout_at =
                    self.sample_clock + self.config.timeout_ms as u64 * SAMPLES_PER_MS;
                self.decoder.reset();
            }
            PhyState::TxAck => {
                self.decoder.reset();
                if self.return_to_wait_ack {
                    self.return_to_wait_ack = false;
                    self.state = PhyState::WaitAck;
                    self.timeout_at =
                        self.sample_clock + self.config.timeout_ms as u64 * SAMPLES_PER_MS;
                } else {
                    self.state = PhyState::Idle;
                    // Drain any pending ACK
                    if let Some(buf) = self.pending_ack.take() {
                        self.tx_buf = buf;
                        self.tx_pos = 0;
                        self.state = PhyState::TxAck;
                    }
                }
            }
            _ => {}
        }
    }

    fn encode_current_fragment(&mut self) {
        let idx = self.tx_frag_index as usize;
        let is_last = idx == self.tx_fragments.len() - 1;
        let more = if is_last { 0u8 } else { 1u8 };

        let header = TYPE_DATA
            | (self.tx_msg_seq << 5)
            | (more << 4)
            | (self.tx_frag_index & FRAG_MASK);

        let mut payload = Vec::with_capacity(1 + self.tx_fragments[idx].len());
        payload.push(header);
        payload.extend_from_slice(&self.tx_fragments[idx]);

        let volume = self.config.volume.min(100);
        self.tx_buf =
            encoder::encode(&payload, volume).expect("fragment encode failed");
        self.tx_pos = 0;
    }

    fn encode_ack(&self, msg_seq: u8, frag_index: u8) -> Vec<f32> {
        let header = TYPE_ACK | (msg_seq << 5) | (frag_index & FRAG_MASK);
        let volume = self.config.volume.min(100);
        encoder::encode(&[header], volume).expect("ACK encode failed")
    }

    fn handle_rx_frame(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }

        let header = payload[0];
        let frame_type = header & TYPE_MASK;
        let msg_seq = (header & MSG_SEQ_BIT) >> 5;
        let more = (header & MORE_BIT) != 0;
        let frag_index = header & FRAG_MASK;
        let data = &payload[1..];

        match frame_type {
            TYPE_DATA => self.handle_rx_data(msg_seq, more, frag_index, data),
            TYPE_ACK => self.handle_rx_ack(msg_seq, frag_index),
            TYPE_NAK => self.handle_rx_nak(),
            _ => {} // reserved, ignore
        }
    }

    fn handle_rx_data(&mut self, msg_seq: u8, more: bool, frag_index: u8, data: &[u8]) {
        // Duplicate check
        if self.last_rx_id == Some((msg_seq, frag_index)) {
            self.queue_ack(msg_seq, frag_index);
            return;
        }

        // Start new reassembly only on fragment 0
        if frag_index == 0 && self.rx_msg_seq != Some(msg_seq) {
            self.rx_buf.clear();
            self.rx_msg_seq = Some(msg_seq);
            self.rx_expected_frag = Some(0);
        }

        // If no active message, only accept fragment 0
        if self.rx_expected_frag.is_none() {
            if frag_index != 0 {
                return;
            }
            self.rx_buf.clear();
            self.rx_msg_seq = Some(msg_seq);
            self.rx_expected_frag = Some(0);
        }

        // Reject mismatched message sequence
        if self.rx_msg_seq != Some(msg_seq) {
            return;
        }

        let expected = self.rx_expected_frag.unwrap_or(0);
        if frag_index != expected {
            // Out-of-order or unexpected fragment — ignore
            return;
        }

        self.rx_buf.extend_from_slice(data);
        self.last_rx_id = Some((msg_seq, frag_index));

        // ACK this fragment
        self.queue_ack(msg_seq, frag_index);

        // Check if message is complete
        if !more {
            let msg = std::mem::take(&mut self.rx_buf);
            self.rx_msg_seq = None;
            self.rx_expected_frag = None;
            self.event_queue.push_back(PhyEvent::Received(msg));
        } else {
            let next = expected.saturating_add(1);
            if next as usize >= MAX_FRAGMENTS {
                // Invalid fragment sequence length — reset reassembly.
                self.rx_buf.clear();
                self.rx_msg_seq = None;
                self.rx_expected_frag = None;
                return;
            }
            self.rx_expected_frag = Some(next);
        }
    }

    fn handle_rx_ack(&mut self, msg_seq: u8, frag_index: u8) {
        if self.state != PhyState::WaitAck {
            return;
        }
        if msg_seq != self.tx_msg_seq || frag_index != self.tx_frag_index {
            return;
        }

        let is_last = self.tx_frag_index as usize == self.tx_fragments.len() - 1;
        if is_last {
            self.state = PhyState::Idle;
            self.tx_fragments.clear();
            self.event_queue.push_back(PhyEvent::SendComplete);
        } else {
            self.tx_frag_index += 1;
            self.retries = 0;
            self.encode_current_fragment();
            self.state = PhyState::TxData;
        }
    }

    fn handle_rx_nak(&mut self) {
        if self.state != PhyState::WaitAck {
            return;
        }
        self.retransmit();
    }

    fn handle_timeout(&mut self) {
        if self.retries >= self.config.max_retries {
            self.state = PhyState::Idle;
            self.tx_fragments.clear();
            self.event_queue.push_back(PhyEvent::SendFailed);
        } else {
            self.retransmit();
        }
    }

    fn retransmit(&mut self) {
        self.retries += 1;
        self.encode_current_fragment();
        self.state = PhyState::TxData;
        self.decoder.reset();
    }

    fn queue_ack(&mut self, msg_seq: u8, frag_index: u8) {
        let ack_audio = self.encode_ack(msg_seq, frag_index);

        match self.state {
            PhyState::Idle => {
                self.tx_buf = ack_audio;
                self.tx_pos = 0;
                self.state = PhyState::TxAck;
                self.return_to_wait_ack = false;
            }
            PhyState::WaitAck => {
                // Received DATA while waiting for our ACK — send ACK, then resume waiting
                self.tx_buf = ack_audio;
                self.tx_pos = 0;
                self.state = PhyState::TxAck;
                self.return_to_wait_ack = true;
            }
            PhyState::TxAck => {
                // Already sending an ACK — queue this one
                self.pending_ack = Some(ack_audio);
            }
            PhyState::TxData => {
                // Shouldn't happen (muted), but queue anyway
                self.pending_ack = Some(ack_audio);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_out_of_order_fragment_ignored() {
        let mut phy = Phy::new(PhyConfig::default());

        // Fragment 1 arrives before fragment 0: should be ignored.
        phy.handle_rx_data(0, true, 1, b"bbb");
        assert!(phy.rx_buf.is_empty());
        assert!(phy.event_queue.is_empty());
        assert!(phy.rx_msg_seq.is_none());
        assert!(phy.rx_expected_frag.is_none());

        // Now send a proper in-order two-fragment message.
        phy.handle_rx_data(0, true, 0, b"aaa");
        assert_eq!(phy.rx_buf.as_slice(), b"aaa");
        assert_eq!(phy.rx_expected_frag, Some(1));

        phy.handle_rx_data(0, false, 1, b"bbb");
        match phy.poll() {
            Some(PhyEvent::Received(data)) => assert_eq!(data, b"aaabbb"),
            other => panic!("expected Received event, got {:?}", other),
        }
    }
}
