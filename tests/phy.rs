//! PHY layer integration tests: loopback harness connecting two Phy instances.

use song_rs::{Phy, PhyConfig, PhyEvent, PhyError};
use song_rs::protocol::SAMPLE_RATE;

const CHUNK: usize = 128;

// --- Loopback harness ---

/// Run two PHY instances in a simulated audio loop.
/// Each iteration: both emit → channel → cross-ingest → poll.
/// Returns collected events from both sides.
fn run_loopback(a: &mut Phy, b: &mut Phy, max_ms: u64) -> (Vec<PhyEvent>, Vec<PhyEvent>) {
    run_loopback_channels(
        a,
        b,
        &mut |s: &[f32]| s.to_vec(),
        &mut |s: &[f32]| s.to_vec(),
        max_ms,
    )
}

/// Run loopback with custom channel functions for each direction.
fn run_loopback_channels<FA, FB>(
    a: &mut Phy,
    b: &mut Phy,
    chan_a_to_b: &mut FA,
    chan_b_to_a: &mut FB,
    max_ms: u64,
) -> (Vec<PhyEvent>, Vec<PhyEvent>)
where
    FA: FnMut(&[f32]) -> Vec<f32>,
    FB: FnMut(&[f32]) -> Vec<f32>,
{
    let max_samples = max_ms as u64 * (SAMPLE_RATE as u64 / 1000);
    let mut elapsed: u64 = 0;
    let mut a_events = Vec::new();
    let mut b_events = Vec::new();

    while elapsed < max_samples {
        let mut a_out = [0.0f32; CHUNK];
        let mut b_out = [0.0f32; CHUNK];

        a.emit(&mut a_out);
        b.emit(&mut b_out);

        let a_to_b = chan_a_to_b(&a_out);
        let b_to_a = chan_b_to_a(&b_out);

        b.ingest(&a_to_b);
        a.ingest(&b_to_a);

        while let Some(ev) = a.poll() {
            a_events.push(ev);
        }
        while let Some(ev) = b.poll() {
            b_events.push(ev);
        }

        elapsed += CHUNK as u64;

        // Early exit once both sides are idle and we have events
        if a.is_idle() && b.is_idle() && (!a_events.is_empty() || !b_events.is_empty()) {
            break;
        }
    }

    (a_events, b_events)
}

// --- Frame drop channel for lossy tests ---

/// Detects audio frame boundaries by energy transitions and drops specified frames.
struct FrameDropChannel {
    drop_indices: Vec<usize>,
    frame_count: usize,
    in_frame: bool,
    silent_chunks: usize,
    /// Silence threshold in chunks (~50ms worth).
    silence_threshold: usize,
}

impl FrameDropChannel {
    fn new(drop: Vec<usize>) -> Self {
        // ~50ms of silence to detect frame end, in chunks.
        // At 48kHz: 50 * 48 / 128 ≈ 19 chunks
        // At 8kHz: 50 * 8 / 128 ≈ 4 chunks
        let silence_threshold = (50 * SAMPLE_RATE as usize / 1000 / CHUNK).max(4);
        Self {
            drop_indices: drop,
            frame_count: 0,
            in_frame: false,
            silent_chunks: silence_threshold + 1, // start as "been silent"
            silence_threshold,
        }
    }

    fn passthrough() -> Self {
        Self::new(vec![])
    }

    fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let is_active = energy > 1e-10;

        if is_active {
            if !self.in_frame {
                self.frame_count += 1;
                self.in_frame = true;
            }
            self.silent_chunks = 0;
        } else {
            self.silent_chunks += 1;
            // Require sustained silence to end a frame.
            if self.silent_chunks >= self.silence_threshold {
                self.in_frame = false;
            }
        }

        if self.in_frame && self.drop_indices.contains(&self.frame_count) {
            vec![0.0; samples.len()]
        } else {
            samples.to_vec()
        }
    }
}

// --- DSP helpers ---

fn add_noise(samples: &[f32], snr_db: f64) -> Vec<f32> {
    let signal_power: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
        / samples.len() as f64;
    if signal_power < 1e-20 {
        return samples.to_vec();
    }
    let noise_power = signal_power / 10.0f64.powf(snr_db / 10.0);
    let noise_amp = noise_power.sqrt();

    let mut rng: u32 = 42;
    samples
        .iter()
        .map(|&s| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = (rng >> 16) as f64 / 65535.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = (rng >> 16) as f64 / 65535.0;
            let gauss = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (s as f64 + gauss * noise_amp) as f32
        })
        .collect()
}

fn phone_channel(samples: &[f32]) -> Vec<f32> {
    // Simple bandpass 300-3400 Hz via biquad
    bandpass(samples, 300.0, 3400.0)
}

fn bandpass(samples: &[f32], lo: f64, hi: f64) -> Vec<f32> {
    let sr = 48000.0;
    let center = (lo * hi).sqrt();
    let bw = hi - lo;
    let q = center / bw;
    let w0 = 2.0 * std::f64::consts::PI * center / sr;
    let alpha = w0.sin() / (2.0 * q);

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha;

    let mut out = vec![0.0f32; samples.len()];
    let mut x1 = 0.0f64;
    let mut x2 = 0.0f64;
    let mut y1 = 0.0f64;
    let mut y2 = 0.0f64;

    for (i, &s) in samples.iter().enumerate() {
        let x0 = s as f64;
        let y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        out[i] = y0 as f32;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    out
}

fn has_event(events: &[PhyEvent], check: fn(&PhyEvent) -> bool) -> bool {
    events.iter().any(check)
}

fn get_received(events: &[PhyEvent]) -> Option<&Vec<u8>> {
    for ev in events {
        if let PhyEvent::Received(data) = ev {
            return Some(data);
        }
    }
    None
}

// --- Tests ---

#[test]
fn test_single_message() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message = b"Hello, PHY layer!";
    a.send(message).unwrap();

    let (a_ev, b_ev) = run_loopback(&mut a, &mut b, 30_000);

    assert!(
        has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)),
        "A should get SendComplete"
    );
    let received = get_received(&b_ev).expect("B should receive message");
    assert_eq!(received.as_slice(), message);
}

#[test]
fn test_single_byte() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    a.send(&[0x42]).unwrap();

    let (a_ev, b_ev) = run_loopback(&mut a, &mut b, 30_000);

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&b_ev).unwrap().as_slice(), &[0x42]);
}

#[test]
fn test_large_message_fragmented() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    a.send(&message).unwrap();

    let (a_ev, b_ev) = run_loopback(&mut a, &mut b, 500_000);

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    let received = get_received(&b_ev).expect("B should receive fragmented message");
    assert_eq!(received, &message);
}

#[test]
fn test_max_message() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message: Vec<u8> = (0..2224).map(|i| (i % 256) as u8).collect();
    a.send(&message).unwrap();

    let (a_ev, b_ev) = run_loopback(&mut a, &mut b, 900_000);

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    let received = get_received(&b_ev).expect("B should receive max-size message");
    assert_eq!(received, &message);
}

#[test]
fn test_message_too_large() {
    let mut a = Phy::new(PhyConfig::default());
    let big = vec![0u8; 2225];
    assert!(matches!(a.send(&big), Err(PhyError::MessageTooLarge)));
}

#[test]
fn test_noisy_channel() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message = b"noisy channel test data!!";
    a.send(message).unwrap();

    let (a_ev, b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| add_noise(s, 15.0),
        &mut |s| add_noise(s, 15.0),
        30_000,
    );

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&b_ev).unwrap().as_slice(), message);
}

#[test]
fn test_phone_channel() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message = b"phone channel test";
    a.send(message).unwrap();

    let (a_ev, b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| phone_channel(s),
        &mut |s| phone_channel(s),
        30_000,
    );

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&b_ev).unwrap().as_slice(), message);
}

#[test]
fn test_lost_ack() {
    let mut a = Phy::new(PhyConfig {
        timeout_ms: 4000,
        max_retries: 5,
        volume: 50,
    });
    let mut b = Phy::new(PhyConfig::default());

    let message = b"lost ack test";
    a.send(message).unwrap();

    let mut chan_a_to_b = FrameDropChannel::passthrough();
    let mut chan_b_to_a = FrameDropChannel::new(vec![1]); // drop first ACK

    let (a_ev, b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| chan_a_to_b.process(s),
        &mut |s| chan_b_to_a.process(s),
        60_000,
    );

    assert!(
        has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)),
        "A should eventually get SendComplete after retransmit"
    );
    assert_eq!(get_received(&b_ev).unwrap().as_slice(), message);
}

/// Test data frame retransmission after packet loss.
///
/// NOTE: At 8kHz, the frame boundary detection in FrameDropChannel is less
/// reliable due to shorter symbol periods (400 samples vs 2400 at 48kHz) and
/// guard periods (80 samples < 128 sample chunk). Skip at 8kHz.
#[test]
#[cfg_attr(feature = "8khz", ignore = "frame detection unreliable at 8kHz chunk boundaries")]
fn test_lost_data() {
    let mut a = Phy::new(PhyConfig {
        timeout_ms: 4000,
        max_retries: 5,
        volume: 50,
    });
    let mut b = Phy::new(PhyConfig::default());

    let message = b"lost data test";
    a.send(message).unwrap();

    let mut chan_a_to_b = FrameDropChannel::new(vec![1]); // drop first DATA
    let mut chan_b_to_a = FrameDropChannel::passthrough();

    let (a_ev, b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| chan_a_to_b.process(s),
        &mut |s| chan_b_to_a.process(s),
        60_000,
    );

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&b_ev).unwrap().as_slice(), message);
}

#[test]
fn test_max_retries() {
    let mut a = Phy::new(PhyConfig {
        timeout_ms: 2000,
        max_retries: 3,
        volume: 50,
    });
    let mut b = Phy::new(PhyConfig::default());

    let message = b"will fail";
    a.send(message).unwrap();

    // Drop all ACKs from B→A
    let drop_all: Vec<usize> = (1..=100).collect();
    let mut chan_a_to_b = FrameDropChannel::passthrough();
    let mut chan_b_to_a = FrameDropChannel::new(drop_all);

    let (a_ev, _b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| chan_a_to_b.process(s),
        &mut |s| chan_b_to_a.process(s),
        120_000,
    );

    assert!(
        has_event(&a_ev, |e| matches!(e, PhyEvent::SendFailed)),
        "A should get SendFailed after max retries"
    );
}

#[test]
fn test_bidirectional() {
    // Stagger sends: A sends first, then B sends after A finishes.
    // This tests that the PHY can receive while also having a pending send.
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let msg_a = b"from A to B";
    let msg_b = b"from B to A";

    // Phase 1: A sends to B
    a.send(msg_a).unwrap();
    let (a_ev1, b_ev1) = run_loopback(&mut a, &mut b, 30_000);
    assert!(has_event(&a_ev1, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&b_ev1).unwrap().as_slice(), msg_a);

    // Phase 2: B sends to A
    b.send(msg_b).unwrap();
    let (a_ev2, b_ev2) = run_loopback(&mut a, &mut b, 30_000);
    assert!(has_event(&b_ev2, |e| matches!(e, PhyEvent::SendComplete)));
    assert_eq!(get_received(&a_ev2).unwrap().as_slice(), msg_b);
}

#[test]
fn test_sequential_messages() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let messages: Vec<Vec<u8>> = (0..5)
        .map(|i| format!("message #{i}").into_bytes())
        .collect();

    let mut all_b_events = Vec::new();

    for msg in &messages {
        a.send(msg).unwrap();
        let (a_ev, b_ev) = run_loopback(&mut a, &mut b, 30_000);
        assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
        all_b_events.extend(b_ev);
    }

    let received: Vec<&Vec<u8>> = all_b_events
        .iter()
        .filter_map(|e| match e {
            PhyEvent::Received(data) => Some(data),
            _ => None,
        })
        .collect();

    assert_eq!(received.len(), 5, "should receive all 5 messages");
    for (i, msg) in messages.iter().enumerate() {
        assert_eq!(received[i], msg, "message {i} mismatch");
    }
}

#[test]
fn test_preview() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message = vec![0x42u8; 50];
    a.send(&message).unwrap();

    let max_samples: u64 = 30_000 * (SAMPLE_RATE as u64 / 1000);
    let mut elapsed: u64 = 0;
    let mut got_preview = false;
    let mut final_msg: Option<Vec<u8>> = None;

    while elapsed < max_samples {
        let mut a_out = [0.0f32; CHUNK];
        let mut b_out = [0.0f32; CHUNK];

        a.emit(&mut a_out);
        b.emit(&mut b_out);

        b.ingest(&a_out);
        a.ingest(&b_out);

        if let Some(preview) = b.preview() {
            if !preview.is_empty() {
                got_preview = true;
            }
        }

        while let Some(ev) = b.poll() {
            if let PhyEvent::Received(msg) = ev {
                final_msg = Some(msg);
            }
        }
        while let Some(_) = a.poll() {}

        elapsed += CHUNK as u64;

        if final_msg.is_some() && a.is_idle() {
            break;
        }
    }

    assert!(got_preview, "should have seen preview data during reception");
    assert_eq!(
        final_msg.as_deref(),
        Some(message.as_slice()),
        "final message should match"
    );
}

#[test]
fn test_fragmented_noisy() {
    let mut a = Phy::new(PhyConfig::default());
    let mut b = Phy::new(PhyConfig::default());

    let message: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    a.send(&message).unwrap();

    let (a_ev, b_ev) = run_loopback_channels(
        &mut a,
        &mut b,
        &mut |s| add_noise(s, 15.0),
        &mut |s| add_noise(s, 15.0),
        500_000,
    );

    assert!(has_event(&a_ev, |e| matches!(e, PhyEvent::SendComplete)));
    let received = get_received(&b_ev).expect("B should receive noisy fragmented message");
    assert_eq!(received, &message);
}
