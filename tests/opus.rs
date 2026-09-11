//! Real Opus codec robustness: encode → libopus → decode.
//!
//! Unlike `robustness.rs` (synthetic DSP approximations), this drives actual
//! libopus at a configurable bitrate, so it exercises SILK/CELT/hybrid mode
//! switching, band folding, and the real quantization noise a VoIP call adds.
//!
//! Primary target: 128 kbps mono, 20 ms frames — the high end of what a
//! conferencing stack will negotiate on a good link.
//!
//! NOTE: 48 kHz mode only, like `robustness.rs` (8 kHz mode lacks the FFT
//! resolution the decoder needs).

#![cfg(not(feature = "8khz"))]

use song_rs::{protocol, Decoder, StreamConfig, StreamRx, StreamTx};
use opus::{Application, Bitrate, Channels};

const SAMPLE_RATE: u32 = 48000;
/// 20 ms frames — the WebRTC/Teams default ptime.
const FRAME_MS: usize = 20;
/// One symbol slot (50 ms voiced + 10 ms guard).
const CHUNK: usize = protocol::SYMBOL_TOTAL_SAMPLES;

// ── Opus channel ─────────────────────────────────────────────────────

/// What one Opus roundtrip cost us.
struct OpusStats {
    /// Number of packets produced.
    packets: usize,
    /// Total compressed bytes across all packets.
    compressed_bytes: usize,
    /// Encoder lookahead in samples (part of the algorithmic delay).
    lookahead: usize,
    /// Wall time spent inside the Opus encoder.
    encode_time: std::time::Duration,
    /// Wall time spent inside the Opus decoder.
    decode_time: std::time::Duration,
    /// Audio duration in seconds.
    audio_secs: f64,
}

impl OpusStats {
    /// Measured bitrate of the compressed stream (bit/s).
    fn actual_bitrate(&self) -> f64 {
        self.compressed_bytes as f64 * 8.0 / self.audio_secs
    }

    /// Algorithmic delay: encoder lookahead + one frame of buffering.
    fn algorithmic_delay_ms(&self) -> f64 {
        self.lookahead as f64 * 1000.0 / SAMPLE_RATE as f64 + FRAME_MS as f64
    }
}

/// Push audio through a real Opus encoder/decoder pair.
///
/// Frames are 20 ms; a short tail is zero-padded to a whole frame. The returned
/// audio is *not* delay-compensated — the modem decoder has to find the
/// preamble through the codec's algorithmic delay, same as on a live call.
fn opus_roundtrip(
    samples: &[f32],
    bitrate: i32,
    app: Application,
) -> (Vec<f32>, OpusStats) {
    let frame = SAMPLE_RATE as usize * FRAME_MS / 1000; // 960 samples
    let mut enc = opus::Encoder::new(SAMPLE_RATE, Channels::Mono, app).unwrap();
    enc.set_bitrate(Bitrate::Bits(bitrate)).unwrap();
    let lookahead = enc.get_lookahead().unwrap() as usize;

    let mut dec = opus::Decoder::new(SAMPLE_RATE, Channels::Mono).unwrap();

    let mut out = Vec::with_capacity(samples.len() + frame);
    let mut packets = 0;
    let mut compressed_bytes = 0;
    let mut encode_time = std::time::Duration::ZERO;
    let mut decode_time = std::time::Duration::ZERO;

    let mut in_frame = vec![0.0f32; frame];
    let mut out_frame = vec![0.0f32; frame];
    let mut packet = vec![0u8; 4000]; // max Opus packet size

    let mut pos = 0;
    while pos < samples.len() {
        let n = frame.min(samples.len() - pos);
        in_frame[..n].copy_from_slice(&samples[pos..pos + n]);
        in_frame[n..].fill(0.0);

        let t0 = std::time::Instant::now();
        let len = enc.encode_float(&in_frame, &mut packet).unwrap();
        encode_time += t0.elapsed();

        let t1 = std::time::Instant::now();
        let decoded = dec.decode_float(&packet[..len], &mut out_frame, false).unwrap();
        decode_time += t1.elapsed();

        out.extend_from_slice(&out_frame[..decoded]);
        packets += 1;
        compressed_bytes += len;
        pos += n;
    }

    let stats = OpusStats {
        packets,
        compressed_bytes,
        lookahead,
        encode_time,
        decode_time,
        audio_secs: samples.len() as f64 / SAMPLE_RATE as f64,
    };
    (out, stats)
}

/// Convenience: 128 kbps VoIP-mode roundtrip, audio only.
fn opus_128k(samples: &[f32]) -> Vec<f32> {
    opus_roundtrip(samples, 128_000, Application::Voip).0
}

// ── DSP helpers (local copies; see robustness.rs) ─────────────────────

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    (*state >> 16) as f32 / 65535.0
}

fn normal(state: &mut u32) -> f32 {
    let u1 = lcg(state).max(1e-10);
    let u2 = lcg(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn add_noise(samples: &[f32], snr_db: f64, seed: u32) -> Vec<f32> {
    let sig_power: f64 =
        samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    let noise_rms = sig_power.sqrt() / 10.0f64.powf(snr_db / 20.0);
    let mut st = seed;
    samples
        .iter()
        .map(|&s| s + normal(&mut st) * noise_rms as f32)
        .collect()
}

fn agc(samples: &[f32], target_rms: f32) -> Vec<f32> {
    let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 1e-10 {
        return samples.to_vec();
    }
    let gain = target_rms / rms;
    samples.iter().map(|&s| (s * gain).clamp(-1.0, 1.0)).collect()
}

// ── Decode helpers ───────────────────────────────────────────────────

fn try_decode(audio: &[f32], payload: &[u8]) -> bool {
    let mut dec = Decoder::new();
    matches!(dec.decode(audio), Ok(Some(ref d)) if d == payload)
}

fn stream_encode(data: &[u8], config: StreamConfig) -> Vec<f32> {
    let mut tx = StreamTx::new(config);
    tx.feed(data);
    tx.finish();

    let mut audio = Vec::new();
    loop {
        let mut buf = vec![0.0f32; CHUNK];
        let n = tx.emit(&mut buf);
        if n == 0 {
            break;
        }
        audio.extend_from_slice(&buf[..n]);
    }
    audio
}

fn stream_decode(audio: &[f32], config: StreamConfig) -> Vec<u8> {
    let mut rx = StreamRx::new(config);
    for chunk in audio.chunks(CHUNK) {
        rx.ingest(chunk);
    }
    rx.read_all()
}

/// Over-the-air duration of a one-shot frame for a given payload size.
fn frame_duration_secs(payload_len: usize) -> f64 {
    let n_ecc = protocol::ecc_bytes_for_length(payload_len);
    let total_bytes = protocol::ENCODED_DATA_OFFSET + payload_len + n_ecc;
    let n_data = song_rs::formant::symbols_for_bytes(total_bytes);
    let symbols =
        protocol::PREAMBLE_LEN + protocol::PILOT_LEN + n_data + protocol::PREAMBLE_LEN;
    (symbols * protocol::SYMBOL_TOTAL_SAMPLES) as f64 / SAMPLE_RATE as f64
}

// ── 128 kbps survival ────────────────────────────────────────────────

#[test]
fn opus_128k_voip_mode() {
    let p = b"opus 128k voip mode";
    let audio = song_rs::encode(p, 50).unwrap();
    assert!(try_decode(&opus_128k(&audio), p), "128 kbps Opus (VOIP app)");
}

#[test]
fn opus_128k_audio_mode() {
    // Application::Audio biases toward CELT/music — a conferencing stack in
    // "high fidelity" mode, or a stream relayed through a media server.
    let p = b"opus 128k audio mode";
    let audio = song_rs::encode(p, 50).unwrap();
    let (out, _) = opus_roundtrip(&audio, 128_000, Application::Audio);
    assert!(try_decode(&out, p), "128 kbps Opus (AUDIO app)");
}

#[test]
fn opus_128k_long_payload() {
    let p = b"This is a longer message pushed through a real libopus encoder at 128 kbps to check for accumulated symbol errors";
    let audio = song_rs::encode(p, 50).unwrap();
    assert!(try_decode(&opus_128k(&audio), p), "128 kbps long payload");
}

#[test]
fn opus_128k_max_payload() {
    let p: Vec<u8> = (0..140).map(|i| 0x20 + (i * 7 % 90) as u8).collect();
    let audio = song_rs::encode(&p, 50).unwrap();
    assert!(try_decode(&opus_128k(&audio), &p), "128 kbps max payload");
}

#[test]
fn opus_128k_with_noise_20db() {
    // Noise on the mic *before* the codec — the realistic ordering.
    let p = b"opus 128k plus room noise";
    let audio = song_rs::encode(p, 50).unwrap();
    let noisy = add_noise(&audio, 20.0, 4242);
    assert!(try_decode(&opus_128k(&noisy), p), "128 kbps + 20 dB SNR");
}

#[test]
fn opus_128k_call_pipeline() {
    // Mic noise → Opus 128 k → receiver AGC.
    let p = b"opus 128k call pipeline";
    let audio = song_rs::encode(p, 50).unwrap();
    let out = add_noise(&audio, 25.0, 777);
    let out = opus_128k(&out);
    let out = agc(&out, 0.3);
    assert!(try_decode(&out, p), "128 kbps call pipeline");
}

#[test]
fn opus_128k_quiet_talker() {
    // Low input level: Opus quantization noise is relatively larger.
    let p = b"opus 128k quiet talker";
    let audio = song_rs::encode(p, 10).unwrap();
    let out = opus_128k(&audio);
    let out = agc(&out, 0.3);
    assert!(try_decode(&out, p), "128 kbps at volume 10");
}

#[test]
fn opus_128k_streaming_codec() {
    let config = StreamConfig::default();
    let data = b"streaming FEC payload through real opus at 128 kbps";
    let audio = stream_encode(data, config.clone());
    let out = stream_decode(&opus_128k(&audio), config);
    assert_eq!(&out[..data.len().min(out.len())], &data[..], "stream through 128 kbps Opus");
}

// ── Bitrate sweep ────────────────────────────────────────────────────

#[test]
fn opus_bitrate_sweep() {
    let p = b"opus bitrate sweep payload";
    let audio = song_rs::encode(p, 50).unwrap();

    eprintln!();
    eprintln!("  {:>9}  {:>9}  {:>8}  {:>6}", "requested", "actual", "packets", "result");
    let mut lowest_pass = None;
    for &br in &[128_000, 96_000, 64_000, 48_000, 32_000, 24_000, 16_000, 12_000, 8_000] {
        let (out, st) = opus_roundtrip(&audio, br, Application::Voip);
        let ok = try_decode(&out, p);
        eprintln!(
            "  {:>6} kbps  {:>6.1} kbps  {:>8}  {:>6}",
            br / 1000,
            st.actual_bitrate() / 1000.0,
            st.packets,
            if ok { "PASS" } else { "FAIL" }
        );
        if ok {
            lowest_pass = Some(br);
        }
    }
    match lowest_pass {
        Some(br) => eprintln!("  lowest passing bitrate: {} kbps", br / 1000),
        None => eprintln!("  no bitrate passed"),
    }

    // The headline requirement.
    let (out, _) = opus_roundtrip(&audio, 128_000, Application::Voip);
    assert!(try_decode(&out, p), "must survive 128 kbps Opus");
}

#[test]
fn opus_128k_snr_sweep() {
    let p = b"opus 128k SNR sweep data";
    let audio = song_rs::encode(p, 50).unwrap();

    eprintln!();
    let mut threshold = None;
    for &snr in &[30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 5.0] {
        let out = opus_128k(&add_noise(&audio, snr, 42));
        let ok = try_decode(&out, p);
        eprintln!("  opus128k + SNR {:>5.1} dB: {}", snr, if ok { "PASS" } else { "FAIL" });
        if ok {
            threshold = Some(snr);
        }
    }
    match threshold {
        Some(t) => eprintln!("  lowest passing SNR through Opus 128k: {:.0} dB", t),
        None => eprintln!("  no SNR passed"),
    }

    assert!(
        try_decode(&opus_128k(&add_noise(&audio, 20.0, 42)), p),
        "128 kbps Opus must survive 20 dB SNR"
    );
}

// ── Latency / throughput ─────────────────────────────────────────────

#[test]
fn opus_128k_latency_throughput() {
    eprintln!();
    eprintln!("=== One-shot frames through Opus 128 kbps (mono, 20 ms ptime) ===");
    eprintln!(
        "  {:>7} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "payload", "air", "goodput", "modem enc", "modem dec", "result"
    );
    eprintln!(
        "  {:>7} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "(bytes)", "(s)", "(bit/s)", "(x RT)", "(x RT)", ""
    );

    let mut last_stats = None;
    for &size in &[1usize, 10, 25, 50, 100, 140] {
        let p: Vec<u8> = (0..size).map(|i| 0x20 + (i * 7 % 90) as u8).collect();

        let t0 = std::time::Instant::now();
        let audio = song_rs::encode(&p, 50).unwrap();
        let modem_enc = t0.elapsed();

        let (out, st) = opus_roundtrip(&audio, 128_000, Application::Voip);

        let t1 = std::time::Instant::now();
        let mut dec = Decoder::new();
        let got = dec.decode(&out).unwrap();
        let modem_dec = t1.elapsed();

        let air = st.audio_secs;
        eprintln!(
            "  {:>7} {:>8.2} {:>9.1} {:>9.0} {:>9.0} {:>8}",
            size,
            air,
            size as f64 * 8.0 / air,
            air / modem_enc.as_secs_f64(),
            air / modem_dec.as_secs_f64(),
            if got.as_deref() == Some(&p[..]) { "PASS" } else { "FAIL" }
        );
        last_stats = Some(st);
    }

    // Streaming codec goodput (no per-frame preamble, no ACK round-trips).
    let config = StreamConfig::default();
    let data: Vec<u8> = (0..200).map(|i| 0x20 + (i * 11 % 90) as u8).collect();
    let audio = stream_encode(&data, config.clone());
    let (out, stream_st) = opus_roundtrip(&audio, 128_000, Application::Voip);
    let t0 = std::time::Instant::now();
    let decoded = stream_decode(&out, config);
    let stream_dec = t0.elapsed();
    let stream_air = stream_st.audio_secs;
    let stream_ok = decoded.len() >= data.len() && decoded[..data.len()] == data[..];

    eprintln!();
    eprintln!("=== Streaming codec (RS(15,11) GF(2^4), interleave 2) through Opus 128 kbps ===");
    eprintln!("  payload            {} bytes", data.len());
    eprintln!("  air time           {:.2} s", stream_air);
    eprintln!(
        "  goodput            {:.1} bit/s ({:.1} byte/s)",
        data.len() as f64 * 8.0 / stream_air,
        data.len() as f64 / stream_air
    );
    eprintln!("  modem decode       {:.0}x real-time", stream_air / stream_dec.as_secs_f64());
    eprintln!("  integrity          {}", if stream_ok { "PASS" } else { "FAIL" });

    // Codec-side numbers.
    let st = last_stats.unwrap();
    eprintln!();
    eprintln!("=== Opus channel cost ===");
    eprintln!(
        "  requested          128.0 kbps  →  actual {:.1} kbps",
        st.actual_bitrate() / 1000.0
    );
    eprintln!(
        "  encoder lookahead  {} samples ({:.2} ms)",
        st.lookahead,
        st.lookahead as f64 * 1000.0 / SAMPLE_RATE as f64
    );
    eprintln!(
        "  algorithmic delay  {:.2} ms (lookahead + {} ms frame)",
        st.algorithmic_delay_ms(),
        FRAME_MS
    );
    eprintln!(
        "  opus encode        {:.0}x real-time",
        st.audio_secs / st.encode_time.as_secs_f64()
    );
    eprintln!(
        "  opus decode        {:.0}x real-time",
        st.audio_secs / st.decode_time.as_secs_f64()
    );

    // Latency budget for a small one-shot frame.
    let air_1b = frame_duration_secs(1) * 1000.0;
    eprintln!();
    eprintln!("=== One-way latency budget, 1-byte frame (excl. network/jitter buffer) ===");
    eprintln!("  air time           {:.0} ms  (8 preamble + 4 data symbols @ 60 ms)", air_1b);
    eprintln!("  opus algorithmic   {:.1} ms", st.algorithmic_delay_ms());
    eprintln!("  modem decode       <1 ms (compute; runs faster than real time)");
    eprintln!("  total              ~{:.0} ms", air_1b + st.algorithmic_delay_ms());

    assert!(stream_ok, "streaming codec must survive 128 kbps Opus");
}

// ── Packet loss + PLC ────────────────────────────────────────────────
//
// Real calls lose packets; the jitter buffer asks libopus for packet-loss
// concealment (PLC), which synthesizes plausible-but-wrong audio in the
// gap. This is harsher than silence: the concealed audio classifies with
// real confidence, so the modem must survive confident wrong symbols, not
// convenient erasures.

/// Opus roundtrip with per-packet loss decided by `lost(packet_idx)`.
/// Lost packets are decoded via PLC (empty payload).
fn opus_roundtrip_lossy(samples: &[f32], lost: impl Fn(usize) -> bool) -> Vec<f32> {
    let frame = SAMPLE_RATE as usize * FRAME_MS / 1000;
    let mut enc = opus::Encoder::new(SAMPLE_RATE, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::Bits(128_000)).unwrap();
    let mut dec = opus::Decoder::new(SAMPLE_RATE, Channels::Mono).unwrap();

    let mut out = Vec::with_capacity(samples.len() + frame);
    let mut in_frame = vec![0.0f32; frame];
    let mut out_frame = vec![0.0f32; frame];
    let mut packet = vec![0u8; 4000];

    let mut pos = 0;
    let mut idx = 0;
    while pos < samples.len() {
        let n = frame.min(samples.len() - pos);
        in_frame[..n].copy_from_slice(&samples[pos..pos + n]);
        in_frame[n..].fill(0.0);
        let len = enc.encode_float(&in_frame, &mut packet).unwrap();
        let decoded = if lost(idx) {
            // Packet lost: packet-loss concealment.
            dec.decode_float(&[], &mut out_frame, false).unwrap()
        } else {
            dec.decode_float(&packet[..len], &mut out_frame, false).unwrap()
        };
        out.extend_from_slice(&out_frame[..decoded]);
        pos += n;
        idx += 1;
    }
    out
}

fn try_decode_with_tail(audio: &[f32], payload: &[u8]) -> bool {
    let mut dec = Decoder::new();
    match dec.decode(audio) {
        Ok(Some(ref d)) => return d == payload,
        Err(_) => return false,
        Ok(None) => {}
    }
    let silence = vec![0.0f32; 10 * CHUNK];
    matches!(dec.decode(&silence), Ok(Some(ref d)) if d == payload)
}

#[test]
fn opus_packet_loss_5pct() {
    let p = b"opus with 5 percent packet loss";
    let audio = song_rs::encode(p, 50).unwrap();
    // Deterministic scattered loss: every 20th packet.
    let out = opus_roundtrip_lossy(&audio, |i| i % 20 == 7);
    assert!(try_decode_with_tail(&out, p), "5% packet loss + PLC");
}

#[test]
fn opus_packet_loss_10pct() {
    // The default (Giant Steps) melody trades a sliver of margin for the
    // tune: periodic 10% loss beats against the byte grid, and one
    // unlucky phase alignment can repeatedly hit the same thin-margin
    // note+vowel windows. Require at least 8 of 10 phases to survive.
    let p = b"opus ten percent loss test";
    let audio = song_rs::encode(p, 50).unwrap();
    let mut pass = 0;
    for phase in 0..10 {
        let out = opus_roundtrip_lossy(&audio, |i| i % 10 == phase);
        if try_decode_with_tail(&out, p) {
            pass += 1;
        }
    }
    eprintln!("default melody, 10% loss: {pass}/10 phases decode");
    assert!(pass >= 8, "default melody should survive most 10% loss phases ({pass}/10)");
}

#[test]
fn opus_packet_loss_10pct_armored() {
    // The armored melody must survive EVERY phase alignment.
    let p = b"opus ten percent loss test";
    let audio = song_rs::encode_armored(p, 50).unwrap();
    for phase in 0..10 {
        let out = opus_roundtrip_lossy(&audio, |i| i % 10 == phase);
        assert!(
            try_decode_with_tail(&out, p),
            "armored melody failed 10% loss at phase {phase}"
        );
    }
}

#[test]
fn opus_packet_loss_burst_100ms() {
    // 5 consecutive lost packets = 100 ms of PLC audio mid-message
    // (~1.7 symbol slots of confident garbage).
    let p = b"opus burst loss one hundred ms";
    let audio = song_rs::encode(p, 50).unwrap();
    let out = opus_roundtrip_lossy(&audio, |i| (40..45).contains(&i));
    assert!(try_decode_with_tail(&out, p), "100 ms burst loss + PLC");
}

#[test]
fn opus_packet_loss_burst_200ms() {
    // 10 consecutive lost packets = 200 ms of PLC audio (~3.3 slots).
    let p = b"opus burst loss two hundred ms!";
    let audio = song_rs::encode(p, 50).unwrap();
    let out = opus_roundtrip_lossy(&audio, |i| (40..50).contains(&i));
    assert!(try_decode_with_tail(&out, p), "200 ms burst loss + PLC");
}
