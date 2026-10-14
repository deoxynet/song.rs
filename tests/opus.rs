//! Real Opus codec robustness: encode → libopus → decode.
//!
//! Unlike `robustness.rs` (synthetic DSP approximations), this drives actual
//! libopus at a configurable bitrate, so it exercises SILK/CELT/hybrid mode
//! switching, band folding, and the real quantization noise a VoIP call adds.
//!
//! Primary target: 128 kbps mono, 20 ms frames — the high end of what a
//! conferencing stack will negotiate on a good link.
//!
//! NOTE: These channel helpers operate at 48 kHz, including RNNoise's required
//! input rate. The core codec has separate 8 kHz coverage.

#![cfg(not(feature = "8khz"))]

use song_rs::{protocol, Decoder, StreamConfig, StreamRx, StreamTx};
use opus::{Application, Bitrate, Channels};
use nnnoiseless::DenoiseState;
use rustfft::{num_complex::Complex, FftPlanner};

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

/// Same voice-optimization model as robustness.rs: F0 boost, F1 suppression,
/// and non-flat gain across F2. Applied to the whole continuous capture.
fn call_spectral_tilt(samples: &[f32], attenuation_db: f64) -> Vec<f32> {
    if attenuation_db == 0.0 {
        return samples.to_vec();
    }
    let n = samples.len();
    let mut bins: Vec<_> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut bins);
    for (i, bin) in bins.iter_mut().enumerate() {
        let hz = i.min(n - i) as f64 * SAMPLE_RATE as f64 / n as f64;
        let gain_db = if hz < 250.0 {
            15.0 * (1.0 - (hz - 208.0).abs() / 50.0).max(0.0)
        } else if (400.0..=850.0).contains(&hz) {
            -attenuation_db
        } else if hz > 850.0 && hz < 2000.0 {
            -attenuation_db * (1.0 - (hz - 850.0) / 1150.0)
        } else if (2000.0..=2500.0).contains(&hz) {
            -attenuation_db * 0.3
        } else {
            0.0
        };
        *bin *= 10.0f64.powf(gain_db / 20.0) as f32;
    }
    planner.plan_fft_inverse(n).process(&mut bins);
    bins.iter().map(|bin| bin.re / n as f32).collect()
}

/// A single continuous RNNoise pass: preserve startup artifacts and delay.
/// Never replay the first frame, reset at the preamble, or realign the output.
fn rnnoise_continuous(samples: &[f32]) -> Vec<f32> {
    let mut denoiser = DenoiseState::new();
    let mut input = [0.0; DenoiseState::FRAME_SIZE];
    let mut output = [0.0; DenoiseState::FRAME_SIZE];
    let mut audio = Vec::with_capacity(samples.len() + input.len());
    for frame in samples.chunks(input.len()) {
        input.fill(0.0);
        for (target, &sample) in input.iter_mut().zip(frame) {
            *target = sample * 32767.0;
        }
        denoiser.process_frame(&mut output, &input);
        audio.extend(output.iter().map(|sample| sample / 32767.0));
    }
    audio
}

fn continuous_call_decodes(
    payload: &[u8],
    track: &[f64],
    bitrate: i32,
    tilt_db: f64,
    case: usize,
) -> bool {
    let prefix = case % 3 * SAMPLE_RATE as usize / 2 + case * 137 % 960;
    continuous_call_decodes_after(payload, track, bitrate, tilt_db, case, prefix)
}

fn continuous_call_decodes_after(
    payload: &[u8],
    track: &[f64],
    bitrate: i32,
    tilt_db: f64,
    case: usize,
    prefix: usize,
) -> bool {
    let volume = if case % 3 == 0 { 10 } else { 50 };
    let voice = song_rs::encode_with_pitch(payload, volume, Some(track)).unwrap();
    continuous_call_decodes_frame_after(payload, &voice, bitrate, tilt_db, case, prefix)
}

fn continuous_call_decodes_frame_after(
    payload: &[u8],
    voice: &[f32],
    bitrate: i32,
    tilt_db: f64,
    case: usize,
    prefix: usize,
) -> bool {
    // Vary both DSP-frame phase and cold/settled startup. The background
    // noise also runs before and after the transmission, as in a live call.
    let mut capture = vec![0.0; prefix];
    capture.extend_from_slice(voice);
    capture.resize(capture.len() + SAMPLE_RATE as usize, 0.0);
    let signal_power = voice.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
        / voice.len() as f64;
    let noise_rms = (signal_power.sqrt() / 10.0f64.powf(25.0 / 20.0)) as f32;
    let mut rng = 9329 + case as u32;
    for sample in &mut capture {
        *sample += normal(&mut rng) * noise_rms;
    }
    let shaped = call_spectral_tilt(&capture, tilt_db);
    let denoised = rnnoise_continuous(&shaped);
    let received = opus_roundtrip(&denoised, bitrate, Application::Voip).0;
    assert!(received.iter().all(|sample| sample.is_finite()));

    let mut decoder = Decoder::new();
    let mut delivered = 0;
    for chunk in received.chunks([128, 511, 1024][case % 3]) {
        // A failed acquisition may occur in background noise. A live
        // receiver must still be able to find the subsequent valid frame.
        if let Ok(Some(bytes)) = decoder.decode(chunk) {
            assert_eq!(bytes, payload, "wrong call payload, case {case}, bitrate {bitrate}, tilt {tilt_db}");
            delivered += 1;
        }
    }
    assert!(delivered <= 1, "duplicate call delivery, case {case}");
    eprintln!("call case={case} bitrate={bitrate} tilt={tilt_db} phase={prefix}: {}", delivered == 1);
    delivered == 1
}

fn check_continuous_call_register(tilt_db: f64) {
    let payload: Vec<u8> = [0x00, 0x55, 0xaa, 0xff, 0x1b, 0xe4, 0x72, 0x8d]
        .into_iter().cycle().take(24).collect();
    let mut failures = Vec::new();
    for midi in 47..=66 {
        let hz = 440.0 * 2.0f64.powf((midi as f64 - 69.0) / 12.0);
        if !continuous_call_decodes(&payload, &vec![hz; payload.len()], 24_000, tilt_db, midi - 47) {
            failures.push(midi);
        }
    }
    assert!(failures.is_empty(), "continuous-call failed pitches: {failures:?}, tilt {tilt_db}");
}

#[test]
fn opus_continuous_call_register() {
    check_continuous_call_register(0.0);
}

#[test]
fn opus_continuous_call_register_tilted() {
    check_continuous_call_register(40.0);
}

#[test]
fn opus_continuous_call_contours() {
    let payload: Vec<u8> = (0..60).map(|i| (i * 73 + 41) as u8).collect();
    let mut failures = Vec::new();
    for contour in 0..4 {
        let track: Vec<f64> = (0..payload.len()).map(|i| {
            let step = match contour {
                0 => i % 20,
                1 => 19 - i % 20,
                2 => if i % 2 == 0 { i / 2 % 10 } else { 19 - i / 2 % 10 },
                _ => (i * 13 + 7) % 20,
            };
            440.0 * 2.0f64.powf((47.0 + step as f64 - 69.0) / 12.0)
        }).collect();
        for bitrate in [16_000, 24_000, 128_000] {
            for tilt in [0.0, 40.0] {
                if !continuous_call_decodes(&payload, &track, bitrate, tilt, contour) {
                    failures.push((contour, bitrate, tilt));
                }
            }
        }
    }
    assert!(failures.is_empty(), "continuous-call failed contours: {failures:?}");
}

#[test]
fn opus_continuous_call_after_long_idle() {
    let payload: Vec<u8> = (0..40).map(|i| (i * 73 + 41) as u8).collect();
    let track = protocol::default_byte_pitch(payload.len());
    for case in [1, 3] {
        for tilt in [0.0, 40.0] {
            let prefix = 5 * SAMPLE_RATE as usize + case * 137;
            assert!(continuous_call_decodes_after(&payload, &track, 24_000, tilt, case, prefix),
                "call after five seconds of background, case {case}, tilt {tilt}");
        }
    }
}

#[test]
fn opus_continuous_call_maximum_frame_after_long_idle() {
    let payload: Vec<u8> = (0..protocol::MAX_LENGTH_VARIABLE)
        .map(|i| (i * 73 + 41) as u8).collect();
    let track: Vec<f64> = (0..payload.len()).map(|i| {
        let midi = 47.0 + ((i * 13 + 7) % 20) as f64;
        440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
    }).collect();
    for case in [1, 3] {
        for tilt in [0.0, 40.0] {
            let prefix = 5 * SAMPLE_RATE as usize + case * 137;
            assert!(continuous_call_decodes_after(&payload, &track, 24_000, tilt, case, prefix),
                "maximum-length call after idle, case {case}, tilt {tilt}");
        }
    }
}

#[test]
fn opus_continuous_call_damaged_header_after_idle() {
    let mut failures = Vec::new();
    for len in [40, protocol::MAX_LENGTH_VARIABLE] {
        let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
        let track: Vec<f64> = (0..len).map(|i| {
            let midi = 47.0 + ((i * 13 + 7) % 20) as f64;
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        }).collect();
        for case in [1, 3] {
            let volume = if case == 3 { 10 } else { 50 };
            let mut frame = song_rs::encode_with_pitch(&payload, volume, Some(&track)).unwrap();
            let mut header = song_rs::reed_solomon::ReedSolomon::new(1, 2).encode(&[len as u8]);
            header[0] ^= 0x5a;
            header[1] ^= 0x5a;
            for (i, vowel) in song_rs::formant::bytes_to_symbols(&header).into_iter().enumerate() {
                let pos = protocol::PREAMBLE_LEN + protocol::PILOT_LEN + i;
                let slot: Vec<f32> = song_rs::formant::synthesize_symbol(
                    vowel, protocol::FRAMING_LICK[pos],
                    volume as f64 / 100.0 * protocol::melody_accent(pos), pos,
                ).into_iter().map(|sample| sample as f32).collect();
                frame[pos * CHUNK..(pos + 1) * CHUNK].copy_from_slice(&slot);
            }
            for tilt in [0.0, 40.0] {
                let prefix = 5 * SAMPLE_RATE as usize + case * 137;
                if !continuous_call_decodes_frame_after(&payload, &frame, 24_000, tilt, case, prefix) {
                    failures.push((len, case, tilt));
                }
            }
        }
    }
    assert!(failures.is_empty(), "damaged-header continuous calls failed: {failures:?}");
}

#[test]
fn opus_continuous_call_sequential_frames_without_resets() {
    let payloads: Vec<Vec<u8>> = [140, 1, 40].into_iter().enumerate()
        .map(|(case, len)| (0..len).map(|i| (i * 73 + 41 + case) as u8).collect())
        .collect();
    let frames: Vec<Vec<f32>> = payloads.iter().enumerate().map(|(case, bytes)| {
        let track: Vec<f64> = (0..bytes.len()).map(|i| {
            let midi = 47.0 + ((i * 13 + case * 7) % 20) as f64;
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        }).collect();
        song_rs::encode_with_pitch(bytes, if case == 1 { 10 } else { 50 }, Some(&track)).unwrap()
    }).collect();
    let mut failures = Vec::new();
    for gap_ms in [0, 180] {
        for tilt in [0.0, 40.0] {
            let mut capture = vec![0.0; 5 * SAMPLE_RATE as usize + 137];
            for (i, frame) in frames.iter().enumerate() {
                if i > 0 {
                    capture.resize(capture.len() + gap_ms * SAMPLE_RATE as usize / 1000, 0.0);
                }
                capture.extend(frame);
            }
            capture.resize(capture.len() + SAMPLE_RATE as usize, 0.0);
            let rms = (frames[0].iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
                / frames[0].len() as f64).sqrt();
            let noise_rms = (rms / 10.0f64.powf(25.0 / 20.0)) as f32;
            let mut rng = 18751;
            for sample in &mut capture {
                *sample += normal(&mut rng) * noise_rms;
            }
            let shaped = call_spectral_tilt(&capture, tilt);
            let denoised = rnnoise_continuous(&shaped);
            let received = opus_roundtrip(&denoised, 24_000, Application::Voip).0;
            let mut decoder = Decoder::new();
            let mut deliveries = Vec::new();
            for chunk in received.chunks(511) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    assert!(payloads.contains(&bytes), "incorrect sequential-call payload");
                    deliveries.push(bytes);
                }
            }
            let lengths: Vec<_> = deliveries.iter().map(Vec::len).collect();
            eprintln!("sequential call gap={gap_ms}ms tilt={tilt}: {lengths:?}");
            if deliveries != payloads {
                failures.push((gap_ms, tilt, lengths));
            }
        }
    }
    assert!(failures.is_empty(), "lost, reordered, or duplicate sequential-call deliveries: {failures:?}");
}

// ── Decode helpers ───────────────────────────────────────────────────

fn try_decode(audio: &[f32], payload: &[u8]) -> bool {
    let mut dec = Decoder::new();
    match dec.decode(audio) {
        Ok(Some(bytes)) => {
            assert_eq!(bytes, payload, "Opus returned an incorrect payload");
            true
        }
        _ => false,
    }
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
    let total_bytes = protocol::EccProfile::Standard.encoded_bytes(payload_len);
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
    let mut one_byte_compute = std::time::Duration::ZERO;
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
        assert_eq!(got.as_deref(), Some(p.as_slice()), "latency benchmark payload, {size} bytes");
        if size == 1 {
            one_byte_compute = modem_dec;
        }

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
    let frame_symbols = 2 * protocol::PREAMBLE_LEN + protocol::PILOT_LEN
        + 4 * protocol::EccProfile::Standard.encoded_bytes(1);
    eprintln!("  air time           {:.0} ms  ({frame_symbols} symbols, including sync/header/CRC/RS)", air_1b);
    eprintln!("  opus algorithmic   {:.1} ms", st.algorithmic_delay_ms());
    eprintln!("  bulk decode        {:.1} ms (measured whole-buffer compute, not end-to-end latency)",
        one_byte_compute.as_secs_f64() * 1000.0);
    eprintln!("  air + opus delay   ~{:.0} ms (excludes receiver scheduling/compute)",
        air_1b + st.algorithmic_delay_ms());

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
        Ok(Some(bytes)) => {
            assert_eq!(bytes, payload, "Opus returned an incorrect payload");
            return true;
        }
        Err(_) => return false,
        Ok(None) => {}
    }
    let silence = vec![0.0f32; 10 * CHUNK];
    match dec.decode(&silence) {
        Ok(Some(bytes)) => {
            assert_eq!(bytes, payload, "Opus returned an incorrect payload on finalization");
            true
        }
        _ => false,
    }
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
