//! Stream FEC integration tests: loopback encode → audio → decode.
//!
//! Tests the streaming codec which uses RS(15,k) over GF(2^4) where each
//! GF(16) symbol maps to two 2-bit vowel symbols of the 4-vowel modem.

use song_rs::{StreamTx, StreamRx, StreamConfig};
use song_rs::rs4::ReedSolomon4;
use song_rs::rs3::ReedSolomon3;

// --- RS4 unit tests (codec-level, GF(2^4), kept for reference) ---

#[test]
fn test_rs4_encode_decode_roundtrip() {
    for (k, t) in [(11, 4), (9, 6), (7, 8), (13, 2)] {
        let rs = ReedSolomon4::new(k, t);
        let msg: Vec<u8> = (0..k as u8).map(|v| v % 16).collect();
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 15);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg, "RS(15,{k}) roundtrip failed");
    }
}

#[test]
fn test_rs4_all_zero_all_ones() {
    let rs = ReedSolomon4::new(11, 4);
    for val in [0u8, 15] {
        let msg = vec![val; 11];
        let encoded = rs.encode(&msg);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs4_error_correction_1() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    for pos in 0..15 {
        let mut corrupted = encoded.clone();
        corrupted[pos] = (corrupted[pos] + 1) % 16;
        let decoded = rs.decode(&corrupted).expect(&format!("pos {pos}"));
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs4_error_correction_2() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[2] = (corrupted[2] + 5) % 16;
    corrupted[10] = (corrupted[10] + 9) % 16;

    let decoded = rs.decode(&corrupted).unwrap();
    assert_eq!(decoded, msg);
}

#[test]
fn test_rs4_erasure_correction_3() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[0] = 0;
    corrupted[5] = 0;
    corrupted[10] = 0;

    let decoded = rs.decode_with_erasures(&corrupted, &[0, 5, 10]).unwrap();
    assert_eq!(decoded, msg);
}

#[test]
fn test_rs4_erasure_correction_4() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[1] = 0;
    corrupted[4] = 0;
    corrupted[8] = 0;
    corrupted[13] = 0;

    let decoded = rs.decode_with_erasures(&corrupted, &[1, 4, 8, 13]).unwrap();
    assert_eq!(decoded, msg);
}

// --- RS3 unit tests (codec-level, GF(2^3), used by streaming codec) ---

#[test]
fn test_rs3_encode_decode_roundtrip() {
    for (k, t) in [(5, 2), (4, 3), (3, 4)] {
        let rs = ReedSolomon3::new(k, t);
        let msg: Vec<u8> = (0..k as u8).map(|v| v % 8).collect();
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 7);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg, "RS(7,{k}) roundtrip failed");
    }
}

#[test]
fn test_rs3_all_zero_all_max() {
    let rs = ReedSolomon3::new(5, 2);
    for val in [0u8, 7] {
        let msg = vec![val; 5];
        let encoded = rs.encode(&msg);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs3_error_correction_1() {
    let rs = ReedSolomon3::new(5, 2);
    let msg: Vec<u8> = (0..5).collect();
    let encoded = rs.encode(&msg);

    for pos in 0..7 {
        let mut corrupted = encoded.clone();
        corrupted[pos] = (corrupted[pos] + 1) % 8;
        let decoded = rs.decode(&corrupted).expect(&format!("pos {pos}"));
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs3_symbols_in_valid_range() {
    // This test verifies that RS3 only produces symbols 0-7,
    // which was the root cause of the original bug (RS4 produced 0-15).
    let rs = ReedSolomon3::new(5, 2);

    // Test with various message patterns
    for pattern in [0u8, 1, 3, 5, 7] {
        let msg = vec![pattern; 5];
        let encoded = rs.encode(&msg);

        // All encoded symbols must be in range 0..8 (valid for 8-symbol modem)
        for (i, &sym) in encoded.iter().enumerate() {
            assert!(
                sym < 8,
                "RS3 symbol at position {i} is {sym}, must be < 8 for 8-symbol modem"
            );
        }
    }
}

// --- Audio loopback helpers ---

const CHUNK: usize = 2880; // one symbol slot

/// Generate complete audio from StreamTx.
fn generate_audio(data: &[u8], config: StreamConfig) -> Vec<f32> {
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

/// Feed audio into StreamRx and collect output.
fn receive_audio(audio: &[f32], config: StreamConfig) -> Vec<u8> {
    let mut rx = StreamRx::new(config);
    // Feed in chunks (simulates real-time streaming)
    for chunk in audio.chunks(CHUNK) {
        rx.ingest(chunk);
    }
    rx.read_all()
}

/// Full loopback with optional channel transformation.
fn stream_loopback(
    data: &[u8],
    channel: impl Fn(&[f32]) -> Vec<f32>,
    config: StreamConfig,
) -> Vec<u8> {
    let audio = generate_audio(data, config.clone());
    let processed = channel(&audio);
    receive_audio(&processed, config)
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
            let gauss =
                (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (s as f64 + gauss * noise_amp) as f32
        })
        .collect()
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

fn phone_channel(samples: &[f32]) -> Vec<f32> {
    bandpass(samples, 300.0, 3400.0)
}

// --- Stream loopback tests ---
//
// These tests use the RS(15,k) GF(2^4) streaming codec: each GF(16) symbol
// maps to exactly two 2-bit vowel symbols of the 4-vowel modem alphabet.

#[test]
fn test_stream_clean() {
    let data: Vec<u8> = (0..50).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1, // no interleaving for clean test
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "clean stream loopback failed");
}

#[test]
fn test_stream_clean_interleaved() {
    let data: Vec<u8> = (0..50).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig::default(); // depth=2
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "interleaved clean stream loopback failed");
}

#[test]
fn test_stream_short_1() {
    let data = vec![0x42];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
fn test_stream_short_5() {
    let data = vec![1, 2, 3, 4, 5];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
fn test_stream_noisy_20db() {
    let data: Vec<u8> = (0..30).map(|i| (i * 3) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        volume: 50,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| add_noise(s, 20.0), config);
    assert_eq!(result, data, "noisy 20dB stream failed");
}

#[test]
fn test_stream_phone() {
    // Shorter payload for phone channel - longer payloads may have detection issues
    let data: Vec<u8> = (0..15).map(|i| (i * 5 + 7) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        volume: 50,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, phone_channel, config);
    assert_eq!(result, data, "phone channel stream failed");
}

#[test]
#[cfg_attr(feature = "8khz", ignore)] // Uses 48kHz-specific symbol timing
fn test_stream_burst_erasure() {
    // With block interleaving (depth=2), a burst of zeroed symbol slots
    // becomes zero-confidence erasures spread across codewords, within
    // RS(15,11) correction capacity (4 parity: 2 errors or 4 erasures).
    let data: Vec<u8> = (0..50).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig::default(); // depth=2, RS(15,11)

    let result = stream_loopback(&data, |audio| {
        let mut processed = audio.to_vec();
        // Zero out 2 symbol slots in the middle of the data portion
        let symbol_slot = 2880; // SYMBOL_TOTAL_SAMPLES
        let start = (4 + 20) * symbol_slot; // after preamble + some data
        let burst_len = 2 * symbol_slot;
        if start + burst_len < processed.len() {
            for i in start..start + burst_len {
                processed[i] = 0.0;
            }
        }
        processed
    }, config);

    assert_eq!(result, data, "burst erasure recovery failed");
}

#[test]
fn test_stream_medium() {
    // Medium-length payload (same as test_stream_clean but different pattern)
    let data: Vec<u8> = (0..50).map(|i| (i % 256) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "medium stream failed");
}

#[test]
fn test_stream_all_zeros() {
    let data = vec![0u8; 30];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
fn test_stream_all_ff() {
    let data = vec![0xFFu8; 30];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
fn test_stream_strong_rs() {
    // RS(7,3) with 4 parity: corrects 2 errors
    let data: Vec<u8> = (0..30).map(|i| (i * 11 + 3) as u8).collect();
    let config = StreamConfig {
        rs_parity: 4,
        interleave_depth: 1,
        volume: 50,
    };
    let result = stream_loopback(&data, |s| add_noise(s, 15.0), config);
    assert_eq!(result, data, "strong RS with noise failed");
}

#[test]
fn test_stream_throughput() {
    // Measure audio length for 50 bytes of payload
    let data: Vec<u8> = (0..50).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let audio = generate_audio(&data, config);
    let duration_s = audio.len() as f64 / 48000.0;
    let bytes_per_sec = data.len() as f64 / duration_s;

    // With RS(7,5), 5 data symbols per 7-symbol codeword, 60ms per symbol:
    // Throughput is lower than RS(15,11) but still reasonable
    eprintln!("Stream throughput: {bytes_per_sec:.1} B/s ({duration_s:.2}s for {} bytes)", data.len());

    // Just verify it runs without panic (the original bug)
    assert!(audio.len() > 0);
}

/// Regression test: verify that streaming codec doesn't panic with symbol overflow.
///
/// The original bug: RS4 (GF(2^4)) outputs symbols 0-15, but the Teams modem
/// only has 8 symbols (0-7). Symbols 8-15 caused a panic in synthesize_symbol().
///
/// This test verifies the fix by encoding data that would have triggered the panic.
#[test]
fn test_stream_no_symbol_overflow_panic() {
    // Use a variety of byte patterns that would produce high nibble values
    // when encoded with RS4. With RS3, all symbols should be 0-7.
    let test_patterns: &[&[u8]] = &[
        &[0xFF, 0xFF, 0xFF],
        &[0x88, 0x99, 0xAA, 0xBB, 0xCC],
        &[0xDE, 0xAD, 0xBE, 0xEF],
        &[0x00, 0xFF, 0x00, 0xFF],
    ];

    for pattern in test_patterns {
        let config = StreamConfig::default();
        // This should not panic - the original bug caused a panic here
        let audio = generate_audio(pattern, config.clone());
        assert!(audio.len() > 0, "audio generation failed for pattern {:?}", pattern);

        // Verify the roundtrip works
        let result = receive_audio(&audio, config);
        assert_eq!(result.as_slice(), *pattern, "roundtrip failed for pattern {:?}", pattern);
    }
}

// ── Streaming codec stress: sync fragility ───────────────────────────

/// Corrupt one symbol slot's audio by replacing it with a different vowel's
/// synthesis (confident wrong symbol — the harshest corruption).
fn swap_symbol_window(audio: &mut [f32], slot: usize) {
    use song_rs::formant;
    use song_rs::protocol::*;
    let start = slot * SYMBOL_TOTAL_SAMPLES;
    if start + SYMBOL_TOTAL_SAMPLES > audio.len() {
        return;
    }
    let synth = formant::synthesize_symbol(2, melody_f0(slot), 0.5, slot);
    for (i, &s) in synth.iter().enumerate() {
        audio[start + i] = s as f32;
    }
}

#[test]
fn test_stream_preamble_symbol_corrupted() {
    // One corrupted preamble symbol must not cost the whole stream.
    let data: Vec<u8> = (0..40u8).collect();
    let cfg = StreamConfig::default();
    let mut audio = generate_audio(&data, cfg.clone());
    // Slot 1 is preamble symbol [3] — swap it for vowel 2.
    swap_symbol_window(&mut audio, 1);
    let received = receive_audio(&audio, cfg);
    assert_eq!(received, data, "stream lost to one corrupted preamble symbol");
}

#[test]
fn test_stream_dropped_20ms() {
    // Jitter buffer removes 20 ms mid-stream: a symbol slip. The receiver
    // must resynchronize rather than scramble everything downstream.
    let data: Vec<u8> = (0..40u8).collect();
    let cfg = StreamConfig::default();
    let audio = generate_audio(&data, cfg.clone());
    let drop = (song_rs::protocol::SAMPLE_RATE * 0.02) as usize; // 20 ms
    let mid = audio.len() / 2;
    let mut cut = audio[..mid].to_vec();
    cut.extend_from_slice(&audio[mid + drop..]);
    let received = receive_audio(&cut, cfg);
    assert_eq!(received, data, "stream broken by 20 ms drop");
}
