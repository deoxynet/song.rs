//! Stream FEC integration tests: loopback encode → audio → decode.
//!
//! Tests the streaming codec which uses RS(15,k) over GF(2^4) where each
//! GF(16) symbol maps to two 2-bit vowel symbols of the 4-vowel modem.

use song_rs::{StreamTx, StreamRx, StreamConfig, StreamProfile};
use song_rs::rs4::ReedSolomon4;
use song_rs::rs3::ReedSolomon3;
use song_rs::protocol::{SAMPLE_RATE, SYMBOL_TOTAL_SAMPLES};

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

const CHUNK: usize = SYMBOL_TOTAL_SAMPLES;

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

#[test]
fn raw_feed_and_audio_callback_boundaries_preserve_the_exact_song() {
    fn drain(tx: &mut StreamTx, audio: &mut Vec<f32>, callback: &mut usize) {
        let mut out = [0.0; 4097];
        loop {
            assert_eq!(tx.emit(&mut []), 0);
            let size = [1, 137, 2048, 4097][*callback % 4];
            *callback += 1;
            let count = tx.emit(&mut out[..size]);
            audio.extend_from_slice(&out[..count]);
            if count == 0 { break; }
        }
    }

    // Cover odd/even RS data lengths, partial interleaver blocks, the maximum
    // payload, and parity extremes without multiplying huge high-parity songs.
    for (parity, depth, length) in [(0, 1, 1), (4, 2, 17), (5, 4, 31), (14, 1, 1), (4, 4, 255)] {
        let config = StreamConfig { rs_parity: parity, interleave_depth: depth, volume: 50 };
        let payload: Vec<_> = (0..length).map(|i| (i * 73 + 19) as u8).collect();
        for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
            let mut baseline = StreamTx::with_profile(config.clone(), profile);
            baseline.feed(&payload);
            baseline.finish();
            let mut expected = Vec::new();
            let mut callback = 0;
            drain(&mut baseline, &mut expected, &mut callback);
            assert!(baseline.is_idle());

            let mut frame = vec![length as u8];
            frame.extend_from_slice(&payload);
            if profile == StreamProfile::Standard {
                frame.extend_from_slice(&song_rs::transfer::crc32(&frame).to_le_bytes());
            }
            for chunk_size in [1, 7, frame.len()] {
                let mut tx = StreamTx::with_profile(config.clone(), profile);
                let mut actual = Vec::new();
                let mut callback = 1;
                // Emitting before data arrives may send sync, but must not
                // restart the melody when raw bytes become available.
                drain(&mut tx, &mut actual, &mut callback);
                assert!(!tx.is_idle());
                for chunk in frame.chunks(chunk_size) {
                    tx.feed_raw(&[]);
                    tx.feed_raw(chunk);
                    drain(&mut tx, &mut actual, &mut callback);
                    assert!(!tx.is_idle(), "input starvation is not end-of-stream");
                }
                tx.finish();
                tx.finish();
                drain(&mut tx, &mut actual, &mut callback);
                assert!(tx.is_idle());
                assert_eq!(actual.len(), expected.len());
                assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{profile:?}, parity {parity}, depth {depth}, length {length}, raw chunks {chunk_size}");
                let mut rx = StreamRx::with_profile(config.clone(), profile);
                for chunk in actual.chunks(511) { rx.ingest(chunk); }
                assert_eq!(rx.read_all(), payload);
                assert!(!rx.is_active());
            }
        }
    }
}

/// Resample the actual transmission, including sync/cadence/trailing silence.
/// A ratio above one means playback runs faster than the capture clock.
fn resample_clock(samples: &[f32], mut ratio: impl FnMut(f64) -> f64) -> Vec<f32> {
    let mut audio = Vec::with_capacity(samples.len());
    let mut position = 0.0;
    while position < (samples.len() - 1) as f64 {
        let index = position as usize;
        let fraction = (position - index as f64) as f32;
        audio.push(samples[index] * (1.0 - fraction) + samples[index + 1] * fraction);
        position += ratio(position / SAMPLE_RATE);
    }
    audio
}

#[test]
fn streaming_tracks_clock_error_through_a_maximum_length_song() {
    let payload: Vec<_> = (0..255).map(|i| (i * 73 + 19) as u8).collect();
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        for ratio in [0.999, 1.001, 0.995, 1.005, 0.98, 1.02] {
            let mut audio = vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES + SYMBOL_TOTAL_SAMPLES / 3];
            audio.extend(resample_clock(&clean, |_| ratio));
            for chunk_size in [511, 4096] {
                let mut rx = StreamRx::new(config.clone());
                for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                assert_eq!(rx.read_all(), payload, "depth {depth}, ratio {ratio}, chunks {chunk_size}");
                assert!(!rx.is_active());
            }
        }
    }
}

#[test]
fn streaming_clock_tracking_survives_wander_channels_and_quiet_phrases() {
    let payload: Vec<_> = (0..128).map(|i| (i * 73 + 19) as u8).collect();
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        for clock in 0..2 {
            for channel in 0..4 {
                let mut song = clean.clone();
                if channel == 3 {
                    // A two-second phrase 40 dB below the surrounding notes.
                    for sample in &mut song[64 * SYMBOL_TOTAL_SAMPLES..96 * SYMBOL_TOTAL_SAMPLES] {
                        *sample *= 0.01;
                    }
                }
                song = resample_clock(&song, |time| match clock {
                    0 => 1.0 + 0.02 * (2.0 * std::f64::consts::PI * 0.07 * time).sin(),
                    _ => if (time / 8.0) as usize % 2 == 0 { 1.015 } else { 0.985 },
                });
                song = match channel {
                    1 => add_noise(&song, 20.0),
                    2 => phone_channel(&song),
                    _ => song,
                };
                let mut audio = vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES + SYMBOL_TOTAL_SAMPLES / 2 - 1];
                audio.extend_from_slice(&song);
                for chunk_size in [137, 2048, audio.len()] {
                    let mut rx = StreamRx::new(config.clone());
                    for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                    let received = rx.read_all();
                    assert_eq!(received, payload,
                        "depth {depth}, clock {clock}, channel {channel}, chunks {chunk_size}, active {}", rx.is_active());
                    assert!(!rx.is_active());
                }
            }
        }
    }
}

#[test]
fn streaming_reacquires_when_neighboring_songs_have_different_clocks() {
    let payloads: Vec<Vec<u8>> = [1usize, 64, 255, 17].into_iter()
        .map(|len| (0..len).map(|i| (i * 73 + len) as u8).collect()).collect();
    for profile in [StreamProfile::Standard, StreamProfile::Legacy] {
        let config = StreamConfig::default();
        let mut audio = vec![0.0; SYMBOL_TOTAL_SAMPLES / 3];
        for (i, payload) in payloads.iter().enumerate() {
            let mut tx = StreamTx::with_profile(config.clone(), profile);
            tx.feed(payload);
            tx.finish();
            let mut song = Vec::new();
            let mut out = [0.0; 2048];
            loop {
                let count = tx.emit(&mut out);
                if count == 0 { break; }
                song.extend_from_slice(&out[..count]);
            }
            audio.extend(resample_clock(&song, |_| [1.02, 0.98, 1.001, 0.995][i]));
            audio.resize(audio.len() + [1, SYMBOL_TOTAL_SAMPLES / 3, SYMBOL_TOTAL_SAMPLES - 1, 0][i], 0.0);
        }
        for chunk_size in [137, 2048, audio.len()] {
            let mut rx = StreamRx::with_profile(config.clone(), profile);
            for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
            assert_eq!(rx.read_all(), payloads.concat(), "{profile:?}, chunks {chunk_size}");
            assert!(!rx.is_active());
        }
    }
}

#[test]
fn streaming_capture_phase_does_not_discard_a_clear_song() {
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        for len in [1usize, 17, 64] {
            let payload: Vec<_> = (0..len).map(|i| (i * 73 + len) as u8).collect();
            let song = generate_audio(&payload, config.clone());
            for idle_slots in [0, 12] {
                for phase in [0, 1, SYMBOL_TOTAL_SAMPLES / 8, SYMBOL_TOTAL_SAMPLES / 4,
                    SYMBOL_TOTAL_SAMPLES / 2 - 1, SYMBOL_TOTAL_SAMPLES / 2,
                    3 * SYMBOL_TOTAL_SAMPLES / 4, SYMBOL_TOTAL_SAMPLES - 1]
                {
                    let mut audio = vec![0.0; idle_slots * SYMBOL_TOTAL_SAMPLES + phase];
                    audio.extend_from_slice(&song);
                    for chunk_size in [137, 2048, audio.len()] {
                        let mut rx = StreamRx::new(config.clone());
                        for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                        assert_eq!(rx.read_all(), payload,
                            "depth {depth}, length {len}, idle {idle_slots}, phase {phase}, chunks {chunk_size}");
                        assert!(!rx.is_active(), "the transmitted trailing silence must release the receiver");
                    }
                }
            }
        }
    }
}

#[test]
fn streaming_realigns_between_songs_after_non_slot_gaps() {
    let payloads: Vec<Vec<u8>> = [1usize, 64, 255, 17].into_iter()
        .map(|len| (0..len).map(|i| (i * 73 + len) as u8).collect()).collect();
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let mut audio = vec![0.0; SYMBOL_TOTAL_SAMPLES / 2 - 1];
        for (i, payload) in payloads.iter().enumerate() {
            audio.extend(generate_audio(payload, config.clone()));
            audio.resize(audio.len() + [1, SYMBOL_TOTAL_SAMPLES / 3, SYMBOL_TOTAL_SAMPLES - 1, 0][i], 0.0);
        }
        for chunk_size in [137, 2048, audio.len()] {
            let mut rx = StreamRx::new(config.clone());
            for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
            assert_eq!(rx.read_all(), payloads.concat(), "depth {depth}, chunks {chunk_size}");
            assert!(!rx.is_active());
        }
    }
}

#[test]
fn streaming_phase_recovery_survives_noise_phone_filtering_and_bad_sync() {
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let payload: Vec<_> = (0..64).map(|i| (i * 73 + 19) as u8).collect();
        let clean = generate_audio(&payload, config.clone());
        let next_payload = b"after the channel";
        let next = generate_audio(next_payload, config.clone());
        for channel in 0..3 {
            let mut song = match channel {
                0 => add_noise(&clean, 20.0),
                1 => phone_channel(&clean),
                _ => clean.clone(),
            };
            if channel == 2 { swap_symbol_window(&mut song, 1); }
            for phase in [1, SYMBOL_TOTAL_SAMPLES / 4, SYMBOL_TOTAL_SAMPLES / 2 - 1,
                3 * SYMBOL_TOTAL_SAMPLES / 4, SYMBOL_TOTAL_SAMPLES - 1]
            {
                let mut audio = vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES + phase];
                audio.extend_from_slice(&song);
                for chunk_size in [137, 2048] {
                    let mut rx = StreamRx::new(config.clone());
                    for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                    assert_eq!(rx.read_all(), payload,
                        "channel {channel}, depth {depth}, phase {phase}, chunks {chunk_size}, active {}", rx.is_active());
                    assert!(!rx.is_active());
                    for chunk in next.chunks(chunk_size) { rx.ingest(chunk); }
                    assert_eq!(rx.read_all(), next_payload, "reacquisition after channel {channel}");
                    assert!(!rx.is_active());
                }
            }
        }
    }
}

fn classify_test_slot(samples: &[f32]) -> (usize, f64) {
    use song_rs::{fft, formant, protocol::SAMPLES_PER_SYMBOL};
    let mut spectrum = vec![0.0; SAMPLES_PER_SYMBOL];
    fft::power_spectrum(&samples[..SAMPLES_PER_SYMBOL], &mut spectrum);
    formant::classify_symbol(&spectrum, &formant::EXPECTED_F2_CENTROIDS)
}

/// Construct a voiced interference case whose measured symbols actually
/// miscorrect under hard RS, but retain an erasure path back to the original.
fn add_recoverable_miscorrection(audio: &mut [f32], config: &StreamConfig, word: usize) {
    use song_rs::{formant, protocol::{melody_f0, PREAMBLE_LEN}};
    let depth = config.interleave_depth as usize;
    let slots: Vec<_> = (0..30).map(|half| {
        PREAMBLE_LEN + 2 * ((word / depth) * depth * 15 + half / 2 * depth + word % depth) + half % 2
    }).collect();
    let clean: Vec<_> = slots.iter().map(|&slot| {
        classify_test_slot(&audio[slot * SYMBOL_TOTAL_SAMPLES..(slot + 1) * SYMBOL_TOTAL_SAMPLES])
    }).collect();
    let original: Vec<_> = clean.chunks_exact(2).map(|p| ((p[0].0 << 2) | p[1].0) as u8).collect();
    let rs = ReedSolomon4::new(11, 4);
    let data = rs.decode(&original).expect("clean audio word");
    assert_eq!(rs.encode(&data), original, "fixture starts from a clean codeword");

    // For each slot/vowel, find a measured, low-confidence mixture that still
    // classifies as that vowel. These are real samples, not injected confidence.
    let variants: Vec<Vec<_>> = slots.iter().map(|&slot| {
        let vowels: Vec<Vec<f32>> = (0..4).map(|vowel| {
            formant::synthesize_symbol(vowel, melody_f0(slot), 0.5, slot)
                .into_iter().map(|s| s as f32).collect()
        }).collect();
        (0..4).map(|vowel| {
            let mut best = (vowels[vowel].clone(), classify_test_slot(&vowels[vowel]));
            for other in 0..4 {
                if other == vowel { continue; }
                for gain in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2] {
                    let mixed: Vec<_> = vowels[vowel].iter().zip(&vowels[other])
                        .map(|(&a, &b)| (a + gain * b) / (1.0 + gain)).collect();
                    let measured = classify_test_slot(&mixed);
                    if measured.0 == vowel && (best.1.0 != vowel || measured.1 < best.1.1) {
                        best = (mixed, measured);
                    }
                }
            }
            best
        }).collect()
    }).collect();

    for delta in 1..16 {
        let mut other_data = data.clone();
        other_data[0] ^= delta;
        let other = rs.encode(&other_data);
        let differences: Vec<_> = original.iter().zip(&other).enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i)).collect();
        assert_eq!(differences.len(), 5);
        for mask in 0u32..32 {
            if mask.count_ones() != 3 { continue; }
            let errors: Vec<_> = differences.iter().enumerate()
                .filter_map(|(bit, &i)| (mask & (1 << bit) != 0).then_some(i)).collect();
            let mut measured = clean.clone();
            for &i in &errors {
                measured[2 * i] = variants[2 * i][(other[i] >> 2) as usize].1;
                measured[2 * i + 1] = variants[2 * i + 1][(other[i] & 3) as usize].1;
            }
            let received: Vec<_> = measured.chunks_exact(2).map(|p| ((p[0].0 << 2) | p[1].0) as u8).collect();
            let mut weakest: Vec<_> = measured.chunks_exact(2).enumerate()
                .map(|(i, p)| (i, p[0].1.min(p[1].1))).filter(|&(_, c)| c < 1.0).collect();
            weakest.sort_by(|a, b| a.1.total_cmp(&b.1));
            let erasures: Vec<_> = weakest.iter().take(4).map(|&(i, _)| i).collect();
            if rs.decode(&received) != Some(other_data.clone())
                || !(1..=erasures.len()).any(|n| rs.decode_with_erasures(&received, &erasures[..n]) == Some(data.clone()))
            {
                continue;
            }
            for &i in &errors {
                for half in 0..2 {
                    let vowel = if half == 0 { other[i] >> 2 } else { other[i] & 3 } as usize;
                    let slot = slots[2 * i + half];
                    audio[slot * SYMBOL_TOTAL_SAMPLES..(slot + 1) * SYMBOL_TOTAL_SAMPLES]
                        .copy_from_slice(&variants[2 * i + half][vowel].0);
                }
            }
            return;
        }
    }
    panic!("no measured miscorrection/erasure fixture at depth {depth}, word {word}");
}

#[test]
fn streaming_crc_recovers_audio_miscorrections_and_keeps_the_next_song() {
    let payload: Vec<_> = (0..40).map(|i| (i * 73 + 19) as u8).collect();
    let previous = b"previous verified song";
    let next = b"the next clean song";
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        let previous_audio = generate_audio(previous, config.clone());
        let next_audio = generate_audio(next, config.clone());
        let mut damaged = clean.clone();
        for word in [1, 2] { add_recoverable_miscorrection(&mut damaged, &config, word); }
        let audio = [previous_audio.as_slice(), damaged.as_slice(), next_audio.as_slice()].concat();
        for chunk_size in [137, 2048, audio.len()] {
            let mut rx = StreamRx::new(config.clone());
            for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
            assert_eq!(rx.read_all(), [previous.as_slice(), payload.as_slice(), next.as_slice()].concat(),
                "depth {depth}, chunks {chunk_size}");
            assert!(!rx.is_active());
        }
    }
}

#[test]
fn streaming_crc_rejects_corrupt_frames_and_preserves_neighboring_songs() {
    let payload: Vec<_> = (0..17).map(|i| (i * 73 + 19) as u8).collect();
    let next = b"the next verified song";
    let mut checked = vec![payload.len() as u8];
    checked.extend_from_slice(&payload);
    checked.extend_from_slice(&song_rs::transfer::crc32(&checked).to_le_bytes());
    assert_ne!(*checked.last().unwrap(), 0, "truncated checksum must differ from padding");
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        let following = generate_audio(next, config.clone());
        for corruption in 0..5 {
            let mut frame = checked.clone();
            match corruption {
                0 => frame[1] ^= 1,
                1 => frame[0] -= 1,
                2 => *frame.last_mut().unwrap() ^= 1,
                3 => frame.truncate(1 + payload.len()), // original Legacy frame
                _ => { frame.pop(); } // truncated checksum, padded by the interleaver
            }
            // Each corrupted frame has newly encoded, valid RS parity. There
            // is no uncorrectable codeword for the previous gap guard to catch.
            let mut tx = StreamTx::new(config.clone());
            for chunk in frame.chunks(3) { tx.feed_raw(chunk); }
            tx.finish();
            let mut bad_audio = Vec::new();
            let mut emit_buf = vec![0.0; 511];
            loop {
                let n = tx.emit(&mut emit_buf);
                if n == 0 { break; }
                bad_audio.extend_from_slice(&emit_buf[..n]);
            }
            if corruption == 0 || corruption == 3 {
                let mut legacy = StreamRx::with_profile(config.clone(), StreamProfile::Legacy);
                legacy.ingest(&bad_audio);
                assert_eq!(legacy.read_all(), &frame[1..1 + payload.len()],
                    "unchecked Legacy must demonstrate that RS alone accepts this frame");
            }
            let audio = [clean.as_slice(), bad_audio.as_slice(), following.as_slice()].concat();
            for chunk_size in [137, 2048, audio.len()] {
                let mut rx = StreamRx::new(config.clone());
                for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                assert_eq!(rx.read_all(), [payload.as_slice(), next.as_slice()].concat(),
                    "depth {depth}, corruption {corruption}, chunks {chunk_size}");
                assert!(!rx.is_active());
            }
        }
    }
}

/// Replace one interleaved RS word with voiced audio that is actually rejected
/// by the production classifier + hard/erasure decoder. Beyond-budget damage
/// alone is not enough: an unchecked RS word can also miscorrect successfully.
fn replace_with_failed_codeword(audio: &mut [f32], config: &StreamConfig, word_index: usize) {
    use song_rs::{fft, formant};
    use song_rs::protocol::{PREAMBLE_LEN, SAMPLES_PER_SYMBOL};

    let depth = config.interleave_depth as usize;
    let slots: Vec<_> = (0..30).map(|half| {
        PREAMBLE_LEN + 2 * ((word_index / depth) * depth * 15 + (half / 2) * depth + word_index % depth) + half % 2
    }).collect();
    let variants: Vec<Vec<_>> = slots.iter().map(|&slot| {
        let mut choices: Vec<_> = (0..4).map(|vowel| {
            // A voiced interferer near G#3, not a modification to StreamTx.
            let samples: Vec<_> = formant::synthesize_symbol(vowel, 208.0, 0.5, slot)
                .into_iter().map(|sample| sample as f32).collect();
            let mut spectrum = vec![0.0; SAMPLES_PER_SYMBOL];
            fft::power_spectrum(&samples[..SAMPLES_PER_SYMBOL], &mut spectrum);
            let (detected, confidence) = formant::classify_symbol(&spectrum, &formant::EXPECTED_F2_CENTROIDS);
            (samples, detected, confidence)
        }).collect();
        // Confident substitutions exercise real RS rejection rather than
        // erasing every parity symbol and fitting an arbitrary codeword.
        if choices.iter().any(|choice| choice.2 == 1.0) {
            choices.retain(|choice| choice.2 == 1.0);
        }
        choices
    }).collect();
    let rs = ReedSolomon4::new(15 - config.rs_parity as usize, config.rs_parity as usize);
    let mut seed = 1923u32;
    for _ in 0..512 {
        let selected: Vec<_> = variants.iter().map(|choices| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            &choices[(seed >> 16) as usize % choices.len()]
        }).collect();
        let word: Vec<_> = selected.chunks_exact(2)
            .map(|pair| ((pair[0].1 << 2) | pair[1].1) as u8).collect();
        let mut weak: Vec<_> = selected.chunks_exact(2).enumerate()
            .map(|(i, pair)| (i, pair[0].2.min(pair[1].2)))
            .filter(|&(_, confidence)| confidence < 1.0).collect();
        weak.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let erasures: Vec<_> = weak.iter().take(config.rs_parity as usize).map(|&(i, _)| i).collect();
        if rs.decode(&word).is_some()
            || (!erasures.is_empty() && rs.decode_with_erasures(&word, &erasures).is_some())
        {
            continue;
        }
        for (&slot, choice) in slots.iter().zip(selected) {
            let start = slot * SYMBOL_TOTAL_SAMPLES;
            audio[start..start + SYMBOL_TOTAL_SAMPLES].copy_from_slice(&choice.0);
        }
        return;
    }
    panic!("no rejected audio fixture at depth {depth}, word {word_index}");
}

#[test]
fn failed_stream_codewords_preserve_positions_and_allow_the_next_song() {
    let payload = vec![0x11; 31];
    let next = b"the following clean song";
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        let next_audio = generate_audio(next, config.clone());
        // Seven data/checksum-bearing words. The eighth is interleaver padding at
        // depths 2/4; damaging it must leave the complete payload readable.
        let indices: &[usize] = if depth == 1 { &[0, 1, 6] } else { &[0, 1, 6, 7] };
        for &index in indices {
            let mut damaged = clean.clone();
            replace_with_failed_codeword(&mut damaged, &config, index);
            for chunk_size in [137, 2048, clean.len()] {
                let mut rx = StreamRx::new(config.clone());
                // A failed reception may neither erase old unread output nor
                // insert a shortened/reordered payload into the output queue.
                rx.ingest(&next_audio);
                for chunk in damaged.chunks(chunk_size) { rx.ingest(chunk); }
                let expected = if index == 7 {
                    [next.as_slice(), payload.as_slice()].concat()
                } else {
                    next.to_vec()
                };
                assert_eq!(rx.read_all(), expected, "depth {depth}, word {index}, chunks {chunk_size}");
                assert!(!rx.is_active());
                rx.ingest(&next_audio);
                assert_eq!(rx.read_all(), next, "the next clean song must still decode");
            }
        }
    }
}

#[test]
fn quiet_musical_phrases_do_not_end_a_stream_or_hide_the_next_one() {
    let payloads: Vec<Vec<u8>> = [54, 23].into_iter()
        .map(|len| (0..len).map(|i| (i * 73 + len) as u8).collect())
        .collect();
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        for gain in [0.1, 0.01, 0.001] {
            let mut audio = generate_audio(&payloads[0], config.clone());
            // A two-second musical phrase is 20/40/60 dB quieter. It is
            // still clean, voiced data, not a dropout or end-of-stream gap.
            let start = 34 * SYMBOL_TOTAL_SAMPLES;
            let end = start + 32 * SYMBOL_TOTAL_SAMPLES;
            for sample in &mut audio[start..end] { *sample *= gain; }
            audio.extend(generate_audio(&payloads[1], config.clone()));
            for chunk_size in [137, 2048, audio.len()] {
                let mut rx = StreamRx::new(config.clone());
                let mut deliveries = Vec::new();
                for chunk in audio.chunks(chunk_size) {
                    rx.ingest(chunk);
                    let bytes = rx.read_all();
                    if !bytes.is_empty() { deliveries.push(bytes); }
                }
                let expected = if chunk_size == audio.len() {
                    // read_all returns a byte queue, not framed messages.
                    vec![payloads.concat()]
                } else {
                    payloads.clone()
                };
                assert_eq!(deliveries, expected, "depth {depth}, gain {gain}, chunks {chunk_size}");
                assert!(!rx.is_active(), "real trailing silence must still release the stream");
            }
        }
    }
}

#[test]
fn consecutive_streams_preserve_unread_bytes_and_partial_reads() {
    let first = b"keep this result until the caller reads it";
    let second = b"the next complete song";
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let first_audio = generate_audio(first, config.clone());
        let second_audio = generate_audio(second, config.clone());
        for already_read in [0, 7, first.len()] {
            let mut rx = StreamRx::new(config.clone());
            rx.ingest(&first_audio);
            let mut prefix = vec![0; already_read];
            assert_eq!(rx.read(&mut prefix), already_read);
            assert_eq!(prefix, &first[..already_read]);
            rx.ingest(&second_audio);
            let expected = [&first[already_read..], second.as_slice()].concat();
            let mut actual = Vec::new();
            loop {
                let mut chunk = [0; 11];
                let count = rx.read(&mut chunk);
                if count == 0 { break; }
                actual.extend_from_slice(&chunk[..count]);
            }
            assert_eq!(actual, expected, "depth {depth}, already read {already_read}");
            assert!(rx.read_all().is_empty(), "delivered bytes are consumed once");
            assert!(!rx.is_active());
        }

        // An incomplete later stream also must not erase completed output.
        let mut rx = StreamRx::new(config);
        rx.ingest(&first_audio);
        rx.ingest(&second_audio[..16 * SYMBOL_TOTAL_SAMPLES]);
        rx.ingest(&vec![0.0; 8 * SYMBOL_TOTAL_SAMPLES]);
        assert_eq!(rx.read_all(), first);
        assert!(!rx.is_active());
    }
}

#[test]
fn finite_capture_glitches_do_not_turn_following_notes_into_silence() {
    let payload: Vec<u8> = (0..54).map(|i| (i * 73 + 19) as u8).collect();
    for depth in [1, 2, 4] {
        let config = StreamConfig { interleave_depth: depth, ..StreamConfig::default() };
        let clean = generate_audio(&payload, config.clone());
        for spike in [1.0, 1e8, f32::MAX, -f32::MAX] {
            let mut audio = clean.clone();
            let slot = 34 * SYMBOL_TOTAL_SAMPLES;
            audio[slot + song_rs::protocol::SAMPLES_PER_SYMBOL / 2] = spike;
            // The damaged symbol is within the FEC budget. The following
            // clean notes must not be discarded just because the glitch
            // raised the receiver's running level estimate.
            for sample in &mut audio[slot + SYMBOL_TOTAL_SAMPLES..slot + 33 * SYMBOL_TOTAL_SAMPLES] {
                *sample *= 0.01;
            }
            for chunk_size in [137, 2048] {
                let mut rx = StreamRx::new(config.clone());
                for chunk in audio.chunks(chunk_size) { rx.ingest(chunk); }
                assert_eq!(rx.read_all(), payload, "depth {depth}, spike {spike}, chunks {chunk_size}");
                assert!(!rx.is_active());
                rx.ingest(&clean);
                assert_eq!(rx.read_all(), payload, "the next stream must remain decodable");
                assert!(!rx.is_active());
            }
        }
    }
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
    let sr = SAMPLE_RATE;
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
fn test_stream_burst_erasure() {
    // With block interleaving (depth=2), a burst of zeroed symbol slots
    // becomes zero-confidence erasures spread across codewords, within
    // RS(15,11) correction capacity (4 parity: 2 errors or 4 erasures).
    let data: Vec<u8> = (0..50).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig::default(); // depth=2, RS(15,11)

    let result = stream_loopback(&data, |audio| {
        let mut processed = audio.to_vec();
        // Zero out 2 symbol slots in the middle of the data portion
        let symbol_slot = SYMBOL_TOTAL_SAMPLES;
        let start = (4 + 20) * symbol_slot; // after preamble + some data
        let burst_len = 2 * symbol_slot;
        assert!(start + burst_len <= processed.len(), "burst must hit the stream");
        processed[start..start + burst_len].fill(0.0);
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
    let duration_s = audio.len() as f64 / SAMPLE_RATE;
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
