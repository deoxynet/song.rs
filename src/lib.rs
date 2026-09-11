//! Vowel-formant modem — encode data as vowel-like audio with melodic pitch.
//!
//! Encode byte payloads to 48 kHz f32 audio using vowel identity (F2 formant)
//! as the sole data channel and decode f32 audio back to bytes.
//! Teams branch: 4 vowels = 2 bits per symbol. Pitch carries no data — the
//! encoder plays a D lydian arpeggio (protocol::MELODY) and detection is
//! F0-agnostic, so pitch distortion can never corrupt data.
//!
//! # Example
//!
//! ```
//! let audio = song_rs::encode(b"hello", 25).unwrap();
//! let mut decoder = song_rs::Decoder::new();
//! let payload = decoder.decode(&audio).unwrap().unwrap();
//! assert_eq!(&payload, b"hello");
//! ```

pub mod protocol;
pub mod reed_solomon;
pub mod rs4;
pub mod rs3;
pub mod dss;
pub mod fft;
pub mod formant;
pub mod encoder;
pub mod decoder;
pub mod phy;
pub mod stream;
pub mod transfer;

pub use decoder::Decoder;
pub use encoder::encode_with_pitch;
pub use phy::{Phy, PhyConfig, PhyEvent, PhyError};
pub use stream::{StreamTx, StreamRx, StreamConfig};

/// Encode a payload into f32 audio samples using the 2-channel vocal modem.
///
/// `payload`: the bytes to transmit (max 140 bytes).
/// `volume`: output volume, 0..=100 (typically 10-50).
///
/// Returns a `Vec<f32>` of audio samples at 48 kHz.
///
/// The payload sings the default melody — the head of Giant Steps
/// (pitch carries no data). For maximum channel margin on worst-case
/// links, use [`encode_armored`].
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, Error> {
    encoder::encode(payload, volume)
}

/// Encode with the margin-verified arpeggio melody throughout — maximum
/// channel robustness (the default for the secret-transfer profile).
pub fn encode_armored(payload: &[u8], volume: u8) -> Result<Vec<f32>, Error> {
    encoder::encode_armored(payload, volume)
}

/// Errors returned by encode/decode operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("payload is empty")]
    EmptyPayload,

    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("invalid volume: {0} (must be 0..=100)")]
    InvalidVolume(u8),

    #[error("decode failed: could not extract valid payload from audio")]
    DecodeFailed,

    #[error("pitch track length {got} does not match payload length {expected}")]
    PitchTrackMismatch { got: usize, expected: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Parameter validation ---

    #[test]
    fn test_encode_empty_payload() {
        assert!(matches!(encode(b"", 25), Err(Error::EmptyPayload)));
    }

    #[test]
    fn test_encode_too_large() {
        let big = vec![0u8; 141];
        assert!(matches!(
            encode(&big, 25),
            Err(Error::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn test_encode_volume_bounds() {
        assert!(encode(b"hello", 0).is_ok());
        assert!(encode(b"hello", 100).is_ok());
        assert!(matches!(
            encode(b"hello", 101),
            Err(Error::InvalidVolume(101))
        ));
        assert!(encode(b"hello", 50).is_ok());
    }

    #[test]
    fn test_encode_various_sizes() {
        for size in 1..=3 {
            let payload = &b"asd"[..size];
            assert!(encode(payload, 25).is_ok(), "encode failed for size {size}");
        }
    }

    // --- Bit packing ---

    #[test]
    fn test_bit_packing_roundtrip() {
        for len in 1..=10 {
            let bytes: Vec<u8> = (0..len).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let syms = formant::bytes_to_symbols(&bytes);
            let back = formant::symbols_to_bytes(&syms, len);
            assert_eq!(bytes, back, "packing roundtrip failed for len {len}");
        }
    }

    // --- Encode size prediction ---

    #[test]
    fn test_encode_size_prediction() {
        let payload = b"a0Z5kR2g";
        let audio = encode(payload, 25).unwrap();

        let data_length = payload.len();
        let n_ecc = protocol::ecc_bytes_for_length(data_length);
        let total_bytes = protocol::ENCODED_DATA_OFFSET + data_length + n_ecc;
        let n_data_symbols = formant::symbols_for_bytes(total_bytes);
        let total_symbols = protocol::PREAMBLE_LEN
            + protocol::PILOT_LEN
            + n_data_symbols
            + protocol::PREAMBLE_LEN;
        let expected_samples = total_symbols * protocol::SYMBOL_TOTAL_SAMPLES;

        assert_eq!(
            audio.len(),
            expected_samples,
            "encode size mismatch: got {} expected {}",
            audio.len(),
            expected_samples
        );
    }

    // --- Round-trip tests ---

    #[test]
    fn test_roundtrip_hello() {
        let payload = b"hello";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_single_byte() {
        let payload = b"A";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_incremental_lengths() {
        let payload_full = b"a0Z5kR2g";
        for length in 1..=payload_full.len() {
            let payload = &payload_full[..length];
            let audio = encode(payload, 25).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "round-trip failed for length {length}"
            );
        }
    }

    #[test]
    fn test_roundtrip_medium() {
        let payload = b"The quick brown fox jumps over the lazy dog!12345";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_max_length() {
        let payload: Vec<u8> = (0..140).map(|i| (i % 256) as u8).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_binary_payload() {
        let payload: Vec<u8> = (0..32).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Streaming decode ---

    #[test]
    fn test_roundtrip_streaming() {
        let payload = b"stream test";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 512;
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_streaming_tiny_chunks() {
        let payload = b"tiny";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 100;
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decoder reset ---

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Decoder::new();
        let payload = b"test";
        let audio = encode(payload, 50).unwrap();
        let _ = decoder.decode(&audio);
        decoder.reset();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Expressive pitch API ---

    #[test]
    fn test_encode_with_pitch_roundtrip() {
        let payload = b"pitch track test";
        let track: Vec<f64> = payload
            .iter()
            .enumerate()
            .map(|(i, _)| protocol::F0_CANDIDATES[i % protocol::F0_CANDIDATES.len()])
            .collect();
        let audio = encode_with_pitch(payload, 50, Some(&track)).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_encode_with_pitch_length_mismatch() {
        let track = vec![146.83f64; 3];
        assert!(matches!(
            encode_with_pitch(b"hello", 50, Some(&track)),
            Err(Error::PitchTrackMismatch { .. })
        ));
    }

    // --- Volume 0 produces silent output ---

    #[test]
    fn test_encode_volume_zero_produces_silence() {
        let audio = encode(b"hi", 0).unwrap();
        assert!(audio.iter().all(|&s| s == 0.0));
    }

    // --- ECC bytes calculation ---

    #[test]
    fn test_ecc_bytes_for_length() {
        assert_eq!(protocol::ecc_bytes_for_length(1), 2);
        assert_eq!(protocol::ecc_bytes_for_length(2), 2);
        assert_eq!(protocol::ecc_bytes_for_length(3), 2);
        assert_eq!(protocol::ecc_bytes_for_length(4), 4);
        assert_eq!(protocol::ecc_bytes_for_length(5), 4);
        assert_eq!(protocol::ecc_bytes_for_length(10), 4);
        assert_eq!(protocol::ecc_bytes_for_length(15), 6);
        assert_eq!(protocol::ecc_bytes_for_length(50), 20);
        assert_eq!(protocol::ecc_bytes_for_length(140), 56);
    }

    // --- Noise robustness ---

    #[test]
    fn test_roundtrip_with_noise_2pct() {
        let payload = b"a0Z5kR2g";
        let mut audio = encode(payload, 25).unwrap();

        let mut rng_state: u32 = 12345;
        for sample in audio.iter_mut() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let frand = (rng_state >> 16) as f32 / 65535.0;
            *sample += (frand - 0.5) * 0.02;
            *sample = sample.clamp(-1.0, 1.0);
        }

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_with_noise_5pct() {
        let payload = b"noise5pct";
        let mut audio = encode(payload, 50).unwrap();

        let mut rng_state: u32 = 54321;
        for sample in audio.iter_mut() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let frand = (rng_state >> 16) as f32 / 65535.0;
            *sample += (frand - 0.5) * 0.05;
            *sample = sample.clamp(-1.0, 1.0);
        }

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Silence returns None ---

    #[test]
    fn test_decode_silence_returns_none() {
        let silence = vec![0.0f32; 48000];
        let mut decoder = Decoder::new();
        let result = decoder.decode(&silence).unwrap();
        assert_eq!(result, None, "silence should produce None, not a decode");
    }

    // --- Random noise no panic ---

    #[test]
    fn test_decode_random_noise_no_panic() {
        let mut rng_state: u32 = 99999;
        let noise: Vec<f32> = (0..48000)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65535.0;
                (f - 0.5) * 2.0
            })
            .collect();
        let mut decoder = Decoder::new();
        let _ = decoder.decode(&noise);
    }

    // --- Sequential decodes ---

    #[test]
    fn test_sequential_decodes() {
        let payloads: &[&[u8]] = &[b"first", b"second", b"third"];
        let mut decoder = Decoder::new();
        for payload in payloads {
            let audio = encode(payload, 50).unwrap();
            decoder.reset();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(*payload),
                "sequential decode failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Volume sweep ---

    #[test]
    fn test_roundtrip_volume_sweep() {
        let payload = b"volume test";
        for volume in [1, 5, 10, 25, 50, 75, 100] {
            let audio = encode(payload, volume).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "round-trip failed at volume {volume}"
            );
        }
    }

    // --- Encode determinism ---

    #[test]
    fn test_encode_deterministic() {
        let payload = b"deterministic";
        let audio1 = encode(payload, 50).unwrap();
        let audio2 = encode(payload, 50).unwrap();
        assert_eq!(audio1.len(), audio2.len());
        for (i, (a, b)) in audio1.iter().zip(audio2.iter()).enumerate() {
            assert_eq!(a, b, "sample {i} differs between two encodes");
        }
    }

    // --- Padded both sides ---

    #[test]
    fn test_roundtrip_padded_both_sides() {
        let payload = b"padded";
        let audio = encode(payload, 50).unwrap();

        // Use window-aligned padding to avoid partial-window misalignment issues
        // at the boundary between silence and audio.
        let aligned_padding = (24000 / crate::protocol::SYMBOL_TOTAL_SAMPLES + 1)
            * crate::protocol::SYMBOL_TOTAL_SAMPLES;
        let mut padded = vec![0.0f32; aligned_padding];
        padded.extend_from_slice(&audio);
        padded.extend(vec![0.0f32; aligned_padding]);

        let mut decoder = Decoder::new();
        let result = decoder.decode(&padded).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- ECC boundary lengths ---

    #[test]
    fn test_roundtrip_ecc_boundaries() {
        let test_lengths = [3, 4, 5, 10, 15, 25, 50, 100, 140];
        for &len in &test_lengths {
            let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + 13) % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "ECC boundary round-trip failed for length {len}"
            );
        }
    }

    // --- All same byte values ---

    #[test]
    fn test_roundtrip_all_zeros() {
        let payload = vec![0u8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_all_ones() {
        let payload = vec![0xFFu8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- 16-bit quantization ---

    #[test]
    fn test_roundtrip_16bit_quantization() {
        let cases: &[&[u8]] = &[b"hello", b"A", b"test1234", b"\x00\xFF\x80\x7F"];
        for &payload in cases {
            let audio = encode(payload, 50).unwrap();

            let quantized: Vec<f32> = audio
                .iter()
                .map(|&s| {
                    let i16_val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    i16_val as f32 / 32768.0
                })
                .collect();

            let mut decoder = Decoder::new();
            let result = decoder.decode(&quantized).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "16-bit quantization round-trip failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Truncated audio ---

    #[test]
    fn test_truncated_audio_no_panic() {
        let payload = b"truncate me";
        let audio = encode(payload, 50).unwrap();

        for &frac in &[0.25, 0.50, 0.75] {
            let end = (audio.len() as f64 * frac) as usize;
            let truncated = &audio[..end];
            let mut decoder = Decoder::new();
            let _ = decoder.decode(truncated);
        }
    }

    // --- DC offset ---

    #[test]
    fn test_roundtrip_with_dc_offset() {
        let payload = b"dc offset";
        let audio = encode(payload, 50).unwrap();

        let offset_audio: Vec<f32> = audio.iter().map(|&s| s + 0.1).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&offset_audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Clipped audio ---

    #[test]
    fn test_roundtrip_clipped_audio() {
        let payload = b"clipped";
        // Volume 50 with ±0.5 clipping: moderate clipping distortion
        let audio = encode(payload, 50).unwrap();
        let clipped: Vec<f32> = audio.iter().map(|&s| s.clamp(-0.5, 0.5)).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&clipped).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Amplitude scaling ---

    #[test]
    fn test_roundtrip_scaled_amplitude() {
        let payload = b"scale test";
        let audio = encode(payload, 50).unwrap();

        let scaled: Vec<f32> = audio.iter().map(|&s| s * 0.1).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&scaled).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Symbol classification across the whole melody ---

    #[test]
    fn test_all_symbols_classify_at_all_melody_notes() {
        let mut failures = Vec::new();
        for pos in 0..protocol::MELODY.len() {
            let f0 = protocol::melody_f0(pos);
            for sym in 0..protocol::NUM_SYMBOLS {
                let slot = formant::synthesize_symbol(sym, f0, 0.5, pos);
                let voiced: Vec<f32> = slot[..protocol::SAMPLES_PER_SYMBOL]
                    .iter()
                    .map(|&s| s as f32)
                    .collect();
                let mut spectrum = vec![0.0f32; voiced.len()];
                fft::power_spectrum(&voiced, &mut spectrum);
                let (detected, _conf) =
                    formant::classify_symbol(&spectrum, &formant::EXPECTED_F2_CENTROIDS);
                if detected != sym {
                    failures.push((f0, sym, detected));
                }
            }
        }
        if !failures.is_empty() {
            for (f0, expected, got) in &failures {
                eprintln!("FAIL vowel {expected} at F0={f0} → {got}");
            }
            panic!("{} (note, vowel) pairs misclassified", failures.len());
        }
    }

    // --- Sample counts consistent ---

    #[test]
    fn test_encode_sample_counts_consistent() {
        for size in [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 140] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();

            let n_ecc = protocol::ecc_bytes_for_length(size);
            let total_bytes = protocol::ENCODED_DATA_OFFSET + size + n_ecc;
            let n_data_symbols = formant::symbols_for_bytes(total_bytes);
            let total_symbols = protocol::PREAMBLE_LEN
                + protocol::PILOT_LEN
                + n_data_symbols
                + protocol::PREAMBLE_LEN;
            let expected = total_symbols * protocol::SYMBOL_TOTAL_SAMPLES;

            assert_eq!(
                audio.len(),
                expected,
                "sample count mismatch for payload size {size}: got {} expected {}",
                audio.len(),
                expected
            );
        }
    }

    // --- Every payload length 1..=140 round-trips ---

    #[test]
    fn test_roundtrip_every_length() {
        let mut failures = Vec::new();
        for len in 1..=140 {
            let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + len * 3) % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            match decoder.decode(&audio) {
                Ok(Some(data)) if data == payload => {}
                Ok(Some(data)) => {
                    failures.push((len, format!("wrong data (got {} bytes)", data.len())));
                }
                Ok(None) => failures.push((len, "None".into())),
                Err(e) => failures.push((len, format!("Err: {e}"))),
            }
        }
        if !failures.is_empty() {
            for (len, msg) in &failures {
                eprintln!("FAIL length {len}: {msg}");
            }
            panic!("{} of 140 lengths failed", failures.len());
        }
    }

    // --- High-pass filter (remove DC and sub-bass, keep speech band) ---

    #[test]
    fn test_roundtrip_highpass() {
        let payload = b"highpass test";
        let audio = encode(payload, 50).unwrap();

        // Simple frequency-domain highpass: remove everything below 100 Hz
        let n = audio.len();
        let hz_per_bin = protocol::SAMPLE_RATE / n as f64;

        use rustfft::{num_complex::Complex, FftPlanner};
        let mut buffer: Vec<Complex<f32>> = audio
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        for (i, c) in buffer.iter_mut().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * hz_per_bin
            } else {
                (n - i) as f64 * hz_per_bin
            };
            if freq < 100.0 {
                *c = Complex::new(0.0, 0.0);
            }
        }

        let ifft = planner.plan_fft_inverse(n);
        ifft.process(&mut buffer);

        let filtered: Vec<f32> = buffer.iter().map(|c| c.re / n as f32).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&filtered).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }
}
