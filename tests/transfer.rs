//! Transfer-profile tests: CRC gating, paranoid ECC, repeat + soft combining.

use song_rs::protocol::*;
use song_rs::transfer::{self, TransferRx};

/// Deterministic white noise, amplitude `amp`.
fn noise(len: usize, amp: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0 * amp
        })
        .collect()
}

/// Replace a sample range with strong noise (simulates a corrupted burst).
fn corrupt_range(audio: &mut [f32], start: usize, len: usize, seed: u32) {
    let end = (start + len).min(audio.len());
    let n = noise(end - start, 0.3, seed);
    audio[start..end].copy_from_slice(&n);
}

/// First data-region sample index (after start preamble + pilot).
fn data_start_sample() -> usize {
    (PREAMBLE_LEN + PILOT_LEN) * SYMBOL_TOTAL_SAMPLES
}

#[test]
fn test_transfer_roundtrip_clean() {
    let key = b"sk-live-4242deadbeefCAFEBABE0123456789";
    let audio = transfer::encode(key, 50).unwrap();
    let mut rx = TransferRx::new();
    let got = rx.ingest(&audio).expect("clean transfer failed");
    assert_eq!(got, key);
}

#[test]
fn test_transfer_max_payload() {
    let key: Vec<u8> = (0..transfer::MAX_TRANSFER_PAYLOAD)
        .map(|i| b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"[i % 32])
        .collect();
    let audio = transfer::encode(&key, 50).unwrap();
    let mut rx = TransferRx::new();
    assert_eq!(rx.ingest(&audio).expect("max payload failed"), key);
    assert!(transfer::encode(&vec![b'x'; transfer::MAX_TRANSFER_PAYLOAD + 1], 50).is_err());
}

#[test]
fn test_transfer_never_delivers_wrong_payload() {
    // Heavily corrupt the audio at increasing severities. The receiver may
    // fail to decode, but must NEVER return bytes different from the key.
    let key = b"sk-live-integrity-gate-test-0001";
    let audio = transfer::encode(key, 50).unwrap();
    let ds = data_start_sample();

    for (i, frac) in [0.3, 0.5, 0.7, 0.9].iter().enumerate() {
        let mut bad = audio.clone();
        let len = ((audio.len() - ds) as f64 * frac) as usize;
        corrupt_range(&mut bad, ds, len, 1000 + i as u32);
        let mut rx = TransferRx::new();
        if let Some(got) = rx.ingest(&bad) {
            assert_eq!(got, key, "wrong payload delivered at corruption {frac}");
        }
    }
}

#[test]
fn test_paranoid_survives_what_standard_cannot() {
    // Corrupt ~30% of the data region (codeword): beyond the standard
    // profile's reach (~40% parity: hard radius plus erasure/Chase
    // assistance tops out near 25%), within the paranoid profile's ~50%.
    let key = b"paranoid profile survival test 123";
    let framed_len = key.len() + 4; // + CRC

    // Standard profile, same framing, same corruption
    let std_audio = song_rs::encode(&{
        let mut f = key.to_vec();
        f.extend_from_slice(&transfer::crc32(key).to_le_bytes());
        f
    }, 50)
    .unwrap();
    let par_audio = transfer::encode(key, 50).unwrap();

    // Corrupt the same fraction of each data region
    let ds = data_start_sample();
    let corrupt_frac = 0.30;
    let std_len = ((std_audio.len() - ds) as f64 * corrupt_frac) as usize;
    let par_len = ((par_audio.len() - ds) as f64 * corrupt_frac) as usize;

    let mut std_bad = std_audio.clone();
    corrupt_range(&mut std_bad, ds, std_len, 7);
    let mut par_bad = par_audio.clone();
    corrupt_range(&mut par_bad, ds, par_len, 7);

    let mut std_dec = song_rs::Decoder::new();
    let std_ok = matches!(
        std_dec.decode(&std_bad),
        Ok(Some(ref d)) if d.len() == framed_len
    );
    assert!(!std_ok, "standard profile unexpectedly survived 30% corruption");

    let mut rx = TransferRx::new();
    let got = rx.ingest(&par_bad).expect("paranoid profile should survive 30%");
    assert_eq!(got, key);
}

#[test]
fn test_soft_combining_across_repeats() {
    // Three copies drowned in heavy independent noise, each too degraded to
    // decode alone. This is combining's real niche — TIME DIVERSITY: the
    // per-symbol classification errors are independent across copies, so a
    // 3-way majority vote cuts the symbol error rate enough for RS to
    // finish the job. (Burst resilience is the solo decoder's own job now:
    // erasures + Chase + brute-forced lengths decode a single copy with a
    // sub-50% burst, so bursts can no longer exercise this path.)
    let key = b"combine-me-9f8e7d6c5b4a39281706";
    let one = transfer::encode(key, 50).unwrap();
    let ds = data_start_sample();

    // Add white noise over the data region only (keep sync clean so every
    // copy locks): SNR low enough that each copy fails alone.
    let add_data_noise = |audio: &[f32], snr_db: f64, seed: u32| -> Vec<f32> {
        let data = &audio[ds..];
        let sig_power: f64 =
            data.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / data.len() as f64;
        let noise_rms = (sig_power.sqrt() / 10.0f64.powf(snr_db / 20.0)) as f32;
        let n = noise(data.len(), 1.0, seed);
        let mut out = audio.to_vec();
        for (i, s) in out[ds..].iter_mut().enumerate() {
            *s += n[i] * noise_rms;
        }
        out
    };

    // −20 dB: at the very edge of solo reach (some copies decode alone,
    // some don't — seed-dependent). The combining path must deliver
    // regardless of which copy first survives. There is deliberately no
    // "each copy must fail alone" assertion: solo decoding has become
    // strong enough (erasures + Chase + brute-forced lengths + validator
    // retry) that no white-noise level cleanly separates solo-fail from
    // combine-pass; the voting mechanism itself is unit-tested in
    // src/transfer.rs against injected confident symbol errors and shifts.
    #[cfg(not(feature = "8khz"))]
    let snr_db = -20.0;
    // 8 kHz windows are 6x shorter: ~8 dB less per-symbol integration gain.
    #[cfg(feature = "8khz")]
    let snr_db = -10.0;
    let copies: Vec<Vec<f32>> = (0..3)
        .map(|i| add_data_noise(&one, snr_db, 977 + i as u32))
        .collect();

    // Combined reception: copies separated by gaps
    let gap = vec![0.0f32; SYMBOL_TOTAL_SAMPLES * 6];
    let mut audio = Vec::new();
    for copy in &copies {
        audio.extend_from_slice(copy);
        audio.extend_from_slice(&gap);
    }

    let mut rx = TransferRx::new();
    let mut result = None;
    for chunk in audio.chunks(SYMBOL_TOTAL_SAMPLES) {
        if let Some(got) = rx.ingest(chunk) {
            result = Some(got);
            break;
        }
    }
    assert_eq!(result.as_deref(), Some(&key[..]), "soft combining failed");
}

#[test]
fn test_combining_garbage_never_false_positives() {
    // Two receptions of pure noise must not combine into a "successful" decode.
    let key = b"never-see-this";
    let one = transfer::encode(key, 50).unwrap();
    let mut rx = TransferRx::new();
    let gap = vec![0.0f32; SYMBOL_TOTAL_SAMPLES * 6];
    for seed in [31u32, 32] {
        let garbage = noise(one.len(), 0.3, seed);
        assert!(rx.ingest(&garbage).is_none());
        assert!(rx.ingest(&gap).is_none());
    }
}

#[test]
fn test_encode_repeated_layout() {
    let key = b"repeat";
    let one = transfer::encode(key, 50).unwrap();
    let three = transfer::encode_repeated(key, 50, 3, 500).unwrap();
    let gap = (SAMPLE_RATE as usize * 500) / 1000;
    assert_eq!(three.len(), one.len() * 3 + gap * 2);

    // And it still decodes from the front
    let mut rx = TransferRx::new();
    assert_eq!(rx.ingest(&three).expect("repeated decode failed"), key);
}

#[test]
fn test_transfer_survives_spectral_tilt() {
    // Strong channel tilt (one-pole lowpass ≈ -6 dB/octave) shifts every
    // vowel's F2 centroid downward; the pilot calibration measures the
    // shifted centers and classification adapts.
    let key = b"tilted-channel-key-0123456789ab";
    let audio = transfer::encode(key, 50).unwrap();
    let mut tilted = Vec::with_capacity(audio.len());
    let mut y = 0.0f32;
    for &s in &audio {
        y = 0.65 * y + 0.35 * s;
        tilted.push(y);
    }
    let mut rx = TransferRx::new();
    assert_eq!(
        rx.ingest(&tilted).expect("tilted channel decode failed"),
        key
    );
}

#[test]
fn test_is_locked_tracks_reception() {
    let key = b"sk-live-lock-state-visibility-01";
    let audio = transfer::encode(key, 50).unwrap();
    let mut rx = TransferRx::new();

    // Idle: nothing ingested yet.
    assert!(!rx.is_locked());

    // Feed the first half — well past the preamble, mid-reception.
    let half = audio.len() / 2;
    assert!(rx.ingest(&audio[..half]).is_none());
    assert!(rx.is_locked(), "should be locked mid-transmission");

    // Feed the rest — decode completes, lock releases.
    let got = rx.ingest(&audio[half..]).expect("full audio should decode");
    assert_eq!(got, key);
    assert!(!rx.is_locked(), "should unlock after successful decode");
}
