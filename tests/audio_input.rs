//! Invalid microphone values must not contaminate later valid audio.

use song_rs::protocol::*;
use song_rs::transfer::{self, TransferRx};
use song_rs::{Decoder, StreamConfig, StreamRx, StreamTx};

const INVALID: [f32; 5] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::from_bits(0x7f80_0001),
    f32::from_bits(0xffc0_0123),
];

fn corrupt_each_slot(audio: &mut [f32]) {
    for (index, slot) in audio.chunks_mut(SYMBOL_TOTAL_SAMPLES).enumerate() {
        if slot.len() > SAMPLES_PER_SYMBOL / 2 {
            slot[SAMPLES_PER_SYMBOL / 2] = INVALID[index % INVALID.len()];
        }
        if slot.len() > SAMPLES_PER_SYMBOL {
            slot[SAMPLES_PER_SYMBOL] = INVALID[(index + 1) % INVALID.len()];
        }
    }
}

fn invalid_idle(extra_samples: usize) -> Vec<f32> {
    (0..4 * SYMBOL_TOTAL_SAMPLES + extra_samples)
        .map(|i| INVALID[i % INVALID.len()])
        .collect()
}

#[test]
fn block_receivers_recover_nonfinite_samples_across_melodies() {
    // Every vowel at every semitone, plus the recognizable head and detuning.
    let payload: Vec<_> = (0..80).map(|i| [0x00, 0x55, 0xaa, 0xff][i % 4]).collect();
    let chromatic: Vec<_> = (0..80)
        .map(|i| 440.0 * 2.0f64.powf((47.0 + (i / 4) as f64 - 69.0) / 12.0))
        .collect();
    let detuned: Vec<_> = chromatic
        .iter()
        .rev()
        .enumerate()
        .map(|(i, hz)| hz * 2.0f64.powf(if i % 2 == 0 { 0.3 / 12.0 } else { -0.3 / 12.0 }))
        .collect();
    for profile in [
        EccProfile::Standard,
        EccProfile::Legacy,
        EccProfile::Paranoid,
    ] {
        for (melody, track) in [
            default_byte_pitch(payload.len()),
            chromatic.clone(),
            detuned.clone(),
        ]
        .iter()
        .enumerate()
        {
            let mut audio =
                song_rs::encoder::encode_with_profile(&payload, 50, Some(track), profile).unwrap();
            corrupt_each_slot(&mut audio);
            let mut input = invalid_idle(137);
            input.extend(audio);
            input.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
            for chunk_size in [1, 437, 2048] {
                let mut decoder = Decoder::with_profile(profile);
                let mut deliveries = Vec::new();
                for chunk in input.chunks(chunk_size) {
                    if let Some(bytes) = decoder.decode(chunk).unwrap() {
                        deliveries.push(bytes);
                    }
                }
                while let Some(bytes) = decoder.decode(&[]).unwrap() {
                    deliveries.push(bytes);
                }
                assert_eq!(
                    deliveries,
                    vec![payload.clone()],
                    "{profile:?}, melody {melody}, chunks {chunk_size}"
                );
                assert!(!decoder.is_receiving());
            }
        }
    }
}

#[test]
fn transfer_recovers_after_invalid_idle_and_sparse_nonfinite_samples() {
    let payload: Vec<_> = (0..transfer::MAX_TRANSFER_PAYLOAD)
        .map(|i| (i * 73 + 17) as u8)
        .collect();
    let mut audio = transfer::encode(&payload, 50).unwrap();
    corrupt_each_slot(&mut audio);
    let mut input = invalid_idle(137);
    input.extend(audio);
    input.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
    for chunk_size in [128, 511, 2048] {
        let mut rx = TransferRx::new();
        let deliveries: Vec<_> = input
            .chunks(chunk_size)
            .filter_map(|chunk| rx.ingest(chunk))
            .collect();
        assert_eq!(deliveries, vec![payload.clone()], "chunks {chunk_size}");
        assert!(!rx.is_locked());
        assert_eq!(
            rx.attempts(),
            0,
            "sparse repairs must not require combining"
        );
    }
}

#[test]
fn streaming_receiver_recovers_nonfinite_samples_and_accepts_the_next_song() {
    for depth in [1, 2, 4] {
        let config = StreamConfig {
            interleave_depth: depth,
            ..StreamConfig::default()
        };
        let payloads: Vec<Vec<_>> = [23, 54]
            .into_iter()
            .map(|len| (0..len).map(|i| (i * 73 + len) as u8).collect())
            .collect();
        let mut input = Vec::new();
        for payload in &payloads {
            input.extend(invalid_idle(0));
            let mut tx = StreamTx::new(config.clone());
            tx.feed(payload);
            tx.finish();
            let mut audio = Vec::new();
            loop {
                let mut buffer = [0.0; 2048];
                let count = tx.emit(&mut buffer);
                if count == 0 {
                    break;
                }
                audio.extend_from_slice(&buffer[..count]);
            }
            corrupt_each_slot(&mut audio);
            input.extend(audio);
            input.extend(vec![0.0; 12 * SYMBOL_TOTAL_SAMPLES]);
        }
        for chunk_size in [1, 437, 2048] {
            let mut rx = StreamRx::new(config.clone());
            let mut deliveries = Vec::new();
            for chunk in input.chunks(chunk_size) {
                rx.ingest(chunk);
                let bytes = rx.read_all();
                if !bytes.is_empty() {
                    deliveries.push(bytes);
                }
            }
            assert_eq!(deliveries, payloads, "depth {depth}, chunks {chunk_size}");
            assert!(!rx.is_active());
        }
    }
}
