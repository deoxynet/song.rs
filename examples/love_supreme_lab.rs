//! A Love Supreme lab: transposed repeat transmissions.
//!
//! In "Acknowledgement" (1964), Coltrane repeats the four-note motif
//! through every key. The transfer profile repeats copies IDENTICALLY, so
//! note-dependent channel damage — reverb comb nulls, narrowband notches —
//! corrupts the same symbols in every copy: correlated errors, the worst
//! case for cross-copy soft combining. Pitch carries no data, so each
//! repeat may be transposed for free; a null that eats one copy's harmonic
//! misses the next copy's. This lab measures identical vs transposed
//! repeats through comb/notch channels via the real `TransferRx`.
//!
//! VERDICT (measured 2026-09-12): directionally strong, not yet baked in.
//! In the transition band (5 ms echo at 0.95 amplitude, -6 dB SNR) single
//! copies fail 3/3 seeds, identical 3-copy repeats recover 1/3, and
//! TRANSPOSED repeats recover 3/3 — correlated comb damage defeats
//! combining; transposition decorrelates it. Three data points is thin
//! evidence for changing `encode_repeated`'s default; widen the matrix
//! before shipping.
//!
//! Usage: cargo run --release --example love_supreme_lab

use song_rs::protocol::SYMBOL_TOTAL_SAMPLES;
use song_rs::transfer::{crc32, TransferRx};

const SAMPLE_RATE: f64 = 48000.0;

/// Per-copy transpositions (semitones) keeping the armored arpeggio
/// D3–G#3–C#4 inside the B2..F#4 compass: 0, +2, -2, +4, -3.
const SHIFTS: [i32; 5] = [0, 2, -2, 4, -3];

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
    let power: f64 =
        samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    let rms = (power.sqrt() / 10.0f64.powf(snr_db / 20.0)) as f32;
    let mut st = seed;
    samples.iter().map(|&s| s + normal(&mut st) * rms).collect()
}

/// Single echo: the comb filter. Nulls sit at odd multiples of
/// 1/(2*delay) — which harmonics die depends on the note being sung.
fn add_echo(samples: &[f32], delay_ms: f64, amplitude: f32) -> Vec<f32> {
    let delay = (SAMPLE_RATE * delay_ms / 1000.0) as usize;
    let mut out = samples.to_vec();
    for i in delay..samples.len() {
        out[i] += samples[i - delay] * amplitude;
    }
    out
}

/// The armored transfer melody with a per-copy transposition, as a
/// per-byte pitch track over payload + CRC bytes.
fn transposed_track(framed_len: usize, shift: i32) -> Vec<f64> {
    let base = [146.83f64, 207.65, 277.18, 207.65]; // D3 G#3 C#4 G#3
    let factor = 2.0f64.powf(shift as f64 / 12.0);
    (0..framed_len).map(|i| base[i % base.len()] * factor).collect()
}

fn encode_copies(payload: &[u8], volume: u8, copies: usize, transpose: bool) -> Vec<f32> {
    let mut framed = payload.to_vec();
    framed.extend_from_slice(&crc32(payload).to_le_bytes());
    let gap = vec![0.0f32; (SAMPLE_RATE as usize * 400) / 1000];
    let mut out = Vec::new();
    for c in 0..copies {
        if c > 0 {
            out.extend_from_slice(&gap);
        }
        let track = if transpose {
            Some(transposed_track(framed.len(), SHIFTS[c % SHIFTS.len()]))
        } else {
            None
        };
        let audio = song_rs::encoder::encode_with_profile(
            &framed,
            volume,
            track.as_deref(),
            song_rs::protocol::EccProfile::Paranoid,
        )
        .unwrap();
        out.extend_from_slice(&audio);
    }
    out
}

fn recovers(audio: &[f32], payload: &[u8]) -> bool {
    let mut rx = TransferRx::new();
    let mut got = None;
    for chunk in audio.chunks(2048) {
        if let Some(bytes) = rx.ingest(chunk) {
            got = Some(bytes);
        }
    }
    if got.is_none() {
        // Trailing silence lets the last copy finalize.
        let silence = vec![0.0f32; 6 * SYMBOL_TOTAL_SAMPLES];
        got = rx.ingest(&silence);
    }
    got.as_deref() == Some(payload)
}

fn main() {
    let payload: Vec<u8> = (0..32).map(|i| (i * 73 + 41) as u8).collect();
    let copies = 3;

    println!(
        "{:<34} {:>6} {:>10} {:>12}",
        "channel", "single", "identical", "transposed"
    );
    let mut id_total = 0usize;
    let mut tr_total = 0usize;
    let mut cases = 0usize;
    for delay_ms in [3.0f64, 5.0, 9.0, 15.0, 25.0] {
        for amp in [0.95f32] {
            for snr in [6.0f64, 3.0, 0.0, -3.0, -6.0] {
                for seed in [11u32, 12, 13] {
                    let channel =
                        |s: &[f32], seed: u32| add_noise(&add_echo(s, delay_ms, amp), snr, seed);
                    let single =
                        recovers(&channel(&encode_copies(&payload, 50, 1, false), seed), &payload);
                    let id = recovers(
                        &channel(&encode_copies(&payload, 50, copies, false), seed), &payload,
                    );
                    let tr = recovers(
                        &channel(&encode_copies(&payload, 50, copies, true), seed), &payload,
                    );
                    // Only the transition band is interesting: single fails.
                    if !single || id != tr {
                        println!(
                            "{:<34} {:>6} {:>10} {:>12}",
                            format!("echo{delay_ms}ms_a{amp}_snr{snr}_s{seed}"),
                            single as u8,
                            id as u8,
                            tr as u8
                        );
                    }
                    if !single {
                        id_total += id as usize;
                        tr_total += tr as usize;
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("\nsingle-copy-failed cases: {cases}; identical recovered {id_total}, transposed recovered {tr_total}");
}
