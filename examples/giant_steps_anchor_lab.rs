//! Giant Steps anchor lab: can the melody LOCALIZE a frame?
//!
//! The structural preamble matcher survives monotone spectral shaping but
//! not comb filtering (echo) — the decoder's known weak flank. The intro
//! Lick + head of Giant Steps is an 88-slot pitch sequence the receiver
//! knows in the default profile. This lab measures whether template
//! alignment against that sequence can find the frame START through
//! channels that defeat vowel-pattern sync, and compares against the real
//! Decoder's end-to-end delivery.
//!
//! VERDICT (measured 2026-09-12): localization works — the Lick+head
//! anchor finds the frame start within a fraction of a slot (|err| <=
//! ~430 samples) through echoes, double echoes, and 0 dB noise — but no
//! tested channel makes the production decoder fail end-to-end, so there
//! is nothing to rescue yet. NOT baked in (house rule: hints must move a
//! measured robustness goal). Revisit if a real acquisition-failure
//! channel is found; the estimator here is ready.
//!
//! Usage: cargo run --release --example giant_steps_anchor_lab

use song_rs::formant;
use song_rs::protocol::*;

const SR: f64 = 48000.0;

const BANK: [f64; 20] = [
    123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00,
    233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99,
];

fn refs() -> Vec<[Vec<f32>; 4]> {
    BANK.iter().map(|&hz| std::array::from_fn(|v| {
        let slot = formant::synthesize_symbol(v, hz, 0.5, 0);
        let voiced: Vec<f32> = slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| s as f32).collect();
        let mut spec = vec![0.0f32; SAMPLES_PER_SYMBOL];
        song_rs::fft::power_spectrum(&voiced, &mut spec);
        spec.iter().map(|&p| p.max(0.0).sqrt()).collect()
    })).collect()
}

const LO: usize = 50;
const HI: usize = 125;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in LO..HI {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn nearest(hz: f64) -> usize {
    BANK.iter().enumerate().min_by(|a, b|
        (a.1 - hz).abs().partial_cmp(&(b.1 - hz).abs()).unwrap()).unwrap().0
}

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
    let power: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
        / samples.len() as f64;
    let rms = (power.sqrt() / 10.0f64.powf(snr_db / 20.0)) as f32;
    let mut st = seed;
    samples.iter().map(|&s| s + normal(&mut st) * rms).collect()
}

fn add_echo(samples: &[f32], delay_ms: f64, amplitude: f32) -> Vec<f32> {
    let delay = (SR * delay_ms / 1000.0) as usize;
    let mut out = samples.to_vec();
    for i in delay..samples.len() {
        out[i] += samples[i - delay] * amplitude;
    }
    out
}

/// Expected pitch of frame-relative slot `k` in a default Standard frame:
/// intro Lick, then the head, one note per payload byte.
fn expected_hz(k: usize, track: &[f64]) -> Option<f64> {
    let intro = PREAMBLE_LEN + PILOT_LEN + 4 * ENCODED_DATA_OFFSET;
    if k < intro {
        Some(FRAMING_LICK[k])
    } else {
        track.get((k - intro) / 4).copied()
    }
}

fn main() {
    let refs = refs();
    let len = 64usize;
    let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
    let frame = song_rs::encode(&payload, 50).unwrap();
    let track = default_byte_pitch(len);
    // Score the intro plus the first 16 head bytes: 88 slots.
    let anchor_slots = PREAMBLE_LEN + PILOT_LEN + 4 * ENCODED_DATA_OFFSET + 4 * 16;

    let channels: Vec<(&str, Box<dyn Fn(&[f32], u32) -> Vec<f32>>)> = vec![
        ("clean", Box::new(|s: &[f32], _| s.to_vec())),
        ("echo25ms_a0.5", Box::new(|s: &[f32], _| add_echo(s, 25.0, 0.5))),
        ("echo25ms_a0.5_n10", Box::new(|s: &[f32], seed| add_noise(&add_echo(s, 25.0, 0.5), 10.0, seed))),
        ("echo60ms_a0.5_n10", Box::new(|s: &[f32], seed| add_noise(&add_echo(s, 60.0, 0.5), 10.0, seed))),
        ("echo25+60_a0.5_n6", Box::new(|s: &[f32], seed| {
            add_noise(&add_echo(&add_echo(s, 25.0, 0.5), 60.0, 0.5), 6.0, seed)
        })),
        ("noise_0db", Box::new(|s: &[f32], seed| add_noise(s, 0.0, seed))),
    ];

    println!("{:<22} {:>6} {:>12} {:>10}", "channel", "seed", "start_err", "decoder");
    for (name, channel) in &channels {
        for seed in [11u32, 12, 13] {
            // Frame embedded after a noise-only prefix, background after.
            let offset = 48_000 + (seed as usize * 977) % 2880;
            let mut audio = vec![0.0f32; offset];
            audio.extend_from_slice(&frame);
            audio.extend(std::iter::repeat(0.0f32).take(48_000));
            let received = channel(&audio, seed);

            // Melody anchor: 8 sample phases x slot grid, sim rows cached
            // per (phase, slot) so each candidate start is a table sum.
            let step = SYMBOL_TOTAL_SAMPLES / 8;
            let expected_notes: Vec<usize> = (0..anchor_slots)
                .map(|k| nearest(expected_hz(k, &track).unwrap()))
                .collect();
            let mut best = (f64::MIN, 0usize);
            for phase in 0..8usize {
                let base = phase * step;
                let slots = (received.len().saturating_sub(base + SAMPLES_PER_SYMBOL))
                    / SYMBOL_TOTAL_SAMPLES;
                if slots < anchor_slots {
                    continue;
                }
                let mut rows: Vec<[f64; 20]> = Vec::with_capacity(slots);
                let mut spec = vec![0.0f32; SAMPLES_PER_SYMBOL];
                for m in 0..slots {
                    let start = base + m * SYMBOL_TOTAL_SAMPLES;
                    song_rs::fft::power_spectrum(
                        &received[start..start + SAMPLES_PER_SYMBOL], &mut spec,
                    );
                    let amp: Vec<f32> = spec.iter().map(|&p| p.max(0.0).sqrt()).collect();
                    let mut row = [0.0f64; 20];
                    for (n, r) in refs.iter().enumerate() {
                        row[n] = (0..4).map(|v| cosine(&amp, &r[v])).fold(0.0f64, f64::max);
                    }
                    rows.push(row);
                }
                for m0 in 0..slots - anchor_slots {
                    let score: f64 = (0..anchor_slots)
                        .map(|k| rows[m0 + k][expected_notes[k]])
                        .sum();
                    if score > best.0 {
                        best = (score, base + m0 * SYMBOL_TOTAL_SAMPLES);
                    }
                }
            }
            let err = best.1 as isize - offset as isize;

            // Baseline: the real decoder end to end.
            let mut decoder = song_rs::Decoder::new();
            let mut delivered = false;
            for chunk in received.chunks(2048) {
                if let Ok(Some(bytes)) = decoder.decode(chunk) {
                    delivered = bytes == payload;
                }
            }
            println!(
                "{:<22} {:>6} {:>12} {:>10}",
                name, seed, format!("{err:+}"), if delivered { "ok" } else { "FAIL" }
            );
        }
    }
    println!("\nstart_err in samples (one slot = {SYMBOL_TOTAL_SAMPLES}); anchor is useful when |err| is small where decoder FAILs");
}
