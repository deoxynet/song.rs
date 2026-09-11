//! Giant Steps as a carrier melody: pitch carries no data, so the encoder
//! may sing Coltrane while the vowels carry the payload.
//!
//! The head is octave-folded into the modem's register (~D3..C#4). The
//! default melody notes are margin-verified for reverb-comb safety; Giant
//! Steps needs ten pitch classes, so this example VERIFIES the result
//! end-to-end through the nasty channels instead of trusting the register.
//!
//! Usage: cargo run --release --example giant_steps [output_dir]

use nnnoiseless::DenoiseState;
use rustfft::{num_complex::Complex, FftPlanner};
use song_rs::Decoder;

const SAMPLE_RATE: f64 = 48000.0;

// ── Note table ───────────────────────────────────────────────────────
//
// Melody taken verbatim from the "Giant Steps" MIDI head (track 0,
// quantized to the byte grid), transposed down an octave plus a major
// third (Giant Steps in G): the TRUE contour — including the rising
// second ending (G5 F5 | Bb5) — then fits the modem's B2..F#4 compass
// exactly, with no octave folds. Pitch carries no data, so key is free.

const B2: f64 = 123.47;
const CS3: f64 = 138.59;
const D3: f64 = 146.83;
const EB3: f64 = 155.56;
const F3: f64 = 174.61;
const FS3: f64 = 185.00;
const G3: f64 = 196.00;
const A3: f64 = 220.00;
const BB3: f64 = 233.08;
const B3: f64 = 246.94;
const CS4: f64 = 277.18;
const D4: f64 = 293.66;
const EB4: f64 = 311.13;
const FS4: f64 = 369.99;

/// The 16-bar head: (note, payload bytes). One byte = 4 symbols = 240 ms
/// = one quarter note, so the half-note pulse runs ~250 bpm in
/// quarter-note terms — right in Giant Steps territory. Sounding note ↔
/// original: D4=F#5, Bb3=D5, G3=B4, Eb3=G4, F#3=Bb4, F3=A4, B2=Eb4,
/// D3=F#4, C#3=F4, B3=Eb5, A3=C#5, Eb4=G5, C#4=F5, F#4=Bb5.
const HEAD: &[(f64, usize)] = &[
    // 1: F#5 D5              2: B4 G4
    (D4, 2), (BB3, 2),        (G3, 2), (EB3, 2),
    // 3: Bb4 (held)          4: B4 A4
    (FS3, 4),                 (G3, 2), (F3, 2),
    // 5: D5 Bb4              6: G4 Eb4
    (BB3, 2), (FS3, 2),       (EB3, 2), (B2, 2),
    // 7: F#4 (held)          8: G4 F4
    (D3, 4),                  (EB3, 2), (CS3, 2),
    // 9: Bb4 (held)          10: B4 A4
    (FS3, 4),                 (G3, 2), (F3, 2),
    // 11: D5 (held)          12: Eb5 C#5
    (BB3, 4),                 (B3, 2), (A3, 2),
    // 13: F#5 (held)         14: G5 F5
    (D4, 4),                  (EB4, 2), (CS4, 2),
    // 15: Bb5 (held)         16: F#5 F#5 (turnaround)
    (FS4, 4),                 (D4, 2), (D4, 2),
];

// ── Channel simulations (as in tests/robustness.rs) ──────────────────

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

fn reverb(samples: &[f32], echoes: &[(f64, f32)]) -> Vec<f32> {
    let mut out = samples.to_vec();
    for &(delay_ms, amp) in echoes {
        let delay = (delay_ms * SAMPLE_RATE / 1000.0) as usize;
        for i in delay..out.len() {
            out[i] += samples[i - delay] * amp;
        }
    }
    out
}

fn teams_spectral_filter(samples: &[f32], f1_atten_db: f64, f0_boost_db: f64) -> Vec<f32> {
    let n = samples.len();
    let hz_per_bin = SAMPLE_RATE / n as f64;
    let mut buf: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, c) in buf.iter_mut().enumerate() {
        let freq = if i <= n / 2 {
            i as f64 * hz_per_bin
        } else {
            (n - i) as f64 * hz_per_bin
        };
        let gain_db = if freq < 250.0 {
            f0_boost_db * (1.0 - (freq - 208.0).abs() / 50.0).max(0.0)
        } else if (400.0..=850.0).contains(&freq) {
            -f1_atten_db
        } else if freq > 850.0 && freq < 2000.0 {
            -f1_atten_db * (1.0 - (freq - 850.0) / 1150.0)
        } else if (2000.0..=2500.0).contains(&freq) {
            -f1_atten_db * 0.3
        } else {
            0.0
        };
        let gain = 10.0f64.powf(gain_db / 20.0) as f32;
        *c = Complex::new(c.re * gain, c.im * gain);
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f32).collect()
}

fn apply_rnnoise(samples: &[f32]) -> Vec<f32> {
    let frame_size = DenoiseState::FRAME_SIZE;
    let mut state = DenoiseState::new();
    let mut out = Vec::with_capacity(samples.len());
    let scale_in = 32767.0f32;
    let scale_out = 1.0 / 32767.0f32;
    let mut input_frame = vec![0.0f32; frame_size];
    let mut output_frame = vec![0.0f32; frame_size];
    let first_len = frame_size.min(samples.len());
    input_frame[..first_len].copy_from_slice(&samples[..first_len]);
    for s in &mut input_frame[..first_len] {
        *s *= scale_in;
    }
    state.process_frame(&mut output_frame, &input_frame);
    state.process_frame(&mut output_frame, &input_frame);
    for s in &output_frame[..first_len] {
        out.push(s * scale_out);
    }
    let mut pos = first_len;
    while pos < samples.len() {
        let chunk_len = frame_size.min(samples.len() - pos);
        input_frame[..chunk_len].copy_from_slice(&samples[pos..pos + chunk_len]);
        for s in &mut input_frame[chunk_len..] {
            *s = 0.0;
        }
        for s in &mut input_frame[..chunk_len] {
            *s *= scale_in;
        }
        state.process_frame(&mut output_frame, &input_frame);
        for s in &output_frame[..chunk_len] {
            out.push(s * scale_out);
        }
        pos += chunk_len;
    }
    out
}

// ── Output ───────────────────────────────────────────────────────────

fn write_wav(path: &str, audio: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-9);
    let gain = 0.6 / peak;
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in audio {
        w.write_sample(((s * gain) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

fn verify(audio: &[f32], payload: &[u8]) -> bool {
    let mut dec = Decoder::new();
    match dec.decode(audio) {
        Ok(Some(ref d)) => return d == payload,
        Err(_) => return false,
        Ok(None) => {}
    }
    let silence = vec![0.0f32; 10 * song_rs::protocol::SYMBOL_TOTAL_SAMPLES];
    matches!(dec.decode(&silence), Ok(Some(ref d)) if d == payload)
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp".to_string());

    // Flatten the head into a per-byte pitch track.
    let track: Vec<f64> = HEAD
        .iter()
        .flat_map(|&(hz, bytes)| std::iter::repeat(hz).take(bytes))
        .collect();

    // Payload sized to the melody: one character per byte-slot.
    let mut msg = b"Giant Steps - John Coltrane (1960) - sung by the vowel modem".to_vec();
    assert!(msg.len() <= track.len(), "message longer than the head");
    msg.resize(track.len(), b' ');

    let audio = song_rs::encoder::encode_with_pitch(&msg, 50, Some(&track)).unwrap();
    println!(
        "encoded {} bytes over {} melody notes, {:.1}s of audio",
        msg.len(),
        HEAD.len(),
        audio.len() as f64 / SAMPLE_RATE
    );

    let channels: Vec<(&str, Vec<f32>)> = vec![
        ("clean", audio.clone()),
        (
            "reverb_large_room",
            reverb(&audio, &[(25.0, 0.25), (60.0, 0.15), (100.0, 0.10), (150.0, 0.05)]),
        ),
        ("noise_10db", add_noise(&audio, 10.0, 42)),
        ("rnnoise_noisy", apply_rnnoise(&add_noise(&audio, 20.0, 42))),
        ("teams_tilt_40db", teams_spectral_filter(&audio, 40.0, 22.0)),
        (
            "teams_full_pipeline",
            apply_rnnoise(&teams_spectral_filter(&audio, 40.0, 15.0)),
        ),
    ];

    for (name, ch_audio) in &channels {
        let ok = verify(ch_audio, &msg);
        let expected = *name != "teams_full_pipeline";
        println!(
            "  {:<22} {}{}",
            name,
            if ok { "decodes OK" } else { "NO DECODE" },
            if ok == expected { "" } else { "  (UNEXPECTED)" }
        );
        if *name == "clean" || *name == "teams_full_pipeline" {
            write_wav(&format!("{dir}/giant_steps_{name}.wav"), ch_audio);
        }
    }
    write_wav(&format!("{dir}/giant_steps.wav"), &audio);
    // The worst-case pipeline (40 dB tilt + RNNoise) measurably DESTROYS
    // certain (note, vowel) windows — every classifier rung fails on the
    // same ~7% of symbols, so this is channel information loss, not decoder
    // slack. That channel is exactly why the DEFAULT melody restricts
    // itself to the margin-verified D lydian notes; expressive tracks trade
    // the last channel for the tune.
    println!("(teams_full_pipeline is expected to fail: see comment in main)");
}
