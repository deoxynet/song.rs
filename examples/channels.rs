//! Channel-degradation listening demo: the same message rendered clean and
//! through progressively nastier channels, each verified to still decode.
//!
//! Usage: cargo run --release --example channels [output_dir]
//! Writes one WAV per channel into output_dir (default /tmp).

use nnnoiseless::DenoiseState;
use rustfft::{num_complex::Complex, FftPlanner};
use song_rs::Decoder;

const SAMPLE_RATE: f64 = 48000.0;

// ── Channel simulations (mirrors tests/robustness.rs) ────────────────

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

/// Teams-like spectral shaping measured from a real call: F0 boosted, the
/// F1 band crushed, and a steep tilt across the F2 decision band.
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

/// Speech-shaped babble at a target signal-to-interference ratio.
fn add_babble(samples: &[f32], sir_db: f64, seed: u32) -> Vec<f32> {
    let mut st = seed;
    let white: Vec<f32> = (0..samples.len()).map(|_| normal(&mut st)).collect();
    let rc = 1.0 / (2.0 * std::f64::consts::PI * 500.0);
    let dt = 1.0 / SAMPLE_RATE;
    let alpha = (dt / (rc + dt)) as f32;
    let mut y = 0.0f32;
    let babble: Vec<f32> = white
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            y += alpha * (x - y);
            let t = i as f64 / SAMPLE_RATE;
            let syllabic = (0.55 + 0.45 * (2.0 * std::f64::consts::PI * 3.5 * t).sin()) as f32;
            y * syllabic
        })
        .collect();
    let sig_power: f64 =
        samples.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / samples.len() as f64;
    let bab_power: f64 =
        babble.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / babble.len() as f64;
    let gain = ((sig_power / bab_power).sqrt() / 10.0f64.powf(sir_db / 20.0)) as f32;
    samples.iter().zip(&babble).map(|(&s, &n)| s + n * gain).collect()
}

// ── Output ───────────────────────────────────────────────────────────

fn write_wav(path: &str, audio: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    // Normalize each clip to a comfortable, consistent playback level.
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
    let msg = b"can you hear me now?";
    let clean = song_rs::encode(msg, 50).unwrap();

    let channels: Vec<(&str, Vec<f32>)> = vec![
        ("1_clean", clean.clone()),
        ("2_noise_10db", add_noise(&clean, 10.0, 42)),
        ("3_babble_10db", add_babble(&clean, 10.0, 4242)),
        ("4_rnnoise_noisy", apply_rnnoise(&add_noise(&clean, 20.0, 42))),
        ("5_teams_tilt_40db", teams_spectral_filter(&clean, 40.0, 22.0)),
        ("6_teams_tilt_55db", teams_spectral_filter(&clean, 55.0, 30.0)),
        (
            "7_teams_full_pipeline",
            apply_rnnoise(&teams_spectral_filter(&clean, 40.0, 15.0)),
        ),
    ];

    for (name, audio) in &channels {
        let ok = verify(audio, msg);
        let path = format!("{dir}/channel_{name}.wav");
        write_wav(&path, audio);
        println!(
            "{:<24} {}  -> {}",
            name,
            if ok { "decodes OK " } else { "NO DECODE " },
            path
        );
    }
}
