//! Minuet in G major, BWV Anh. 114 (c. 1725, from the Anna Magdalena Bach
//! notebook; attributed to Christian Petzold) — public domain. First 16
//! bars from memory, transposed down an octave plus a tone (Minuet in F)
//! to fit the modem's compass exactly with the true contour.
//!
//! One byte = one eighth note (240 ms), so quarter = 2 bytes, bar = 6.
//!
//! Usage: cargo run --release --example minuet [out_dir]

use song_rs::Decoder;

const E3: f64 = 164.81;
const F3: f64 = 174.61;
const G3: f64 = 196.00;
const A3: f64 = 220.00;
const BB3: f64 = 233.08;
const C4: f64 = 261.63;
const D4: f64 = 293.66;
const E4: f64 = 329.63;
const F4: f64 = 349.23;

/// (note, eighth-note slots). Sounding key F; names in comments give the
/// original G-major spelling.
const MELODY: &[(f64, usize)] = &[
    // m1: D5(q) G4 A4 B4 C5(e)
    (C4, 2), (F3, 1), (G3, 1), (A3, 1), (BB3, 1),
    // m2: D5(q) G4(q) G4(q)
    (C4, 2), (F3, 2), (F3, 2),
    // m3: E5(q) C5 D5 E5 F#5(e)
    (D4, 2), (BB3, 1), (C4, 1), (D4, 1), (E4, 1),
    // m4: G5(q) G4(q) G4(q)
    (F4, 2), (F3, 2), (F3, 2),
    // m5: C5(q) D5 C5 B4 A4(e)
    (BB3, 2), (C4, 1), (BB3, 1), (A3, 1), (G3, 1),
    // m6: B4(q) C5 B4 A4 G4(e)
    (A3, 2), (BB3, 1), (A3, 1), (G3, 1), (F3, 1),
    // m7: F#4(q) G4 A4 B4 G4(e)
    (E3, 2), (F3, 1), (G3, 1), (A3, 1), (F3, 1),
    // m8: B4(q) A4(h) — half cadence
    (A3, 2), (G3, 4),
    // m9-m12 = m1-m4
    (C4, 2), (F3, 1), (G3, 1), (A3, 1), (BB3, 1),
    (C4, 2), (F3, 2), (F3, 2),
    (D4, 2), (BB3, 1), (C4, 1), (D4, 1), (E4, 1),
    (F4, 2), (F3, 2), (F3, 2),
    // m13: C5(q) D5 C5 B4 A4(e)
    (BB3, 2), (C4, 1), (BB3, 1), (A3, 1), (G3, 1),
    // m14: B4(q) C5 B4 A4 G4(e)
    (A3, 2), (BB3, 1), (A3, 1), (G3, 1), (F3, 1),
    // m15: A4(q) B4 A4 G4 F#4(e)
    (G3, 2), (A3, 1), (G3, 1), (F3, 1), (E3, 1),
    // m16: G4 (dotted half) — final cadence
    (F3, 6),
];

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp".to_string());
    let track: Vec<f64> = MELODY
        .iter()
        .flat_map(|&(hz, n)| std::iter::repeat(hz).take(n))
        .collect();

    let mut msg =
        b"Minuet in G, BWV Anh. 114 (Petzold c.1725), Anna Magdalena Bach notebook".to_vec();
    assert!(msg.len() <= track.len());
    msg.resize(track.len(), b' ');

    let audio = song_rs::encode_with_pitch(&msg, 50, Some(&track)).unwrap();

    let ok = {
        let mut dec = Decoder::new();
        match dec.decode(&audio) {
            Ok(Some(ref d)) => d == &msg,
            Err(_) => false,
            Ok(None) => {
                let silence = vec![0.0f32; 10 * song_rs::protocol::SYMBOL_TOTAL_SAMPLES];
                matches!(dec.decode(&silence), Ok(Some(ref d)) if d == &msg)
            }
        }
    };

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-9);
    let gain = 0.6 / peak;
    let path = format!("{dir}/minuet_modem.wav");
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for &s in &audio {
        w.write_sample(((s * gain) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
    println!(
        "{} bytes as {:.1}s, decode {}, wrote {path}",
        msg.len(),
        audio.len() as f64 / 48000.0,
        if ok { "OK" } else { "FAILED" }
    );
    assert!(ok);
}
