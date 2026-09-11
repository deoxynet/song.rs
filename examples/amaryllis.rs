//! "Amaryllis" (Ballet Comique de la Reine, 1581; Henry Ghys's Air de
//! Louis XIII arrangement, 1868 — public domain, and the rice-cooker
//! melody of legend), hand-transcribed from memory. A-section, C major,
//! down two octaves into the modem compass.
//!
//! Usage: cargo run --release --example amaryllis [out_dir]

use song_rs::Decoder;

const B2: f64 = 123.47;
const C3: f64 = 130.81;
const D3: f64 = 146.83;
const E3: f64 = 164.81;
const F3: f64 = 174.61;
const G3: f64 = 196.00;

/// One note per payload byte (240 ms ≈ a gavotte quarter note).
const MELODY: &[f64] = &[
    // 1: E D E C            2: E D E C
    E3, D3, E3, C3,          E3, D3, E3, C3,
    // 3: E D E G            4: E D C — (held)
    E3, D3, E3, G3,          E3, D3, C3, C3,
    // 5: D C D B            6: D C D B
    D3, C3, D3, B2,          D3, C3, D3, B2,
    // 7: D C D F            8: E D C — (held)
    D3, C3, D3, F3,          E3, D3, C3, C3,
];

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp".to_string());
    let mut msg = b"amaryllis 1581 - the rice cooker".to_vec();
    msg.resize(MELODY.len(), b' ');

    let audio = song_rs::encode_with_pitch(&msg, 50, Some(MELODY)).unwrap();

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
    let path = format!("{dir}/amaryllis_memory.wav");
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for &s in &audio {
        w.write_sample(((s * gain) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
    println!("{} bytes, decode {}, wrote {path}", msg.len(), if ok { "OK" } else { "FAILED" });
    assert!(ok);
}
