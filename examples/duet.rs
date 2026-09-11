//! Two-robot conversation demo: writes a WAV of a short back-and-forth.
//!
//! Usage: cargo run --example duet [output.wav]
//! Then verify both messages decode: the file round-trips through Decoder.

use song_rs::Decoder;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/song_rs_duet.wav".to_string());

    let exchange: &[&[u8]] = &[
        b"hey! ready for the demo?",
        b"born ready. counting you in",
        b"3.. 2.. 1..",
        b"transmitting in style",
    ];

    let gap = vec![0.0f32; 48000 / 2]; // 500 ms between turns
    let mut audio: Vec<f32> = gap.clone();
    for msg in exchange {
        audio.extend(song_rs::encode(msg, 40).unwrap());
        audio.extend(gap.iter());
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&path, spec).unwrap();
    for &s in &audio {
        w.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();

    // Verify every message decodes from the combined audio
    let mut decoder = Decoder::new();
    let mut decoded = Vec::new();
    for chunk in audio.chunks(2880) {
        if let Ok(Some(data)) = decoder.decode(chunk) {
            decoded.push(data);
        }
    }
    println!(
        "wrote {path}: {:.1}s, {}/{} messages decode",
        audio.len() as f64 / 48000.0,
        decoded.len(),
        exchange.len()
    );
    for (i, d) in decoded.iter().enumerate() {
        println!("  [{}] {}", i, String::from_utf8_lossy(d));
    }
}
