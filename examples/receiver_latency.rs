//! Measure receiver work in browser-sized callbacks without altering audio.
//! Run: cargo run --release --example receiver_latency
//! Add -- --transfer to measure the web app's TransferRx path instead.
//! Add --clock for the 64/maximum-byte clock-drift browser workloads.
//! Add --echo for 25/60 ms half-amplitude echoes combined with clock drift.
//! Also supports --features 8khz. These are native measurements, not browser
//! timings. Queue debt models a single consumer of regularly arriving chunks.

use song_rs::{formant, protocol, transfer, Decoder};
use std::time::{Duration, Instant};

enum Receiver {
    Standard(Decoder),
    Transfer(transfer::TransferRx),
}

fn measure(name: &str, audio: &[f32], expected: Option<&[u8]>, use_transfer: bool) {
    let mut receiver = if use_transfer {
        Receiver::Transfer(transfer::TransferRx::new())
    } else {
        Receiver::Standard(Decoder::new())
    };
    let mut times = Vec::new();
    let mut deliveries = Vec::new();
    let mut errors = 0;
    let mut debt = 0.0f64;
    let mut peak_debt = 0.0f64;
    for chunk in audio.chunks(2048) {
        let start = Instant::now();
        let result = match &mut receiver {
            Receiver::Standard(rx) => rx.decode(chunk),
            Receiver::Transfer(rx) => Ok(rx.ingest(chunk)),
        };
        let elapsed = start.elapsed();
        times.push(elapsed);
        debt = (debt + elapsed.as_secs_f64() - chunk.len() as f64 / protocol::SAMPLE_RATE).max(0.0);
        peak_debt = peak_debt.max(debt);
        match result {
            Ok(Some(bytes)) => deliveries.push(bytes),
            Ok(None) => {}
            Err(_) => errors += 1,
        }
    }
    let expected: Vec<Vec<u8>> = expected.into_iter().map(<[u8]>::to_vec).collect();
    assert_eq!(
        deliveries, expected,
        "{name}: incorrect, lost, or duplicate delivery"
    );
    let compute: Duration = times.iter().copied().sum();
    let seconds = audio.len() as f64 / protocol::SAMPLE_RATE;
    times.sort_unstable();
    let p99 = times[(times.len() - 1) * 99 / 100];
    let peak = times.last().unwrap();
    let attempts = match &receiver {
        Receiver::Standard(_) => 0,
        Receiver::Transfer(rx) => rx.attempts(),
    };
    println!(
        "{name:<24} audio={seconds:>6.1}s compute={:>7.3}s speed={:>6.1}x p99={:>7.2}ms max={:>8.2}ms queue_debt={:>8.2}ms api_errors={errors} combining_attempts={attempts}",
        compute.as_secs_f64(), seconds / compute.as_secs_f64(),
        p99.as_secs_f64() * 1000.0, peak.as_secs_f64() * 1000.0, peak_debt * 1000.0,
    );
}

fn background(slots: usize) -> Vec<f32> {
    let slot: Vec<f32> = formant::synthesize_symbol(2, 185.0, 0.08, 0)
        .into_iter()
        .map(|sample| sample as f32)
        .collect();
    slot.iter()
        .copied()
        .cycle()
        .take(slots * protocol::SYMBOL_TOTAL_SAMPLES)
        .collect()
}

fn main() {
    let use_transfer = std::env::args().any(|arg| arg == "--transfer");
    let use_echo = std::env::args().any(|arg| arg == "--echo");
    println!(
        "{} Hz, 2048-sample callbacks ({:.2} ms), {}",
        protocol::SAMPLE_RATE,
        2048.0 / protocol::SAMPLE_RATE * 1000.0,
        if use_transfer {
            "TransferRx / Paranoid + CRC"
        } else {
            "Standard / CRC"
        },
    );
    if use_echo || std::env::args().any(|arg| arg == "--clock") {
        let maximum = if use_transfer { transfer::MAX_TRANSFER_PAYLOAD } else { protocol::MAX_LENGTH_VARIABLE };
        for length in [64, maximum] {
            let payload: Vec<u8> = (0..length).map(|i| (i * 73 + length) as u8).collect();
            let frame = if use_transfer {
                transfer::encode(&payload, 50).unwrap()
            } else {
                song_rs::encode(&payload, 50).unwrap()
            };
            for ratio in [0.98, 1.02] {
                let delay_ms = if ratio < 1.0 { 25 } else { 60 };
                let mut transmitted = frame.clone();
                if use_echo {
                    let delay = delay_ms * protocol::SAMPLE_RATE as usize / 1000;
                    transmitted.resize(frame.len() + delay, 0.0);
                    for (i, &sample) in frame.iter().enumerate() { transmitted[i + delay] += sample * 0.5; }
                }
                let warped: Vec<f32> = (0..((transmitted.len() - 1) as f64 / ratio).ceil() as usize).map(|i| {
                    let position = i as f64 * ratio;
                    let index = position as usize;
                    let fraction = position - index as f64;
                    (transmitted[index] as f64 * (1.0 - fraction) + transmitted[index + 1] as f64 * fraction) as f32
                }).collect();
                let mut audio = if use_echo { vec![0.0; 8 * protocol::SYMBOL_TOTAL_SAMPLES] }
                    else { warped[..protocol::SYMBOL_TOTAL_SAMPLES].repeat(100) };
                audio.resize(audio.len() + protocol::SYMBOL_TOTAL_SAMPLES / 3, 0.0);
                audio.extend(warped);
                audio.resize(audio.len() + 12 * protocol::SYMBOL_TOTAL_SAMPLES, 0.0);
                let label = if use_echo { format!("{length} bytes, {delay_ms}ms echo, clock {ratio}") }
                    else { format!("{length} bytes, clock {ratio}") };
                measure(&label, &audio, Some(&payload), use_transfer);
            }
        }
        return;
    }
    let melody: Vec<u8> = (0..64).map(|i| (i * 73 + 41) as u8).collect();
    let mut clean = if use_transfer {
        transfer::encode(&melody, 50).unwrap()
    } else {
        song_rs::encode(&melody, 50).unwrap()
    };
    clean.resize(clean.len() + 10 * protocol::SYMBOL_TOTAL_SAMPLES, 0.0);
    measure(
        if use_transfer {
            "clean transfer"
        } else {
            "clean Giant Steps"
        },
        &clean,
        Some(&melody),
        use_transfer,
    );

    let max_payload = if use_transfer {
        transfer::MAX_TRANSFER_PAYLOAD
    } else {
        protocol::MAX_LENGTH_VARIABLE
    };
    let payload: Vec<u8> = (0..max_payload).map(|i| (i * 73 + 41) as u8).collect();
    let track: Vec<f64> = (0..payload.len())
        .map(|i| {
            let midi = 47.0 + ((i * 13 + 7) % 20) as f64;
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        })
        .collect();
    let mut after_idle = background(100);
    after_idle.extend(if use_transfer {
        transfer::encode(&payload, 50).unwrap()
    } else {
        song_rs::encode_with_pitch(&payload, 50, Some(&track)).unwrap()
    });
    after_idle.resize(after_idle.len() + 10 * protocol::SYMBOL_TOTAL_SAMPLES, 0.0);
    measure(
        "maximum frame after idle",
        &after_idle,
        Some(&payload),
        use_transfer,
    );

    // Both workloads cross their profile's maximum recording deadline.
    let mut idle = background(if use_transfer { 1200 } else { 1000 });
    idle.resize(idle.len() + 10 * protocol::SYMBOL_TOTAL_SAMPLES, 0.0);
    measure("periodic background only", &idle, None, use_transfer);
}
