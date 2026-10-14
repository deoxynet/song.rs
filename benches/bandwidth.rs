use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use song_rs::{protocol, formant};

const PAYLOAD_SIZES: &[usize] = &[1, 5, 10, 25, 50, 100, 140];

/// Compute the over-the-air duration for a given payload size.
fn audio_duration_secs(payload_len: usize) -> f64 {
    let total_bytes = protocol::EccProfile::Standard.encoded_bytes(payload_len);
    let n_data_symbols = formant::symbols_for_bytes(total_bytes);
    let total_symbols = protocol::PREAMBLE_LEN + protocol::PILOT_LEN
        + n_data_symbols + protocol::PREAMBLE_LEN;
    let total_samples = total_symbols * protocol::SYMBOL_TOTAL_SAMPLES;
    total_samples as f64 / protocol::SAMPLE_RATE
}

/// Print the protocol bandwidth table once before benchmarks run.
fn print_protocol_table() {
    println!();
    println!("=== Protocol Bandwidth (CRC-protected Vowel Modem, 2 bit/sym) ===");
    println!(
        "{:>7} {:>5} {:>8} {:>10} {:>8} {:>12}",
        "Payload", "ECC", "Symbols", "Samples", "Duration", "Bitrate"
    );
    println!(
        "{:>7} {:>5} {:>8} {:>10} {:>8} {:>12}",
        "(bytes)", "(bytes)", "", "", "(sec)", "(bit/s)"
    );
    println!("{}", "-".repeat(58));
    for &size in PAYLOAD_SIZES {
        let n_ecc = protocol::EccProfile::Standard.ecc_bytes(size);
        let total_bytes = protocol::EccProfile::Standard.encoded_bytes(size);
        let n_data_symbols = formant::symbols_for_bytes(total_bytes);
        let total_symbols = protocol::PREAMBLE_LEN + protocol::PILOT_LEN
            + n_data_symbols + protocol::PREAMBLE_LEN;
        let total_samples = total_symbols * protocol::SYMBOL_TOTAL_SAMPLES;
        let duration = total_samples as f64 / protocol::SAMPLE_RATE;
        let payload_bits = size * 8;
        let bitrate = payload_bits as f64 / duration;
        println!(
            "{:>7} {:>5} {:>8} {:>10} {:>8.2} {:>8.1}",
            size, n_ecc, total_symbols, total_samples, duration, bitrate,
        );
    }
    println!();
}

fn make_payload(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 7 + 13) % 256) as u8).collect()
}

fn bench_encode(c: &mut Criterion) {
    print_protocol_table();

    let mut group = c.benchmark_group("encode");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        let duration = audio_duration_secs(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| song_rs::encode(payload, 50).unwrap());
        });
        let encode_time = {
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = song_rs::encode(&payload, 50).unwrap();
            }
            start.elapsed().as_secs_f64() / 10.0
        };
        println!(
            "  encode/{size}: audio {duration:.2}s, encode {encode_time:.4}s -> {:.0}x real-time",
            duration / encode_time
        );
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        let audio = song_rs::encode(&payload, 50).unwrap();
        let duration = audio_duration_secs(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &audio, |b, audio| {
            b.iter(|| {
                let mut decoder = song_rs::Decoder::new();
                decoder.decode(audio).unwrap().unwrap();
            });
        });
        let decode_time = {
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let mut decoder = song_rs::Decoder::new();
                decoder.decode(&audio).unwrap().unwrap();
            }
            start.elapsed().as_secs_f64() / 10.0
        };
        println!(
            "  decode/{size}: audio {duration:.2}s, decode {decode_time:.4}s -> {:.0}x real-time",
            duration / decode_time
        );
    }
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                let audio = song_rs::encode(payload, 50).unwrap();
                let mut decoder = song_rs::Decoder::new();
                decoder.decode(&audio).unwrap().unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
