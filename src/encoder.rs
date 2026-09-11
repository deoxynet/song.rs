//! Encoder: payload → f32 audio samples (vowel modem with melodic pitch).

use crate::formant::{self, PREAMBLE_END, PREAMBLE_START};
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Encode a payload into f32 audio samples.
///
/// `volume` is 0..=100 (typically 10-50).
/// Returns a Vec<f32> containing the complete waveform at 48 kHz.
///
/// The payload is sung to the default melody — the Giant Steps head
/// (`protocol::GIANT_STEPS`); pitch carries no data, and sync, header,
/// and parity stay on the margin-verified arpeggio. Use
/// [`encode_armored`] when the audio must survive worst-case channels
/// (heavy spectral tilt plus an aggressive denoiser) with maximum margin.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    let track = default_byte_pitch(payload.len());
    encode_with_pitch(payload, volume, Some(&track))
}

/// Encode with the armored melody throughout: the payload follows the
/// margin-verified arpeggio (`protocol::MELODY`) instead of the default
/// Giant Steps head. Maximum channel margin; used by the transfer profile.
pub fn encode_armored(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    encode_with_pitch(payload, volume, None)
}

/// Encode with an expressive per-byte pitch track.
///
/// `byte_pitch[i]` is the F0 (Hz) for the 4 symbols carrying payload byte `i`
/// (must have `payload.len()` entries). Preamble, header, and RS parity
/// symbols keep the default melody. Pitch carries no data — the decoder is
/// F0-agnostic — so callers may drive it semantically (e.g. token surprisal),
/// but notes should come from the verified-safe set (`protocol::MELODY` /
/// `F0_CANDIDATES`): other notes can degrade vowel separation under reverb.
pub fn encode_with_pitch(
    payload: &[u8],
    volume: u8,
    byte_pitch: Option<&[f64]>,
) -> Result<Vec<f32>, crate::Error> {
    encode_with_profile(payload, volume, byte_pitch, EccProfile::Standard)
}

/// Encode with an explicit ECC profile (receiver must use the same profile).
pub fn encode_with_profile(
    payload: &[u8],
    volume: u8,
    byte_pitch: Option<&[f64]>,
    profile: EccProfile,
) -> Result<Vec<f32>, crate::Error> {
    let data_length = payload.len();
    if data_length == 0 {
        return Err(crate::Error::EmptyPayload);
    }
    if data_length > MAX_LENGTH_VARIABLE {
        return Err(crate::Error::PayloadTooLarge {
            size: data_length,
            max: MAX_LENGTH_VARIABLE,
        });
    }
    if volume > 100 {
        return Err(crate::Error::InvalidVolume(volume));
    }

    let send_volume = volume as f64 / 100.0;

    // --- Step 1: Prepare tx_data = [length_byte, payload_bytes...] ---
    let mut tx_data = vec![0u8; data_length + 1];
    tx_data[0] = data_length as u8;
    tx_data[1..=data_length].copy_from_slice(payload);

    // --- Step 2: Reed-Solomon encode ---
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let encoded_length = rs_length.encode(&tx_data[0..1]);

    let n_ecc = profile.ecc_bytes(data_length);
    let rs_data = ReedSolomon::new(data_length, n_ecc);
    let encoded_data = rs_data.encode(&tx_data[1..=data_length]);

    let mut data_encoded = Vec::with_capacity(ENCODED_DATA_OFFSET + data_length + n_ecc);
    data_encoded.extend_from_slice(&encoded_length);
    data_encoded.extend_from_slice(&encoded_data);

    // --- Step 3: Convert bytes to 2-bit symbol sequence ---
    let data_symbols = formant::bytes_to_symbols(&data_encoded);

    // --- Step 4: Build symbol sequence: preamble + pilot + data + preamble ---
    let total_symbols = PREAMBLE_LEN + PILOT_LEN + data_symbols.len() + PREAMBLE_LEN;
    let total_samples = total_symbols * SYMBOL_TOTAL_SAMPLES;

    let mut output = Vec::with_capacity(total_samples);

    if let Some(bp) = byte_pitch {
        if bp.len() != data_length {
            return Err(crate::Error::PitchTrackMismatch {
                got: bp.len(),
                expected: data_length,
            });
        }
    }

    // Synthesize and append a full symbol slot (voiced window + release tail).
    // Accent follows the melody position; pitch follows the melody unless the
    // caller supplied a per-payload-byte track.
    let mut position = 0usize;
    let mut emit_symbol = |sym_idx: usize, f0_override: Option<f64>| {
        let volume = send_volume * melody_accent(position);
        let f0 = f0_override.unwrap_or_else(|| melody_f0(position));
        let slot = formant::synthesize_symbol(sym_idx, f0, volume, position);
        position += 1;
        for &s in &slot {
            output.push(s as f32);
        }
    };

    // Start preamble
    for &sym in &PREAMBLE_START {
        emit_symbol(sym, None);
    }

    // Calibration pilot: known symbols the receiver uses to measure each
    // vowel's centroid through the live channel (and to let AGC settle).
    for &sym in &PILOT {
        emit_symbol(sym, None);
    }

    // Data symbols: byte b of data_encoded spans symbols 4b..4b+4.
    // Bytes 0..3 are the length header, then the payload, then RS parity —
    // only payload bytes get the expressive pitch track.
    for (j, &sym) in data_symbols.iter().enumerate() {
        let byte_idx = j / (8 / BITS_PER_SYMBOL);
        let f0 = byte_pitch.and_then(|bp| {
            byte_idx
                .checked_sub(ENCODED_DATA_OFFSET)
                .and_then(|i| bp.get(i))
                .copied()
        });
        emit_symbol(sym, f0);
    }

    // End preamble
    for &sym in &PREAMBLE_END {
        emit_symbol(sym, None);
    }

    Ok(output)
}
