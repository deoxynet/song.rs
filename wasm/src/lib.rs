//! WASM bindings for the song-rs secret-transfer profile.
//!
//! Exports exactly the "Audio contract" from docs/design/README.md:
//! 48 kHz mono f32 audio, max 120-byte payloads, crc32 fingerprinting.

use wasm_bindgen::prelude::*;

/// Encode one copy of the payload as 48 kHz mono f32 audio.
#[wasm_bindgen]
pub fn transfer_encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, JsError> {
    song_rs::transfer::encode(payload, volume).map_err(|e| JsError::new(&e.to_string()))
}

/// Encode `copies` copies of the payload separated by `gap_ms` of silence.
#[wasm_bindgen]
pub fn transfer_encode_repeated(
    payload: &[u8],
    volume: u8,
    copies: u32,
    gap_ms: u32,
) -> Result<Vec<f32>, JsError> {
    song_rs::transfer::encode_repeated(payload, volume, copies as usize, gap_ms)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// CRC-32 of the payload; the UI derives the fingerprint from this.
#[wasm_bindgen]
pub fn crc32(payload: &[u8]) -> u32 {
    song_rs::transfer::crc32(payload)
}

/// Maximum transfer payload size in bytes (120).
#[wasm_bindgen]
pub fn max_transfer_payload() -> usize {
    song_rs::transfer::MAX_TRANSFER_PAYLOAD
}

/// Streaming receiver. Feed capture chunks to `ingest`; returns the verified
/// payload bytes once, then null forever after.
#[wasm_bindgen]
pub struct WasmTransferRx {
    inner: song_rs::transfer::TransferRx,
}

#[wasm_bindgen]
impl WasmTransferRx {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: song_rs::transfer::TransferRx::new(),
        }
    }

    /// Ingest a chunk of 48 kHz mono f32 samples.
    /// Returns the verified payload (`Uint8Array`) exactly once, else null.
    pub fn ingest(&mut self, chunk: &[f32]) -> Option<Vec<u8>> {
        self.inner.ingest(chunk)
    }

    /// Number of failed copy attempts so far.
    pub fn attempts(&self) -> usize {
        self.inner.attempts()
    }

    /// True while the decoder is locked onto an incoming copy.
    pub fn is_locked(&self) -> bool {
        self.inner.is_locked()
    }
}

impl Default for WasmTransferRx {
    fn default() -> Self {
        Self::new()
    }
}
