// Classic worker: no module-worker or SharedArrayBuffer requirement.
// Only local, build-generated glue is loaded. Audio stays on this device.
let receiver = null;
let generation = 0;
let initialized = false;

function fail() {
  // Do not serialize audio, payloads, or internal error details into logs.
  self.postMessage({ type: 'error' });
  self.close();
}

self.onmessage = ({ data }) => {
  try {
    if (!initialized) throw new Error('Decoder not initialized');
    if (data.type === 'reset') {
      receiver?.free();
      receiver = new wasm_bindgen.WasmTransferRx();
      generation = data.generation;
      return;
    }
    if (data.type !== 'ingest' || data.generation !== generation || !receiver
        || !(data.samples instanceof Float32Array) || data.samples.length > 2048) {
      throw new Error('Invalid decoder command');
    }
    const payload = receiver.ingest(data.samples);
    self.postMessage({
      type: 'processed', generation, sequence: data.sequence,
      attempts: receiver.attempts(), locked: receiver.is_locked(), payload,
    }, payload ? [payload.buffer] : []);
  } catch (_) {
    fail();
  }
};

try {
  importScripts('../pkg-worker/song_rs_wasm.js');
  wasm_bindgen({ module_or_path: new URL('../pkg-worker/song_rs_wasm_bg.wasm', self.location.href) })
    .then(() => {
      initialized = true;
      self.postMessage({ type: 'ready', version: 1 });
    }).catch(fail);
} catch (_) {
  fail();
}
