// Real browser + production classic Worker + production WASM, no fake decoder.
import init, { transfer_encode, transfer_encode_repeated } from '../pkg/song_rs_wasm.js';
import { ReceiverWorker } from '../js/receiver.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

export async function waitUntil(predicate, message, timeoutMs = 60000) {
  const deadline = performance.now() + timeoutMs;
  while (!predicate()) {
    if (performance.now() > deadline) throw new Error(`Timed out: ${message}`);
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

function padded(frame, idleSlots = 0, invalid = false, phase = 0) {
  const lead = idleSlots * 2880 + (invalid ? 137 : 0) + phase;
  const input = new Float32Array(lead + frame.length + 12 * 2880);
  for (let slot = 0; slot < idleSlots; slot++) input.set(frame.subarray(0, 2880), slot * 2880);
  input.set(frame, lead);
  if (invalid) {
    const bad = [NaN, Infinity, -Infinity];
    for (let i = 0; i < lead; i++) input[i] = bad[i % 3];
    for (let start = lead, i = 0; start < lead + frame.length; start += 2880, i++) {
      input[start + 1200] = bad[i % 3];
      input[start + 2400] = bad[(i + 1) % 3];
    }
  }
  return input;
}

// Model a playback/capture sample-clock mismatch without changing the sender.
function resampled(frame, ratio) {
  return Float32Array.from({ length: Math.ceil((frame.length - 1) / ratio) }, (_, i) => {
    const position = i * ratio;
    const index = Math.floor(position);
    const fraction = position - index;
    return frame[index] * (1 - fraction) + frame[index + 1] * fraction;
  });
}

async function receive(input, expected, chunkSize, { reset = false, heartbeat = false, label = 'worker queue to drain' } = {}) {
  const deliveries = [], errors = [];
  const rx = new ReceiverWorker({
    onDecoded: bytes => deliveries.push(Array.from(bytes)), onError: error => errors.push(error.message),
  });
  let timer;
  let beats = 0, maxGapMs = 0, previous = performance.now();
  try {
    if (reset) {
      await rx.ready;
      rx.ingest(input);
      rx.reset(); // old in-flight work/queued audio must not replay in this session
    }
    if (heartbeat) {
      timer = setInterval(() => {
        const now = performance.now();
        maxGapMs = Math.max(maxGapMs, now - previous);
        previous = now;
        beats++;
      }, 10);
    }
    // Also exercise pre-ready buffering. Capture can reuse its storage as soon
    // as ingest returns; the wrapper must not transfer caller-owned memory.
    for (let i = 0; i < input.length; i += chunkSize) {
      check(rx.ingest(input.subarray(i, i + chunkSize)), 'audio rejected');
    }
    input.fill(0);
    await rx.ready;
    await waitUntil(() => rx.pendingSamples === 0 || errors.length, label);
    check(errors.length === 0, `decoder errors: ${errors.join(', ')}`);
    check(JSON.stringify(deliveries) === JSON.stringify(expected.map(bytes => Array.from(bytes))),
      `wrong deliveries: wanted lengths ${expected.map(bytes => bytes.length)}, got ${deliveries.map(bytes => bytes.length)}`);
    check(!rx.is_locked(), 'decoder still locked after trailing silence');
    if (heartbeat) {
      check(beats >= 2, 'main-thread heartbeat did not run during decoding');
      check(maxGapMs < 500, `main-thread stalled for ${maxGapMs.toFixed(1)} ms`);
    }
    return { deliveries: deliveries.length, attempts: rx.attempts(), beats,
      maxGapMs: Number(maxGapMs.toFixed(1)) };
  } finally {
    clearInterval(timer);
    rx.free();
  }
}

export async function runEncoderValidation() {
  await init();
  const empty = new Uint8Array();
  const calls = [() => transfer_encode(empty, 50),
    ...[0, 1, 3].map(copies => () => transfer_encode_repeated(empty, 50, copies, 180))];
  for (const call of calls) {
    let error;
    try { call(); } catch (caught) { error = caught; }
    check(error instanceof Error && error.message === 'payload is empty',
      'empty transfer must report a recoverable JS error');
  }
  // A rejected call must leave the WASM sender usable, including a zero byte.
  const payload = new Uint8Array([0]);
  const single = transfer_encode(payload, 50);
  const repeated = transfer_encode_repeated(payload, 50, 1, 180);
  check(single.length > 0 && single.every(Number.isFinite), 'valid sender audio was damaged');
  check(single.length === repeated.length && single.every((sample, i) => sample === repeated[i]),
    'one repeated copy changed the song');
  return { rejectedEmpty: calls.length, validSamples: single.length };
}

export async function runWorkerSmoke() {
  await init();
  const payloads = [1, 64, 120].map(length =>
    Uint8Array.from({ length }, (_, i) => (i * 73 + length) & 255));
  const frames = payloads.map(payload => transfer_encode(payload, 50));
  let roundtrips = 0;
  for (let i = 0; i < payloads.length; i++) {
    for (const chunkSize of [128, 511, 2048]) {
      await receive(padded(frames[i], chunkSize === 511 ? 100 : 0), [payloads[i]], chunkSize);
      roundtrips++;
    }
    await receive(padded(frames[i], 4, true), [payloads[i]], 511);
    roundtrips++;
  }

  const order = [0, 1, 2, 0];
  const continuous = new Float32Array(order.reduce((sum, i) => sum + frames[i].length + 28800, 0));
  let pos = 0;
  for (const i of order) { continuous.set(frames[i], pos); pos += frames[i].length + 28800; }
  const sequential = await receive(continuous, order.map(i => payloads[i]), continuous.length);
  const reset = await receive(padded(frames[0]), [payloads[0]], 2048, { reset: true });

  // A repeated vowel followed by only an end marker must not become a
  // failed-copy reference. Keep the same receiver for the following frame.
  const unanchoredSlots = 40 + 4 + 10;
  const afterUnanchored = new Float32Array(unanchoredSlots * 2880 + frames[2].length + 28800);
  const vowelTwo = frames[0].subarray(6 * 2880, 7 * 2880); // calibration pilot's vowel 2
  for (let slot = 0; slot < 40; slot++) afterUnanchored.set(vowelTwo, slot * 2880);
  afterUnanchored.set(frames[0].subarray(frames[0].length - 4 * 2880), 40 * 2880);
  afterUnanchored.set(frames[2], unanchoredSlots * 2880);
  const unanchored = await receive(afterUnanchored, [payloads[2]], 511);
  check(unanchored.attempts === 0, 'unanchored background was counted as a failed copy');

  // This crosses the decoder's recording deadline: an expensive background
  // search must not stall the UI, invent a payload, or prevent the next song.
  const background = new Float32Array(1200 * 2880 + 28800);
  for (let slot = 0; slot < 1200; slot++) background.set(frames[0].subarray(0, 2880), slot * 2880);
  const recovered = new Float32Array(background.length + frames[0].length + 28800);
  recovered.set(background);
  recovered.set(frames[0], background.length);
  const recovery = await receive(recovered, [payloads[0]], 2048, { heartbeat: true });
  return { roundtrips, sequential, reset, unanchored, periodicBackgroundSeconds: 72, recovery };
}

// Run clock cases independently so one timeout does not hide other checks.
export async function runWorkerClockCase(length, ratio) {
  await init();
  const payload = Uint8Array.from({ length }, (_, i) => (i * 73 + length) & 255);
  const frame = transfer_encode(payload, 50);
  const started = performance.now();
  const result = await receive(padded(resampled(frame, ratio), 100, false, 960), [payload], 511,
    { label: `${length}-byte transfer at clock ratio ${ratio}` });
  return { ...result, elapsedMs: Math.round(performance.now() - started) };
}

export async function runWorkerEchoCase(length, ratio, delayMs) {
  await init();
  const payload = Uint8Array.from({ length }, (_, i) => (i * 73 + length) & 255);
  const frame = transfer_encode(payload, 50);
  const delay = Math.round(delayMs * 48);
  const echo = new Float32Array(frame.length + delay);
  for (let i = 0; i < frame.length; i++) {
    echo[i] += frame[i];
    echo[i + delay] += frame[i] * 0.5;
  }
  const started = performance.now();
  const result = await receive(padded(resampled(echo, ratio), 0, false, 960), [payload], 511,
    { label: `${length}-byte transfer with ${delayMs} ms echo at clock ratio ${ratio}` });
  return { ...result, elapsedMs: Math.round(performance.now() - started) };
}

export async function runWorkerClockSequence() {
  await init();
  const payloads = [120, 1, 1].map(length =>
    Uint8Array.from({ length }, (_, i) => (i * 73 + length) & 255));
  const frames = payloads.map(payload => transfer_encode(payload, 50));
  // No end marker and no inter-song silence. Delayed completion of the
  // slow frame must preserve the next sender, including repeated payloads.
  const clocked = [resampled(frames[0].subarray(0, frames[0].length - 4 * 2880), 0.98),
    resampled(frames[1], 1.02), frames[2]];
  const input = new Float32Array(960 + clocked.reduce((sum, frame) => sum + frame.length, 0) + 12 * 2880);
  let position = 960;
  for (const frame of clocked) { input.set(frame, position); position += frame.length; }
  const started = performance.now();
  const result = await receive(input, payloads, 511, { label: 'consecutive clocked transfers without an end marker' });
  return { ...result, elapsedMs: Math.round(performance.now() - started) };
}

export async function runWorkerSilentGapCase() {
  await init();
  const slot = 2880;
  const payloads = [Uint8Array.from({ length: 13 }, (_, i) => (i * 73 + 19) & 255),
    Uint8Array.of(0x53)];
  const frame = transfer_encode(payloads[0], 50);
  const vowelTwo = frame.slice(6 * slot, 7 * slot);
  // Destroy the length header without silencing it or changing the sync,
  // then omit the end marker. Delivery needs the checked quiet-gap path.
  for (let i = 12; i < 24; i++) frame.set(vowelTwo, i * slot);
  const body = frame.subarray(0, frame.length - 4 * slot);
  const beforeThirdSlot = new Float32Array(body.length + 2 * slot);
  beforeThirdSlot.set(body);
  const deliveries = [], errors = [];
  const rx = new ReceiverWorker({
    onDecoded: bytes => deliveries.push(Array.from(bytes)), onError: error => errors.push(error.message),
  });
  const started = performance.now();
  const feed = async (input, label) => {
    for (let i = 0; i < input.length; i += 511) {
      check(rx.ingest(input.subarray(i, i + 511)), 'silent-gap audio rejected');
    }
    input.fill(0);
    await waitUntil(() => rx.pendingSamples === 0 || errors.length, label);
    check(errors.length === 0, `silent-gap decoder errors: ${errors.join(', ')}`);
  };
  try {
    await rx.ready;
    await feed(beforeThirdSlot, 'damaged-header body and first two silent slots');
    check(deliveries.length === 0, 'delivery occurred before the three-slot completion check');
    await feed(new Float32Array(slot), 'third silent slot');
    check(JSON.stringify(deliveries) === JSON.stringify([Array.from(payloads[0])]),
      'the first transfer did not complete before the next sender');
    await feed(padded(transfer_encode(payloads[1], 50)), 'next sender after three-slot gap');
    check(JSON.stringify(deliveries) === JSON.stringify(payloads.map(bytes => Array.from(bytes))),
      'wrong, lost, or duplicate silent-gap deliveries');
    check(rx.attempts() === 0, 'successful quiet completion was counted as a failed copy');
    check(!rx.is_locked(), 'silent-gap receiver stayed locked after the next frame');
    return { deliveries: deliveries.length, completedAfterSilentSlots: 3, attempts: rx.attempts(),
      elapsedMs: Math.round(performance.now() - started) };
  } finally { rx.free(); }
}
