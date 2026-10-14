import test from 'node:test';
import assert from 'node:assert/strict';
import { ReceiverWorker } from '../js/receiver.js';

class FakeWorker {
  messages = [];
  terminated = false;
  postMessage(message, transfer = []) {
    this.messages.push(structuredClone(message, { transfer }));
  }
  terminate() { this.terminated = true; }
  emit(message) { this.onmessage?.({ data: message }); }
  finish({ payload = null, attempts = 0, locked = false } = {}) {
    const command = this.messages.findLast(message => message.type === 'ingest');
    this.emit({ type: 'processed', generation: command.generation, sequence: command.sequence,
      payload, attempts, locked });
  }
}

function setup(options = {}) {
  const worker = new FakeWorker();
  const deliveries = [];
  const errors = [];
  const rx = new ReceiverWorker({ workerFactory: () => worker,
    onDecoded: bytes => deliveries.push(bytes), onError: error => errors.push(error), ...options });
  return { worker, rx, deliveries, errors };
}

test('queues through startup and preserves every sample, FIFO, and caller buffer', async () => {
  const { worker, rx } = setup();
  try {
    const storage = Float32Array.from({ length: 5100 }, (_, i) => i - 2500);
    storage[150] = -0;
    const input = storage.subarray(100, 5100);
    const expected = new Uint32Array(input.slice().buffer);
    rx.ingest(input);
    storage.fill(17); // transport owns a copy, not this reused capture buffer
    assert.equal(worker.messages.length, 0);
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    const actual = [];
    while (rx.pendingSamples) {
      const packet = worker.messages.at(-1);
      assert.equal(packet.type, 'ingest');
      actual.push(...new Uint32Array(packet.samples.buffer));
      const before = worker.messages.length;
      assert.equal(rx.inflight.sequence, packet.sequence);
      worker.finish();
      assert.ok(worker.messages.length <= before + 1, 'only one audio packet may be in flight');
    }
    assert.deepEqual(actual, Array.from(expected));
    assert.equal(storage.length, 5100, 'caller memory is never detached');
    assert.equal(rx.backlogSeconds, 0);
  } finally { rx.free(); }
});

test('reset rejects stale results while preserving post-reset audio and status', async () => {
  const { worker, rx, deliveries } = setup();
  try {
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    rx.ingest(new Float32Array(3000));
    const old = { ...rx.inflight };
    rx.reset();
    rx.ingest(Float32Array.of(42, 43));
    const stale = Uint8Array.of(7);
    worker.emit({ type: 'processed', generation: old.generation, sequence: old.sequence,
      payload: stale, attempts: 99, locked: true });
    assert.deepEqual(deliveries, []);
    assert.deepEqual(Array.from(stale), [0]);
    assert.equal(rx.attempts(), 0);
    assert.equal(rx.is_locked(), false);
    const next = worker.messages.at(-1);
    assert.equal(next.generation, 1);
    assert.deepEqual(Array.from(next.samples), [42, 43]);
    worker.finish({ payload: Uint8Array.of(9), attempts: 2, locked: false });
    assert.deepEqual(deliveries.map(bytes => Array.from(bytes)), [[9]]);
    assert.equal(rx.attempts(), 2);
    assert.equal(rx.pendingSamples, 0);
  } finally { rx.free(); }
});

test('delivery does not discard queued audio for subsequent transmissions', async () => {
  const { worker, rx, deliveries } = setup();
  try {
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    rx.ingest(new Float32Array(4096));
    worker.finish({ payload: Uint8Array.of(1) });
    assert.equal(rx.pendingSamples, 2048);
    worker.finish({ payload: Uint8Array.of(2) });
    assert.deepEqual(deliveries.map(bytes => Array.from(bytes)), [[1], [2]]);
    assert.equal(rx.pendingSamples, 0);
  } finally { rx.free(); }
});

test('reset during initialization keeps only the newest explicit session', async () => {
  const { worker, rx } = setup();
  try {
    rx.ingest(Float32Array.of(1));
    rx.reset();
    rx.reset();
    rx.ingest(Float32Array.of(2));
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    assert.deepEqual(worker.messages[0], { type: 'reset', generation: 2 });
    assert.deepEqual(Array.from(worker.messages[1].samples), [2]);
    worker.finish();
  } finally { rx.free(); }
});

test('queue bound includes in-flight audio and overflow fails visibly once', async () => {
  const { worker, rx, errors } = setup({ maxPendingSamples: 4 });
  worker.emit({ type: 'ready', version: 1 });
  await rx.ready;
  assert.equal(rx.ingest(Float32Array.of(1, 2)), true);
  assert.equal(rx.ingest(Float32Array.of(3, 4)), true);
  assert.equal(rx.ingest(Float32Array.of(5)), false);
  assert.equal(worker.terminated, true);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].name, 'DecoderWorkerError');
  assert.match(errors[0].message, /fell behind/);
  assert.equal(rx.pendingSamples, 0);
  assert.equal(rx.ingest(Float32Array.of(6)), false);
  assert.equal(errors.length, 1);
});

test('worker load errors, timeout, and message errors terminate and reject startup', async () => {
  for (const kind of ['load', 'message', 'timeout', 'protocol']) {
    const { worker, rx, errors } = setup({ startupTimeoutMs: kind === 'timeout' ? 5 : 30000 });
    const rejection = assert.rejects(rx.ready, { name: 'DecoderWorkerError' });
    if (kind === 'load') worker.onerror({ preventDefault() {} });
    if (kind === 'message') worker.onmessageerror();
    if (kind === 'protocol') worker.emit({ type: 'ready', version: 9 });
    await rejection;
    assert.equal(worker.terminated, true);
    assert.equal(errors.length, 1);
  }
});

test('post failure and unexpected acknowledgement fail without a hidden queue', async () => {
  for (const kind of ['post', 'ack']) {
    const { worker, rx, errors } = setup();
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    if (kind === 'post') worker.postMessage = () => { throw new Error('unavailable'); };
    rx.ingest(Float32Array.of(1));
    if (kind === 'ack') worker.emit({ type: 'processed', sequence: 99, generation: 0 });
    assert.equal(errors.length, 1);
    assert.equal(rx.pendingSamples, 0);
    assert.equal(worker.terminated, true);
  }
});

test('free cancels startup and stale callbacks cannot deliver', async () => {
  const { worker, rx, deliveries } = setup();
  const callback = worker.onmessage;
  const rejection = assert.rejects(rx.ready, /closed/);
  rx.free();
  rx.free();
  await rejection;
  callback({ data: { type: 'ready', version: 1 } });
  callback({ data: { type: 'processed', payload: Uint8Array.of(1) } });
  assert.equal(deliveries.length, 0);
  assert.equal(worker.messages.length, 0);
});

test('unavailable worker construction is identified as a decoder failure', () => {
  assert.throws(() => new ReceiverWorker({
    workerFactory: () => { throw new Error('workers unavailable'); },
  }), { name: 'DecoderWorkerError' });
});

test('reset inside a delivery callback discards only the old session', async () => {
  const { worker, rx } = setup({ onDecoded: () => {
    rx.reset();
    rx.ingest(Float32Array.of(7));
  } });
  try {
    worker.emit({ type: 'ready', version: 1 });
    await rx.ready;
    rx.ingest(new Float32Array(4096));
    worker.finish({ payload: Uint8Array.of(1) });
    assert.equal(rx.pendingSamples, 1);
    assert.deepEqual(Array.from(worker.messages.at(-1).samples), [7]);
    assert.equal(worker.messages.at(-1).generation, 1);
    worker.finish();
    assert.equal(rx.pendingSamples, 0);
  } finally { rx.free(); }
});
