import test from 'node:test';
import assert from 'node:assert/strict';
import { AudioEngine } from '../js/audio.js';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function fakeStream() {
  const track = { stops: 0, stop() { this.stops++; } };
  return { track, getTracks: () => [track] };
}

function fixture(t, { sampleRate = 48000 } = {}) {
  const permissions = [];
  const modules = [];
  const sources = [];
  const nodes = [];
  const ctx = {
    sampleRate, destination: {}, resumes: 0,
    resume() { this.resumes++; return Promise.resolve(); },
    audioWorklet: { addModule(url) {
      const result = deferred();
      modules.push({ ...result, url });
      return result.promise;
    } },
    createMediaStreamSource(stream) {
      const source = { stream, disconnected: false, connect() {},
        disconnect() { this.disconnected = true; } };
      sources.push(source);
      return source;
    },
  };
  const replacements = {
    window: { AudioContext: class { constructor() { return ctx; } } },
    navigator: { mediaDevices: { getUserMedia(constraints) {
      const result = deferred();
      permissions.push({ ...result, constraints });
      return result.promise;
    } } },
    AudioWorkletNode: class {
      constructor() { this.port = {}; this.disconnected = false; nodes.push(this); }
      connect() {}
      disconnect() { this.disconnected = true; }
    },
  };
  for (const [key, value] of Object.entries(replacements)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else delete globalThis[key];
    });
  }
  const engine = new AudioEngine();
  t.after(() => engine.stopRx());
  return { engine, permissions, modules, sources, nodes, ctx };
}

test('stopping during permission setup closes the eventual stream', async t => {
  const { engine, permissions, modules } = fixture(t);
  const stream = fakeStream();
  const starting = engine.startRx({ onChunk() { assert.fail('cancelled capture'); } });
  const rejection = assert.rejects(starting, { name: 'AbortError' });
  engine.stopRx();
  permissions[0].resolve(stream);
  await rejection;
  assert.equal(stream.track.stops, 1);
  assert.equal(modules.length, 0);
  assert.equal(engine.rx, null);
});

test('new capture wins when an older permission request resolves last', async t => {
  const { engine, permissions, modules, nodes, ctx } = fixture(t);
  const old = fakeStream(), current = fakeStream();
  const chunks = [];
  const first = engine.startRx({ onChunk() { assert.fail('old callback'); } });
  const rejection = assert.rejects(first, { name: 'AbortError' });
  const second = engine.startRx({ onChunk: chunk => chunks.push(chunk) });
  assert.equal(ctx.resumes, 2);
  assert.deepEqual(permissions[1].constraints.audio, {
    echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    channelCount: 1, sampleRate: 48000,
  });
  permissions[1].resolve(current);
  await Promise.resolve();
  modules[0].resolve();
  await second;
  const callback = nodes[0].port.onmessage;
  const chunk = Float32Array.of(0.1, -0.2);
  callback({ data: chunk });
  assert.equal(chunks[0], chunk);
  permissions[0].resolve(old);
  await rejection;
  assert.equal(engine.rx.stream, current);
  assert.equal(old.track.stops, 1);
  assert.equal(current.track.stops, 0);
  engine.stopRx();
  callback({ data: chunk });
  assert.equal(chunks.length, 1, 'late port messages are ignored');
  assert.equal(current.track.stops, 1);
  assert.equal(nodes[0].disconnected, true);
});

test('overlapping captures share worklet loading without reviving cancelled audio', async t => {
  const { engine, permissions, modules, sources } = fixture(t);
  const old = fakeStream(), current = fakeStream();
  const first = engine.startRx({ onChunk() {} });
  const rejection = assert.rejects(first, { name: 'AbortError' });
  permissions[0].resolve(old);
  await Promise.resolve();
  engine.stopRx();
  const second = engine.startRx({ onChunk() {} });
  permissions[1].resolve(current);
  await Promise.resolve();
  assert.equal(modules.length, 1);
  modules[0].resolve();
  await rejection;
  await second;
  assert.equal(old.track.stops, 1);
  assert.equal(sources.length, 1);
  assert.equal(engine.rx.stream, current);
});

test('failed worklet loading releases capture and permits a clean retry', async t => {
  const { engine, permissions, modules } = fixture(t);
  const old = fakeStream(), current = fakeStream();
  const first = engine.startRx({ onChunk() {} });
  const rejection = assert.rejects(first, /module unavailable/);
  permissions[0].resolve(old);
  await Promise.resolve();
  modules[0].reject(new Error('module unavailable'));
  await rejection;
  assert.equal(old.track.stops, 1);
  assert.equal(engine.rx, null);
  assert.equal(engine.workletLoading, null);
  const second = engine.startRx({ onChunk() {} });
  permissions[1].resolve(current);
  await Promise.resolve();
  assert.equal(modules.length, 2);
  modules[1].resolve();
  await second;
  assert.equal(engine.rx.stream, current);
});

test('44.1 kHz capture is resampled continuously across callback boundaries', async t => {
  const { engine, permissions, modules, nodes } = fixture(t, { sampleRate: 44100 });
  const output = [];
  const starting = engine.startRx({ onChunk: chunk => output.push(...chunk) });
  permissions[0].resolve(fakeStream());
  await Promise.resolve();
  modules[0].resolve();
  await starting;
  const input = Float32Array.from({ length: 4411 }, (_, i) => i);
  for (let offset = 0; offset < input.length; offset += 137) {
    nodes[0].port.onmessage({ data: input.subarray(offset, offset + 137) });
  }
  assert.ok(Math.abs(output.length - 4800) <= 1);
  output.forEach((value, i) => assert.ok(Math.abs(value - i * 44100 / 48000) < 0.001));
});
