// Optional, dependency-free Chrome smoke test. Requires Node >= 22 and both
// WASM builds. Uses an isolated temporary profile, muted output, and fake mic.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = fileURLToPath(new URL('../', import.meta.url));
const chrome = process.env.CHROME_BIN || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
let blockWorkerBuild = false;
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.wasm': 'application/wasm' };
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!pathname.startsWith('/fixture/')) throw new Error('not found');
    const path = resolve(root, pathname.slice('/fixture/'.length) || 'index.html');
    if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) throw new Error('not found');
    if (blockWorkerBuild && pathname.includes('/pkg-worker/')) throw new Error('test outage');
    const data = await readFile(path);
    res.writeHead(200, { 'Content-Type': mime[extname(path)] || 'application/octet-stream',
      'Cache-Control': 'no-store' });
    res.end(data);
  } catch (_) { res.writeHead(404); res.end('Not found'); }
});

class DevTools {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.pending = new Map();
    this.id = 0;
    this.ready = new Promise((resolveReady, reject) => {
      this.socket.addEventListener('open', resolveReady, { once: true });
      this.socket.addEventListener('error', reject, { once: true });
    });
    this.socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
      else pending.resolve(message.result);
    });
    this.socket.addEventListener('close', () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(new Error('DevTools disconnected'));
      }
      this.pending.clear();
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    return new Promise((resolveResult, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`DevTools timeout: ${method}`));
      }, 180000);
      this.pending.set(id, { resolve: resolveResult, reject, timer });
      this.socket.send(JSON.stringify({ id, method, params, sessionId }));
    });
  }
  async evaluate(expression, sessionId) {
    const response = await this.send('Runtime.evaluate', {
      expression, awaitPromise: true, returnByValue: true,
    }, sessionId);
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
}

let child, cdp, profile;
try {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  profile = await mkdtemp(join(tmpdir(), 'song-worker-browser-'));
  console.log(`Isolated Chrome profile: ${profile}`);
  child = spawn(chrome, ['--headless=new', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-background-timer-throttling',
    '--mute-audio', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
    '--autoplay-policy=no-user-gesture-required', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const endpoint = await new Promise((resolveEndpoint, reject) => {
    let stderr = '';
    const timer = setTimeout(() => reject(new Error('Chrome startup timeout')), 30000);
    const finish = (error, url) => {
      clearTimeout(timer);
      if (error) reject(error); else resolveEndpoint(url);
    };
    child.once('error', error => finish(error));
    child.once('exit', code => finish(new Error(`Chrome exited: ${code}`)));
    child.stderr.on('data', data => {
      stderr = (stderr + data).slice(-16000);
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) finish(null, match[1]);
    });
  });
  cdp = new DevTools(endpoint);
  await cdp.ready;
  console.log(JSON.stringify(await cdp.send('Browser.getVersion')));
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__testWorkers = [];
    window.__testStreams = [];
    window.__testAudioPackets = 0;
    const RealWorker = window.Worker;
    window.Worker = class extends RealWorker {
      constructor(...args) { super(...args); window.__testWorkers.push(this); }
      postMessage(message, transfers) {
        if (message.type === 'ingest') window.__testAudioPackets++;
        return super.postMessage(message, transfers);
      }
    };
    const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      const stream = await getUserMedia(constraints);
      window.__testStreams.push(stream);
      return stream;
    };
  ` }, sessionId);
  const origin = `http://127.0.0.1:${server.address().port}`;
  await cdp.send('Page.navigate', { url: `${origin}/fixture/` }, sessionId);
  // Each evaluation is scoped to the local test page, never a user's browser.
  const evaluate = expression => cdp.evaluate(expression, sessionId);
  await evaluate(`new Promise(resolve => {
    if (document.readyState === 'complete') resolve();
    else window.addEventListener('load', resolve, { once: true });
  })`);
  const encoderResult = await evaluate(`(async () => {
    const suite = await import('./tests/worker-browser.js');
    return suite.runEncoderValidation();
  })()`);
  console.log(JSON.stringify({ encoder: encoderResult }));
  assert.equal(encoderResult.rejectedEmpty, 4);
  const workerResult = await evaluate(`(async () => {
    const suite = await import('./tests/worker-browser.js');
    window.__waitUntil = suite.waitUntil;
    return suite.runWorkerSmoke();
  })()`);
  console.log(JSON.stringify({ worker: workerResult }));
  assert.equal(workerResult.roundtrips, 12);

  // The page has loaded only its sender WASM. A missing worker build must fail
  // visibly before capture starts, leave sending usable, and permit retry.
  blockWorkerBuild = true;
  const startupFailure = await evaluate(`(async () => {
    document.getElementById('micBtn').click();
    await __waitUntil(() => document.getElementById('micStatusText').textContent === 'Decoder unavailable', 'worker failure UI');
    const input = document.getElementById('secret');
    input.value = 'synthetic test payload';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return { streams: __testStreams.length, canSend: !document.getElementById('playBtn').disabled };
  })()`);
  assert.deepEqual(startupFailure, { streams: 0, canSend: true });
  blockWorkerBuild = false;
  const capture = await evaluate(`(async () => {
    const { ReceiverWorker } = await import('./js/receiver.js');
    const ingest = ReceiverWorker.prototype.ingest;
    ReceiverWorker.prototype.ingest = function(samples) {
      window.__testReceiver = this;
      return window.__testPauseCapture ? true : ingest.call(this, samples);
    };
    window.__testInjectAudio = samples => ingest.call(window.__testReceiver, samples);
    const packets = __testAudioPackets;
    document.getElementById('micBtn').click();
    await __waitUntil(() => document.getElementById('micStatusText').textContent === 'Microphone on'
      && __testAudioPackets > packets, 'capture into worker');
    return { streams: __testStreams.length, tracksLive: __testStreams[0].getTracks().every(t => t.readyState === 'live') };
  })()`);
  assert.deepEqual(capture, { streams: 1, tracksLive: true });
  const results = await evaluate(`(async () => {
    // Pause only the fake capture feed. All injected samples still traverse
    // the production FIFO, Worker, decoder, callbacks, and result UI.
    window.__testPauseCapture = true;
    const { transfer_encode } = await import('./pkg/song_rs_wasm.js');
    const encoder = new TextEncoder();
    const frame = text => {
      const audio = transfer_encode(encoder.encode(text), 50);
      const input = new Float32Array(audio.length + 28800);
      input.set(audio);
      return input;
    };
    const rx = __testReceiver;
    rx.reset();
    const generation = rx.generation;
    const first = frame('first synthetic result');
    const second = frame('second synthetic result');
    const input = new Float32Array(first.length + second.length);
    input.set(first);
    input.set(second, first.length);
    __testInjectAudio(input);
    await __waitUntil(() => rx.pendingSamples === 0, 'consecutive UI deliveries');
    if (document.getElementById('resValue').value !== 'first synthetic result') throw new Error('result not latched');
    if (rx.generation !== generation) throw new Error('delivery discarded queued capture');

    // Hold a real verified worker response until after the user's clear.
    // This is a deterministically delayed message, not a fabricated payload.
    const worker = rx.worker;
    const callback = worker.onmessage;
    let held;
    worker.onmessage = event => {
      if (event.data.type === 'processed' && event.data.payload) held = event;
      else callback(event);
    };
    __testInjectAudio(first);
    await __waitUntil(() => held, 'delayed verified worker response');
    document.getElementById('resClear').click();
    worker.onmessage = callback;
    callback(held);
    await __waitUntil(() => rx.pendingSamples === 0, 'discard stale response');
    if (!document.getElementById('zoneResult').hidden || document.getElementById('resValue').value !== '') {
      throw new Error('clear restored stale result');
    }
    if (held.data.payload.some(byte => byte !== 0)) throw new Error('stale payload not wiped');
    __testInjectAudio(second);
    await __waitUntil(() => rx.pendingSamples === 0, 'post-clear reception');
    if (document.getElementById('resValue').value !== 'second synthetic result') throw new Error('new session failed');
    window.__testPauseCapture = false;
    return { firstResultLatched: true, staleClearRejected: true, postClearReception: true };
  })()`);
  console.log(JSON.stringify({ results }));
  const runtimeFailure = await evaluate(`(async () => {
    __testWorkers.at(-1).dispatchEvent(new MessageEvent('message', { data: { type: 'error' } }));
    await __waitUntil(() => document.getElementById('micStatusText').textContent === 'Decoder unavailable', 'capture failure UI');
    return __testStreams.every(stream => stream.getTracks().every(t => t.readyState === 'ended'));
  })()`);
  assert.equal(runtimeFailure, true);
  await evaluate(`(async () => {
    document.getElementById('micBtn').click();
    await __waitUntil(() => __testStreams.length === 2
      && document.getElementById('micStatusText').textContent === 'Microphone on', 'second capture retry');
  })()`);
  console.log(JSON.stringify({ app: { startupFailure, capture, runtimeFailureStopsTracks: runtimeFailure, retry: true } }));
  await evaluate(`document.getElementById('micBtn').click()`);
  const clockResults = [];
  for (const length of [64, 120]) {
    for (const ratio of [0.98, 1.02]) {
      let result;
      try {
        result = await evaluate(`(async () => {
          const suite = await import('./tests/worker-browser.js');
          return suite.runWorkerClockCase(${length}, ${ratio});
        })()`);
      } catch (error) {
        result = { error: error.message };
      }
      clockResults.push({ length, ratio, ...result });
      console.log(JSON.stringify({ clock: clockResults.at(-1) }));
    }
  }
  assert.equal(clockResults.filter(result => result.error).length, 0, 'all four clock-drift cases must pass');
  const echoResults = [];
  for (const length of [4, 16, 64, 120]) {
    for (const [ratio, delayMs] of [[0.98, 25], [1.02, 60]]) {
      let result;
      try {
        result = await evaluate(`(async () => {
          const suite = await import('./tests/worker-browser.js');
          return suite.runWorkerEchoCase(${length}, ${ratio}, ${delayMs});
        })()`);
      } catch (error) {
        result = { error: error.message };
      }
      echoResults.push({ length, ratio, delayMs, ...result });
      console.log(JSON.stringify({ echo: echoResults.at(-1) }));
    }
  }
  assert.equal(echoResults.filter(result => result.error).length, 0, 'all eight echo/clock cases must pass');
  const clockSequence = await evaluate(`(async () => {
    const suite = await import('./tests/worker-browser.js');
    return suite.runWorkerClockSequence();
  })()`);
  console.log(JSON.stringify({ clockSequence }));
  assert.equal(clockSequence.deliveries, 3);
  const silentGap = await evaluate(`(async () => {
    const suite = await import('./tests/worker-browser.js');
    return suite.runWorkerSilentGapCase();
  })()`);
  console.log(JSON.stringify({ silentGap }));
  assert.equal(silentGap.deliveries, 2);
  assert.equal(silentGap.completedAfterSilentSlots, 3);
  await cdp.send('Target.closeTarget', { targetId });
} finally {
  cdp?.socket.close();
  if (child?.pid && child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'exit');
    child.kill('SIGTERM');
    const timer = setTimeout(() => child.kill('SIGKILL'), 5000);
    await exited;
    clearTimeout(timer);
  }
  server.closeAllConnections();
  await new Promise(resolveClose => server.close(resolveClose));
  // Only this invocation's mkdtemp-created profile, after its browser exits.
  // Never remove or reuse an existing/personal Chrome profile.
  if (profile) await rm(profile, { recursive: true, force: true });
}
