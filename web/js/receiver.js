// Main-thread transport only. WASM decoding never runs in this module.
const RATE = 48000;
const PACKET_SAMPLES = 2048;
const MAX_PENDING_SAMPLES = 120 * RATE;

export class ReceiverWorker {
  constructor({ onDecoded, onError, workerFactory = (url) => new Worker(url),
    maxPendingSamples = MAX_PENDING_SAMPLES, startupTimeoutMs = 30000 } = {}) {
    this.onDecoded = onDecoded;
    this.onError = onError;
    this.maxPendingSamples = maxPendingSamples;
    this.generation = 0;
    this.sequence = 0;
    this.queue = [];
    this.queueHead = 0;
    this.pendingSamples = 0;
    this.inflight = null;
    this.started = false;
    this.closed = false;
    this.state = { attempts: 0, locked: false };
    try {
      this.worker = workerFactory(new URL('./receiver-worker.js', import.meta.url));
    } catch (_) {
      const error = new Error('Background decoder could not start. Check browser worker support and retry.');
      error.name = 'DecoderWorkerError';
      throw error;
    }
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.worker.onmessage = (event) => this._message(event.data);
    this.worker.onerror = (event) => {
      event.preventDefault?.();
      this._fail('Background decoder could not run. Rebuild the WASM files and retry.');
    };
    this.worker.onmessageerror = () => this._fail('Background decoder message could not be read.');
    this.startupTimer = setTimeout(() => this._fail('Background decoder startup timed out.'), startupTimeoutMs);
  }

  attempts() { return this.state.attempts; }
  is_locked() { return this.state.locked; }
  get backlogSeconds() { return this.pendingSamples / RATE; }

  // Copies only this view, never transfers/detaches a caller's buffer (which
  // may be shared, reused, or backed by WASM memory). Keep all samples in order.
  ingest(samples) {
    if (this.closed) return false;
    if (!(samples instanceof Float32Array)) throw new TypeError('Expected Float32Array audio');
    if (this.pendingSamples + samples.length > this.maxPendingSamples) {
      this._fail('Background decoder fell behind. Reception stopped; retry on a faster device.');
      return false;
    }
    for (let start = 0; start < samples.length; start += PACKET_SAMPLES) {
      this.queue.push(samples.slice(start, start + PACKET_SAMPLES));
    }
    this.pendingSamples += samples.length;
    this._pump();
    return true;
  }

  // Explicit session boundary (clear, pause/resume, retry). In-flight work may
  // finish, but its bytes/status cannot enter the new session. Reset commands
  // precede new audio on the same FIFO port. Verified delivery itself does NOT
  // reset: queued audio can contain the next transmission.
  reset() {
    if (this.closed) return;
    this.generation += 1;
    this.queue = [];
    this.queueHead = 0;
    this.pendingSamples = this.inflight?.samples ?? 0;
    this.state = { attempts: 0, locked: false };
    if (this.started) this._post({ type: 'reset', generation: this.generation });
  }

  free() {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.startupTimer);
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
    this.queue = [];
    this.queueHead = 0;
    this.pendingSamples = 0;
    this.inflight = null;
    this.state = { attempts: 0, locked: false };
    this.rejectReady(new Error('Background decoder closed'));
  }

  _fail(message) {
    if (this.closed) return;
    const error = new Error(message);
    error.name = 'DecoderWorkerError';
    this.rejectReady(error);
    this.free();
    this.onError?.(error);
  }

  _post(message, transfers = []) {
    try {
      this.worker.postMessage(message, transfers);
      return true;
    } catch (_) {
      this._fail('Audio could not be sent to the background decoder.');
      return false;
    }
  }

  _pump() {
    if (this.closed || !this.started || this.inflight || this.queueHead === this.queue.length) return;
    const samples = this.queue[this.queueHead];
    this.queue[this.queueHead++] = null;
    if (this.queueHead === this.queue.length) {
      this.queue = [];
      this.queueHead = 0;
    } else if (this.queueHead >= 256) {
      this.queue = this.queue.slice(this.queueHead);
      this.queueHead = 0;
    }
    const sequence = ++this.sequence;
    this.inflight = { sequence, generation: this.generation, samples: samples.length };
    this._post({ type: 'ingest', generation: this.generation, sequence, samples }, [samples.buffer]);
  }

  _message(message) {
    if (this.closed) return;
    if (message?.type === 'error') {
      this._fail('Background decoder failed. Reception stopped; retry.');
      return;
    }
    if (message?.type === 'ready' && !this.started && message.version === 1) {
      clearTimeout(this.startupTimer);
      this.started = true;
      if (!this._post({ type: 'reset', generation: this.generation })) return;
      this.resolveReady();
      this._pump();
      return;
    }
    const pending = this.inflight;
    if (message?.type !== 'processed' || !pending || message.sequence !== pending.sequence
        || message.generation !== pending.generation) {
      this._fail('Background decoder returned an unexpected acknowledgement.');
      return;
    }
    this.inflight = null;
    this.pendingSamples -= pending.samples;
    try {
      if (message.generation === this.generation) {
        this.state = { attempts: message.attempts, locked: message.locked };
        if (message.payload) this.onDecoded?.(message.payload);
      } else {
        message.payload?.fill(0);
      }
    } finally {
      this._pump();
    }
  }
}
