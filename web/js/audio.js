// Audio engine: 48 kHz mono f32 plumbing per the audio contract.
// - TX: schedules encoded copies back to back with a gap; a setInterval keeps
//   copies scheduled ahead even while the tab is hidden.
// - RX: AudioWorklet capture (all processing constraints off), ~2048-sample
//   batches, linear-resampled to 48 kHz if the context refused 48000.

const TARGET_RATE = 48000;

class LinearResampler {
  constructor(from, to) {
    this.ratio = from / to;
    this.pos = 0;
    this.last = 0;
    this.hasLast = false;
  }

  process(input) {
    if (input.length === 0) return input;
    const pre = this.hasLast ? 1 : 0;
    const L = input.length + pre;
    const at = (k) => (k < pre ? this.last : input[k - pre]);
    const out = [];
    let p = this.pos;
    while (p < L - 1) {
      const idx = Math.floor(p);
      const frac = p - idx;
      out.push(at(idx) * (1 - frac) + at(idx + 1) * frac);
      p += this.ratio;
    }
    this.pos = p - (L - 1); // last input sample becomes v[0] next chunk
    this.last = input[input.length - 1];
    this.hasLast = true;
    return Float32Array.from(out);
  }
}

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.tx = null;
    this.rx = null;
    this.workletLoaded = false;
    this.level = { fill: 0, zone: 'quiet', peak: 0 };
  }

  ensureCtx() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      try {
        this.ctx = new AC({ sampleRate: TARGET_RATE });
      } catch (_) {
        this.ctx = new AC(); // platform refused the rate option entirely
      }
    }
    return this.ctx;
  }

  // ---- TX ----------------------------------------------------------------

  async startTx({ encode, volume, gapMs }) {
    const ctx = this.ensureCtx();
    await ctx.resume();
    const out = ctx.createGain();
    out.connect(ctx.destination);
    const tx = (this.tx = {
      encode,
      volume,
      dirty: false,
      gap: gapMs / 1000,
      out,
      sched: [],
      buffer: null,
      encVol: volume,
      copyDur: 0,
      nextIndex: 1,
      nextTime: 0,
      t0: 0,
      stopped: false,
    });
    this._txEncode();
    tx.t0 = ctx.currentTime + 0.15;
    tx.nextTime = tx.t0;
    this._txSchedule();
    // rAF (which drives txTick) pauses while the tab is hidden; a timer keeps
    // copies scheduled ahead regardless. Audio-producing pages are exempt from
    // intensive timer throttling, so 500 ms holds in the background too.
    tx.timer = setInterval(() => this._txSchedule(), 500);
    tx.onVisibility = () => this._txSchedule();
    document.addEventListener('visibilitychange', tx.onVisibility);
  }

  _txEncode() {
    const tx = this.tx;
    const samples = tx.encode(tx.volume); // Float32Array @48k
    const buf = this.ctx.createBuffer(1, samples.length, TARGET_RATE);
    buf.copyToChannel(samples, 0);
    tx.buffer = buf;
    tx.encVol = tx.volume; // volume the buffer was encoded at
    tx.copyDur = samples.length / TARGET_RATE;
  }

  // Volume changes are heard immediately: the encoder maps volume linearly to
  // amplitude (volume/100), so scaling each already-scheduled copy's gain by
  // newVolume/encodedVolume is exact. The buffer is also re-encoded at the
  // next copy boundary so gains return to 1.
  setTxVolume(v) {
    if (!this.tx) return;
    this.tx.volume = v;
    this.tx.dirty = true; // re-encode at the exact volume from the next copy
    for (const c of this.tx.sched) {
      c.gain.gain.value = c.encVol > 0 ? v / c.encVol : 1;
    }
  }

  // Schedules copies ahead of the playhead. Driven by a setInterval (plus
  // visibilitychange) so it keeps running while the tab is hidden.
  _txSchedule() {
    const tx = this.tx;
    if (!tx || tx.stopped) return;
    const t = this.ctx.currentTime;
    // If scheduling fell behind (hidden tab, suspended context), drop the
    // missed copies: sources started in the past all begin playing at once
    // and would overlap.
    if (tx.nextTime < t) tx.nextTime = t + 0.1;
    while (tx.nextTime < t + 3.0) {
      if (tx.dirty) {
        this._txEncode();
        tx.dirty = false;
      }
      const src = this.ctx.createBufferSource();
      src.buffer = tx.buffer;
      // per-copy gain so setTxVolume can rescale copies already scheduled
      const gain = this.ctx.createGain();
      gain.gain.value = tx.encVol > 0 ? tx.volume / tx.encVol : 1;
      src.connect(gain);
      gain.connect(tx.out);
      src.start(tx.nextTime);
      tx.sched.push({ start: tx.nextTime, dur: tx.copyDur, index: tx.nextIndex, src, gain, encVol: tx.encVol });
      tx.nextTime += tx.copyDur + tx.gap;
      tx.nextIndex += 1;
      while (tx.sched.length > 3) tx.sched.shift();
    }
  }

  // Per-frame readout while looping; reports position only, no scheduling.
  txTick() {
    const tx = this.tx;
    if (!tx || tx.stopped || tx.sched.length === 0) return null;
    const t = this.ctx.currentTime;
    let cur = null;
    let next = null;
    for (const c of tx.sched) {
      if (t >= c.start) cur = c;
      else if (!next) next = c;
    }
    if (!cur) {
      cur = tx.sched[0];
      next = tx.sched[1] ?? null;
    }
    const copyEnd = cur.start + cur.dur;
    const inGap = t > copyEnd;
    return {
      copyIndex: cur.index,
      playhead: Math.min(Math.max(t - cur.start, 0), cur.dur),
      copyDur: cur.dur,
      elapsed: Math.max(t - tx.t0, 0),
      inGap,
      gapRemaining: inGap
        ? Math.max(0, (next?.start ?? (copyEnd + tx.gap)) - t)
        : 0,
    };
  }

  stopTx() {
    const tx = this.tx;
    if (!tx) return;
    tx.stopped = true;
    clearInterval(tx.timer);
    document.removeEventListener('visibilitychange', tx.onVisibility);
    for (const c of tx.sched) {
      try { c.src.stop(); } catch (_) { /* already ended */ }
    }
    try { tx.out.disconnect(); } catch (_) {}
    this.tx = null;
  }

  // ---- RX ----------------------------------------------------------------

  async startRx({ onChunk }) {
    const ctx = this.ensureCtx();
    // Even after a user gesture, some engines can hold resume() while the
    // permission sheet is open. Don't await it here; the app watches the
    // context state and retries on the next interaction if needed.
    ctx.resume().catch(() => {});
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: TARGET_RATE,
      },
    });
    let srcNode;
    let node;
    try {
      if (!this.workletLoaded) {
        await ctx.audioWorklet.addModule(new URL('./worklet.js', import.meta.url));
        this.workletLoaded = true;
      }
      srcNode = ctx.createMediaStreamSource(stream);
      node = new AudioWorkletNode(ctx, 'akt-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      srcNode.connect(node);
      node.connect(ctx.destination); // silent output keeps the node pulled
    } catch (err) {
      try { node?.disconnect(); } catch (_) {}
      try { srcNode?.disconnect(); } catch (_) {}
      for (const track of stream.getTracks()) track.stop();
      throw err;
    }
    const resampler = ctx.sampleRate !== TARGET_RATE
      ? new LinearResampler(ctx.sampleRate, TARGET_RATE)
      : null;
    node.port.onmessage = (e) => {
      let chunk = e.data;
      this._updateLevel(chunk);
      if (resampler) chunk = resampler.process(chunk);
      if (chunk.length) onChunk(chunk);
    };
    this.rx = { stream, srcNode, node };
    this.level = { fill: 0, zone: 'quiet', peak: 0 };
  }

  stopRx() {
    const rx = this.rx;
    if (!rx) return;
    rx.node.port.onmessage = null;
    try { rx.node.disconnect(); } catch (_) {}
    try { rx.srcNode.disconnect(); } catch (_) {}
    for (const t of rx.stream.getTracks()) t.stop();
    this.rx = null;
    this.level = { fill: 0, zone: 'quiet', peak: 0 };
  }

  _updateLevel(chunk) {
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < chunk.length; i++) {
      const v = chunk[i];
      sum += v * v;
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    const rms = Math.sqrt(sum / chunk.length);
    const db = 20 * Math.log10(rms + 1e-8);
    const target = Math.min(1, Math.max(0, (db + 50) / 50));
    const prev = this.level.fill;
    const fill = target > prev ? target : prev * 0.92; // fast attack, slow decay
    let zone;
    if (peak > 0.99 || fill >= 0.78) zone = 'clip';
    else if (fill < 0.22) zone = 'quiet';
    else zone = 'good';
    this.level = { fill, zone, peak };
  }
}
