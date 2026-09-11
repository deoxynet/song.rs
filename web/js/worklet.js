// AudioWorklet capture processor: batches mic frames into ~2048-sample
// Float32Array chunks and posts them to the main thread. Outputs silence.
const BATCH = 2048;

class AktCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BATCH);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let i = 0;
      while (i < ch.length) {
        const take = Math.min(BATCH - this.n, ch.length - i);
        this.buf.set(ch.subarray(i, i + take), this.n);
        this.n += take;
        i += take;
        if (this.n === BATCH) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(BATCH);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('akt-capture', AktCaptureProcessor);
