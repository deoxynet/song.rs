// AKT — one screen, two explicit roles, one verified result.
// The sender loops acoustic copies until stopped. The receiver keeps merging
// incomplete copies and only returns CRC-verified bytes.
import init, { transfer_encode, crc32, max_transfer_payload, WasmTransferRx } from '../pkg/song_rs_wasm.js';
import { crcHex, crcWords } from './fingerprint.js';
import { AudioEngine } from './audio.js';

const $ = (id) => document.getElementById(id);
const enc = new TextEncoder();
const dec = new TextDecoder();
const engine = new AudioEngine();

let MAX = 120;
const GAP_MS = 500;
const copySeconds = (len) => (8 * len + 60) * 0.06;
const LOCK_GRACE_MS = 1500;
const STALL_AFTER = 5;
const TIPS = 'Set sender volume to 80% · reduce distance to 30 cm · reduce background noise · align speaker with microphone.';
const QUIET_TIP = 'Reduce distance or increase sender volume.';
const MERGE_DECAY_FALLBACK_MS = 30000;

const S = {
  payload: '',
  byteLength: 0,
  overLimit: false,
  revealed: false,
  volume: 50,
  playing: false,
  mic: 'off',              // off | requesting | on | denied
  micError: null,
  suspended: false,
  rx: null,
  attempts: 0,
  mergeLine: null,
  mergeSetAt: 0,
  prevAttemptAt: -Infinity,
  copyEstMs: 0,
  lastLockedAt: -Infinity,
  decoded: null,
  decodedBytes: null,
  okCopies: 0,
  okFp: null,
};

// ---- small render helpers ------------------------------------------------

function setText(el, value) {
  if (el.textContent !== value) el.textContent = value;
}

function makeFingerprint(bytes) {
  const c = crc32(bytes);
  return { hex: crcHex(c), words: crcWords(c) };
}

function renderFingerprint(el, fp) {
  el.replaceChildren(...fp.words.map((word) => {
    const span = document.createElement('span');
    span.textContent = word;
    return span;
  }));
  el.setAttribute('aria-label', fp.words.join(', '));
}

function setReceiverStatus(text, state) {
  setText($('micStatusText'), text);
  $('micStatus').dataset.state = state;
}

function setLine(text, tips = null, tipsAccent = false) {
  setText($('liveLine'), text);
  const t = $('liveTips');
  if (tips) {
    setText(t, tips);
    t.classList.toggle('is-alert', tipsAccent);
    t.hidden = false;
  } else {
    setText(t, '');
    t.classList.remove('is-alert');
    t.hidden = true;
  }
}

function setMicButton({ hidden, disabled = false, text = 'Enable microphone' }) {
  const b = $('micBtn');
  b.hidden = hidden;
  b.disabled = disabled;
  setText($('micBtnText'), text);
  syncPlayPrimacy();
}

// ---- send ---------------------------------------------------------------

function updateInput() {
  const bytes = enc.encode(S.payload).length;
  const over = bytes > MAX;
  S.byteLength = bytes;
  S.overLimit = over;

  const input = $('secret');
  input.type = S.revealed ? 'text' : 'password';
  input.setAttribute('aria-invalid', String(over));
  input.readOnly = S.playing;

  const show = $('showBtn');
  show.disabled = bytes === 0;
  setText(show, S.revealed ? 'Hide' : 'Show');
  show.setAttribute('aria-pressed', String(S.revealed));

  setText($('countLabel'), over ? 'Payload too large' : 'UTF-8 payload');
  setText($('count'), `${bytes} / ${MAX} B`);
  $('count').classList.toggle('is-over', over);
  $('clearInputBtn').hidden = bytes === 0;
  $('clearInputBtn').disabled = S.playing;

  const meter = $('byteMeter');
  meter.classList.toggle('over', over);
  meter.setAttribute('aria-valuemax', String(MAX));
  meter.setAttribute('aria-valuenow', String(Math.min(bytes, MAX)));
  $('byteMeterFill').style.width = `${Math.min(100, (bytes / MAX) * 100)}%`;

  const limitError = $('limitError');
  if (over) {
    const excess = bytes - MAX;
    setText(limitError, `Remove ${excess} ${excess === 1 ? 'byte' : 'bytes'} to send. Some characters use more than one byte.`);
    limitError.hidden = false;
  } else {
    setText(limitError, '');
    limitError.hidden = true;
  }

  const fingerprint = $('sendFingerprint');
  if (bytes > 0 && !over) {
    try {
      const fp = makeFingerprint(enc.encode(S.payload));
      renderFingerprint($('fpWords'), fp);
      setText($('fpHex'), fp.hex);
      fingerprint.hidden = false;
    } catch (_) {
      fingerprint.hidden = true;
    }
  } else {
    $('fpWords').replaceChildren();
    setText($('fpHex'), '');
    fingerprint.hidden = true;
  }

  setText(
    $('copyTime'),
    bytes === 0
      ? 'Enter a payload to estimate transfer time'
      : `About ${copySeconds(bytes).toFixed(1)} sec per copy · 0.5 sec gap`,
  );

  $('playBtn').disabled = S.playing ? false : (over || bytes === 0);
  syncPlayPrimacy();
}

function renderSending() {
  $('playBtn').classList.add('is-sending');
  setText($('playBtnText'), 'Stop transmission');
  $('sendProgress').hidden = false;
  $('secret').readOnly = true;
  renderReceiverPaused();
}

function resetSendProgress() {
  $('sendProgress').hidden = true;
  setText($('sendState'), 'Transmitting copy 1');
  setText($('sendTime'), '0.0 / 0.0 s');
  $('sendProgressFill').style.width = '0%';
  $('sendProgressBar').setAttribute('aria-valuenow', '0');
}

function updateSendProgress(st) {
  let progress;
  if (st.inGap) {
    progress = 100;
    setText($('sendState'), `Copy ${st.copyIndex} complete`);
    setText($('sendTime'), `Next copy in ${st.gapRemaining.toFixed(1)} s`);
  } else {
    progress = st.copyDur > 0 ? Math.min(100, (st.playhead / st.copyDur) * 100) : 0;
    setText($('sendState'), `Transmitting copy ${st.copyIndex}`);
    setText($('sendTime'), `${st.playhead.toFixed(1)} / ${st.copyDur.toFixed(1)} s`);
  }
  $('sendProgressFill').style.width = `${progress.toFixed(1)}%`;
  $('sendProgressBar').setAttribute('aria-valuenow', String(Math.round(progress)));
}

async function togglePlay() {
  if (S.playing) {
    stopPlaying();
    return;
  }
  if (S.overLimit || S.byteLength === 0) return;

  const button = $('playBtn');
  const error = $('sendError');
  error.hidden = true;
  setText(error, '');
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  setText($('playBtnText'), 'Starting transmission…');

  try {
    const bytes = enc.encode(S.payload);
    await engine.startTx({
      encode: (vol) => transfer_encode(bytes, vol),
      volume: S.volume,
      gapMs: GAP_MS,
    });
    S.playing = true;
    if (S.rx) freshReceiver();
    renderSending();
  } catch (err) {
    console.error(err);
    setText(error, 'Audio output could not start. Check browser audio settings and retry.');
    error.hidden = false;
    setText($('playBtnText'), 'Start transmission');
  } finally {
    button.removeAttribute('aria-busy');
    updateInput();
  }
}

function stopPlaying() {
  engine.stopTx();
  S.playing = false;
  $('playBtn').classList.remove('is-sending');
  setText($('playBtnText'), 'Start transmission');
  $('secret').readOnly = false;
  resetSendProgress();
  updateInput();
  if (S.mic === 'on') freshReceiver();
  restoreReceiver();
}

// ---- listen --------------------------------------------------------------

function renderMicOff() {
  setReceiverStatus('Microphone off', 'off');
  $('soundOrbit').classList.remove('is-live');
  setLine('Receiver inactive', 'Enable the microphone to begin receiving.');
  setMicButton({ hidden: false, text: 'Enable microphone' });
  $('permissionNote').hidden = false;
  setText($('permissionNote'), 'Microphone access is required for reception.');
  $('meterBlock').hidden = true;
}

function renderMicRequesting() {
  setReceiverStatus('Requesting access', 'busy');
  $('soundOrbit').classList.remove('is-live');
  setLine('Microphone permission required', 'Allow microphone access to continue.');
  setMicButton({ hidden: false, disabled: true, text: 'Requesting access…' });
  $('permissionNote').hidden = true;
  $('meterBlock').hidden = true;
}

function micFailureCopy(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return {
        status: 'Microphone blocked',
        line: 'Microphone access is off',
        tip: 'Allow microphone access in your browser settings, then try again.',
      };
    case 'NotFoundError':
      return {
        status: 'No microphone',
        line: 'No microphone was found',
        tip: 'Connect or enable a microphone, then try again.',
      };
    case 'NotReadableError':
    case 'AbortError':
      return {
        status: 'Microphone busy',
        line: 'The microphone is in use',
        tip: 'Close other apps using the microphone, then try again.',
      };
    default:
      return {
        status: 'Microphone unavailable',
        line: 'Reception could not start',
        tip: 'Check microphone access and retry.',
      };
  }
}

function renderMicDenied(err = S.micError) {
  const copy = micFailureCopy(err);
  setReceiverStatus(copy.status, 'error');
  $('soundOrbit').classList.remove('is-live');
  setLine(copy.line, copy.tip, true);
  setMicButton({ hidden: false, text: 'Retry microphone' });
  $('permissionNote').hidden = true;
  $('meterBlock').hidden = true;
}

function renderMicOn() {
  setReceiverStatus('Microphone on', 'live');
  $('soundOrbit').classList.add('is-live');
  setLine('Waiting for signal', 'Start transmission on the sending device.');
  setMicButton({ hidden: true });
  $('permissionNote').hidden = true;
  $('meterBlock').hidden = false;
}

function renderReceiverPaused() {
  setReceiverStatus(S.mic === 'on' ? 'Reception paused' : 'Transmitter active', 'busy');
  $('soundOrbit').classList.remove('is-live');
  setLine('Transmitting from this device', 'Reception resumes when transmission stops.');
  setMicButton({ hidden: true });
  $('permissionNote').hidden = true;
  $('meterBlock').hidden = true;
}

function restoreReceiver() {
  if (S.playing) {
    renderReceiverPaused();
  } else if (S.mic === 'on') {
    syncSuspended();
  } else if (S.mic === 'denied') {
    renderMicDenied();
  } else if (S.mic === 'requesting') {
    renderMicRequesting();
  } else {
    renderMicOff();
  }
}

function freshReceiver() {
  if (S.rx) {
    try { S.rx.free(); } catch (_) { /* already freed */ }
  }
  S.rx = new WasmTransferRx();
  S.attempts = 0;
  S.mergeLine = null;
  S.mergeSetAt = 0;
  S.prevAttemptAt = -Infinity;
  S.copyEstMs = 0;
  S.lastLockedAt = -Infinity;
}

async function requestMic() {
  if (S.mic === 'requesting' || S.mic === 'on' || S.playing) return;
  S.mic = 'requesting';
  S.micError = null;
  renderMicRequesting();
  try {
    await engine.startRx({ onChunk: handleChunk });
  } catch (err) {
    console.error(err);
    S.mic = 'denied';
    S.micError = err;
    renderMicDenied(err);
    return;
  }
  S.mic = 'on';
  freshReceiver();
  engine.ctx.onstatechange = syncSuspended;
  syncSuspended();
  resumeOnGesture();
}

function syncSuspended() {
  if (S.mic !== 'on' || S.playing) return;
  S.suspended = engine.ctx.state !== 'running';
  if (S.suspended) {
    setReceiverStatus('Audio processing paused', 'busy');
    $('soundOrbit').classList.remove('is-live');
    setLine('Resume audio processing', 'Interact with this page to resume microphone processing.');
    $('meterBlock').hidden = true;
  } else {
    renderMicOn();
  }
}

function resumeOnGesture() {
  const ctx = engine.ctx;
  if (!ctx || ctx.state === 'running') return;
  const kick = () => ctx.resume().then(detach).catch(() => {});
  const detach = () => {
    for (const event of ['pointerdown', 'keydown']) {
      document.removeEventListener(event, kick, { capture: true });
    }
  };
  for (const event of ['pointerdown', 'keydown']) {
    document.addEventListener(event, kick, { capture: true });
  }
}

function handleChunk(chunk) {
  if (S.playing || !S.rx) return;
  const result = S.rx.ingest(chunk);
  if (result) onDecoded(result);
}

function updateMeter(fill, zone, attempting) {
  const percent = Math.min(100, Math.max(0, fill * 100));
  const meter = $('levelMeter');
  $('levelFill').style.width = `${percent.toFixed(1)}%`;
  $('levelNeedle').style.left = `${Math.min(99.7, percent).toFixed(1)}%`;
  meter.dataset.zone = zone;
  meter.setAttribute('aria-valuenow', String(Math.round(percent)));

  const labels = { quiet: attempting ? 'Low' : 'No signal', good: 'Good', clip: 'Clipping' };
  setText($('levelWord'), labels[zone]);

  const hint = $('meterHint');
  let text;
  let alert = false;
  if (zone === 'clip') {
    text = 'Input is clipping. Lower sender volume or increase distance.';
    alert = true;
  } else if (zone === 'quiet' && attempting) {
    text = 'Input level is low. Reduce distance or increase sender volume.';
    alert = true;
  } else if (zone === 'good' && attempting) {
    text = 'Input level is within the recommended range.';
  } else if (zone === 'good') {
    text = 'Microphone input is active.';
  } else {
    text = 'Waiting for an input signal.';
  }
  setText(hint, text);
  hint.classList.toggle('is-alert', alert);
}

// ---- verified result -----------------------------------------------------

function onDecoded(bytes) {
  // A verified result belongs to the user until they explicitly clear it.
  // Keep listening, but never replace or re-announce that result when the
  // sender is still looping (or a second sender starts nearby).
  if (S.decoded !== null) {
    if (typeof bytes.fill === 'function') bytes.fill(0);
    freshReceiver();
    return;
  }
  S.decodedBytes = bytes;
  S.decoded = dec.decode(bytes);
  S.okCopies = S.rx.attempts() + 1;
  S.okFp = makeFingerprint(bytes);
  freshReceiver();
  renderResult(true);
}

function syncPlayPrimacy() {
  const sendable = S.playing || (S.byteLength > 0 && !S.overLimit);
  const resultOpen = !$('zoneResult').hidden;
  const sendPrimary = sendable && !resultOpen;
  const listenPrimary = !sendable && !resultOpen;
  $('playBtn').classList.toggle('btn-primary', sendPrimary);
  $('playBtn').classList.toggle('btn-secondary', !sendPrimary);
  $('micBtn').classList.toggle('btn-primary', listenPrimary);
  $('micBtn').classList.toggle('btn-secondary', !listenPrimary);
}

function renderResult(announce = false) {
  const copies = S.okCopies;
  setText($('resHead'), `Verified after ${copies} ${copies === 1 ? 'copy' : 'copies'}`);
  const value = $('resValue');
  value.type = 'password';
  value.value = S.decoded ?? '';
  setText($('resReveal'), 'Reveal');
  $('resReveal').setAttribute('aria-pressed', 'false');
  if (S.okFp) {
    renderFingerprint($('resFp'), S.okFp);
    setText($('resHex'), S.okFp.hex);
  }
  $('zoneResult').hidden = false;
  $('fpWords').classList.add('fp-dim');
  syncPlayPrimacy();

  if (announce) {
    requestAnimationFrame(() => {
      $('zoneResult').scrollIntoView({
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        block: 'center',
      });
      $('resultTitle').focus({ preventScroll: true });
    });
  }
}

function clearResult() {
  if (S.decodedBytes) S.decodedBytes.fill(0);
  S.decoded = null;
  S.decodedBytes = null;
  S.okFp = null;
  S.okCopies = 0;

  const value = $('resValue');
  value.value = '';
  value.type = 'password';
  setText($('resReveal'), 'Reveal');
  $('resReveal').setAttribute('aria-pressed', 'false');
  $('resFp').replaceChildren();
  setText($('resHex'), '');
  setText($('resHead'), '');
  setText($('copyStatus'), '');
  $('zoneResult').hidden = true;
  $('fpWords').classList.remove('fp-dim');
  syncPlayPrimacy();
  $('receiveTitle').focus({ preventScroll: true });
}

// ---- frame loop ----------------------------------------------------------

function frame() {
  const now = performance.now();

  if (S.playing) {
    const state = engine.txTick();
    if (state) updateSendProgress(state);
  } else if (S.mic === 'on' && S.rx && !S.suspended) {
    const locked = S.rx.is_locked();
    const attempts = S.rx.attempts();
    if (locked) S.lastLockedAt = now;

    if (attempts > S.attempts) {
      S.mergeLine = `Copy ${attempts} incomplete. Waiting for next copy…`;
      S.mergeSetAt = now;
      if (Number.isFinite(S.prevAttemptAt)) S.copyEstMs = now - S.prevAttemptAt;
      S.prevAttemptAt = now;
      S.lastLockedAt = -Infinity;
    }
    S.attempts = attempts;

    const hearing = now - S.lastLockedAt < LOCK_GRACE_MS;
    if (S.mergeLine && !hearing) {
      const decay = S.copyEstMs > 0 ? 2 * S.copyEstMs : MERGE_DECAY_FALLBACK_MS;
      if (now - Math.max(S.mergeSetAt, S.lastLockedAt) > decay) S.mergeLine = null;
    }

    const quiet = engine.level.zone === 'quiet';
    let tips = null;
    if (attempts >= STALL_AFTER) {
      tips = quiet ? `${QUIET_TIP} ${TIPS}` : TIPS;
    } else if (S.mergeLine && quiet) {
      tips = QUIET_TIP;
    }

    if (hearing) {
      setLine('Signal detected. Receiving copy…', tips ?? 'Maintain device position.', quiet && tips !== null);
    } else if (S.mergeLine) {
      setLine(S.mergeLine, tips ?? 'Copies are combined automatically.', quiet && tips !== null);
    } else {
      setLine('Waiting for signal', 'Start transmission on the sending device.');
    }

    const attempting = hearing || S.mergeLine !== null;
    updateMeter(engine.level.fill, engine.level.zone, attempting);
  }

  requestAnimationFrame(frame);
}

// ---- wiring --------------------------------------------------------------

function wire() {
  for (const form of document.querySelectorAll('form')) {
    form.addEventListener('submit', (event) => event.preventDefault());
  }

  $('secret').addEventListener('input', (event) => {
    S.payload = event.target.value;
    updateInput();
  });
  $('secret').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !$('playBtn').disabled) $('playBtn').click();
  });
  $('showBtn').addEventListener('click', () => {
    S.revealed = !S.revealed;
    updateInput();
  });
  $('clearInputBtn').addEventListener('click', () => {
    if (S.playing) return;
    S.payload = '';
    S.revealed = false;
    $('secret').value = '';
    updateInput();
    $('secret').focus();
  });
  $('playBtn').addEventListener('click', () => { void togglePlay(); });

  $('vol').addEventListener('input', (event) => {
    S.volume = Number(event.target.value);
    setText($('volLabel'), `${S.volume}%`);
    event.target.style.setProperty('--volume', `${S.volume}%`);
    engine.setTxVolume(S.volume);
  });
  $('vol').style.setProperty('--volume', `${S.volume}%`);

  $('micBtn').addEventListener('click', () => { void requestMic(); });

  $('resReveal').addEventListener('click', () => {
    const value = $('resValue');
    const reveal = value.type === 'password';
    value.type = reveal ? 'text' : 'password';
    setText($('resReveal'), reveal ? 'Mask' : 'Reveal');
    $('resReveal').setAttribute('aria-pressed', String(reveal));
  });

  $('resCopy').addEventListener('click', async () => {
    if (S.decoded === null) return;
    const button = $('resCopy');
    const flash = (message, accessibleMessage) => {
      setText(button, message);
      setText($('copyStatus'), accessibleMessage);
      setTimeout(() => {
        setText(button, 'Copy payload');
        setText($('copyStatus'), '');
      }, 1600);
    };

    try {
      await navigator.clipboard.writeText(S.decoded);
      flash('Copied', 'Payload copied to clipboard.');
      return;
    } catch (_) {
      // Clipboard API is unavailable on non-secure origins; use a short-lived,
      // visually hidden field and wipe it before removal.
    }

    let copied = false;
    const scratch = document.createElement('textarea');
    scratch.value = S.decoded;
    scratch.setAttribute('readonly', '');
    scratch.style.position = 'fixed';
    scratch.style.opacity = '0';
    document.body.appendChild(scratch);
    scratch.select();
    try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
    scratch.value = '';
    scratch.remove();
    flash(
      copied ? 'Copied' : 'Copy failed',
      copied ? 'Payload copied to clipboard.' : 'Could not copy the payload.',
    );
  });

  $('resClear').addEventListener('click', clearResult);
}

// ---- debug state forcing -------------------------------------------------

const DEBUG = new URLSearchParams(location.search).get('debug') === '1';
const DEBUG_FP = { hex: '3fa291c7', words: ['RIVER', 'CLAMP', 'OTTER', 'VAULT'] };

function debugHooks() {
  return {
    setPayload(text) {
      $('secret').value = text;
      S.payload = text;
      updateInput();
    },
    setLine(text, tips) {
      setLine(text, tips ?? null);
    },
    setPlaying(copy, pos, dur) {
      $('sendProgress').hidden = false;
      updateSendProgress({
        copyIndex: copy,
        playhead: pos,
        copyDur: dur,
        inGap: false,
        gapRemaining: 0,
      });
      setText($('playBtnText'), 'Stop transmission');
      $('playBtn').classList.add('is-sending');
      renderReceiverPaused();
    },
    setAttempts(n) {
      S.attempts = n;
      S.mergeLine = n > 0 ? `Copy ${n} incomplete. Waiting for next copy…` : null;
      S.mic = 'on';
      S.suspended = false;
      renderMicOn();
      setLine(
        S.mergeLine ?? 'Waiting for signal',
        n >= STALL_AFTER
          ? TIPS
          : n > 0
            ? 'Copies are combined automatically.'
            : 'Start transmission on the sending device.',
      );
    },
    setMicDenied() {
      S.mic = 'denied';
      S.micError = { name: 'NotAllowedError' };
      renderMicDenied();
    },
    injectLevel(value, attempting = false) {
      $('meterBlock').hidden = false;
      const fill = Math.min(1, Math.max(0, value));
      const zone = fill >= 0.78 ? 'clip' : fill < 0.22 ? 'quiet' : 'good';
      updateMeter(fill, zone, attempting);
    },
    showResult(value, copies = 1) {
      S.decoded = value;
      S.decodedBytes = enc.encode(value);
      S.okCopies = copies;
      try {
        S.okFp = makeFingerprint(S.decodedBytes);
      } catch (_) {
        S.okFp = DEBUG_FP;
      }
      renderResult();
    },
    clearResult,
  };
}

// ---- boot ---------------------------------------------------------------

(async () => {
  try {
    await init();
    MAX = max_transfer_payload();
  } catch (err) {
    console.error(err);
    if (!DEBUG) {
      $('bootErr').hidden = false;
      return;
    }
  }

  wire();
  updateInput();
  renderMicOff();

  if (DEBUG) {
    window.__akt = debugHooks();
    return;
  }

  requestAnimationFrame(frame);
})();
