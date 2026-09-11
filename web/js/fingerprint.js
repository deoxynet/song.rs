// Fingerprint = crc32 of the payload, rendered two ways:
//  - hex: 8 lowercase hex chars
//  - words: four words, one per crc byte (big-endian), from the fixed wordlist
import { WORDS } from './words.js';

export function crcHex(crc) {
  return (crc >>> 0).toString(16).padStart(8, '0');
}

export function crcWords(crc) {
  const c = crc >>> 0;
  const bytes = [(c >>> 24) & 0xff, (c >>> 16) & 0xff, (c >>> 8) & 0xff, c & 0xff];
  return bytes.map((b) => WORDS[b].toUpperCase());
}
