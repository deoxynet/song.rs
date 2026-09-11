# AKT — Simplified single-screen spec (supersedes the 11-state board for layout/flow)

Principle: two people in a room; one has a string, the other needs it. The UI is
"play it there, hear it here." Roles emerge from behavior — never from navigation.

Keep the Modernist design system (modernist-styles.css) and ALL of the state-board
README's visual rules (tokens, 2px rules, zero radius, flush-left buttons, mono
readouts, red discipline, no security theater) and its audio contract. What changes
is the information architecture:

## One screen, three zones, both devices identical

1. **Field + play.** One `.input` (masked, show toggle, live byte counter ≤120 B)
   and one primary button: `play` → loops copies with 500 ms gaps until pressed
   again (`stop`). Volume slider beside it. The moment text is entered, the
   **fingerprint words** (crc32 → four words, Archivo 800, the largest type on
   screen) appear directly under the field.
2. **Live line + meter.** The page requests the mic once on load (constraints per
   the audio contract) and is ALWAYS listening — no receive mode. One sentence in
   mono that reflects reality: `listening` → `hearing something…` →
   `copy 2 didn't land — merging with the next` → escalating tips inline after ~5
   failed copies (never a separate screen, never styled as an error). The 3-zone
   level meter lives here, always visible. While playing, the line shows
   `playing copy 3 · 14.0/22.0 s` instead (a device does not decode its own sound).
3. **Result card.** On verified decode: the value (masked + reveal + copy button),
   fingerprint words huge for cross-check, `verified · copy N`, and `clear` (wipes
   value AND all metadata from DOM/memory — audited requirement). Card persists
   until cleared; listening continues behind it.

Mic denied → sender-only: zone 2 shows `mic off — this device can only send` with
a re-request affordance. Internal decoder states (lock/merge/attempts) drive only
the sentence in zone 2; they are never shown as screens or state ids. Keep the
`?debug=1` `window.__akt` hook (adapted: force zone contents). Keep the reference
footer (air-time table + interface signatures) collapsed behind a `reference`
disclosure at the bottom.
