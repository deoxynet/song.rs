// Build both bindings from the same Rust source; no JS bundler required.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const crate = fileURLToPath(new URL('../wasm', import.meta.url));
for (const [target, directory] of [['web', 'pkg'], ['no-modules', 'pkg-worker']]) {
  execFileSync('wasm-pack', ['build', crate, '--target', target,
    '--out-dir', fileURLToPath(new URL(`./${directory}`, import.meta.url)),
    ...process.argv.slice(2)], { stdio: 'inherit' });
}
