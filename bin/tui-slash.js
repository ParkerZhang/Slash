#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const child = spawn('npx', ['tsx', 'src/index.tsx'], {
    cwd: projectRoot,
    stdio: 'inherit',
});

child.on('error', (err) => {
    console.error('Failed to start tui-slash:', err.message);
    process.exit(1);
});

child.on('exit', (code) => {
    process.exit(code ?? 0);
});
