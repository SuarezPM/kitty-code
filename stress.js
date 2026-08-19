#!/usr/bin/env node
// Stress test contra el modelo local vía el proxy medidor (endpoint OpenAI-compatible)
// Uso: node stress.js [--concurrency N] [--iterations M]
const http = require('http');

const BASE = 'http://127.0.0.1:8765/v1/chat/completions';
const MODEL = 'qwen38-27b';
const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === '--concurrency') args.concurrency = parseInt(process.argv[++i], 10);
  else if (a === '--iterations') args.iterations = parseInt(process.argv[++i], 10);
}
const CONC = args.concurrency || 4;
const ITERS = args.iterations || 6;

function call(messages, maxTokens) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ model: MODEL, messages, stream: false, max_tokens: maxTokens, temperature: 0.7 });
    const req = http.request(BASE, { method: 'POST', headers: { 'Content-Type': 'application/json', 'User-Agent': 'stress-test' } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        try {
          const j = JSON.parse(Buffer.concat(chunks).toString());
          resolve({ pt: j.usage.prompt_tokens, ct: j.usage.completion_tokens, ms: j.usage.total_time ? j.usage.total_time * 1000 : 0, text: (j.choices[0]?.message?.content || '').slice(0, 60) });
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.end(body);
  });
}

async function single(ms) {
  const t0 = Date.now();
  const r = await call([{ role: 'user', content: 'Escribe una lista de 10 ideas para mejorar la productividad en el desarrollo de software. Formato: una idea por línea.' }], ms || 400);
  const d = (Date.now() - t0) / 1000;
  console.log(`single: ${r.ct} tok en ${d.toFixed(1)}s → ${(r.ct / d).toFixed(2)} t/s (wall)`);
  return r.ct / d;
}

async function concurrent(n, per, label) {
  const t0 = Date.now();
  const rs = await Promise.all(Array.from({ length: n }, () => call([{ role: 'user', content: 'Explica brevemente qué es la recursión en programación, con un ejemplo corto en Python.' }], per)));
  const d = (Date.now() - t0) / 1000;
  const totalTok = rs.reduce((a, r) => a + r.ct, 0);
  const perReq = rs.map(r => r.ct / (d / 1)).sort((a, b) => a - b);
  console.log(`[${label}] ${n} req concurrentes: total ${totalTok} tok en ${d.toFixed(1)}s → agg ${(totalTok / d).toFixed(2)} t/s | peor req ${perReq[0].toFixed(2)} t/s | mejor ${perReq[n - 1].toFixed(2)} t/s`);
  return totalTok / d;
}

async function longContext() {
  const filler = 'La gestión de memoria en sistemas modernos es un tema fascinante. ' .repeat(400); // ~8000 tokens aprox
  const t0 = Date.now();
  const r = await call([{ role: 'user', content: filler + '\n\nResume en una frase: ¿de qué trata el texto anterior?' }], 150);
  const d = (Date.now() - t0) / 1000;
  console.log(`longctx: prompt ${r.pt} tok, gen ${r.ct} tok en ${d.toFixed(1)}s → prefill ~${(r.pt / d).toFixed(0)} t/s incl. gen`);
  console.log(`  respuesta: ${r.text.slice(0, 80)}`);
}

(async () => {
  console.log(`=== STRESS Kitty-Code (concurrency=${CONC}, iterations=${ITERS}) ===`);
  await single();
  await concurrent(2, 200, '2req');
  await concurrent(4, 200, '4req');
  await concurrent(8, 200, '8req');
  await longContext();
  await single(1200);
  console.log('=== STRESS DONE ===');
})().catch((e) => { console.error('FALLO:', e.message); process.exit(1); });