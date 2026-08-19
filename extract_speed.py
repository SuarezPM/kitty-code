#!/usr/bin/env python3
"""Extrae métricas por request del journal de ollama (print_timing).
Uso: extract_speed.py "SINCE" [UNTIL]
Clasifica: decode (eval time final), prefill (prompt eval time final). Ignora progress.
"""
import re, subprocess, sys, statistics

since = sys.argv[1]
until = sys.argv[2] if len(sys.argv) > 2 else 'now'
out = subprocess.run(['sudo', 'journalctl', '-u', 'ollama', '--since', since, '--until', until],
                     capture_output=True, text=True).stdout
decode, prefill, prompt_proc = [], [], []
n_req = 0
for line in out.splitlines():
    if 'print_timing' not in line:
        continue
    m = re.search(r'(\d+\.\d+) tokens per second', line)
    if not m:
        continue
    v = float(m.group(1))
    if 'prompt eval time' in line:
        prefill.append(v)
        n_req += 1
    elif 'prompt processing' in line:
        prompt_proc.append(v)
    else:
        decode.append(v)
        n_req += 1

def stat(name, vals):
    if not vals:
        print(f'{name}: sin datos')
        return
    vals.sort()
    n = len(vals)
    print(f'{name}: n={n} min={vals[0]:.2f} p50={statistics.median(vals):.2f} p90={vals[int(n*0.9)-1]:.2f} max={vals[-1]:.2f} mean={statistics.mean(vals):.2f}')

print(f'=== ventana {since} -> {until} (requests finales: {n_req}) ===')
stat('DECODE  (gen t/s)', decode)
stat('PREFILL (prompt t/s)', prefill)
stat('PROMPT-PROC progress t/s', prompt_proc)
