const http = require('http');
const fs = require('fs');

const UPSTREAM = { host: '127.0.0.1', port: 11434 };
const PORT = 8765;
const LOG = fs.createWriteStream('/tmp/opencode/proxy_speed.log', { flags: 'a' });

function log(line) {
  LOG.write(new Date().toISOString() + ' ' + line + '\n');
}

const server = http.createServer((req, res) => {
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    const body = Buffer.concat(chunks);
    const t0 = Date.now();
    const ua = (req.headers['user-agent'] || 'unknown').slice(0, 40);
    const model = (() => { try { return JSON.parse(body).model; } catch { return '?'; } })();
    const stream = (() => { try { return JSON.parse(body).stream; } catch { return false; } })();

    const upstream = http.request({
      host: UPSTREAM.host, port: UPSTREAM.port,
      path: req.url, method: req.method, headers: { ...req.headers, host: `127.0.0.1:${UPSTREAM.port}` },
    }, (up) => {
      res.writeHead(up.statusCode, up.headers);
      if (stream && up.headers['content-type'] && up.headers['content-type'].includes('text/event-stream')) {
        let buf = '';
        let usage = null;
        up.on('data', (c) => {
          res.write(c);
          buf += c.toString();
          const idx = buf.lastIndexOf('data: ');
          if (idx >= 0) {
            const lineStr = buf.slice(idx + 6).split('\n')[0].trim();
            if (lineStr && lineStr !== '[DONE]') {
              try {
                const j = JSON.parse(lineStr);
                if (j.usage) usage = j.usage;
              } catch {}
            }
          }
        });
        up.on('end', () => {
          res.end();
          const ms = Date.now() - t0;
          const pt = usage ? usage.prompt_tokens : 0;
          const ct = usage ? usage.completion_tokens : 0;
          log(`stream ua=${ua} model=${model} path=${req.url} pt=${pt} ct=${ct} ms=${ms} tps=${(ct / (ms / 1000)).toFixed(2)}`);
        });
      } else {
        const rbuf = [];
        up.on('data', (c) => { res.write(c); rbuf.push(c); });
        up.on('end', () => {
          res.end();
          const ms = Date.now() - t0;
          let pt = 0, ct = 0;
          try {
            const j = JSON.parse(Buffer.concat(rbuf).toString());
            if (j.usage) { pt = j.usage.prompt_tokens; ct = j.usage.completion_tokens; }
          } catch {}
          log(`json ua=${ua} model=${model} path=${req.url} pt=${pt} ct=${ct} ms=${ms} tps=${(ct / (ms / 1000)).toFixed(2)}`);
        });
      }
    });
    upstream.on('error', (e) => { res.writeHead(502); res.end('proxy error'); log(`error ${e.message}`); });
    upstream.end(body);
  });
});
server.listen(PORT, '127.0.0.1', () => log(`proxy listening on ${PORT}`));