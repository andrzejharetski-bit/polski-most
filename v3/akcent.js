/* Polski Most — silnik akcentu (on-device, bez serwera).
 * Whisper base (open-source, ONNX) uruchomiony w przeglądarce przez onnxruntime-web.
 * Oczekiwane zdanie podajemy dekoderowi (teacher forcing) i czytamy, jak prawdopodobne jest każde słowo
 * przy nagraniu ucznia. Wynik porównujemy z tym samym zdaniem nagranym przez polskiego lektora (kalibracja),
 * a warianty z typowymi błędami rosyjskojęzycznych (ł→l, ą→a, sz→s…) wskazują konkretną głoskę. */
(function (G) {
  "use strict";
  var SR = 16000, NFFT = 400, HOP = 160, NMEL = 80, NB = NFFT / 2 + 1, WINDOW_S = 10;
  var BASE = "", CFG = null, ort = null, ENC = null, DEC = null, TOK = null, REFS = {}, READY = false, LOADING = null;

  /* ---------- DSP: log-mel identical to openai/whisper ---------- */
  var MEL, COS, SIN, WIN;
  function hzToMel(f) { var fsp = 200 / 3, minHz = 1000, minMel = minHz / fsp, ls = Math.log(6.4) / 27; return f >= minHz ? minMel + Math.log(f / minHz) / ls : f / fsp; }
  function melToHz(m) { var fsp = 200 / 3, minHz = 1000, minMel = minHz / fsp, ls = Math.log(6.4) / 27; return m >= minMel ? minHz * Math.exp(ls * (m - minMel)) : fsp * m; }
  function initDSP() {
    if (MEL) return;
    var fft = new Float64Array(NB), i, j;
    for (i = 0; i < NB; i++) fft[i] = i * (SR / 2) / (NB - 1);
    var mmin = hzToMel(0), mmax = hzToMel(SR / 2), mf = new Float64Array(NMEL + 2);
    for (i = 0; i < NMEL + 2; i++) mf[i] = melToHz(mmin + (mmax - mmin) * i / (NMEL + 1));
    MEL = new Float32Array(NMEL * NB);
    for (i = 0; i < NMEL; i++) {
      var en = 2 / (mf[i + 2] - mf[i]);
      for (j = 0; j < NB; j++) {
        var lo = (fft[j] - mf[i]) / (mf[i + 1] - mf[i]), up = (mf[i + 2] - fft[j]) / (mf[i + 2] - mf[i + 1]);
        MEL[i * NB + j] = Math.max(0, Math.min(lo, up)) * en;
      }
    }
    WIN = new Float32Array(NFFT); for (i = 0; i < NFFT; i++) WIN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT);
    COS = new Float32Array(NB * NFFT); SIN = new Float32Array(NB * NFFT);
    for (i = 0; i < NB; i++) for (j = 0; j < NFFT; j++) { var a = 2 * Math.PI * i * j / NFFT; COS[i * NFFT + j] = Math.cos(a); SIN[i * NFFT + j] = Math.sin(a); }
  }
  function logMel(audio, seconds) {
    initDSP();
    var n = SR * seconds, a = new Float32Array(n + NFFT), i, k, m, f;
    var len = Math.min(n, audio.length);
    for (i = 0; i < len; i++) a[i + NFFT / 2] = audio[i];
    for (i = 1; i <= NFFT / 2; i++) { a[NFFT / 2 - i] = a[NFFT / 2 + i]; a[NFFT / 2 + n - 1 + i] = a[NFFT / 2 + n - 1 - i]; } // reflect
    var T = n / HOP, out = new Float32Array(NMEL * T), frame = new Float32Array(NFFT), pw = new Float32Array(NB), maxv = -1e9;
    var lastActive = Math.ceil((len + NFFT) / HOP) + 1;
    for (f = 0; f < T; f++) {
      if (f > lastActive) { for (m = 0; m < NMEL; m++) out[m * T + f] = -10; continue; }
      var off = f * HOP, nz = false;
      for (i = 0; i < NFFT; i++) { frame[i] = a[off + i] * WIN[i]; if (frame[i] !== 0) nz = true; }
      if (!nz) { for (m = 0; m < NMEL; m++) out[m * T + f] = -10; continue; }
      for (k = 0; k < NB; k++) {
        var re = 0, im = 0, base = k * NFFT;
        for (i = 0; i < NFFT; i++) { re += frame[i] * COS[base + i]; im -= frame[i] * SIN[base + i]; }
        pw[k] = re * re + im * im;
      }
      for (m = 0; m < NMEL; m++) {
        var s = 0, mb = m * NB; for (k = 0; k < NB; k++) s += MEL[mb + k] * pw[k];
        var v = Math.log10(Math.max(s, 1e-10)); out[m * T + f] = v; if (v > maxv) maxv = v;
      }
    }
    for (i = 0; i < out.length; i++) { var x = Math.max(out[i], maxv - 8); out[i] = (x + 4) / 4; }
    return { data: out, T: T };
  }

  /* ---------- tokenizer: byte-level BPE (tiktoken ranks) ---------- */
  function Tok(text) {
    this.rank = new Map(); this.inv = [];
    var lines = text.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var p = lines[i].trim().split(/\s+/); if (p.length !== 2) continue;
      var b, id = +p[1]; try { b = atob(p[0]); } catch (e) { continue; } this.rank.set(b, id); this.inv[id] = b;
    }
  }
  var PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu, ENCODER = new TextEncoder(), DECODER = new TextDecoder();
  Tok.prototype.bpe = function (bytes) {
    var parts = []; for (var i = 0; i < bytes.length; i++) parts.push(String.fromCharCode(bytes[i]));
    while (parts.length > 1) {
      var best = -1, bi = -1;
      for (var j = 0; j < parts.length - 1; j++) { var r = this.rank.get(parts[j] + parts[j + 1]); if (r !== undefined && (best < 0 || r < best)) { best = r; bi = j; } }
      if (bi < 0) break;
      parts.splice(bi, 2, parts[bi] + parts[bi + 1]);
    }
    var self = this; return parts.map(function (p) { return self.rank.get(p); });
  };
  Tok.prototype.encode = function (text) { var out = [], m = text.match(PAT) || []; for (var i = 0; i < m.length; i++) out = out.concat(this.bpe(ENCODER.encode(m[i]))); return out; };
  Tok.prototype.decode = function (ids) {
    var s = ""; for (var i = 0; i < ids.length; i++) s += this.inv[ids[i]] || "";
    var b = new Uint8Array(s.length); for (var j = 0; j < s.length; j++) b[j] = s.charCodeAt(j);
    return DECODER.decode(b);
  };

  /* ---------- model loading (cached on the phone after the first download) ---------- */
  function fetchCached(url, onProgress) {
    var cacheName = "pm-akcent-v1";
    var getCache = (G.caches && G.isSecureContext) ? caches.open(cacheName) : Promise.resolve(null);
    return getCache.then(function (cache) {
      var hit = cache ? cache.match(url) : Promise.resolve(null);
      return hit.then(function (r) {
        if (r) return r.arrayBuffer().then(function (b) { if (onProgress) onProgress(b.byteLength, b.byteLength); return b; });
        return fetch(url).then(function (resp) {
          if (!resp.ok) throw new Error("HTTP " + resp.status + " " + url);
          var total = +resp.headers.get("content-length") || 0, got = 0, chunks = [], rd = resp.body && resp.body.getReader ? resp.body.getReader() : null;
          if (!rd) return resp.arrayBuffer();
          function pump() { return rd.read().then(function (x) { if (x.done) return; chunks.push(x.value); got += x.value.length; if (onProgress) onProgress(got, total); return pump(); }); }
          return pump().then(function () { var buf = new Uint8Array(got), o = 0; chunks.forEach(function (c) { buf.set(c, o); o += c.length; }); return buf.buffer; });
        }).then(function (buf) { if (cache) { try { cache.put(url, new Response(buf.slice(0))); } catch (e) {} } return buf; });
      });
    });
  }
  function load(opts) {
    if (READY) return Promise.resolve(true);
    if (LOADING) return LOADING;
    opts = opts || {}; BASE = opts.base || BASE; var prog = opts.onProgress || function () {};
    ort = G.ort; if (!ort) return Promise.reject(new Error("onnxruntime-web not loaded"));
    ort.env.wasm.wasmPaths = new URL(BASE + "ort/", G.location.href).href; ort.env.wasm.numThreads = (G.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1);
    var sizes = { enc: 29.1e6, dec: 51.4e6 }, got = { enc: 0, dec: 0 };
    function upd() { prog(Math.min(1, (got.enc + got.dec) / (sizes.enc + sizes.dec))); }
    LOADING = Promise.all([
      fetch(BASE + "model/base-config.json").then(function (r) { return r.json(); }),
      fetch(BASE + "model/base-tokens.txt").then(function (r) { return r.text(); }),
      fetch(BASE + "model/refs.json").then(function (r) { return r.ok ? r.json() : {}; }).catch(function () { return {}; }),
      fetchCached(BASE + "model/base-encoder.int8.onnx", function (g) { got.enc = g; upd(); }),
      fetchCached(BASE + "model/base-decoder.small.onnx", function (g) { got.dec = g; upd(); })
    ]).then(function (r) {
      CFG = r[0]; TOK = new Tok(r[1]); REFS = r[2] || {};
      var so = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      return Promise.all([ort.InferenceSession.create(r[3], so), ort.InferenceSession.create(r[4], so)]);
    }).then(function (s) { ENC = s[0]; DEC = s[1]; READY = true; prog(1); return true; })
      .catch(function (e) { LOADING = null; throw e; });
    return LOADING;
  }

  /* ---------- inference ---------- */
  var ZK = null, ZV = null;
  function zeros() {
    if (!ZK) { var n = CFG.n_layer * CFG.n_ctx * CFG.n_state; ZK = new ort.Tensor("float32", new Float32Array(n), [CFG.n_layer, 1, CFG.n_ctx, CFG.n_state]); ZV = new ort.Tensor("float32", new Float32Array(n), [CFG.n_layer, 1, CFG.n_ctx, CFG.n_state]); }
    return [ZK, ZV];
  }
  function encode(audio) {
    var mel = logMel(audio, WINDOW_S);
    return ENC.run({ mel: new ort.Tensor("float32", mel.data, [1, NMEL, mel.T]) }).then(function (o) { return { k: o.n_layer_cross_k, v: o.n_layer_cross_v }; });
  }
  function prefix() { return [CFG.sot, CFG.pl, CFG.transcribe, CFG.notimestamps]; }
  function decRun(tokens, k, v, cross, offset) {
    return DEC.run({ tokens: new ort.Tensor("int64", BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]), in_n_layer_self_k_cache: k, in_n_layer_self_v_cache: v,
      n_layer_cross_k: cross.k, n_layer_cross_v: cross.v, offset: new ort.Tensor("int64", BigInt64Array.from([BigInt(offset)]), [1]) });
  }
  function logSoftmaxAt(lg, row, V) {
    var base = row * V, mx = -1e30, i, s = 0, arg = 0;
    for (i = 0; i < V; i++) if (lg[base + i] > mx) { mx = lg[base + i]; arg = i; }
    for (i = 0; i < V; i++) s += Math.exp(lg[base + i] - mx);
    var lse = mx + Math.log(s); return { lse: lse, arg: arg };
  }
  function forced(cross, ids) {
    var z = zeros(), pre = prefix(), seq = pre.concat(ids);
    return decRun(seq, z[0], z[1], cross, 0).then(function (o) {
      var lg = o.logits.data, V = o.logits.dims[2], out = [], tgt = ids.concat([CFG.eot]);
      for (var j = 0; j < tgt.length; j++) { var row = pre.length - 1 + j, r = logSoftmaxAt(lg, row, V); out.push({ t: tgt[j], lp: lg[row * V + tgt[j]] - r.lse, top: r.arg }); }
      return out;
    });
  }
  function greedy(cross, maxLen) {
    var z = zeros(), seq = prefix(), out = [];
    return decRun(seq, z[0], z[1], cross, 0).then(function step(o) {
      var lg = o.logits.data, V = o.logits.dims[2], T = o.logits.dims[1], best = -1e30, arg = 0, base = (T - 1) * V;
      for (var i = 0; i < V; i++) if (lg[base + i] > best) { best = lg[base + i]; arg = i; }
      if (arg === CFG.eot || out.length >= maxLen) return TOK.decode(out);
      out.push(arg);
      return decRun([arg], o.out_n_layer_self_k_cache, o.out_n_layer_self_v_cache, cross, seq.length + out.length - 1).then(step);
    });
  }

  /* ---------- scoring ---------- */
  var WORD = /\p{L}+/gu;
  var SUBS = [["ą", ["a", "on", "om"]], ["ę", ["e", "en", "em"]], ["ł", ["l"]], ["rz", ["r"]], ["sz", ["s", "ś"]], ["cz", ["c", "ć"]], ["dź", ["dz", "dzi"]],
    ["ś", ["s", "sz"]], ["ć", ["c", "cz"]], ["ź", ["z", "ż"]], ["ż", ["z"]], ["ń", ["n"]], ["y", ["i"]], ["ó", ["o"]], ["h", ["g"]]];
  function variants(word) {
    var lw = word.toLowerCase(), out = [], seen = {};
    SUBS.forEach(function (s) {
      var i = lw.indexOf(s[0]);
      while (i >= 0) { s[1].forEach(function (t) { var v = word.slice(0, i) + t + word.slice(i + s[0].length); if (v.toLowerCase() !== lw && !seen[v]) { seen[v] = 1; out.push({ v: v, sound: s[0], said: t }); } }); i = lw.indexOf(s[0], i + 1); }
    });
    return out;
  }
  function normKey(t) { return String(t).toLowerCase().replace(/[^\p{L}\s]/gu, " ").replace(/\s+/g, " ").trim(); }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i].lp; return s; }
  function wordScore(delta) { return Math.max(0, Math.min(100, Math.round(100 * (1 - (-delta - 0.3) / 3)))); }

  function assess(audio, text, opts) {
    opts = opts || {};
    if (!READY) return Promise.reject(new Error("engine not ready"));
    var t0 = performance.now(), words = (text.match(WORD) || []), ids = TOK.encode(" " + text.trim()), cross, base, perWord = words.map(function () { return []; });
    return encode(audio).then(function (c) { cross = c; return forced(cross, ids); }).then(function (res) {
      base = res;
      var wi = 0, consumed = "";
      for (var j = 0; j < res.length - 1; j++) {
        var letters = (TOK.decode([res[j].t]).match(WORD) || []).join(""); if (!letters) continue;
        while (wi < words.length && consumed.length >= words[wi].length) { wi++; consumed = ""; }
        if (wi >= words.length) break;
        perWord[wi].push(res[j].lp); consumed += letters;
      }
      var total = sum(res), ref = REFS[normKey(text)] || null;
      var W = words.map(function (w, k) {
        var lps = perWord[k], lp = lps.length ? lps.reduce(function (a, b) { return a + b; }, 0) / lps.length : -5;
        var delta = ref && ref[k] !== undefined ? lp - ref[k] : lp + 0.6;
        return { w: w, lp: lp, delta: delta, score: wordScore(delta), st: delta >= -0.8 ? "ok" : delta >= -1.8 ? "accent" : "bad", diag: null };
      });
      // which sound went wrong: try typical substitutions for weak words (or all, if asked)
      var jobs = [];
      W.forEach(function (x, k) {
        if (!opts.allVariants && x.st === "ok") return;
        variants(x.w).forEach(function (v) { jobs.push({ k: k, v: v }); });
      });
      var chain = Promise.resolve();
      jobs.forEach(function (jb) {
        chain = chain.then(function () {
          var parts = words.slice(); parts[jb.k] = jb.v.v;
          var alt = text.replace(new RegExp("(^|[^\\p{L}])" + words[jb.k] + "(?![\\p{L}])", "u"), function (m, p) { return p + jb.v.v; });
          return forced(cross, TOK.encode(" " + alt.trim())).then(function (r2) {
            var gain = sum(r2) - total, x = W[jb.k];
            if (gain > 0.5 && (!x.diag || gain > x.diag.gain)) x.diag = { heard: jb.v.v, sound: jb.v.sound, said: jb.v.said, gain: gain };
          });
        });
      });
      return chain.then(function () {
        W.forEach(function (x) { if (x.diag) { x.st = "bad"; x.score = Math.min(x.score, 45); } });
        var sc = Math.round(W.reduce(function (a, x) { return a + x.score; }, 0) / Math.max(1, W.length));
        return (opts.heard === false ? Promise.resolve("") : greedy(cross, 30)).then(function (h) {
          return { words: W, score: sc, heard: (h || "").trim(), ms: Math.round(performance.now() - t0), calibrated: !!ref };
        });
      });
    });
  }

  /* ---------- microphone: record until a pause (or tap), 16 kHz mono ---------- */
  function Recorder() { this.stream = null; this.ctx = null; this.chunks = []; this.rate = 48000; this.on = false; }
  Recorder.prototype.start = function (onLevel, onAutoStop) {
    var self = this; self.chunks = []; self.on = true;
    try { if (navigator.audioSession) navigator.audioSession.type = "play-and-record"; } catch (e) {}
    return navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } }).then(function (st) {
      self.stream = st; var AC = G.AudioContext || G.webkitAudioContext; self.ctx = new AC(); self.rate = self.ctx.sampleRate;
      var src = self.ctx.createMediaStreamSource(st), proc = self.ctx.createScriptProcessor(4096, 1, 1), spoke = false, quiet = 0, t = 0, noise = 0.004;
      proc.onaudioprocess = function (e) {
        if (!self.on) return; var x = e.inputBuffer.getChannelData(0), c = new Float32Array(x.length), rms = 0;
        for (var i = 0; i < x.length; i++) { c[i] = x[i]; rms += x[i] * x[i]; }
        rms = Math.sqrt(rms / x.length); self.chunks.push(c); t += x.length / self.rate; if (onLevel) onLevel(rms);
        if (t < 0.4) noise = Math.max(noise, rms * 1.8);
        if (rms > Math.max(0.012, noise * 2)) { spoke = true; quiet = 0; } else if (spoke) quiet += x.length / self.rate;
        if ((spoke && quiet > 1.0) || t > 9.5) { if (onAutoStop) onAutoStop(); }
      };
      src.connect(proc); proc.connect(self.ctx.destination); self.proc = proc; self.src = src;
    });
  };
  Recorder.prototype.stop = function () {
    var self = this; self.on = false;
    try { self.proc && self.proc.disconnect(); self.src && self.src.disconnect(); } catch (e) {}
    try { self.stream && self.stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) {}
    try { self.ctx && self.ctx.close(); } catch (e) {}
    try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch (e) {}
    var n = 0; self.chunks.forEach(function (c) { n += c.length; });
    var all = new Float32Array(n), o = 0; self.chunks.forEach(function (c) { all.set(c, o); o += c.length; });
    return resample(all, self.rate);
  };
  function resample(x, rate) {
    if (rate === SR) return trim(x);
    var ratio = rate / SR, n = Math.floor(x.length / ratio), out = new Float32Array(n), w = Math.max(1, Math.round(ratio));
    for (var i = 0; i < n; i++) { var c = i * ratio, s = 0, k = 0, a = Math.floor(c - w / 2); for (var j = a; j < a + w; j++) if (j >= 0 && j < x.length) { s += x[j]; k++; } out[i] = k ? s / k : 0; }
    return trim(out);
  }
  function trim(x) { // cut leading/trailing silence, keep 0.25 s margin; normalise level
    var thr = 0.01, a = 0, b = x.length - 1; while (a < b && Math.abs(x[a]) < thr) a++; while (b > a && Math.abs(x[b]) < thr) b--;
    a = Math.max(0, a - SR / 4); b = Math.min(x.length, b + SR / 4); var y = x.slice(a, b), mx = 0;
    for (var i = 0; i < y.length; i++) mx = Math.max(mx, Math.abs(y[i])); if (mx > 0 && mx < 0.5) for (i = 0; i < y.length; i++) y[i] *= 0.5 / mx;
    return y;
  }

  G.Akcent = { load: load, assess: assess, Recorder: Recorder, ready: function () { return READY; }, setRefs: function (r) { REFS = r || {}; }, _logMel: logMel, _tok: function () { return TOK; }, _forced: function (a, t) { return encode(a).then(function (c) { return forced(c, TOK.encode(" " + t.trim())); }); } };
})(typeof window !== "undefined" ? window : this);
