// Feedback Lab, static edition. All learning happens in this file and probe.js.
(function () {
  "use strict";
  const { TextPriorProbe } = window.Probe;
  const $ = (id) => document.getElementById(id);

  const state = {
    index: null, cache: {}, task: null, probe: null,
    feedback: new Map(), skipped: new Set(), selected: null,
    history: [], justFlipped: new Set(), order: "uncertainty", status: "",
  };

  function decode(b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return new Float32Array(bytes.buffer);
  }

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "className") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
  }

  function creditText(credit) {
    const title = credit.title ? `“${credit.title}”` : "Photo";
    return credit.author ? `${title} by ${credit.author}` : title;
  }

  function licenseName(url) {
    const match = /licenses\/([a-z-]+)\/([\d.]+)/.exec(url || "");
    return match ? `CC ${match[1].toUpperCase()} ${match[2]}` : (url || "");
  }

  // --- model -----------------------------------------------------------------

  const heldOut = () => state.task.photos.filter((p) => p.split === "eval");
  const pool = () => state.task.photos.filter((p) => p.split === "pool");

  function retrain() {
    const labeled = [...state.feedback.keys()].sort();
    const byId = state.task.byId;
    state.probe.fit(labeled.map((id) => byId[id].vector), labeled.map((id) => state.feedback.get(id)));
  }

  function metrics() {
    let tp = 0, tn = 0, fp = 0, fn = 0, btp = 0, btn = 0, bfp = 0, bfn = 0;
    for (const photo of heldOut()) {
      const p = state.probe.score(photo.vector) >= 0.5;
      const b = state.probe.score(photo.vector, true) >= 0.5;
      const t = photo.label === 1;
      if (t) { p ? tp++ : fn++; b ? btp++ : bfn++; } else { p ? fp++ : tn++; b ? bfp++ : btn++; }
    }
    const balanced = (tp, tn, fp, fn) =>
      tp + fn && tn + fp ? (tp / (tp + fn) + tn / (tn + fp)) / 2 : null;
    return {
      labels: state.feedback.size, n: tp + tn + fp + fn,
      correct: tp + tn, baseCorrect: btp + btn,
      balanced: balanced(tp, tn, fp, fn), baseBalanced: balanced(btp, btn, bfp, bfn),
    };
  }

  function nextPhoto() {
    const current = state.selected;
    if (current && !state.feedback.has(current) && !state.skipped.has(current)) return current;
    const candidates = pool().filter((p) => !state.feedback.has(p.id) && !state.skipped.has(p.id));
    if (!candidates.length) return (state.selected = null);
    if (state.order === "random") {
      state.selected = candidates[Math.floor(Math.random() * candidates.length)].id;
    } else {
      let best = candidates[0], bestGap = Infinity;
      for (const photo of candidates) {
        const gap = Math.abs(state.probe.score(photo.vector) - 0.5);
        if (gap < bestGap) { best = photo; bestGap = gap; }
      }
      state.selected = best.id;
    }
    return state.selected;
  }

  function label(value) {
    if (!state.selected) return;
    const held = heldOut();
    const before = held.map((p) => state.probe.score(p.vector) >= 0.5);
    const started = performance.now();
    state.feedback.set(state.selected, value);
    retrain();
    const elapsed = performance.now() - started;
    const after = held.map((p) => state.probe.score(p.vector) >= 0.5);
    state.justFlipped = new Set(held.filter((_, i) => before[i] !== after[i]).map((p) => p.id));
    const fixed = held.filter((p, i) => before[i] !== after[i] && after[i] === (p.label === 1)).length;
    const name = value === 1 ? state.task.positive_name : state.task.negative_name;
    const flips = state.justFlipped.size;
    state.status = flips
      ? `Labeled <b>${name}</b>: flipped <b>${flips}</b> held-out photo${flips === 1 ? "" : "s"} (⚡): ` +
        `${fixed} now right, ${flips - fixed} now wrong.`
      : `Labeled <b>${name}</b>: no held-out photos changed.`;
    state.status += `<br><span class="note">Retrained in ${elapsed.toFixed(0)} ms.</span>`;
    state.selected = null;
    state.history.push(metrics());
    render();
  }

  // --- task loading ----------------------------------------------------------

  async function loadTask(key) {
    const entry = state.index.tasks.find((t) => t.key === key);
    if (!state.cache[key]) {
      const task = await (await fetch(entry.file)).json();
      task.byId = {};
      for (const photo of task.photos) {
        photo.vector = decode(photo.embedding);
        task.byId[photo.id] = photo;
      }
      task.prior = decode(task.prior);
      state.cache[key] = task;
    }
    state.task = state.cache[key];
    startExperiment();
    history.replaceState(null, "", `#${key}`);
  }

  function startExperiment() {
    const lam = parseFloat($("regularization").value) || state.index.regularization;
    state.probe = new TextPriorProbe(state.task.prior, lam, state.index.optimizer_steps);
    state.feedback = new Map();
    state.skipped = new Set();
    state.justFlipped = new Set();
    state.selected = null;
    state.status = "";
    state.history = [metrics()];
    render();
  }

  // --- rendering -------------------------------------------------------------

  function renderTasks() {
    const nav = $("tasks");
    nav.replaceChildren(...state.index.tasks.map((t) => {
      const button = el("button", { type: "button", text: t.title });
      button.setAttribute("aria-pressed", String(state.task && state.task.key === t.key));
      button.addEventListener("click", () => loadTask(t.key));
      return button;
    }));
  }

  function renderSummary(m) {
    const { positive_name: yes, negative_name: no } = state.task;
    const pct = (v) => (v === null ? "—" : `${Math.round(v * 100)}%`);
    const yesCount = [...state.feedback.values()].filter((v) => v === 1).length;
    $("summary").replaceChildren(
      el("div", { className: "headline",
        text: `${yes} vs ${no}: ${m.correct}/${m.n} held-out photos correct ` +
              `(zero-shot: ${m.baseCorrect}/${m.n}) · balanced accuracy ${pct(m.balanced)} ` +
              `(zero-shot ${pct(m.baseBalanced)})` }),
      el("div", { className: "detail",
        text: `${yes} = “${state.task.positive}” · ${no} = “${state.task.negative}” · ` +
              `${m.labels} labels given (${yesCount} ${yes}, ${m.labels - yesCount} ${no})` }),
    );
  }

  function renderLabeler() {
    const { positive_name: yes, negative_name: no } = state.task;
    $("label-yes").textContent = `✓ ${yes}`;
    $("label-no").textContent = `✓ ${no}`;
    const id = nextPhoto();
    const img = $("current");
    for (const button of [$("label-yes"), $("label-no"), $("skip")]) button.disabled = !id;
    if (!id) {
      img.removeAttribute("src");
      $("prompt").textContent = "No more photos to label for this task.";
      return;
    }
    const photo = state.task.byId[id];
    const score = state.probe.score(photo.vector);
    const guess = score >= 0.5 ? yes : no;
    const sure = Math.round(Math.max(score, 1 - score) * 100);
    img.src = photo.image;
    img.title = `${creditText(photo.credit)} (${licenseName(photo.credit.license)})`;
    $("prompt").replaceChildren(
      "Model's guess: ", el("b", { text: guess }), ` (${sure}% sure).`, el("br"),
      "Is this ", el("b", { text: yes }), " or ", el("b", { text: no }), "?",
    );
  }

  function renderGalleries() {
    const { positive_name: yes, negative_name: no } = state.task;
    const scored = heldOut().map((p) => ({
      photo: p, p: state.probe.score(p.vector), b: state.probe.score(p.vector, true),
    })).sort((a, z) => z.p - a.p);
    for (const [side, name, keep] of [["yes", yes, (s) => s.p >= 0.5], ["no", no, (s) => s.p < 0.5]]) {
      const items = scored.filter(keep);
      const wrong = items.filter((s) => (s.p >= 0.5) !== (s.photo.label === 1)).length;
      $(`title-${side}`).textContent = `Model says ${name} · ${items.length} photos, ${wrong} wrong`;
      const figures = items.map((s) => {
        const correct = (s.p >= 0.5) === (s.photo.label === 1);
        const flipped = state.justFlipped.has(s.photo.id);
        let caption = `${Math.round(Math.max(s.p, 1 - s.p) * 100)}% sure`;
        if (flipped) caption = `⚡ ${caption}`;
        if ((s.p >= 0.5) !== (s.b >= 0.5)) caption += ` · was ${s.b >= 0.5 ? yes : no}`;
        const img = el("img", { src: s.photo.image, loading: "lazy",
          alt: `Held-out photo, model says ${name}`,
          title: `${creditText(s.photo.credit)} (${licenseName(s.photo.credit.license)})` });
        return el("figure", { className: `${correct ? "right" : "wrong"}${flipped ? " flipped" : ""}` },
          img, el("figcaption", { text: caption }));
      });
      $(`gallery-${side}`).replaceChildren(
        ...(figures.length ? figures : [el("p", { className: "empty", text: "None" })]));
    }
  }

  function renderCurve() {
    const svg = $("curve");
    const ns = "http://www.w3.org/2000/svg";
    const W = 320, H = 170, L = 34, R = 10, T = 12, B = 30;
    const n = state.history[0].n || 1;
    const lowest = Math.min(...state.history.map((m) => Math.min(m.correct, m.baseCorrect)));
    const floor = Math.max(0, lowest - 4);
    const maxX = Math.max(5, state.history.length - 1);
    const x = (i) => L + (i / maxX) * (W - L - R);
    const y = (v) => T + (1 - (v - floor) / (n - floor || 1)) * (H - T - B);
    const node = (tag, attrs, text) => {
      const e = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      if (text !== undefined) e.textContent = text;
      return e;
    };
    const parts = [
      node("line", { x1: L, x2: W - R, y1: y(n), y2: y(n), stroke: "#e2e8f0" }),
      node("line", { x1: L, x2: W - R, y1: y(floor), y2: y(floor), stroke: "#cbd5e1" }),
      node("text", { x: L - 6, y: y(n) + 4, "text-anchor": "end", "font-size": 10, fill: "#64748b" }, n),
      node("text", { x: L - 6, y: y(floor) + 4, "text-anchor": "end", "font-size": 10, fill: "#64748b" }, floor),
      node("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 10,
        fill: "#64748b" }, "labels given → held-out photos correct"),
      node("line", { x1: L, x2: W - R, y1: y(state.history[0].baseCorrect),
        y2: y(state.history[0].baseCorrect), stroke: "#94a3b8", "stroke-dasharray": "4 3" }),
      node("polyline", { fill: "none", stroke: "#0d9488", "stroke-width": 2,
        points: state.history.map((m, i) => `${x(i)},${y(m.correct)}`).join(" ") }),
    ];
    state.history.forEach((m, i) => parts.push(node("circle", { cx: x(i), cy: y(m.correct), r: 2.5,
      fill: "#0d9488" })));
    svg.replaceChildren(...parts);
  }

  function renderCredits() {
    $("descriptions").textContent =
      `${state.task.positive_name}: “${state.task.positive}” · ` +
      `${state.task.negative_name}: “${state.task.negative}”`;
    const items = state.task.photos.map((p) => {
      const c = p.credit;
      const li = el("li");
      li.append(c.url ? el("a", { href: c.url, rel: "noopener", text: creditText(c) }) : creditText(c));
      if (c.license) li.append(" · ", el("a", { href: c.license, rel: "noopener", text: licenseName(c.license) }));
      return li;
    });
    $("credit-list").replaceChildren(...items);
  }

  function render() {
    const m = state.history[state.history.length - 1];
    renderTasks();
    renderSummary(m);
    renderLabeler();
    renderGalleries();
    renderCurve();
    renderCredits();
    $("status").innerHTML = state.status;  // composed above from fixed text and task names
  }

  // --- wiring ----------------------------------------------------------------

  async function main() {
    state.index = await (await fetch("data/index.json")).json();
    $("regularization").value = state.index.regularization;
    $("label-yes").addEventListener("click", () => label(1));
    $("label-no").addEventListener("click", () => label(0));
    $("skip").addEventListener("click", () => {
      if (state.selected) state.skipped.add(state.selected);
      state.selected = null;
      render();
    });
    $("reset").addEventListener("click", startExperiment);
    $("regularization").addEventListener("change", startExperiment);
    for (const radio of document.querySelectorAll('input[name="order"]')) {
      radio.addEventListener("change", () => { state.order = radio.value; state.selected = null; render(); });
    }
    $("simulate").addEventListener("click", () => {
      for (let i = 0; i < 5; i++) {
        const id = nextPhoto();
        if (!id) break;
        state.feedback.set(id, state.task.byId[id].label);
        state.selected = null;
        retrain();
        state.history.push(metrics());
      }
      state.justFlipped = new Set();
      state.status = "Auto-labeled up to 5 photos using the reviewed labels.";
      render();
    });
    const requested = location.hash.slice(1);
    const first = state.index.tasks.some((t) => t.key === requested) ? requested : state.index.tasks[0].key;
    await loadTask(first);
  }

  main().catch((error) => {
    $("summary").textContent = `Could not load the demo data: ${error.message}`;
  });
})();
