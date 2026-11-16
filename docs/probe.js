// Text-prior logistic probe: a line-for-line port of online_ml/probe.py.
// Works in the browser (window.Probe) and in Node (module.exports) for tests.
(function (root) {
  "use strict";

  function sigmoid(z) {
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }

  function unit(vector) {
    let norm = 0;
    for (let i = 0; i < vector.length; i++) norm += vector[i] * vector[i];
    norm = Math.sqrt(norm);
    if (!(norm > 1e-12)) throw new Error("Embeddings must have nonzero norm.");
    const out = new Float64Array(vector.length);
    for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
    return out;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  class TextPriorProbe {
    constructor(prior, regularization = 0.1, steps = 300, balanced = true) {
      if (!(regularization > 0) || !isFinite(regularization) || !(steps >= 1)) {
        throw new Error("Regularization and optimizer steps must be positive.");
      }
      this.prior = Float64Array.from(prior);
      this.regularization = regularization;
      this.steps = steps;
      this.balanced = balanced;
      this.weights = Float64Array.from(this.prior);
      this.bias = 0;
    }

    // Rebuild from the prior on every call; label order cannot affect the result.
    fit(embeddings, labels) {
      const x = embeddings.map(unit);
      const y = Float64Array.from(labels);
      const n = x.length;
      const d = this.prior.length;
      this.weights = Float64Array.from(this.prior);
      this.bias = 0;
      if (!n) return this;
      // Class-balanced, mean-one sample weights.
      const counts = {};
      for (const label of y) counts[label] = (counts[label] || 0) + 1;
      const k = Object.keys(counts).length;
      const w = y.map((label) => (this.balanced ? n / (k * counts[label]) : 1));
      const lam = this.regularization;
      const rate = 1 / (0.5 + lam);
      const errors = new Float64Array(n);
      const dw = new Float64Array(d);
      for (let step = 0; step < this.steps; step++) {
        let db = 0;
        for (let i = 0; i < n; i++) {
          errors[i] = w[i] * (sigmoid(dot(x[i], this.weights) + this.bias) - y[i]);
          db += errors[i];
        }
        db = db / n + lam * this.bias;
        let largest = Math.abs(db);
        for (let j = 0; j < d; j++) {
          let g = 0;
          for (let i = 0; i < n; i++) g += x[i][j] * errors[i];
          dw[j] = g / n + lam * (this.weights[j] - this.prior[j]);
          largest = Math.max(largest, Math.abs(dw[j]));
        }
        for (let j = 0; j < d; j++) this.weights[j] -= rate * dw[j];
        this.bias -= rate * db;
        if (largest < 1e-8) break;
      }
      return this;
    }

    score(embedding, baseline = false) {
      const x = unit(embedding);
      return baseline ? sigmoid(dot(x, this.prior)) : sigmoid(dot(x, this.weights) + this.bias);
    }
  }

  const api = { TextPriorProbe, sigmoid, unit };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Probe = api;
})(typeof window !== "undefined" ? window : globalThis);
