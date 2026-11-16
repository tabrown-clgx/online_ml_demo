import time
from pathlib import Path

import numpy as np

from .data import ImageRecord, pixel_hash
from .presets import PROMPT_TEMPLATES
from .probe import TextPriorProbe, classification_metrics, unit_rows
from .tracking import RunLog


class Experiment:
    """One fixed binary task. Only explicit feedback enters the training set."""

    def __init__(
        self, records: list[ImageRecord], embeddings: np.ndarray, prior: np.ndarray,
        *, preset: str, positive: str, negative: str, encoder_identity: dict,
        runs_dir: str | Path = "runs", regularization: float = 0.1,
        optimizer_steps: int = 300, seed: int = 42, strategy: str = "manual",
        content_hashes: dict[str, str] | None = None,
    ):
        if any(row.preset != preset for row in records):
            raise ValueError("An experiment contains only one preset.")
        if len({r.id for r in records}) != len(records):
            raise ValueError("Duplicate image IDs.")
        if any(r.split == "eval" and not r.reviewed for r in records):
            raise ValueError("Evaluation labels must be reviewed.")
        self.probe = TextPriorProbe(prior, regularization, optimizer_steps)
        embeddings = unit_rows(embeddings)
        if embeddings.shape != (len(records), len(prior)):
            raise ValueError("Expected one embedding per manifest row, matching the text prior.")
        self.records = {row.id: row for row in records}
        self.embeddings = {row.id: vector for row, vector in zip(records, embeddings, strict=True)}
        self.content_hashes = content_hashes if content_hashes is not None else {
            row.id: pixel_hash(row.path) for row in records
        }
        if set(self.content_hashes) != set(self.records):
            raise ValueError("Every record needs a content identity.")
        if len(set(self.content_hashes.values())) != len(records):
            raise ValueError("Duplicate image content; possible train/evaluation leakage.")
        self.feedback: dict[str, int] = {}
        self.history: list[dict] = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.config = {
            "schema_version": 1, "preset": preset, "positive": positive, "negative": negative,
            "encoder": encoder_identity, "regularization": regularization,
            "optimizer_steps": optimizer_steps, "seed": seed, "strategy": strategy,
            "prompt_templates": list(PROMPT_TEMPLATES),
            "evaluation": sorted([
                {"id": r.id, "content_hash": self.content_hashes[r.id], "label": r.label}
                for r in records if r.split == "eval"
            ], key=lambda row: row["id"]),
            "pool": sorted([
                {"id": r.id, "content_hash": self.content_hashes[r.id],
                 "reviewed": r.reviewed, "label": r.label if r.reviewed else None}
                for r in records if r.split == "pool"
            ], key=lambda row: row["id"]),
        }
        self.log = RunLog(runs_dir, self.config)
        self._snapshot(0.0)

    def ids(self, split: str) -> list[str]:
        return sorted(i for i, record in self.records.items() if record.split == split)

    def features(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, len(self.probe.prior)))
        return np.stack([self.embeddings[i] for i in ids])

    def _snapshot(self, update_ms: float) -> dict:
        ids = self.ids("eval")
        x = self.features(ids)
        y = np.array([self.records[i].label for i in ids])
        result = {
            "step": len(self.history), "n_labels": len(self.feedback),
            "n_positive": sum(self.feedback.values()),
            "n_negative": len(self.feedback) - sum(self.feedback.values()),
            "update_ms": update_ms,
            "baseline": classification_metrics(y, self.probe.scores(x, baseline=True)),
            "probe": classification_metrics(y, self.probe.scores(x)),
        }
        self.history.append(result)
        self.log.write("metrics", result)
        return result

    def _rebuild(self) -> dict:
        start = time.perf_counter()
        ids = sorted(self.feedback)
        self.probe.fit(self.features(ids), np.array([self.feedback[i] for i in ids]))
        return self._snapshot((time.perf_counter() - start) * 1000)

    def label(self, image_id: str, value: int | None, *, source: str = "human") -> dict:
        if image_id not in self.records:
            raise ValueError("Unknown image.")
        if self.records[image_id].split != "pool":
            raise ValueError("Held-out evaluation images can never be used for feedback.")
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Feedback must be 0, 1, or None to remove a label.")
        if value is None:
            self.feedback.pop(image_id, None)
        else:
            self.feedback[image_id] = value
        self.log.write("feedback", {
            "image_id": image_id, "content_hash": self.content_hashes[image_id],
            "label": value, "source": source,
        })
        return self._rebuild()

    def reset(self) -> dict:
        self.feedback.clear()
        self.rng = np.random.default_rng(self.seed)
        self.log.write("reset", {})
        return self._rebuild()

    def rank_pool(self, strategy: str = "uncertainty", *, reviewed_only: bool = False) -> list[str]:
        ids = [i for i in self.ids("pool") if i not in self.feedback
               and (not reviewed_only or self.records[i].reviewed)]
        if strategy == "random":
            return list(self.rng.permutation(ids))
        if strategy != "uncertainty":
            raise ValueError("Strategy must be random or uncertainty.")
        uncertainty = np.abs(self.probe.scores(self.features(ids)) - 0.5)
        return [ids[i] for i in np.argsort(uncertainty, kind="stable")]

    def simulate(self, steps: int, strategy: str = "uncertainty") -> None:
        if steps < 0:
            raise ValueError("Simulation steps cannot be negative.")
        self.log.write("simulation", {"steps_requested": steps, "strategy": strategy})
        for _ in range(steps):
            ids = self.rank_pool(strategy, reviewed_only=True)
            if not ids:
                break
            image_id = ids[0]
            self.label(image_id, self.records[image_id].label, source="simulated-reviewed-label")

    def add_upload(self, image, embedding: np.ndarray) -> str:
        digest = pixel_hash(image)
        for image_id, content in self.content_hashes.items():
            if content == digest:
                if self.records[image_id].split == "eval":
                    raise ValueError("This upload duplicates a held-out evaluation image.")
                return image_id
        image_id = f"upload-{digest[:20]}"
        vector = unit_rows(np.asarray(embedding)[None, :])[0]
        if vector.shape != self.probe.prior.shape:
            raise ValueError("Upload embedding dimension does not match the encoder.")
        self.records[image_id] = ImageRecord(
            id=image_id, path="", preset=self.config["preset"], label=0
        )
        self.embeddings[image_id] = vector
        self.content_hashes[image_id] = digest
        self.log.write("upload", {"image_id": image_id, "content_hash": digest})
        return image_id


def create_experiment(encoder, records: list[ImageRecord], *, preset: str, **kwargs) -> Experiment:
    rows = [row for row in records if row.preset == preset]
    prior = encoder.text_prior(kwargs["positive"], kwargs["negative"])
    embeddings = encoder.encode_images([row.path for row in rows])
    return Experiment(rows, embeddings, prior, preset=preset,
                      encoder_identity=encoder.identity, **kwargs)


def synthetic_experiment(*, seed: int = 42, **kwargs) -> Experiment:
    """Clearly synthetic numerical fixture; not a substitute for SigLIP2 evaluation."""
    rng = np.random.default_rng(seed)
    x = unit_rows(rng.normal(size=(240, 16)))
    labels = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    prior = np.zeros(16)
    prior[0], prior[2] = 1.0, 3.0  # Intentionally imperfect synthetic text prior.
    records = [ImageRecord(
        id=f"synthetic-{i:03d}", path="", preset="synthetic", label=int(label),
        split="eval" if i < 80 else "pool", reviewed=True, source="synthetic",
    ) for i, label in enumerate(labels)]
    # No image files exist for this numerical fixture.
    return Experiment(
        records, x, prior, preset="synthetic", positive="synthetic positive",
        negative="synthetic negative", encoder_identity={"backend": "synthetic", "seed": seed},
        content_hashes={r.id: r.id for r in records}, seed=seed, **kwargs,
    )
