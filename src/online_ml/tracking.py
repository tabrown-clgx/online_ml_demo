"""Append-only local experiment logs. No images or image paths are logged."""

import csv
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import fingerprint

_WRITE_LOCK = threading.Lock()


class RunLog:
    def __init__(self, directory: str | Path, config: dict):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.path = directory / f"{self.id}.jsonl"
        self.write("config", config)

    def write(self, event: str, payload: dict) -> None:
        record = {
            "event": event, "time": datetime.now(timezone.utc).isoformat(),
            "run_id": self.id, **payload,
        }
        with _WRITE_LOCK, self.path.open("a") as target:
            target.write(json.dumps(record, sort_keys=True, default=str, allow_nan=False) + "\n")


def read_run(path: str | Path) -> tuple[dict, list[dict]]:
    config = None
    metrics = []
    with Path(path).open() as source:
        for line in source:
            record = json.loads(line)
            if record["event"] == "config":
                config = record
            elif record["event"] == "metrics":
                metrics.append(record)
    if config is None:
        raise ValueError(f"Not an experiment log: {path}")
    return config, metrics


def export_csv(path: str | Path) -> Path:
    config, metrics = read_run(path)
    output = Path(path).with_suffix(".csv")
    rows = []
    for metric in metrics:
        row = {k: v for k, v in metric.items() if not isinstance(v, dict)}
        row.update({"preset": config["preset"], "strategy": config.get("strategy", "manual")})
        for kind in ("baseline", "probe"):
            row.update({f"{kind}_{k}": v for k, v in metric[kind].items()})
        rows.append(row)
    if not rows:
        raise ValueError("Run has no metrics to export.")
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def comparison_key(config: dict) -> str:
    """Only compare the same task, baseline, feature space, and exact held-out set."""
    return fingerprint({k: config[k] for k in (
        "preset", "positive", "negative", "encoder", "evaluation", "prompt_templates"
    )})


def comparable_runs(directory: str | Path, config: dict) -> list[tuple[dict, list[dict]]]:
    result = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        try:
            other, history = read_run(path)
            if comparison_key(other) == comparison_key(config):
                result.append((other, history))
        except (ValueError, KeyError):
            continue
    return result
