import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from online_ml.probe import TextPriorProbe

PROBE_JS = Path(__file__).resolve().parents[1] / "docs" / "probe.js"
NODE = shutil.which("node")

RUNNER = """
const { TextPriorProbe } = require(process.argv[1]);
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map((c) => {
  const p = new TextPriorProbe(c.prior, c.lam, c.steps, c.balanced).fit(c.x, c.y);
  return { weights: Array.from(p.weights), bias: p.bias,
           scores: c.x.map((v) => p.score(v)), baseline: c.x.map((v) => p.score(v, true)) };
});
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_browser_probe_matches_python_probe():
    rng = np.random.default_rng(3)
    cases = []
    for n, d, balanced, lam, labels in [
        (12, 16, True, 0.1, None), (12, 16, False, 0.1, None), (30, 32, True, 0.01, None),
        (5, 8, True, 0.1, [1, 1, 1, 1, 1]), (0, 8, True, 0.1, []), (9, 768, True, 0.1, None),
    ]:
        x = rng.normal(size=(n, d))
        y = labels if labels is not None else [int(v) for v in rng.integers(0, 2, n)]
        if labels is None and n:
            y[0], y[1] = 1, 0  # ensure both classes appear
        cases.append({"x": x.tolist(), "y": y, "prior": rng.normal(size=d).tolist(),
                      "lam": lam, "steps": 300, "balanced": balanced})
    result = subprocess.run([NODE, "-e", RUNNER, str(PROBE_JS)], input=json.dumps(cases),
                            capture_output=True, text=True, check=True)
    for case, js in zip(cases, json.loads(result.stdout), strict=True):
        probe = TextPriorProbe(np.array(case["prior"]), case["lam"], case["steps"],
                               balanced=case["balanced"])
        x = np.array(case["x"]).reshape(len(case["y"]), len(case["prior"]))
        probe.fit(x, np.array(case["y"]))
        np.testing.assert_allclose(js["weights"], probe.weights, atol=1e-9)
        assert js["bias"] == pytest.approx(probe.bias, abs=1e-9)
        if len(x):
            np.testing.assert_allclose(js["scores"], probe.scores(x), atol=1e-9)
            np.testing.assert_allclose(js["baseline"], probe.scores(x, baseline=True),
                                       atol=1e-9)


def test_export_web_writes_self_contained_site_data(encoder, records, tmp_path):
    from online_ml.web import export_web

    credits = {r.id: {"author": f"Author {r.id}", "author_url": "", "url": "https://x",
                      "license": "https://creativecommons.org/licenses/by/2.0/",
                      "title": ""} for r in records}
    index = export_web(encoder, records, tmp_path, credits=credits)
    assert [t["key"] for t in index["tasks"]] == ["people"]
    task = json.loads((tmp_path / "data" / "people.json").read_text())
    reviewed = [r for r in records if r.reviewed]
    assert len(task["photos"]) == len(reviewed)  # unreviewed candidates are left out
    assert task["positive_name"] == "People"
    first = task["photos"][0]
    assert (tmp_path / first["image"]).exists()
    assert first["credit"]["author"].startswith("Author")
    vector = np.frombuffer(__import__("base64").b64decode(first["embedding"]), dtype="<f4")
    assert vector.shape == (3,)
    with Image.open(tmp_path / first["image"]) as image:
        assert max(image.size) <= 400 and not image.getexif()


def test_export_web_requires_credits_for_open_images(encoder, records, tmp_path):
    from dataclasses import replace

    from online_ml.web import export_web

    rows = [replace(r, source="open-images-v7/validation") for r in records]
    with pytest.raises(ValueError, match="author credit"):
        export_web(encoder, rows, tmp_path)
