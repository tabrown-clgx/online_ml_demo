# Online Learning · Feedback Lab

A local Gradio demo of rapid learning from photo feedback. A frozen
[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) encoder provides image
and text embeddings. Compare a text-only zero-shot baseline with a tiny linear
probe that rebuilds after every label—without fine-tuning the encoder.

## Quick start

Python 3.10+ is required; Python 3.11 is recommended. Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[model,app,dev]"
online-ml app
```

Open http://127.0.0.1:7860. The page starts an experiment automatically; label photos
positive or negative and watch the held-out predictions change. The first start
downloads the model from Hugging Face; later runs use the local model cache. The
device is chosen automatically (CUDA, then Apple-silicon GPU via MPS, then CPU);
override with `--device cpu|mps|cuda`. Embeddings are cached per device, so switching
device re-encodes photos once.

Without a dataset manifest, the app supports uploads and feedback, but **does not
claim an evaluation score**. Follow the dataset workflow below for held-out metrics
and automatic feedback simulation. No photos, reviewed dataset, or pretrained
weights are bundled in the repository.

### Offline numerical check

You can exercise training, tracking, and CSV export without downloading a model:

```bash
python -m pip install -e ".[dev]"
online-ml simulate --synthetic --steps 20 --seed 42
pytest -q
```

This uses generated numerical embeddings and an intentionally imperfect prior.
Its metrics are **not SigLIP2 or real-image performance**. Tests use synthetic
fixtures; when model dependencies are installed, they also exercise tiny randomly
initialized Transformers image/text towers without network access.

## How learning works

1. Encode each image once and L2-normalize its embedding.
2. Encode positive and negative descriptions with three templates: the raw text,
   `a photo of {description}`, and `an image showing {description}`. Average each
   group's normalized text embeddings.
3. Use a two-description softmax for the zero-shot score. Equivalently,
   `sigmoid(x @ w₀)`, where `w₀ = exp(logit_scale) * (t_positive - t_negative)`.
   SigLIP's common logit bias cancels in this relative comparison.
4. Initialize the probe at `w₀`, with intercept zero. At **zero feedback labels**,
   probe and zero-shot predictions match exactly.
5. After each added, corrected, or removed label, deterministically refit on all
   current feedback with full-batch gradient descent:

   ```text
   weighted_mean(binary_cross_entropy(x @ w + b, y))
       + λ/2 * (||w - w₀||² + b²)
   ```

   Labels are **class-balanced**: each class present in the feedback carries equal
   total weight, so a run of mostly-negative labels cannot drag every prediction
   toward "negative".

The default `λ=0.1` keeps the probe close to the text prior while allowing
adaptation. It was chosen to stop the first few labels from swinging predictions
wildly; on the bundled hard split, `λ=0.01` let one task drop from 15/20 to 10/20
early on. The trade-off: a strong prior learns less from many labels (the people
task gained less at 0.1 than at 0.01). Adjust it under Settings. One-class
feedback is allowed because training is anchored to the prior; you do not have to
supply both classes first. Only explicit feedback enters training—not source
annotations or evaluation labels.

This is **online interaction through repeated small-batch refits**, not encoder
training or a streaming-SGD benchmark. Rebuilding avoids stale updates after label
corrections and makes results independent of prior fits. The default optimizer has
300 steps and a fixed safe step-size bound for normalized inputs.

Both models use a 0.5 decision threshold. Their displayed scores are **not
calibrated probabilities**: the baseline is relative to the chosen descriptions,
and the probe is a regularized logistic model fitted on very little data.

## Build a small Open Images V7 dataset

Start with one preset, preferably `people` or `animals`, before collecting all four.
The optional FiftyOne downloader may need platform-specific MongoDB support; see
[FiftyOne installation](https://docs.voxel51.com/getting_started/install.html).
It also downloads source annotation metadata, which can be much larger than the
bounded set of images.

Downloaded media uses FiftyOne's configured dataset zoo cache (typically
`~/fiftyone`), not the manifest output directory. Set
`FIFTYONE_DATASET_ZOO_DIR=/absolute/path/to/cache` before running `prepare` to
choose a different cache. The manifest references those cached image files;
keep them in place after preparation.

### 1. Download candidates

```bash
python -m pip install -e ".[data]"
online-ml prepare --preset people --per-class 80 --output data/candidates.jsonl
```

`--per-class` bounds each **binary class**, not each Open Images source class.
`--preset all` requests all presets. The script uses `fiftyone.zoo`, validation-split
images, explicit positive source annotations, and content deduplication. It never
interprets a missing annotation as a negative. Source labels are only retrieval
hints: all downloaded rows start with `reviewed: false` in the feedback pool.
Some requested candidates may be unavailable, overlap, or fail the source-label
filter; fewer images than requested is expected.

### 2. Verify binary labels

```bash
online-ml review --manifest data/candidates.jsonl
```

Open http://127.0.0.1:7861. Review the visible content against the descriptions.
**Positive** and **Negative** save your decision immediately. **Skip / ambiguous**
leaves the row unreviewed and unavailable for evaluation or automatic simulation.
Unreviewed pool images can still receive explicit feedback in the main app.
Close the review server before splitting or launching experiments. Use only one
review server on a manifest at a time.

Aim for at least 40–60 reviewed images per class, with variety beyond the retrieval
classes. Review overlapping concepts carefully: a city street can contain people,
trees, animals, or a place of worship. Broad presets are not exhaustive ontologies.
Choose labels based on the stated binary task; skip genuinely ambiguous examples.

### 3. Freeze a stratified evaluation set

```bash
online-ml split --manifest data/candidates.jsonl --output data/demo.jsonl \
  --eval-per-class 20 --seed 42
```

This reserves 40 reviewed evaluation images per preset and retains feedback
examples in both classes. The script refuses to overwrite an existing output or
resplit a manifest that already has evaluation images. Keep `data/demo.jsonl`
unchanged throughout a comparison.

Add `--select uncertain` to hold out, per preset and class, the reviewed images
whose zero-shot score is closest to 0.5 instead of a random sample. This makes a
harder test set where feedback has room to help, but it is **biased against
zero-shot by construction**: its baseline score understates typical performance.
Use it to demonstrate learning, not to report how good zero-shot is.

`uncertain` has a side effect: it moves every borderline photo into the held-out
set, leaving the labeling pool with only photos the model is already sure about,
so "least confident first" has nothing useful to ask. Prefer `--select shared`:
it ranks photos the same way but alternates them between held-out and the
labeling pool (topping up small classes from the easiest photos), so both sets
get borderline cases and active learning still works. On the bundled data, with
20 held-out photos per task:

| Task | Zero-shot | Least confident, 3 labels | Random, 3 labels (10 seeds) | Least confident, 20 | Random, 20 |
|---|---|---|---|---|---|
| People | 16 | 19 | 16.7 | 18 | 17.2 |
| Religion | 16 | 16 | 15.9 | 17 | 17.0 |
| Pets | 18 | 20 | 18.6 | 20 | 19.0 |
| Scenery | 15 | 17 | 16.6 | 17 | 16.9 |

Active learning helps most in the first few labels; by 20 labels random sampling
has mostly caught up.

Only reviewed images can enter evaluation. Byte checksums detect modified source
files. Decoded-pixel identities reject duplicate images within each preset, even
when file metadata or formats differ. The app also rejects an uploaded copy of a
held-out image. These checks do **not** catch crops, resizing, or lossy recompression;
review near-duplicates and related shots manually before splitting.

### 4. Cache embeddings and validate before using the UI

```bash
online-ml precompute --manifest data/demo.jsonl
online-ml evaluate --manifest data/demo.jsonl --preset people
online-ml simulate --manifest data/demo.jsonl --preset people \
  --strategy random --steps 20 --seed 42
online-ml simulate --manifest data/demo.jsonl --preset people \
  --strategy uncertainty --steps 20 --seed 42
online-ml app --manifest data/demo.jsonl --preset people
```

Simulation reveals labels only from the **reviewed feedback pool**, never from
evaluation. Uncertainty sampling selects the score closest to 0.5; random sampling
is seeded. Each simulation writes a new run and a CSV. Compare both curves rather
than assuming uncertainty selection always helps. For a meaningful strategy
comparison, also use the same pool, seed, regularization, and label budget; the
run metadata records them. Try several seeds, not just a favorable example.

Image embeddings are individual `.npy` files under `.cache/embeddings/`. Cache
identities include image pixels, checkpoint revision, preprocessing, library
versions, and device. An upload needs encoding only if it is not already cached.
Use `--revision <Hugging-Face-commit-SHA>` consistently to pin a model; resolved
revisions are recorded automatically. Precomputing images avoids repeated image
tower inference, but the encoder still loads for text prompts and new uploads.

### Alternative: bring your own reviewed images

Put positive and negative photos in separate folders, then import them:

```bash
online-ml import-images --preset people --positive-dir /path/to/positive \
  --negative-dir /path/to/negative --output data/local.jsonl --reviewed
online-ml split --manifest data/local.jsonl --output data/local-demo.jsonl \
  --eval-per-class 20
```

`--reviewed` is your explicit confirmation that you checked every folder label.
Omit it to use the review app first. Imports reference original files rather than
copying them. Use `--license` to record your source's usage terms. Duplicate images
are rejected, including duplicates across positive and negative folders.

### Manifest format

Manifests are UTF-8 JSONL: one object per image and preset. Paths are relative to
the manifest file (absolute paths are accepted when reading). The example below
illustrates the schema; it is **not** a bundled downloadable/reviewed image list.

```json
{"id":"example-id","path":"images/example.jpg","preset":"people","label":1,"split":"pool","reviewed":false,"source":"open-images-v7/validation","source_url":"https://example.org/original-photo","license":"verify original attribution","source_class":"Person","sha256":""}
```

`label` is integer 0 or 1. For unreviewed rows it is only a suggested label.
`split` is `pool` or `eval`. Empty checksums are filled when loading; generated
manifests record actual file hashes. Source URLs and license notes remain in the
manifest. Preserve the original image attribution and check the applicable license
before sharing images; a dataset-level CC-BY notice is not an attribution record.

## Presets

| Key | Preset | Positive description | Negative description |
| --- | --- | --- | --- |
| `religion` | Religious Symbols & Crosses | religious symbol, cross, or place of worship | ordinary everyday object or plain scenery |
| `people` | People & Crowds | person, portrait, or crowd of people | empty landscape or object without people |
| `animals` | Wild & Domestic Animals | animal, pet, bird, or wildlife | plants, buildings, or inanimate objects |
| `nature` | Scenic Nature | natural landscape, forest, mountain, or water | indoor room, city street, or human portrait |
| `pets` | Pets vs Wild Animals | pet dog or cat | wild animal or bird in nature |
| `scenery` | Natural vs Landscaped | wild natural landscape, forest, coast, or river | landscaped resort, swimming pool, or garden |

`animals` and `nature` turned out too easy for a feedback demo: zero-shot was
already perfect or near-perfect on their held-out photos. `pets` and `scenery` are
harder replacements built from the same downloads plus extra candidates. Farm
animals (horses, donkeys, cattle) are excluded from `pets`, which has too few
for a third class. In `scenery`, palm trees and greenery appear on both sides, so
the label is about whether the scene is wild or designed. The app's task menu
lists only presets that appear in the loaded manifest.

The Open Images subset exposed by FiftyOne covers 600 boxable classes, not every
concept in these descriptions. **There is no direct cross/place-of-worship class**
in that subset. `religion` retrieves Building/Sculpture candidates only; many will
be negatives or irrelevant. Expect manual collection/import for adequate coverage.
Likewise, Tree/Palm tree are only hints for `nature`, not reliable scenic labels.
Do not infer people's religious identity from appearance.

Prompts are editable. Apply changes by starting a new experiment, and preserve the
meaning of the reviewed dataset labels. For an unrelated custom binary task, launch
without a manifest and use uploads, or add an explicit new preset and review its
dataset. Arbitrary new semantics cannot reuse the old evaluation labels.

## UI walkthrough

Everything is on one page, and the experiment starts automatically when it opens:

- **Task buttons (top):** switch between tasks in one click; the experiment
  restarts for the new task.
- **Summary line:** held-out photos correct now vs. zero-shot, and how many labels
  you have given.
- **Left, "Label this photo":** the next photo to label (most uncertain first by
  default), with the model's guess in the task's own terms (for example "Pet, 73%
  sure"). Click the class button (**✓ Pet** / **✓ Wild**, **✓ People** /
  **✓ No people**, …) or **Skip**; the probe rebuilds and the page updates
  immediately. After each label, the panel reports its effect, for example
  "flipped **3** held-out photos (⚡): 2 now right, 1 now wrong". A small learning
  curve sits underneath.
- **Right, held-out photos:** every evaluation photo, split by what the model
  currently says (for example **Model says Pet** / **Model says Wild**). A **green
  box** means the model is right; **red** means wrong. Captions show how sure the
  model is, plus `was …` when a photo has flipped since zero-shot, and ⚡ on the
  photos your last label flipped.
- **Settings (collapsed):** labeling order (least confident first or random),
  descriptions, prior strength, restart, reset, photo upload, simulated labels and
  CSV export. Uploads are only ever labeled, never held out.

Metrics use the full evaluation set. A 20-image set is useful for illustration,
but one error changes accuracy by 5 percentage points. The first one or two labels
can swing many predictions at once, especially on an `uncertain` split where most
held-out scores sit near 0.5. Balanced accuracy requires both classes and exposes
constant-class predictors. Improvement is **not guaranteed**. Repeatedly inspecting
results makes the held-out set a validation set; reserve a separate untouched test
set for stronger claims.

## Lightweight experiment tracking

Each run is an append-only `runs/<timestamp>-<id>.jsonl` file recording:

- Resolved model revision, encoder/preprocessing configuration, prompt templates
- Descriptions, optimizer settings, regularization, seed, and initial strategy
- Fixed evaluation identities/labels and initial pool identities/reviewed labels
- Feedback additions, corrections/removals, reset and simulation events
- Metrics after every update, class counts, and probe rebuild timing

Logs omit raw image pixels and filesystem paths. Upload events record content
identity, not original photos. Automatic comparisons require identical task,
prompts, encoder identity, templates, and evaluation image identities/labels.
Pools/settings may differ intentionally; inspect metadata when interpreting curves.

```bash
online-ml export runs/<run-id>.jsonl
```

CSV files sit next to their logs. Runs can be compared/exported, but **resuming a
saved UI run is not implemented**. Reproduce dataset simulations with the same
manifest, pinned model, arguments, and dependency versions. Uploaded-image runs
cannot be reconstructed from logs alone because the photos are not persisted.

## Static web version (GitHub Pages)

`docs/` is a self-contained copy of the demo that runs entirely in the browser,
with no Python and no model download. The encoder is frozen, so every photo's
SigLIP2 embedding and each task's text prior are computed once and shipped as
data; the browser only runs the small probe (`docs/probe.js`, a line-for-line port
of `probe.py`, checked against it in `tests/test_web.py`). It has the same layout:
task buttons, the labeling panel with flip reporting, the two boxed held-out
galleries, a learning curve, and settings for labeling order, prior strength,
reset and auto-labeling. Uploading your own photos is not available there, since
that would need the image encoder in the browser.

Rebuild the site data after changing the photos or splits:

```bash
curl -LO https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
online-ml export-web --manifest data/four-shared.jsonl --output docs \
  --credits validation-images-with-rotation.csv
python -m http.server --directory docs   # preview at http://localhost:8000
```

This rewrites `docs/data/` (task JSON with embeddings and labels) and `docs/img/`
(photos resized to 400 px, EXIF stripped); the page files themselves are not
generated. The credits CSV supplies each Open Images photo's author, original page
and license, shown under **Photo credits** and as hover text; export refuses to
publish an Open Images photo without that credit. Only reviewed photos are
exported. To publish, push the repository to GitHub and set **Settings → Pages →
Build and deployment** to *Deploy from a branch*, branch `main`, folder `/docs`.

## Privacy and resource notes

- Both apps bind to `127.0.0.1`, never request a public Gradio share link, and are
  intended for a single local user—not production or multi-tenant deployment.
- Model/data preparation requires network access. Inference is local after files
  are downloaded. Hugging Face and Gradio may perform library telemetry/version
  checks; set `HF_HUB_DISABLE_TELEMETRY=1` and `GRADIO_ANALYTICS_ENABLED=False` to
  disable their optional telemetry. Use cached files/network controls for offline use.
- Upload decoding honors EXIF orientation, then strips metadata. The original
  Gradio temporary upload is deleted after processing. Metadata-free display copies
  can remain in Gradio's temporary cache until its hourly cleanup. Browser session
  state expires one hour after closure; new experiments discard their upload state.
- Derived upload embeddings persist in `.cache/embeddings/`; feedback/content hashes
  persist in `runs/`. These are still derived user data. Remove the relevant local
  files yourself when they are no longer wanted. Dataset originals are not modified.
- Dataset/model download size, first-load latency, and inference time depend on
  hardware. This repository does not claim a CPU latency target or real-data gain.

## Project layout and development

```text
src/online_ml/
  encoder.py      Frozen encoder, prompt ensemble, versioned image embedding cache
  probe.py        Deterministic text-prior logistic probe and binary metrics
  data.py         Manifest I/O, image identities, fixed reviewed splits
  prepare.py      Optional Open Images download and local folder import
  experiment.py   Feedback, active selection, simulation, evaluation isolation
  tracking.py     JSONL logs, compatible-run matching, CSV export
  app.py          Session-isolated Gradio feedback demo
  review.py       Human label-review UI
  web.py          Static-site export (embeddings, priors, photos, credits)
  cli.py          Command-line entry points
docs/             GitHub Pages site: index.html, app.js, probe.js, style.css,
                  plus generated data/ and img/
tests/            Offline unit, model API, UI callback, and browser-probe tests
```

```bash
python -m pip install -e ".[app,model,dev]"
ruff check src tests
pytest -q
online-ml --help
```

The default checkpoint is configurable with `--model`. Model dependencies and
FiftyOne remain optional so numerical tests, manifest utilities, and tracking do
not require downloading a vision model or running a database.
