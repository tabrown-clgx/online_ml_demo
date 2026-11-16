"""Single-page Gradio demo. Each browser session owns its feedback and in-memory uploads."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .data import ImageRecord, clean_image
from .experiment import Experiment, create_experiment
from .plots import learning_curve
from .presets import PRESETS
from .tracking import export_csv

GREEN, RED = "#16a34a", "#dc2626"
THUMB = 240


@dataclass
class Session:
    experiment: Experiment | None = None
    uploads: dict[str, Image.Image] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    selected: str = ""
    strategy: str = "uncertainty"
    just_flipped: set[str] = field(default_factory=set)


def _percent(value):
    return "—" if value is None else f"{value:.0%}"


def build_app(
    encoder, records: list[ImageRecord], *, runs_dir: str | Path = "runs",
    initial_preset: str = "people", regularization: float = 0.1, seed: int = 42,
    optimizer_steps: int = 300, initial_positive: str | None = None,
    initial_negative: str | None = None,
):
    import gradio as gr

    thumbs: dict[str, Image.Image] = {}

    def thumbnail(session: Session, image_id: str) -> Image.Image:
        if image_id in session.uploads:
            image = session.uploads[image_id].copy()
            image.thumbnail((THUMB, THUMB))
            return image
        path = session.experiment.records[image_id].path
        if path not in thumbs:
            image = clean_image(path)
            image.thumbnail((THUMB, THUMB))
            thumbs[path] = image
        return thumbs[path]

    def full_image(session: Session, image_id: str) -> Image.Image:
        if image_id in session.uploads:
            return session.uploads[image_id]
        return clean_image(session.experiment.records[image_id].path)

    def next_photo(session: Session) -> str:
        experiment = session.experiment
        current = session.selected
        valid = (current in experiment.records and experiment.records[current].split == "pool"
                 and current not in experiment.feedback and current not in session.skipped)
        if not valid:
            ranked = [i for i in experiment.rank_pool(session.strategy)
                      if i not in session.skipped]
            session.selected = ranked[0] if ranked else ""
        return session.selected

    def held_out(session: Session):
        experiment = session.experiment
        preset = PRESETS[experiment.config["preset"]]
        names = {True: preset.positive_name, False: preset.negative_name}
        ids = experiment.ids("eval")
        if not ids:
            return [], [], 0, 0
        x = experiment.features(ids)
        probe = experiment.probe.scores(x)
        base = experiment.probe.scores(x, baseline=True)
        predicted_pos, predicted_neg = [], []
        wrong_pos = wrong_neg = 0
        for image_id, p, b in sorted(zip(ids, probe, base, strict=True), key=lambda t: -t[1]):
            truth = experiment.records[image_id].label
            correct = (p >= 0.5) == bool(truth)
            boxed = ImageOps.expand(thumbnail(session, image_id), border=10,
                                    fill=GREEN if correct else RED)
            confidence = p if p >= 0.5 else 1 - p
            caption = f"{confidence:.0%} sure"
            if image_id in session.just_flipped:
                caption = "⚡ " + caption
            if (p >= 0.5) != (b >= 0.5):
                caption += f" · was {names[bool(b >= 0.5)]}"
            if p >= 0.5:
                predicted_pos.append((boxed, caption))
                wrong_pos += not correct
            else:
                predicted_neg.append((boxed, caption))
                wrong_neg += not correct
        return predicted_pos, predicted_neg, wrong_pos, wrong_neg

    def render(session, status=""):
        experiment = session.experiment
        if experiment is None:
            empty = gr.Gallery(value=[])
            return (session, None, "Loading…", "Starting…", empty, empty,
                    learning_curve([]), status, gr.Button(), gr.Button())
        preset = PRESETS[experiment.config["preset"]]
        yes, no = preset.positive_name, preset.negative_name
        image_id = next_photo(session)
        if image_id:
            x = experiment.features([image_id])
            score = float(experiment.probe.scores(x)[0])
            current = full_image(session, image_id)
            guess, confidence = (yes, score) if score >= 0.5 else (no, 1 - score)
            prompt = (f"Model's guess: **{guess}** ({confidence:.0%} sure).  \n"
                      f"Is this **{yes}** or **{no}**?")
        else:
            current, prompt = None, "No more photos to label. Upload more in Settings."
        positives, negatives, wrong_pos, wrong_neg = held_out(session)
        metric = experiment.history[-1]
        probe, base = metric["probe"], metric["baseline"]
        n = probe["n"]
        correct = probe["tp"] + probe["tn"]
        base_correct = base["tp"] + base["tn"]
        summary = (
            f"## {yes} vs {no}: **{correct}/{n}** held-out photos correct "
            f"(zero-shot: {base_correct}/{n}) · "
            f"balanced accuracy **{_percent(probe['balanced_accuracy'])}** "
            f"(zero-shot {_percent(base['balanced_accuracy'])})\n"
            f"**{yes}** = *{experiment.config['positive']}* · "
            f"**{no}** = *{experiment.config['negative']}* · "
            f"{metric['n_labels']} labels given "
            f"({metric['n_positive']} {yes}, {metric['n_negative']} {no})"
        )
        pos_gallery = gr.Gallery(
            value=positives, label=f"Model says {yes} · {len(positives)} photos, "
                                   f"{wrong_pos} wrong")
        neg_gallery = gr.Gallery(
            value=negatives, label=f"Model says {no} · {len(negatives)} photos, "
                                   f"{wrong_neg} wrong")
        return (session, current, prompt, summary, pos_gallery, neg_gallery,
                learning_curve(experiment.history), status,
                gr.Button(value=f"✓ {yes}"), gr.Button(value=f"✓ {no}"))

    def start(session, preset, positive, negative, strength, strategy):
        try:
            experiment = create_experiment(
                encoder, records, preset=preset, positive=positive, negative=negative,
                runs_dir=runs_dir, regularization=float(strength),
                optimizer_steps=optimizer_steps, seed=seed,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc
        return render(Session(experiment=experiment, strategy=strategy))

    def require(session):
        if session.experiment is None:
            raise gr.Error("The experiment is still starting.")
        return session.experiment

    def feedback(session, value):
        experiment = require(session)
        if not session.selected:
            raise gr.Error("No photo to label.")
        preset = PRESETS[experiment.config["preset"]]
        ids = experiment.ids("eval")
        x = experiment.features(ids)
        before_pred = experiment.probe.scores(x) >= 0.5 if ids else []
        started = time.perf_counter()
        experiment.label(session.selected, value)
        elapsed = (time.perf_counter() - started) * 1000
        session.selected = ""
        if not ids:
            return render(session, f"Updated in {elapsed:.0f} ms.")
        after_pred = experiment.probe.scores(x) >= 0.5
        truth = [experiment.records[i].label == 1 for i in ids]
        flipped = [i for i, b, a in zip(ids, before_pred, after_pred, strict=True) if b != a]
        fixed = sum(a == t for i, b, a, t in zip(ids, before_pred, after_pred, truth, strict=True)
                    if b != a)
        session.just_flipped = set(flipped)
        name = preset.positive_name if value == 1 else preset.negative_name
        if not flipped:
            effect = f"Labeled **{name}**: no held-out photos changed."
        else:
            effect = (f"Labeled **{name}**: flipped **{len(flipped)}** held-out "
                      f"photo{'s' if len(flipped) != 1 else ''} (⚡) — "
                      f"{fixed} now right, {len(flipped) - fixed} now wrong.")
        return render(session, f"{effect}  \nUpdated in {elapsed:.0f} ms.")

    def skip(session):
        require(session)
        if session.selected:
            session.skipped.add(session.selected)
            session.selected = ""
        return render(session)

    def reset(session):
        require(session).reset()
        session.skipped.clear()
        session.just_flipped.clear()
        session.selected = ""
        return render(session, "Feedback cleared; back to zero-shot.")

    def set_strategy(session, strategy):
        session.strategy = strategy
        if session.experiment is None:
            return render(session)
        session.selected = ""
        return render(session)

    def upload(session, files):
        experiment = require(session)
        if not files:
            raise gr.Error("Choose one or more images first.")
        added, errors = 0, []
        for filename in files:
            try:
                image = clean_image(filename)
                embedding = encoder.encode_images([image])[0]
                image_id = experiment.add_upload(image, embedding)
                session.uploads[image_id] = image
                session.skipped.discard(image_id)
                session.selected = image_id
                added += 1
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(str(exc))
            finally:
                # Keep only metadata-free pixels in memory, not Gradio's temp copy.
                Path(filename).unlink(missing_ok=True)
        return (*render(session, f"Loaded {added} photos. " + " ".join(errors)), None)

    def simulate(session, count, strategy):
        experiment = require(session)
        before = len(experiment.feedback)
        experiment.simulate(int(count), strategy)
        session.selected = ""
        session.just_flipped.clear()
        return render(session, f"Simulated {len(experiment.feedback) - before} labels from "
                               "reviewed pool labels.")

    def export(session):
        return str(export_csv(require(session).log.path))

    def preset_changed(key):
        p = PRESETS[key]
        return (gr.Textbox(value=p.positive, label=f"Text description of {p.positive_name}"),
                gr.Textbox(value=p.negative, label=f"Text description of {p.negative_name}"))

    css = """
    .gradio-container {max-width: 1600px !important}
    footer {display: none !important}
    #label-photo img {object-fit: contain}
    """
    preset = PRESETS[initial_preset]
    available = [k for k in PRESETS if any(r.preset == k for r in records)] or list(PRESETS)
    if initial_preset not in available:
        available.insert(0, initial_preset)
    with gr.Blocks(title="Feedback Lab", theme=gr.themes.Soft(primary_hue="teal"),
                   css=css, delete_cache=(3600, 3600)) as app:
        state = gr.State(value=Session(), time_to_live=3600)
        task = gr.Radio(choices=[(PRESETS[k].title, k) for k in available],
                        value=initial_preset, label="Task", elem_id="task-picker")
        summary = gr.Markdown("Starting… the first load takes about 10 seconds.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Label this photo")
                current = gr.Image(elem_id="label-photo", interactive=False, height=320,
                                   show_label=False, show_download_button=False)
                prompt = gr.Markdown()
                with gr.Row():
                    pos_button = gr.Button(f"✓ {preset.positive_name}", variant="primary")
                    neg_button = gr.Button(f"✓ {preset.negative_name}", variant="primary")
                skip_button = gr.Button("Skip", size="sm")
                status = gr.Markdown()
                curve = gr.Plot(value=learning_curve([]), show_label=False)
            with gr.Column(scale=3):
                gr.Markdown("### Held-out photos  ·  🟩 model is right  🟥 model is wrong")
                pos_gallery = gr.Gallery(label=f"Model says {preset.positive_name}", columns=6,
                                         height="auto", object_fit="contain",
                                         allow_preview=False)
                neg_gallery = gr.Gallery(label=f"Model says {preset.negative_name}", columns=6,
                                         height="auto", object_fit="contain",
                                         allow_preview=False)
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                strategy = gr.Radio([("Least confident first (active learning)", "uncertainty"),
                                     ("Random", "random")], value="uncertainty",
                                    label="Which photos to label")
                strength = gr.Number(value=regularization, label="Text-prior regularization",
                                     minimum=0.0001, maximum=10)
            with gr.Row():
                positive = gr.Textbox(value=initial_positive or preset.positive,
                                      label=f"Text description of {preset.positive_name}")
                negative = gr.Textbox(value=initial_negative or preset.negative,
                                      label=f"Text description of {preset.negative_name}")
            with gr.Row():
                start_button = gr.Button("Restart with these settings", variant="primary")
                reset_button = gr.Button("Reset feedback")
            with gr.Row():
                files = gr.File(label="Add your own photos to label", file_count="multiple",
                                file_types=["image"])
                with gr.Column():
                    upload_button = gr.Button("Load photos")
                    count = gr.Slider(1, 50, value=10, step=1, label="Simulate labels")
                    sim_button = gr.Button("Simulate feedback")
                    export_button = gr.Button("Export metrics CSV")
                    download = gr.File(label="Metrics CSV", interactive=False)

        outputs = [state, current, prompt, summary, pos_gallery, neg_gallery, curve, status,
                   pos_button, neg_button]
        settings = [state, task, positive, negative, strength, strategy]
        app.load(start, settings, outputs)
        start_button.click(start, settings, outputs, api_name="start_experiment")
        task.change(preset_changed, task, [positive, negative]).then(start, settings, outputs)
        pos_button.click(lambda s: feedback(s, 1), state, outputs, api_name="label_positive")
        neg_button.click(lambda s: feedback(s, 0), state, outputs, api_name="label_negative")
        skip_button.click(skip, state, outputs, api_name="skip")
        reset_button.click(reset, state, outputs, api_name="reset")
        strategy.change(set_strategy, [state, strategy], outputs)
        upload_button.click(upload, [state, files], [*outputs, files], api_name="upload")
        sim_button.click(simulate, [state, count, strategy], outputs, api_name="simulate")
        export_button.click(export, state, download, api_name="export")
    return app
