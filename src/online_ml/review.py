"""Explicit human review before labels may enter evaluation or simulation."""

from dataclasses import replace
from pathlib import Path

from .data import clean_image, load_manifest, save_manifest
from .presets import PRESETS


def build_review_app(manifest: str | Path):
    import gradio as gr

    path = Path(manifest).resolve()
    records = load_manifest(path)
    indices = [i for i, row in enumerate(records) if row.split == "pool" and not row.reviewed]

    def show(position):
        """Image, description, and the two class names for the button captions."""
        if position >= len(indices):
            return (None, "Review complete. Stop this server, then create the fixed split.",
                    "—", "—")
        row = records[indices[position]]
        preset = PRESETS[row.preset]
        yes, no = preset.positive_name, preset.negative_name
        description = (
            f"### {position + 1} / {len(indices)} · {preset.title}\n"
            f"**{yes}:** {preset.positive}  \n**{no}:** {preset.negative}\n\n"
            f"Candidate `{row.id}` · source class: {row.source_class or 'local'}  \n"
            f"Suggested label (not verified): **{yes if row.label else no}**\n\n"
            f"{preset.note}\n\nSkip ambiguous or irrelevant images; they stay unreviewed."
        )
        return clean_image(row.path), description, yes, no

    def view(position):
        image, description, yes, no = show(position)
        return image, description, gr.Button(value=yes), gr.Button(value=no)

    def decide(position, label):
        if position < len(indices) and label is not None:
            index = indices[position]
            records[index] = replace(records[index], label=label, reviewed=True)
            save_manifest(path, records)
        position += 1
        return position, *view(position)

    with gr.Blocks(title="Review dataset labels") as app:
        gr.Markdown(
            "# Review candidate labels\n"
            "Your clicks save immediately to the manifest. Run one review server at a time, "
            "before launching experiments. Existing evaluation rows cannot be edited here."
        )
        position = gr.State(0)
        initial_image, initial_description, yes, no = show(0)
        with gr.Row():
            image = gr.Image(value=initial_image, interactive=False, label="Candidate image")
            description = gr.Markdown(initial_description)
        with gr.Row():
            first = gr.Button(yes, variant="primary")
            second = gr.Button(no, variant="primary")
            skip = gr.Button("Skip / ambiguous")
        outputs = [position, image, description, first, second]
        first.click(lambda p: decide(p, 1), position, outputs)
        second.click(lambda p: decide(p, 0), position, outputs)
        skip.click(lambda p: decide(p, None), position, outputs)
    return app
