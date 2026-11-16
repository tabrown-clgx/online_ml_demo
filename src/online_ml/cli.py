import argparse
import json
from pathlib import Path

from .data import load_manifest, make_split, save_manifest
from .presets import DEFAULT_MODEL, PRESETS


def _model_options(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main", help="Pin a commit SHA for reproduction")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA, then Apple MPS, then CPU), cpu, mps, or cuda")
    parser.add_argument("--cache-dir", default=".cache/embeddings")


def _encoder(args):
    from .encoder import SiglipEncoder

    return SiglipEncoder(args.model, revision=args.revision,
                         device=args.device, cache_dir=args.cache_dir)


def _experiment_options(parser):
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preset", choices=PRESETS, default="people")
    parser.add_argument("--positive", help="Positive description override (keep task semantics)")
    parser.add_argument("--negative", help="Negative description override (keep task semantics)")
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--optimizer-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    _model_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local SigLIP2 feedback-learning demo")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download unreviewed Open Images V7 candidates")
    prepare.add_argument("--output", type=Path, default=Path("data/candidates.jsonl"))
    prepare.add_argument("--preset", choices=[*PRESETS, "all"], default="people")
    prepare.add_argument("--per-class", type=int, default=80,
                         help="Maximum candidates per binary class and preset")
    prepare.add_argument("--seed", type=int, default=42)

    importing = sub.add_parser("import-images", help="Import positive and negative image folders")
    importing.add_argument("--output", type=Path, required=True)
    importing.add_argument("--preset", choices=PRESETS, required=True)
    importing.add_argument("--positive-dir", type=Path, required=True)
    importing.add_argument("--negative-dir", type=Path, required=True)
    importing.add_argument("--reviewed", action="store_true",
                           help="Confirm that you personally verified the folder labels")
    importing.add_argument("--license", default="user-supplied; verify permission")

    review = sub.add_parser("review", help="Review candidate labels in a local Gradio app")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--port", type=int, default=7861)

    split = sub.add_parser("split", help="Create an immutable stratified evaluation split")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--eval-per-class", type=int, default=20)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--select", choices=["random", "uncertain", "shared"], default="random",
        help="uncertain: hold out the photos whose zero-shot score is closest to 0.5 "
             "(harder test set; biased against zero-shot by construction). shared: rank "
             "the same way but alternate photos between held-out and the labeling pool, "
             "so active learning still has borderline photos to ask about",
    )
    _model_options(split)
    precompute = sub.add_parser("precompute", help="Cache dataset image embeddings")
    precompute.add_argument("--manifest", type=Path, required=True)
    _model_options(precompute)

    evaluate = sub.add_parser("evaluate", help="Evaluate zero-shot on the fixed held-out set")
    _experiment_options(evaluate)
    simulate = sub.add_parser("simulate", help="Simulate feedback using reviewed pool labels")
    _experiment_options(simulate)
    simulate.add_argument("--synthetic", action="store_true",
                          help="Use numerical test embeddings, not SigLIP2 or real images")
    simulate.add_argument("--steps", type=int, default=20)
    simulate.add_argument("--strategy", choices=["random", "uncertainty"], default="uncertainty")

    app = sub.add_parser("app", help="Launch the local Gradio demo; manifest is optional")
    _experiment_options(app)
    app.add_argument("--port", type=int, default=7860)
    export = sub.add_parser("export", help="Export an experiment JSONL to CSV")
    export.add_argument("run", type=Path)

    web = sub.add_parser("export-web", help="Build the static GitHub Pages data (docs/)")
    web.add_argument("--manifest", type=Path, required=True)
    web.add_argument("--output", type=Path, default=Path("docs"))
    web.add_argument("--credits", type=Path,
                     help="Open Images image metadata CSV with authors and licenses "
                          "(validation-images-with-rotation.csv)")
    _model_options(web)
    return parser


def _run(args):
    if args.command == "prepare":
        from .prepare import download_candidates

        keys = list(PRESETS) if args.preset == "all" else [args.preset]
        rows = download_candidates(args.output, keys, args.per_class, args.seed)
        print(f"Saved {len(rows)} unreviewed candidates to {args.output}")
    elif args.command == "import-images":
        from .prepare import import_folders

        rows = import_folders(
            args.output, args.preset, args.positive_dir, args.negative_dir,
            reviewed=args.reviewed, license_text=args.license,
        )
        print(f"Saved {len(rows)} images to {args.output}; reviewed={args.reviewed}")
    elif args.command == "split":
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}.")
        rows = load_manifest(args.manifest)
        scores = None
        if args.select in ("uncertain", "shared"):
            from .probe import sigmoid

            encoder = _encoder(args)
            scores = {}
            for key in sorted({r.preset for r in rows}):
                group = [r for r in rows if r.preset == key and r.reviewed]
                prior = encoder.text_prior(PRESETS[key].positive, PRESETS[key].negative)
                values = sigmoid(encoder.encode_images([r.path for r in group]) @ prior)
                scores.update({(key, r.id): float(v) for r, v in zip(group, values, strict=True)})
        rows = make_split(rows, args.eval_per_class, args.seed, scores,
                          share=args.select == "shared")
        save_manifest(args.output, rows)
        print(f"Saved {sum(r.split == 'eval' for r in rows)} evaluation images "
              f"({args.select} selection) to {args.output}")
    elif args.command == "review":
        from .review import build_review_app

        build_review_app(args.manifest).queue().launch(
            server_name="127.0.0.1", server_port=args.port, share=False
        )
    elif args.command == "precompute":
        rows = load_manifest(args.manifest)
        encoder = _encoder(args)
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            encoder.encode_images([row.path for row in batch])
            print(f"Cached {min(start + 8, len(rows))}/{len(rows)} images", flush=True)
        print(json.dumps(encoder.identity, indent=2, default=str))
    elif args.command == "export":
        from .tracking import export_csv

        print(export_csv(args.run))
    elif args.command == "export-web":
        from .web import export_web, load_open_images_credits

        credits = load_open_images_credits(args.credits) if args.credits else {}
        index = export_web(_encoder(args), load_manifest(args.manifest), args.output,
                           credits=credits)
        for task in index["tasks"]:
            print(f"{task['key']}: {task['held_out']} held-out, {task['pool']} to label")
        print(f"Wrote {args.output}/data and {args.output}/img")
    elif args.command == "app":
        from .app import build_app

        rows = load_manifest(args.manifest) if args.manifest else []
        build_app(_encoder(args), rows, runs_dir=args.runs_dir, initial_preset=args.preset,
                  regularization=args.regularization, seed=args.seed,
                  optimizer_steps=args.optimizer_steps,
                  initial_positive=args.positive, initial_negative=args.negative).queue().launch(
            server_name="127.0.0.1", server_port=args.port, share=False,
        )
    elif args.command in {"evaluate", "simulate"}:
        from .experiment import create_experiment, synthetic_experiment
        from .tracking import export_csv

        options = dict(runs_dir=args.runs_dir, regularization=args.regularization,
                       optimizer_steps=args.optimizer_steps, seed=args.seed,
                       strategy=getattr(args, "strategy", "baseline"))
        if getattr(args, "steps", 0) < 0:
            raise ValueError("Simulation steps cannot be negative.")
        if getattr(args, "synthetic", False):
            if args.manifest or args.positive or args.negative:
                raise ValueError("Synthetic mode cannot be combined with a manifest or prompts.")
            experiment = synthetic_experiment(**options)
            print("SYNTHETIC NUMERICAL CHECK — not SigLIP2 performance", flush=True)
        else:
            if args.manifest is None:
                raise ValueError("Provide --manifest (or --synthetic for a numerical simulation).")
            rows = load_manifest(args.manifest)
            if not any(r.preset == args.preset and r.split == "eval" for r in rows):
                raise ValueError(
                    "No evaluation images for this preset. Review and split the dataset."
                )
            preset = PRESETS[args.preset]
            experiment = create_experiment(
                _encoder(args), rows, preset=args.preset,
                positive=args.positive or preset.positive,
                negative=args.negative or preset.negative,
                **options,
            )
        if not experiment.ids("eval"):
            raise ValueError("No evaluation images for this preset. Review and split the dataset.")
        if args.command == "simulate":
            experiment.simulate(args.steps, args.strategy)
        print(json.dumps(experiment.history[-1], indent=2))
        print(f"Run: {experiment.log.path}\nCSV: {export_csv(experiment.log.path)}")


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
