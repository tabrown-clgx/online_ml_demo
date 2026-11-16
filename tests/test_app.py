import asyncio
from copy import deepcopy

import pytest
from PIL import Image

pytest.importorskip("gradio")
pytest.importorskip("matplotlib")

from gradio.state_holder import SessionState  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from online_ml.app import Session, build_app  # noqa: E402
from online_ml.data import save_manifest  # noqa: E402
from online_ml.review import build_review_app  # noqa: E402


def test_app_callbacks_label_upload_simulate_reset_and_export(encoder, records, tmp_path):
    app = build_app(encoder, records, runs_dir=tmp_path / "runs")
    callbacks = {block.api_name: block.fn for block in app.fns.values()}
    result = callbacks["start_experiment"](
        Session(), "people", "people", "empty", 0.01, "uncertainty"
    )
    assert len(result) == 10
    assert result[8].value == "✓ People" and result[9].value == "✓ No people"
    assert result[4].label.startswith("Model says People")
    assert result[5].label.startswith("Model says No people")
    assert "positive" not in result[2].lower() and "negative" not in result[2].lower()
    session = result[0]
    assert session.selected
    deepcopy(session)  # Gradio State requires session values to be deepcopy-compatible.
    positives, negatives = result[4].value, result[5].value
    assert len(positives) + len(negatives) == 4  # every held-out image, split by prediction
    original = session.selected
    result = callbacks["label_positive"](session)
    assert session.experiment.feedback[original] == 1
    assert result[7].startswith("Labeled **People**")
    marked = sum(item["caption"].startswith("⚡") if isinstance(item, dict)
                 else item.caption.startswith("⚡")
                 for gallery in (result[4], result[5]) for item in gallery.value)
    assert marked == len(session.just_flipped)
    if session.just_flipped:
        assert f"flipped **{len(session.just_flipped)}**" in result[7]
    assert session.selected and session.selected != original
    skipped = session.selected
    callbacks["skip"](session)
    assert skipped not in session.experiment.feedback
    assert session.selected != skipped
    callbacks["simulate"](session, 2, "random")
    assert len(session.experiment.feedback) == 3
    upload = tmp_path / "private-upload.png"
    Image.new("RGB", (10, 10), "purple").save(upload)
    result = callbacks["upload"](session, [str(upload)])
    assert len(result) == 11
    assert result[-1] is None
    assert not upload.exists()
    assert session.selected.startswith("upload-")
    callbacks["label_negative"](session)
    assert len(session.experiment.feedback) == 4
    assert callbacks["export"](session).endswith(".csv")
    callbacks["reset"](session)
    assert not session.experiment.feedback
    app.close()


def test_held_out_boxes_are_green_when_correct_and_red_when_wrong(encoder, records, tmp_path):
    from online_ml.app import GREEN, RED

    app = build_app(encoder, records, runs_dir=tmp_path / "runs")
    callbacks = {block.api_name: block.fn for block in app.fns.values()}
    result = callbacks["start_experiment"](
        Session(), "people", "people", "empty", 0.01, "uncertainty"
    )
    experiment = result[0].experiment
    metric = experiment.history[-1]["probe"]
    reds = []
    for gallery in (result[4], result[5]):
        colours = []
        for item in gallery.value:
            path = item["image"]["path"] if isinstance(item, dict) else item.image.path
            pixel = Image.open(path).convert("RGB").getpixel((0, 0))
            # Gradio stores gallery images lossily; classify by nearest box colour.
            colours.append(min((GREEN, RED), key=lambda c: sum(
                (int(c[k:k + 2], 16) - v) ** 2 for k, v in zip((1, 3, 5), pixel, strict=True)
            )))
        assert set(colours) <= {GREEN, RED}
        wrong = int(gallery.label.split(", ")[1].split(" ")[0])
        assert colours.count(RED) == wrong
        reds.append(wrong)
    assert reds == [metric["fp"], metric["fn"]]
    app.close()


def test_review_changes_only_pool_labels(encoder, records, tmp_path):
    path = tmp_path / "manifest.jsonl"
    save_manifest(path, records)
    app = build_review_app(path)
    assert app.config["components"]
    app.close()


def test_app_http_routes_build_api_schema_without_loading_model(tmp_path):
    app = build_app(None, [], runs_dir=tmp_path)
    with TestClient(app.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        config = client.get("/config")
        assert config.status_code == 200
        info = client.get("/gradio_api/info")
        assert info.status_code == 200
        assert "/start_experiment" in info.json()["named_endpoints"]
    app.close()


def test_gradio_process_api_initializes_independent_session_state(encoder, tmp_path):
    app = build_app(encoder, [], runs_dir=tmp_path)
    functions = {block.api_name: key for key, block in app.fns.items()}
    state_id = next(key for key, block in app.blocks.items() if block.get_block_name() == "state")
    first, second = SessionState(app), SessionState(app)

    async def exercise():
        result = await app.process_api(
            functions["start_experiment"],
            [None, "people", "person", "empty", 0.01, "uncertainty"],
            state=first,
        )
        assert result["data"][0] is None  # Stateful values never go to the browser as JSON.
        assert first[state_id].experiment is not None
        assert second[state_id].experiment is None
        assert first[state_id] is not second[state_id]

    asyncio.run(exercise())
    app.close()
