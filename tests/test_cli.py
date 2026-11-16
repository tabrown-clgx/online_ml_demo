import pytest

from online_ml.cli import main
from online_ml.data import save_manifest


def test_synthetic_cli_and_export(tmp_path, capsys):
    main(["simulate", "--synthetic", "--steps", "3", "--runs-dir", str(tmp_path)])
    assert "SYNTHETIC NUMERICAL CHECK" in capsys.readouterr().out
    log = next(tmp_path.glob("*.jsonl"))
    assert log.with_suffix(".csv").exists()
    main(["export", str(log)])
    assert ".csv" in capsys.readouterr().out


def test_missing_manifest_is_actionable(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["evaluate"])
    assert exc.value.code == 2
    assert "Provide --manifest" in capsys.readouterr().err


def test_split_will_not_overwrite_source(records, tmp_path):
    source = tmp_path / "manifest.jsonl"
    save_manifest(source, records)
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main(["split", "--manifest", str(source), "--output", str(source)])
    assert source.read_bytes() == before
