import subprocess
import sys

from detection_lab.models.prepare import Status, prepare_model, status_for
from detection_lab.models.registry import MODEL_REGISTRY, checkpoint_path, model_root


def test_model_registry_has_exactly_the_expected_candidates():
    assert [model.id for model in MODEL_REGISTRY] == [
        "rfdetr_large", "rfdetr_xlarge", "rfdetr_2xlarge", "yolo26x", "dfine_x",
    ]


def test_model_root_honours_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VEO_MATCH_MODELS_ROOT", str(tmp_path / "models"))
    assert model_root() == (tmp_path / "models").resolve()


def test_registry_import_does_not_import_ml_runtime_modules():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import detection_lab.models.registry; print('rfdetr' in sys.modules, 'ultralytics' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"


def test_missing_model_reports_missing_without_download(tmp_path):
    model = MODEL_REGISTRY[-1]
    status = status_for(model, tmp_path)
    assert status.state == "MISSING"
    assert checkpoint_path(model, tmp_path).name in status.detail


def test_rfdetr_plus_requires_explicit_acceptance(tmp_path):
    status = status_for(MODEL_REGISTRY[1], tmp_path)
    assert status.state == "LICENSE REQUIRED"
    prepared = prepare_model(MODEL_REGISTRY[1], tmp_path, accepted_rfdetr_plus_license=False)
    assert prepared.state == "LICENSE REQUIRED"
