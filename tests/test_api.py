import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["MTVQA_STORAGE"] = str(tmp_path_factory.mktemp("storage"))
    import importlib
    import src.api.app as appmod
    importlib.reload(appmod)
    return TestClient(appmod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "Not for clinical use" in r.json()["disclaimer"]


def test_upload_rejects_non_dicom(client):
    r = client.post("/studies/upload", files=[("files", ("x.txt", io.BytesIO(b"not dicom"), "text/plain"))])
    assert r.status_code == 400


def test_validation_errors(client):
    assert client.post("/vqa/ask", json={"study_id": "zz", "question": "hi there"}).status_code == 422
    assert client.post("/vqa/ask", json={"study_id": "0123456789abcdef"}).status_code == 422
    assert client.post("/anomaly-detection", json={"study_id": "0123456789abcdef"}).status_code == 404
    assert client.get("/results/0123456789abcdef").status_code == 404
