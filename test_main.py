import io 
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def create_fake_image_bytes() -> bytes:
    """Crea una imagen válida en memoria, sin necesidad de un archivo real en disco."""
    img = Image.new("RGB", (256, 256), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def test_root_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API Funciona"}

def test_predict_with_valid_image_returns_200(client):
    image_bytes = create_fake_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200

    data = response.json()
    assert "is_anomalous" in data
    assert "score" in data
    assert "contour_image_base64" in data
    assert "heatmap_image_base64" in data

def test_predict_rejects_invalid_contest_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"Esto no es una imagen", "image/plain")},
    )
    assert response.status_code == 415

def test_predict_rejects_corrupted_image(client):
    response = client.post(
        "/predict",
        files={"file": ("fake.png", b"Esto no es un PNG de verdad", "image/png")},
    )
    assert response.status_code == 400

def test_predict_preview_return_png(client):
    image_bytes = create_fake_image_bytes()
    response = client.post(
        "/predict/preview",
        files={"file": ("test.png", image_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "X-Is-Anomalous" in response.headers
    assert "X-Score" in response.headers