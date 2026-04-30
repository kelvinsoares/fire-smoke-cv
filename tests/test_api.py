from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def create_test_image() -> BytesIO:
    image = Image.new("RGB", (640, 480), color="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "predict_json" in data
    assert "predict_image" in data


def test_predict_endpoint_returns_json() -> None:
    image = create_test_image()

    with TestClient(app) as client:
        response = client.post(
            "/predict?confidence=0.25",
            files={"file": ("test.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200

    data = response.json()

    assert "filename" in data
    assert "detections_count" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_predict_image_endpoint_returns_image() -> None:
    image = create_test_image()

    with TestClient(app) as client:
        response = client.post(
            "/predict/image?confidence=0.25",
            files={"file": ("test.jpg", image, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_invalid_file_type_returns_400() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")},
        )

    assert response.status_code == 400
