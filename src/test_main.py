import os.path
from datetime import timedelta
from io import StringIO, BytesIO

from fastapi.testclient import TestClient
from httpx import Response

from main import app
from main import httpx


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
client = TestClient(app)


class MockBaseAsyncClient(object):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        return


def mock_post(monkeypatch, status_code=200, json_content=None, raise_object=None):
    class MockAsyncClient(MockBaseAsyncClient):
        async def post(self, *args, **kwargs):
            if raise_object:
                raise raise_object
            response = Response(status_code, json=json_content or {})
            response.elapsed = timedelta()
            return response
    monkeypatch.setattr(httpx, 'AsyncClient', MockAsyncClient)

    buff = BytesIO(b'Just a test')
    return client.post("/api/v1/resnet", files={"img": ("filename", buff, "image/jpeg")})


class TestAPI:
    pass

    def test_without_send_file(self):
        response = client.post("/api/v1/resnet", files=[])
        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "loc": ["body", "img"],
                    "msg":"field required",
                    "type":"value_error.missing"
                }
            ]
        }

    def test_invalid_content(self, monkeypatch):
        buff = StringIO('Just a test')
        response = client.post("/api/v1/resnet", files={"img": ("filename", buff, "text/plain")})
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid content type"}

    def test_connection_error(self, monkeypatch):
        response = mock_post(monkeypatch, raise_object=httpx.HTTPError('Just a test'))

        assert response.status_code == 500
        assert response.json() == {"detail": "Cannot connect to TensorFlow"}

    def test_wrong_status_without_error(self, monkeypatch):
        response = mock_post(monkeypatch, status_code=400)

        assert response.status_code == 500
        assert response.json() == {"detail": "Error processing file"}

    def test_wrong_status_with_error(self, monkeypatch):
        response = mock_post(monkeypatch, status_code=400, json_content={'error': 'error from TensorFlow'})

        assert response.status_code == 500
        assert response.json() == {"detail": "error from TensorFlow"}

    def test_wrong_body(self, monkeypatch):
        response = mock_post(monkeypatch)

        assert response.status_code == 500
        assert response.json() == {"detail": "No predictions received"}

    def test_no_prediction(self, monkeypatch):
        response = mock_post(monkeypatch, json_content={"predictions": []})

        assert response.status_code == 500
        assert response.json() == {"detail": "No predictions received"}

    def test_wrong_prediction_list(self, monkeypatch):
        response = mock_post(monkeypatch, json_content={"predictions": ['Wrong type']})

        assert response.status_code == 500
        assert response.json() == {"detail": "Unexpected response from TensorFlow"}

    def test_wrong_classes_type(self, monkeypatch):
        response = mock_post(monkeypatch, json_content={"predictions": [{'classes': 'Int expected'}]})

        assert response.status_code == 500
        assert response.json() == {"detail": "Unexpected response from TensorFlow"}

    def test_good_response(self, monkeypatch):
        response = mock_post(monkeypatch, json_content={"predictions": [{'classes': 100}]})

        assert response.status_code == 200
        assert response.json() == {"class": 100}
