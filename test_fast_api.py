# test_fast_api.py

from fastapi.testclient import TestClient
from fast_api import app

client = TestClient(app)


def test_chat():
    # Send a test request to the /chat/ endpoint
    response = client.post(
        "/chat/",
        json={"user_input": "Hello! How are you?"}
    )

    # Check that a response was received
    assert response.status_code == 200

    # Check that the response contains the 'response' key
    assert "response" in response.json()

    # Check that the 'response' key in the response is not an empty string
    assert response.json()["response"], "The response should not be empty."
