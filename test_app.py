# test_app.py

import pytest
from app import chatbot_response

def test_chatbot_response():
    user_input = "Hello! How are you?"
    actual_response = chatbot_response(user_input)

    assert actual_response != "", "The response should not be empty"
