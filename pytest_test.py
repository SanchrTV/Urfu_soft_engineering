import pytest
import httpx
from fastapi import FastAPI
from app import app  # Импортируйте ваш экземпляр FastAPI

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/chat/", json={"user_input": "Привет"})
        assert response.status_code == 200
        assert "response" in response.json()

@pytest.mark.asyncio
async def test_input_validation():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/chat/", json={"user_input": ""})
        assert response.status_code == 422  # Предполагая, что вы обрабатываете это как ошибку валидации

@pytest.mark.asyncio
async def test_model_integration():
    # Здесь вы должны имитировать ответ модели для тестирования интеграции
    # Это требует более глубокой настройки для имитации поведения модели
    pass

@pytest.mark.asyncio
async def test_error_handling():
    # Этот тест включает в себя вызов ошибки и проверку того, как ваше приложение с ней справляется
    pass

#запуск тестов:
#pytest pytest_test.py