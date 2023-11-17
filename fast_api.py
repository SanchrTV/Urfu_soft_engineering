from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

# @app.post("/chat/")
# async def chat(chat_request: ChatRequest):
#     # добавить код для обработки входящего сообщения и генерации ответа модели
#     pass

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    #необходимо импортировать tokenizer и model из другого файла в репозитории
    input_ids = tokenizer.encode(chat_request.user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"response": response}

#для запуска в терминале: 
# uvicorn main:app --reload