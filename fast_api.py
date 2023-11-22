from fastapi import FastAPI
from pydantic import BaseModel

from app import tokenizer, model

app = FastAPI()


class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(chat_request: ChatRequest):

    input_ids = tokenizer.encode(chat_request.user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"response": response}

# для запуска в терминале:
# uvicorn main:app --reload
