from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def correct_text(input_text):
    """Исправляет входной текст с помощью TextBlob и возвращает его."""
    text_blob = TextBlob(input_text)
    return str(text_blob.correct())

def chatbot_response(input_text):
    """Генерирует ответ чат-бота после возможного исправления входного текста."""
    corrected_input_text = correct_text(input_text)
    if corrected_input_text != input_text:
        print("Исправленный текст:", corrected_input_text)
    input_ids = tokenizer.encode(corrected_input_text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

def main():
    """Основная функция для управления общением с чат-ботом."""
    print("Чат-бот активирован. Напишите что-нибудь, чтобы начать разговор (введите 'exit', чтобы завершить)")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'exit':
            print("Чат-бот деактивирован.")
            break
        response = chatbot_response(user_input)
        print("Чат-бот: ", response)

if __name__ == "__main__":
    main()
