import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from textblob import TextBlob

# Загрузка предобученной модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def correct_text(input_text):
    """
    Функция для корректировки введенного текста с использованием библиотеки TextBlob.

    Аргументы:
    input_text (str): Входной текст для корректировки.

    Возвращает:
    str: Исправленный текст.
    """
    text_blob = TextBlob(input_text)
    corrected_text = text_blob.correct()
    return str(corrected_text)

def chatbot_response(input_text):
    """
    Функция для генерации ответа чат-бота на основе введенного текста.

    Аргументы:
    input_text (str): Входной текст для генерации ответа.

    Возвращает:
    str: Сгенерированный ответ чат-бота.
    """
    corrected_input_text = correct_text(input_text)
    if corrected_input_text != input_text:
        st.write("Исправленный текст:", corrected_input_text)
    input_ids = tokenizer.encode(corrected_input_text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Интерфейс Streamlit
st.title("Чат-бот с корректировкой текста")

# Инициализация истории разговора
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""

# Поле для ввода текста пользователем
user_input = st.text_input("Вы: ", key="input_text")

if st.button("Отправить"):
    # Получение ответа чат-бота
    response = chatbot_response(user_input)

    # Обновление истории разговора
    st.session_state.conversation_history += f"Вы: {user_input}<br>Чат-бот: {response}<hr>"

    # Отображение истории разговора с использованием st.markdown и HTML форматирования
    st.markdown("#### История разговора:", unsafe_allow_html=True)
    st.markdown(st.session_state.conversation_history, unsafe_allow_html=True)
