import streamlit as st

# Заголовок приложения
st.title("Пообщаемся?")

# Текстовое поле для ввода пользователя
user_input = st.text_area("Введи сообщение здесь:")

# Кнопка для преобразования текста
if st.button("Получить ответ"):
    # Преобразование текста в нижний регистр
    lowercase_text = user_input.lower()
    # Отображение преобразованного текста
    st.text_area("Ответ модели:", value=lowercase_text, height=150)