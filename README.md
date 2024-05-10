# Документация к чат-боту app.py

## Обзор

Данная программа на Python реализует чат-бота, используя сочетание библиотеки `transformers` от Hugging Face, библиотеки `TextBlob` для обработки естественного языка (NLP), библиотеки `streamlit`для создания веб-приложения. Библиотека `transformers` используется для загрузки предварительно обученной модели каузального языка, которая генерирует ответы для диалога. `TextBlob` применяется для предварительной коррекции текста, что позволяет обеспечить более грамматически корректный ввод в модель. `streamlit` предназначена для разработки интерактивных веб-приложений с использованием языка Python.

## Особенности

- **Коррекция текста**: Программа использует `TextBlob` для исправления базовых грамматических ошибок во вводимом тексте перед его обработкой.
- **Модель для диалога**: Использует токенизатор `codegen-350M-mono` от Salesforce и модель `DialoGPT-medium` от Microsoft для генерации диалогового текста.
- **Интерактивный чат**: Запускает интерактивную оболочку, которая позволяет пользователю вести диалог с ботом и получать ответы в реальном времени.
- **Причинно-следственное предсказание**: "AutoModelForCausalLM" представляет из себя тип модели машинного обучения для причинно-следственного предсказания последовательности текста, а "AutoTokenizer" используется для токенизации, то есть преобразования текста в подходящий для модели формат.

## Требования

- Python версии 3.x
- Библиотека `transformers`
- Библиотека `TextBlob`
- Библиотека `streamlit`

Кроме того, для использования коррекции текста с помощью `TextBlob` могут потребоваться ресурсы `nltk`, такие как `punkt`, `averaged_perceptron_tagger` и `wordnet`.

## Установка
Установите необходимые библиотеки с помощью `pip`:
pip install -r requirements.txt

## Запуск
streamlit run app.py --server.port 8502

## Использование
Для использования чат-бота запустите скрипт. После инициализации чат-бот предложит ввести текст. Он исправит ваш ввод, если это необходимо, и затем предоставит ответ. Чтобы выйти из чата, введите 'exit'.

## Функции
correct_text(input_text: str) -> str
Принимает строку input_text, исправляет грамматику с использованием TextBlob и возвращает исправленный текст в виде строки.

chatbot_response(input_text: str) -> str
Принимает строку input_text, исправляет текст с помощью функции correct_text и генерирует ответ с помощью предварительно обученной языковой модели. Возвращает ответ чат-бота в виде строки.

## Выход
Чтобы выйти из процесса Streamlit, который запущен в командной строке (терминале), вы обычно можете использовать сочетание клавиш Ctrl + C. Это стандартный способ прерывания выполнения большинства процессов в терминале.

# Улучшения чат-бота

## Описание

Данный документ описывает ряд улучшений для скрипта чат-бота на Python, использующего модели трансформеров и TextBlob для коррекции текста и генерации ответов.

## Предложенные изменения

### 1. Очистка кода и организация
- **Удаление комментариев**: Удаляются комментарии загрузки `nltk`, так как они не используются.
- **Организация импортов**: Все импорты находятся в начале файла и не дублируются.

### 2. Повышение эффективности
- **Инициализация модели и токенизатора**: Модель и токенизатор инициализируются один раз за время выполнения скрипта, чтобы избежать повторной загрузки.

### 3. Конфигурация токенизатора и модели
- **Правильная установка `padding_side`**: Проверьте, требует ли модель изменения стороны заполнения.

### 4. Читаемость кода
- **Улучшенные комментарии**: Добавлены подробные комментарии на русском языке к критическим участкам кода.

## Заключение

Эти улучшения направлены на повышение эффективности, надежности и удобства использования чат-бота, а также на улучшение качества взаимодействия с пользователем.
