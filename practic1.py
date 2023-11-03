from transformers import pipeline
model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
model("Привет, ты мне нравишься!")
# [{'label': 'love', 'score': 0.5955629944801331}]