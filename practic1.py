from transformers import pipeline
model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
model("Мне нравится машинное обуччение")
