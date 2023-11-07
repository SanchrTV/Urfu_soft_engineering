from transformers import pipeline

model = pipeline("sentiment-analysis",
                 "blanchefort/rubert-base-cased-sentiment")
print(model("Мне нравится машинное обучение"))
