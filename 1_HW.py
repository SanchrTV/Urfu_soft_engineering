"""
https://huggingface.co/seara/rubert-tiny2-russian-sentiment

This is RuBERT-tiny2 model fine-tuned for sentiment classification of short Russian texts. The task is a multi-class classification with the following labels:

0: neutral
1: positive
2: negative

Label to Russian label:

neutral: нейтральный
positive: позитивный
negative: негативный

"""
from transformers import pipeline
model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
model("Привет, ты мне нравишься!")

