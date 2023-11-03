#!/usr/bin/env python
# coding: utf-8


from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(text=['''Привет! Это документ для первой практики по курсу "программная инженерия"!'''],
    				return_tensors="pt")

speech_values = model.generate(**inputs, do_sample=True)