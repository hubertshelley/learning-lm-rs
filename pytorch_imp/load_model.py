import math

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch

# excute from project directory.
model_directory = "./models/story"
# model_directory = "./models/chat"

model = AutoModelForCausalLM.from_pretrained(model_directory)

tokenizer = AutoTokenizer.from_pretrained(model_directory)
text = "<|start_story|>Once upon a time, "
# text = """<|im_start|>system
# You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.<|im_end|>
# <|im_start|>user
# Hey! Got a question for you!<|im_end|>
# <|im_start|>assistant
# """
inputs = tokenizer(text, return_tensors="pt")

# 执行推理
out_puts = model.generate(inputs.input_ids, max_length=500, do_sample=True, top_p=0.55, top_k=35, temperature=0.65)

result = tokenizer.batch_decode(out_puts)[0]

print(result)
