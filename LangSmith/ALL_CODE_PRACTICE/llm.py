from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

inputs = tokenizer(
    "Explain gradient descent in simple terms",
    return_tensors="pt"
).to("cuda")

output = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
