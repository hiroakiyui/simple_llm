from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

input_ids = tokenizer("こんにちは", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

