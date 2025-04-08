from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルのローカル保存パス（TinyLlama）
model_path = "./model"

# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# 入力文（好きな文に変更OK）
user_input = "広島でおすすめのラーメン屋は？"

# プロンプト構成（シンプルに）
prompt = f"ユーザー: {user_input}\nアシスタント:"

# トークン化してモデルに渡す
inputs = tokenizer(prompt, return_tensors="pt")

# 推論（文章生成）
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

# 結果をデコードして表示
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("===== TinyLlamaの応答 =====")
print(output_text)

