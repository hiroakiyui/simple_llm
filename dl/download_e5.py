from sentence_transformers import SentenceTransformer

# 保存先ディレクトリ（絶対パス）
model_path = "/Users/<ユーザー名>/Documents/dev/llm-env"

# e5-small のダウンロード＆保存
model = SentenceTransformer("intfloat/e5-small", cache_folder=model_path)

print("✅ e5-small モデルをダウンロードしました")

