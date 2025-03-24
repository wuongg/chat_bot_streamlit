from llama_cpp import Llama
import os
import os
import gdown

MODEL_ID = "1D8kAbej21_l4vqFCzHJB461k2tMiRe6b"  # Thay bằng ID Google Drive của bạn
MODEL_NAME = "phi-2.Q6_K.gguf"
MODEL_PATH = os.path.join("models", MODEL_NAME)  # Lưu vào thư mục models

# Tạo thư mục nếu chưa có
os.makedirs("models", exist_ok=True)

# Kiểm tra nếu model chưa tồn tại thì tải xuống
if not os.path.exists(MODEL_PATH):
    print(f"🔽 Đang tải model từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
else:
    print(f"✅ Model đã có sẵn tại {MODEL_PATH}")
n_threads = os.cpu_count()
GPU_LAYERS = 40

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=n_threads, batch_size=4096, n_gpu_layers=GPU_LAYERS)  # Điều chỉnh n_threads cho CPU

def generate_answer_fast(question, context, max_context_length=500):
    # Limit context length to prevent overflow
    context = " ".join(context.split()[:max_context_length])

    # Optimized prompt with clear and concise instructions
    prompt = f"""You are a traffic law expert AI assistant.
Below is relevant information:
---
{context}
---
Answer the question in a SHORT and PRECISE manner:
Question: {question}
Answer:"""

    response = llm(prompt, max_tokens=100, stop=["Question:"],temperature=0)  # Limit max_tokens
    return response["choices"][0]["text"].strip()
