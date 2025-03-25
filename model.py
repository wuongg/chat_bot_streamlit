import os
from llama_cpp import Llama

# Định nghĩa đường dẫn mô hình (từ Dockerfile)
MODEL_PATH = "/app/model.pth"

# Kiểm tra xem file mô hình có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại {MODEL_PATH}. Kiểm tra lại Dockerfile!")

# Cấu hình Llama
n_threads = os.cpu_count()
GPU_LAYERS = 0  # Để 0 nếu Railway không có GPU

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=n_threads, batch_size=4096, n_gpu_layers=GPU_LAYERS)

def generate_answer_fast(question, context, max_context_length=500):
    context = " ".join(context.split()[:max_context_length])
    prompt = f"""You are a traffic law expert AI assistant.
Below is relevant information:
---
{context}
---
Answer the question in a SHORT and PRECISE manner:
Question: {question}
Answer:"""
    
    response = llm(prompt, max_tokens=100, stop=["Question:"], temperature=0)
    return response["choices"][0]["text"].strip()
