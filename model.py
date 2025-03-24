import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Tên repository chứa mô hình (nếu bạn upload mô hình của riêng bạn thì thay đổi repo_id)
repo_id = "TheBloke/phi-2-GGUF"  # Repo chứa mô hình phi-2 GGUF
model_filename = "phi-2.Q6_K.gguf"  # Tên file mô hình

# Tải xuống mô hình
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

print(f"Mô hình đã được tải về: {model_path}")
# Load mô hình vào Llama
n_threads = os.cpu_count()
GPU_LAYERS = 40

llm = Llama(model_path="phi-2.Q6_K.gguf", n_ctx=4096, n_threads=n_threads, batch_size=4096, n_gpu_layers=GPU_LAYERS)

def generate_answer_fast(question, context, max_context_length=500):
    # Giới hạn độ dài context
    context = " ".join(context.split()[:max_context_length])

    # Prompt tối ưu
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
