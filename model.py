import os
from llama_cpp import Llama

# Ghép file từ các phần nhỏ
def merge_model_parts(output_file="phi-2.Q6_K.gguf"):
    part_files = sorted([f for f in os.listdir() if f.startswith("phi-2.Q6_K.gguf.part_")])
    
    with open(output_file, "wb") as outfile:
        for part in part_files:
            with open(part, "rb") as infile:
                outfile.write(infile.read())

    print(f"Merged {len(part_files)} parts into {output_file}")

# Chạy ghép file nếu mô hình chưa tồn tại
if not os.path.exists("phi-2.Q6_K.gguf"):
    merge_model_parts()

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
