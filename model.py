wget --no-check-certificate "https://drive.google.com/file/d/1D8kAbej21_l4vqFCzHJB461k2tMiRe6b/view?usp=sharing" -O phi-2.Q6_K.gguf
from llama_cpp import Llama
import os
n_threads = os.cpu_count()
GPU_LAYERS = 40

llm = Llama(model_path="phi-2.Q6_K.gguf", n_ctx=4096, n_threads=n_threads, batch_size=4096, n_gpu_layers=GPU_LAYERS)  # Điều chỉnh n_threads cho CPU

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
