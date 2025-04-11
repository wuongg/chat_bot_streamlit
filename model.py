from llama_cpp import Llama
import os
import requests

MODEL_PATH = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("⏬ Model not found. Downloading...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download model: {e}\nPlease download manually from:\n{MODEL_URL}")

download_model_if_missing()

n_threads = os.cpu_count()
GPU_LAYERS = 40

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=n_threads,
    batch_size=4096,
    n_gpu_layers=GPU_LAYERS
)

def generate_answer_fast(question, context, max_context_length=500):
    # Giới hạn độ dài context
    context = " ".join(context.split()[:max_context_length])

    # Prompt với yêu cầu rõ ràng tránh dùng "Yes"/"No"
    prompt = f"""You are an AI assistant specializing in Vietnam's traffic laws.

🛑 Do **not** begin your answer with "Yes" or "No".  
Instead, explain the legal situation based strictly on the context below.

Your response must be:  
✅ **Accurate and legally sound**  
✅ **Based strictly on the legal information provided**  
✅ **Include relevant legal citations when applicable**  
✅ **Avoid speculation or invented content**  

### 📘 Legal Context:
---
{context}
---

### 📝 Answering Instructions:
1. Provide a **clear and comprehensive explanation** based solely on the laws cited.  
2. Mention **specific articles, decrees, or penalties** when appropriate.  
3. If the question lacks enough detail or legal grounding, respond with:  
   _"I need more legal information to give an accurate answer."_  
4. Never invent or assume any legal information.

Now, answer the following question thoroughly:

**Q: {question} (under Vietnam’s traffic law)**

**A:**
"""

    # Gọi model (giả sử `llm` là callable)
    response = llm(
        prompt,
        max_tokens=200,
        stop=["Q:"],
        temperature=0.0,
        top_p=1.0,
        top_k=0
    )

    return response["choices"][0]["text"].strip()

