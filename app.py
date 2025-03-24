from Faiss import retrieve_text_faiss
from model import generate_answer_fast
import streamlit as st
import time
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



st.title("🚦 Chatbot Luật Giao Thông")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        context = retrieve_text_faiss(prompt)
        answer = generate_answer_fast(prompt, context)
        response = st.write(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
