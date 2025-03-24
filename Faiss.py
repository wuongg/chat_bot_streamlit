from prepare_data import process_folder
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from docx import Document

model = SentenceTransformer("all-MiniLM-L6-v2")

# Đọc dữ liệu từ folder chứa các file .docx
folder_path = "ta"  # Đổi thành đường dẫn thư mục của bạn
index_path = os.path.join(folder_path, "faiss_index.bin")

if os.path.exists(index_path):
    # Nếu FAISS index đã tồn tại, tải lên
    index = faiss.read_index(index_path)
    print("Đã tải FAISS index từ tệp faiss_index.bin")
    
    # Giả sử ta cũng cần nạp lại documents để sử dụng sau này
    documents = process_folder(folder_path)
else:
    # Nếu FAISS index chưa tồn tại, tạo mới
    documents = process_folder(folder_path)
    
    # Mã hóa văn bản thành vector embeddings
    embeddings = np.array([model.encode(doc) for doc in documents], dtype=np.float32)
    
    # Tạo FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Lưu FAISS index vào file
    faiss.write_index(index, index_path)
    print("Đã tạo và lưu FAISS index vào faiss_index.bin")

# Hàm tìm kiếm văn bản gần nhất
def retrieve_text_faiss(user_question, top_k=3):
    query_vector = np.array([model.encode(user_question)], dtype=np.float32)
    _, idx = index.search(query_vector, k=top_k)
    return "\n\n".join([documents[i] for i in idx[0]])