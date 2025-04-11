import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from docx import Document

# Load mô hình embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Đường dẫn thư mục chứa file .docx
folder_path = "ta"

# Đường dẫn file lưu dữ liệu
index_file = "faiss_index.bin"
embeddings_file = "embeddings.npy"
documents_file = "documents.pkl"

# Hàm đọc nội dung từ file .docx
def read_docx(file_path):
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

# Đọc và xử lý toàn bộ file trong thư mục
def process_folder(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            content = read_docx(file_path)
            if content:
                documents.extend(content)  # Chia nhỏ từng đoạn văn
                filenames.extend([filename] * len(content))
    return documents, filenames

# Kiểm tra nếu đã có dữ liệu lưu sẵn
if os.path.exists(index_file) and os.path.exists(embeddings_file) and os.path.exists(documents_file):
    print("🔄 Đang tải FAISS index và dữ liệu đã lưu...")
    
    # Load FAISS index
    index = faiss.read_index(index_file)
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    
    # Load danh sách tài liệu
    with open(documents_file, "rb") as f:
        documents, filenames = pickle.load(f)
    
    print("✅ Tải dữ liệu hoàn tất!")
else:
    print("📂 Đang xử lý tài liệu và tạo FAISS index...")

    # Đọc dữ liệu từ thư mục
    documents, filenames = process_folder(folder_path)

    # Mã hóa văn bản thành embeddings
    embeddings = np.array(model.encode(documents, batch_size=32, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)

    # Thiết lập FAISS Index IVF + HNSW
    d = embeddings.shape[1]
    nlist = 100  # Số lượng cụm trong IVF
    quantizer = faiss.IndexHNSWFlat(d, 32)  # HNSW với 32 neighbors
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Huấn luyện và thêm dữ liệu vào index
    index.train(embeddings)
    index.add(embeddings)

    # Cải thiện độ chính xác bằng cách tăng số cụm được duyệt khi tìm kiếm
    index.nprobe = 10  

    # Lưu FAISS index
    faiss.write_index(index, index_file)
    
    # Lưu embeddings
    np.save(embeddings_file, embeddings)
    
    # Lưu danh sách tài liệu
    with open(documents_file, "wb") as f:
        pickle.dump((documents, filenames), f)

    print("✅ Lưu dữ liệu hoàn tất!")

# Hàm tìm kiếm văn bản gần nhất
def retrieve_text_faiss(user_question, top_k=10):
    query_vector = np.array([model.encode(user_question, normalize_embeddings=True)], dtype=np.float32)
    distances, indices = index.search(query_vector, k=top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):  # Kiểm tra tránh lỗi index out of range
            score = 1 - distances[0][i]  # Chuyển khoảng cách L2 thành độ tương đồng
            results.append(f"[{filenames[idx]}] (Score: {score:.4f})\n{documents[idx]}")

    return "\n\n".join(results) if results else "Không tìm thấy văn bản liên quan."
