import numpy as np
import os
from docx import Document

# Hàm đọc nội dung từ file .docx
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# Đọc tất cả các file .docx trong thư mục
def process_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            content = read_docx(file_path)
            if content:
                documents.append(content)
    return documents