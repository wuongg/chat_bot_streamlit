# Sử dụng Python 3.12 làm base image
FROM python:3.12

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn vào container
COPY . .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt wget để tải mô hình
RUN apt-get update && apt-get install -y wget

# Tải mô hình từ GitHub Releases
RUN wget -O model_part_000.part "https://github.com/wuongg/chat_bot_streamlit/releases/download/v1.0.0/model_part_000.part"
RUN wget -O model_part_001.part "https://github.com/wuongg/chat_bot_streamlit/releases/download/v1.0.0/model_part_001.part"

# Ghép file thành model.pth
RUN cat model_part_000.part model_part_001.part > model.pth

# Xóa các file không cần thiết để tiết kiệm dung lượng
RUN rm model_part_000.part model_part_001.part

# Mở cổng 8501 (mặc định của Streamlit)
EXPOSE 8501

# Chạy ứng dụng
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
