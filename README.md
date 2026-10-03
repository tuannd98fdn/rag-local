# Trợ Lý Tài Liệu Cục Bộ (Local RAG) Với Ollama, ChromaDB & Streamlit

Dự án hands-on thực hành xây dựng ứng dụng RAG (Retrieval-Augmented Generation) hoàn toàn offline trên máy cá nhân, đi kèm bài viết:
👉 **[Tự xây trợ lý hỏi đáp tài liệu với RAG, Ollama và Python](https://tuannd.vercel.app/blog/rag-tutorial-build-tro-ly-tai-lieu)**

---

## 🛠️ Công Nghệ Sử Dụng

- **Python 3.10+**
- **[Streamlit](https://streamlit.io/)**: Giao diện hỏi đáp và upload tài liệu trực quan.
- **[Ollama](https://ollama.com/)**: Chạy model ngôn ngữ và model embedding offline.
  - Chat model mặc định: `qwen2.5:3b`
  - Embedding model mặc định: `nomic-embed-text`
- **[ChromaDB](https://www.trychroma.com/)**: Vector database cục bộ, lưu trữ embedding và tìm kiếm tương đồng (cosine distance).
- **[pypdf](https://pypdf.readthedocs.io/)**: Trích xuất nội dung từ tài liệu PDF.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Chuẩn bị Ollama & Tải Models

Khởi động Ollama (nếu chưa chạy dạng service):
```bash
ollama serve
```

Tải 2 mô hình cần thiết:
```bash
# Model tạo câu trả lời
ollama pull qwen2.5:3b

# Model biến văn bản thành vector embedding
ollama pull nomic-embed-text
```

Kiểm tra danh sách mô hình đã tải:
```bash
ollama list
```

---

### 2. Thiết Lập Môi Trường Python

Clone repository và truy cập thư mục:
```bash
git clone https://github.com/tuannd98fdn/rag-local.git
cd rag-local
```

Khởi tạo và kích hoạt môi trường ảo:
```bash
# Trên macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Trên Windows PowerShell
# python -m venv .venv
# .venv\Scripts\Activate.ps1
```

Cài đặt các thư viện phụ thuộc:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### 3. Chạy Ứng Dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở tại `http://localhost:8501`.

---

## ⚙️ Biến Môi Trường (Tùy Chọn)

Bạn có thể tùy chỉnh các cấu hình thông qua biến môi trường:

```bash
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="qwen2.5:3b"
export OLLAMA_EMBED_MODEL="nomic-embed-text"
```
