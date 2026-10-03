import hashlib
import os
from io import BytesIO
from pathlib import Path

import chromadb
import requests
import streamlit as st
from pypdf import PdfReader

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DB_PATH = Path(__file__).resolve().parent / ".chroma"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def ollama_embed(texts: list[str]) -> list[list[float]]:
    cleaned: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        value = text.strip()
        if value:
            cleaned.append(value)

    if not cleaned:
        raise ValueError("Không có nội dung để embedding.")

    def request_batch(batch: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch, "truncate": True},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama embed lỗi: {response.status_code} - {response.text[:500]}")
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama không trả về embedding hợp lệ: {payload}")
        if len(embeddings) != len(batch):
            raise RuntimeError("Số lượng embedding không khớp với số đoạn văn bản.")
        return embeddings

    all_embeddings: list[list[float]] = []
    batch_size = 8
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        try:
            all_embeddings.extend(request_batch(batch))
        except RuntimeError as error:
            if len(batch) == 1:
                raise
            halfway = max(1, len(batch) // 2)
            left = batch[:halfway]
            right = batch[halfway:]
            all_embeddings.extend(ollama_embed(left))
            all_embeddings.extend(ollama_embed(right))
            break

    if len(all_embeddings) != len(cleaned):
        raise RuntimeError("Ollama trả về số embedding thấp hơn số văn bản hợp lệ.")
    return all_embeddings


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    raise ValueError(f"Định dạng chưa được hỗ trợ: {suffix}")


def split_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - CHUNK_OVERLAP

    safe_chunks: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if len(chunk) > 2500:
            for sub in chunk.split(". "):
                sub = sub.strip()
                if sub and len(sub) <= 2500:
                    safe_chunks.append(sub)
            continue
        safe_chunks.append(chunk)
    return safe_chunks


def index_file(filename: str, content: bytes) -> int:
    text = extract_text(filename, content)
    chunks = split_text(text)
    if not chunks:
        return 0

    collection = get_collection()
    collection.delete(where={"source": filename})
    embeddings = ollama_embed(chunks)
    ids = [
        hashlib.sha256(f"{filename}:{index}:{chunk}".encode()).hexdigest()
        for index, chunk in enumerate(chunks)
    ]
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": filename} for _ in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def looks_like_vietnamese(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    vietnamese_chars = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
    has_vietnamese_chars = any(ch in vietnamese_chars for ch in lowered)
    has_vietnamese_markers = any(marker in lowered for marker in ["điều", "trong", "nếu", "sử dụng", "hệ thống", "trả lời", "tài liệu"])
    return has_vietnamese_chars or has_vietnamese_markers


def answer_question(question: str) -> tuple[str, list[dict[str, str]]]:
    collection = get_collection()
    if collection.count() == 0:
        return "Chưa có tài liệu nào được lập chỉ mục. Hãy tải lên file trước.", []

    query_embedding = ollama_embed([question])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(4, collection.count()),
        include=["documents", "metadatas"],
    )

    retrieved = list(zip(result["documents"][0], result["metadatas"][0]))
    preferred = [(doc, meta) for doc, meta in retrieved if looks_like_vietnamese(doc)]
    if preferred:
        retrieved = preferred

    documents = [doc for doc, _ in retrieved]
    metadatas = [meta for _, meta in retrieved]
    sources = [
        {"label": f"[{index}] {meta['source']}", "text": doc}
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
    ]
    context = "\n\n".join(f"[{idx}] {item['text']}" for idx, item in enumerate(sources, start=1))

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý hỏi đáp tài liệu. Hãy trả lời bằng tiếng Việt hoàn toàn. "
                        "Chỉ đưa ra câu trả lời dựa trên NGỮ CẢNH dưới đây. "
                        "Nếu ngữ cảnh có nhiều ngôn ngữ, hãy ưu tiên các đoạn tiếng Việt và bỏ qua các đoạn không phải tiếng Việt. "
                        "Không lẫn tiếng Anh, tiếng Trung hoặc ngôn ngữ khác trong câu trả lời. "
                        "Nếu không có thông tin trong ngữ cảnh, hãy nói rõ là tài liệu chưa cung cấp đáp án. "
                        "Trích dẫn nguồn bằng nhãn [1], [2] tương ứng với đoạn đã dùng."
                    ),
                },
                {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}"},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["message"]["content"], sources


st.set_page_config(page_title="Trợ lý tài liệu", page_icon="📚")
st.title("Trợ lý tài liệu cục bộ")
st.caption(f"Chat: {CHAT_MODEL} · Embedding: {EMBED_MODEL} · Ollama: {OLLAMA_URL}")

with st.sidebar:
    st.subheader("Thêm tài liệu")
    uploads = st.file_uploader(
        "Chọn PDF, TXT hoặc Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Lập chỉ mục", disabled=not uploads):
        for upload in uploads or []:
            try:
                count = index_file(upload.name, upload.getvalue())
                if count:
                    st.success(f"{upload.name}: đã lưu {count} đoạn.")
                else:
                    st.warning(f"{upload.name}: không có nội dung để lập chỉ mục.")
            except Exception as error:
                st.error(f"Không xử lý được {upload.name}: {error}")

    try:
        st.caption(f"Tổng số đoạn trong kho: {get_collection().count()}")
    except Exception:
        st.caption("Kho vector chưa sẵn sàng.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Đoạn tài liệu được truy xuất"):
                for source in message["sources"]:
                    st.markdown(f"**{source['label']}**")
                    st.caption(source["text"])

if question := st.chat_input("Bạn muốn hỏi gì trong tài liệu?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tìm đoạn liên quan và tạo câu trả lời..."):
                answer, sources = answer_question(question)
            st.markdown(answer)
            if sources:
                with st.expander("Đoạn tài liệu được truy xuất"):
                    for source in sources:
                        st.markdown(f"**{source['label']}**")
                        st.caption(source["text"])
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except requests.RequestException as error:
            st.error(f"Không gọi được Ollama. Hãy kiểm tra Ollama đang chạy và model đã được cài. Chi tiết: {error}")
