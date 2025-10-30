# Knowledge Management RAG Chatbot (v1)

The **Knowledge Management RAG Chatbot** helps a **small or mid-sized organization** query its internal knowledge using a conversational interface. It combines a FastAPI backend, a vector database (Chroma), and a ChatGPT-like UI to deliver Retrieval‑Augmented Generation (RAG) out of the box. It also includes an ingestion script to embed and ingest Word, Excel and PDF documents into the vector store.

## 🧩 Current Capabilities (v1)

- Quickly deploy with 1-touch setup script and automated document ingestion.
- Query your knowledge base and create reports via ChatGPT-like UI (LibreChat).
- Ingest additional Word, Excel, PDF documents into knowledge base anytime via script.
- FastAPI backend for retrieval-augmented generation (RAG).
- Semantic search with optional keyword matching and metadata filters.
- Re-ranking of results on retrieval for more relevant LLM responses.
- Config-driven OpenAI API usage for embeddings and generation.
- Dockerized services: Chroma, backend API, LibreChat, and MongoDB for UI.
- Local file storage for uploads and processed artifacts.

## 🧱 Software Stack

| Component | Technology | Purpose |
|---|---|---|
| **UI** | LibreChat | Frontend chat interface and session management |
| **Backend API** | FastAPI | RAG endpoints: ingest, query, health |
| **Vector Database** | Chroma | Embeddings storage and similarity search |
| **Operational DB** | MongoDB (LibreChat) | Stores UI conversations and settings |
| **Embeddings** | multilingual-e5-base (configurable) | Semantic representations of documents |
| **Re-ranking** | ms-marco-MiniLM-L-6-v2 (configurable) | Re-ranking of retrieved sources |
| **Generator** | GPT-5 via API (configurable) | LLM used for answers |
| **Orchestration** | Docker + Docker Compose | Local/cloud deployment |
| **Scripts** | Python + Bash | Document ingestion and local setup |

## ⚙️ Installation & Local Setup

```bash
# clone
git clone https://github.com/your-org/km-rag-chatbot.git
cd km-rag-chatbot

# Install Docker if needed

# Review and update environment variables (such as your OpenAI API key) in .env, config.py and docker-compose.yml

# Place any Word, Excel and PDF documents you'd like to ingest on setup in ./data/documents

# local setup (Mac/Linux)
chmod +x setup-local.sh
./setup-local.sh
```

Default endpoints:
- Frontend (LibreChat): http://localhost:3080
- Vector store (Chroma): http://localhost:8000
- Backend API (FastAPI): http://localhost:8001
- MongoDB (LibreChat): http://localhost:27017

## 📥 Document Ingestion

Use `ingest_documents.py` for manual ingestion of Word, Excel and PDF documents into the vector store. See the script's help for usage details. The script is automatically called on initial setup with setup_local.sh.

## 🔮 Future Updates

- Hybrid retrieval (BM25 + semantic) and advanced reranking.
- Auth integration (OIDC / SSO) and role-based access controls.
- Connectors via MCP (web, cloud drives, knowledge bases).
- Report generation to Word/Excel/PowerPoint via function calling.
- Robust citations and source tracing in answers.
- Monitoring and analytics (e.g., Langfuse) for quality and usage.
- Cloud object storage (S3-compatible) for documents and embeddings snapshots.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

Licensed under the MIT License. See `LICENSE`.
