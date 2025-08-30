Studymate — PDF Q&A Assistant (Local RAG)

Studymate lets you upload academic PDFs and ask questions about them.
It extracts text (PyMuPDF), builds embeddings (Sentence-Transformers), indexes with FAISS, and generates answers with a local Hugging Face Transformers model.
No external inference API is required.

Features

Multi-PDF upload and parsing (PyMuPDF)

Overlapping chunking with page/file attribution

Embeddings with sentence-transformers/all-MiniLM-L6-v2

Fast similarity search with FAISS

Local LLM generation using transformers (Granite or any HF model)

Chat-style conversation with follow-ups and conversation memory

Downloadable chat history (JSON)