# Advanced RAG (Retrieval-Augmented Generation) Repository

A comprehensive collection of state-of-the-art RAG implementations showcasing various techniques from basic PDF question-answering to advanced multimodal and graph-based retrieval systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains multiple RAG implementations demonstrating different approaches to information retrieval and generation:

1. **PDF Chatbot** - Simple question-answering system for PDF documents
2. **Hybrid Search RAG** - Combines keyword and semantic search for improved retrieval
3. **Vision RAG** - Multimodal system handling text and images using CLIP embeddings
4. **LangChain Multimodal RAG** - Advanced PDF processing with table and image extraction
5. **LightRAG** - Graph-based RAG with knowledge graph construction
6. **Knowledge Graph LLMs** - Advanced knowledge graph integration with LLMs

## 📁 Projects

### 1. PDF Chatbot (`pdf chatbot/`)
**Difficulty:** Beginner  
**Description:** A straightforward implementation for querying PDF documents using OpenAI embeddings and vector search.

**Features:**
- PDF text extraction
- Vector embeddings with OpenAI
- Semantic search using Chroma
- Interactive Q&A interface

**Tech Stack:**
- LangChain
- OpenAI
- Chroma Vector Database
- PyPDF

[📖 Detailed Documentation](./pdf%20chatbot/README.md)

---

### 2. Hybrid Search RAG (`Hybrid Search Rag/`)
**Difficulty:** Intermediate  
**Description:** Implements hybrid search combining BM25 (keyword-based) and semantic search for enhanced retrieval accuracy.

**Features:**
- Dual retrieval strategy (keyword + semantic)
- Ensemble retriever with weighted scoring
- Reranking capabilities
- Optimized for diverse query types

**Tech Stack:**
- LangChain
- BM25 Algorithm
- Sentence Transformers
- FAISS/Chroma

[📖 Detailed Documentation](./Hybrid%20Search%20Rag/README.md)

---

### 3. Vision RAG (Multimodal) (`Hybrid Search Rag/`)
**Difficulty:** Advanced  
**Description:** Multimodal RAG system that processes both text and images, enabling queries across different content types.

**Features:**
- PDF text and image extraction using PyMuPDF
- Image embedding with CLIP (OpenAI/HuggingFace)
- Dual vector stores for text and images
- Multimodal retrieval and generation
- Image summarization with GPT-4 Vision

**Tech Stack:**
- LangChain
- CLIP (Contrastive Language-Image Pre-training)
- PyMuPDF (fitz)
- GPT-4 Vision
- Chroma Vector Database

**Notebooks:**
- `Advanced_RAG_Hybrid_Search_RAG.ipynb` - Hybrid search implementation
- `Vision_RAG.ipynb` - Multimodal RAG with CLIP embeddings

[📖 Detailed Documentation](./Hybrid%20Search%20Rag/README.md)

---

### 4. LangChain Multimodal RAG (`multimodal rag/`)
**Difficulty:** Advanced  
**Description:** Production-ready multimodal RAG using the `unstructured` library for advanced PDF parsing with table detection and image extraction.

**Features:**
- High-resolution PDF parsing with table structure inference
- Automatic image extraction and encoding
- OCR for text in images (Tesseract)
- Separate summarization for text, tables, and images
- Multi-vector retrieval strategy
- Support for Groq (fast inference) and OpenAI models

**Tech Stack:**
- LangChain
- Unstructured.io library
- Poppler (PDF rendering)
- Tesseract OCR
- Table Transformer models
- ChromaDB with multi-vector retrieval
- Groq API (LLaMA models)
- OpenAI GPT-4o-mini

**Files:**
- `langchain_multimodal.ipynb` - Complete implementation
- `langchain_multimodal (1).ipynb` - Working version with dependencies configured

**System Requirements:**
- Poppler binaries (Windows: via winget)
- Tesseract OCR (Windows: via winget)
- Python 3.11+ (required for unstructured library compatibility)

[📖 Detailed Documentation](./multimodal%20rag/README.md)

---

### 5. LightRAG (`LightRAG-main/`)
**Difficulty:** Expert  
**Description:** State-of-the-art graph-based RAG system that constructs and queries knowledge graphs for enhanced contextual understanding.

**Features:**
- Automatic knowledge graph construction from documents
- Entity and relationship extraction
- Graph-based retrieval with community detection
- Multiple search modes (naive, local, global, hybrid)
- Support for various LLM backends
- Incremental graph updates
- Visual graph exploration

**Tech Stack:**
- NetworkX for graph operations
- Neo4j/Native graph storage
- Multiple LLM support (OpenAI, Ollama, HuggingFace)
- Embedding models integration
- Community detection algorithms

[📖 Detailed Documentation](./LightRAG-main/README.md)

---

### 6. Knowledge Graph LLMs (`knowledge-graph-llms-main/`)
**Difficulty:** Expert  
**Description:** Advanced integration of knowledge graphs with LLMs for structured reasoning and query answering.

**Features:**
- Knowledge graph creation from unstructured text
- Graph query generation from natural language
- Multi-hop reasoning capabilities
- Entity linking and disambiguation
- Structured data extraction

**Tech Stack:**
- Graph databases (Neo4j, RDF)
- SPARQL/Cypher query languages
- Entity extraction models
- LLM integration for reasoning

[📖 Detailed Documentation](./knowledge-graph-llms-main/README.md)

---

## 🛠️ Environment Setup

### Prerequisites

- Python 3.11 or higher
- UV package manager (recommended) or pip
- Git
- Windows: Poppler and Tesseract (for multimodal RAG)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AnandaRimal/Rag.git
cd Rag
```

2. **Install UV package manager:**
```bash
# Windows (PowerShell)
pip install uv

# Or download from: https://github.com/astral-sh/uv
```

3. **Create and activate virtual environment:**
```bash
# Using UV
uv venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
.venv\Scripts\activate.bat   # Windows CMD
# or
source .venv/bin/activate    # Linux/Mac
```

4. **Install dependencies:**
```bash
# Install all dependencies
uv pip install -r requirements.txt

# Or for specific projects, navigate to project folder and install
```

### System Tools (For Multimodal RAG)

**Windows:**
```powershell
# Install Poppler
winget install oschwartz10612.Poppler

# Install Tesseract OCR
winget install UB-Mannheim.TesseractOCR

# Refresh PATH (or restart terminal)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

**Linux:**
```bash
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
```

**macOS:**
```bash
brew install poppler tesseract libmagic
```

### API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Groq: https://console.groq.com/keys
- Google (Gemini): https://makersuite.google.com/app/apikey
- Cohere: https://dashboard.cohere.com/api-keys
- HuggingFace: https://huggingface.co/settings/tokens

## 🚀 Quick Start

### Running PDF Chatbot
```bash
cd "pdf chatbot"
jupyter notebook "Ask A Book Questions.ipynb"
```

### Running Hybrid Search RAG
```bash
cd "Hybrid Search Rag"
jupyter notebook Advanced_RAG_Hybrid_Search_RAG.ipynb
```

### Running Vision RAG
```bash
cd "Hybrid Search Rag"
jupyter notebook Vision_RAG.ipynb
```

### Running LangChain Multimodal RAG
```bash
cd "multimodal rag/multimodal advance"
jupyter notebook "langchain_multimodal (1).ipynb"
```

### Running LightRAG
```bash
cd LightRAG-main
# See LightRAG-main/README.md for detailed instructions
python examples/lightrag_api_openai_compatible_demo.py
```

## 💡 Technologies Used

### Core Frameworks
- **LangChain**: LLM application framework
- **LlamaIndex**: Data framework for LLM applications
- **Unstructured**: Document parsing and preprocessing

### Vector Databases
- **Chroma**: Lightweight, in-memory vector database
- **FAISS**: Facebook AI Similarity Search
- **Pinecone**: Cloud-native vector database
- **Weaviate**: Open-source vector search engine

### LLM Providers
- **OpenAI**: GPT-4, GPT-3.5-Turbo, text-embedding-ada-002
- **Groq**: Fast inference with LLaMA models
- **Google Gemini**: Multimodal AI models
- **Cohere**: Embedding and reranking models
- **HuggingFace**: Open-source models

### Document Processing
- **PyMuPDF (fitz)**: PDF text and image extraction
- **Unstructured.io**: Advanced document parsing
- **PyPDF**: Basic PDF operations
- **Poppler**: PDF rendering engine
- **Tesseract**: OCR engine

### Computer Vision
- **CLIP**: Contrastive Language-Image Pre-training
- **Table Transformer**: Table detection and structure recognition
- **LayoutParser**: Document layout analysis

### Graph Technologies
- **NetworkX**: Python graph library
- **Neo4j**: Graph database
- **RDF/SPARQL**: Semantic web standards

### ML & Deep Learning
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace transformers library
- **Sentence Transformers**: Sentence embeddings
- **ONNX Runtime**: Model inference optimization

## 📊 Performance Comparisons

| Project | Retrieval Speed | Accuracy | Multimodal | Graph Support | Complexity |
|---------|----------------|----------|------------|---------------|------------|
| PDF Chatbot | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | ❌ | Low |
| Hybrid Search | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ❌ | Medium |
| Vision RAG | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ | High |
| LangChain Multimodal | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ | High |
| LightRAG | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ | Very High |
| Knowledge Graph | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ | Very High |

## 🎓 Learning Path

**Beginner → Intermediate → Advanced → Expert**

1. Start with **PDF Chatbot** to understand basic RAG concepts
2. Move to **Hybrid Search RAG** to learn about retrieval strategies
3. Explore **Vision RAG** for multimodal understanding
4. Try **LangChain Multimodal RAG** for production-ready systems
5. Master **LightRAG** and **Knowledge Graph LLMs** for advanced applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution
- New RAG techniques and implementations
- Performance optimizations
- Documentation improvements
- Bug fixes and testing
- Additional examples and tutorials

## 📝 Project Structure

```
Rag/
├── .env                          # API keys configuration
├── .gitignore                    # Git ignore rules
├── .venv/                        # Virtual environment (UV)
├── README.md                     # This file
├── pyproject.toml                # Project dependencies (UV)
├── uv.lock                       # Dependency lock file
│
├── pdf chatbot/                  # Basic PDF Q&A
│   ├── Ask A Book Questions.ipynb
│   └── README.md
│
├── Hybrid Search Rag/            # Hybrid retrieval + Vision RAG
│   ├── Advanced_RAG_Hybrid_Search_RAG.ipynb
│   ├── Vision_RAG.ipynb
│   └── README.md
│
├── multimodal rag/               # Advanced multimodal with tables
│   └── multimodal advance/
│       ├── langchain_multimodal.ipynb
│       ├── langchain_multimodal (1).ipynb
│       ├── content/
│       │   └── attention.pdf
│       └── README.md
│
├── LightRAG-main/                # Graph-based RAG
│   ├── examples/
│   ├── lightrag/
│   ├── README.md
│   └── ... (see LightRAG docs)
│
└── knowledge-graph-llms-main/    # Knowledge graph integration
    ├── examples/
    ├── README.md
    └── ...
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors with `unstructured`:**
- Ensure Python 3.11+ is installed
- Install system dependencies (Poppler, Tesseract)
- Use compatible version: `uv pip install "unstructured[pdf]==0.15.13"`

**2. CUDA/GPU Issues:**
- Install appropriate PyTorch version for your CUDA version
- For CPU-only: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`

**3. API Rate Limits:**
- Use Groq for faster, free inference
- Implement caching for embeddings
- Use batch processing for large documents

**4. Memory Issues:**
- Use chunking for large documents
- Reduce batch size in embedding generation
- Use quantized models when possible

## 📚 Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [Unstructured Docs](https://docs.unstructured.io/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Groq Docs](https://console.groq.com/docs)

### Tutorials
- [RAG from Scratch (LangChain)](https://python.langchain.com/docs/tutorials/rag/)
- [Building Multimodal RAG](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [Graph RAG Overview](https://microsoft.github.io/graphrag/)

### Papers
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Transformer architecture
- ["Retrieval-Augmented Generation"](https://arxiv.org/abs/2005.11401) - RAG paper
- ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) - CLIP paper

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- LangChain team for the excellent framework
- Unstructured.io for document processing tools
- OpenAI, Groq, and other LLM providers
- Open-source community for various libraries and tools

## 📧 Contact

- **Repository Owner:** AnandaRimal
- **Repository:** https://github.com/AnandaRimal/Rag
- **Issues:** https://github.com/AnandaRimal/Rag/issues

---

**⭐ If you find this repository helpful, please consider giving it a star!**

Last Updated: November 6, 2025
