# PDF Chatbot - Simple Question-Answering System

A beginner-friendly implementation of RAG (Retrieval-Augmented Generation) for querying PDF documents using LangChain and OpenAI.

## 📋 Overview

This project demonstrates the fundamentals of RAG by creating an interactive Q&A system that can answer questions about the content of PDF documents. It's perfect for beginners learning about vector databases, embeddings, and LLM-based question answering.

## ✨ Features

- **PDF Text Extraction**: Automatically extracts and processes text from PDF files
- **Vector Embeddings**: Converts text into semantic embeddings using OpenAI
- **Semantic Search**: Finds relevant context using similarity search in Chroma vector database
- **Conversational Interface**: Natural language Q&A powered by GPT models
- **Memory**: Maintains conversation history for contextual responses
- **Easy Setup**: Minimal configuration required

## 🏗️ Architecture

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Text Extraction │ (PyPDF/LangChain)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Text Chunking  │ (RecursiveCharacterTextSplitter)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Embeddings    │ (OpenAI text-embedding-ada-002)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vector Database  │ (Chroma)
└────────┬─────────┘
         │
         ▼
   ┌─────────┐
   │  Query  │
   └────┬────┘
        │
        ▼
┌──────────────────┐
│  Retrieval       │ (Similarity Search)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Generation  │ (GPT-3.5/GPT-4)
└────────┬─────────┘
         │
         ▼
    ┌────────┐
    │ Answer │
    └────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Jupyter Notebook

### Installation

1. **Navigate to the project directory:**
```bash
cd "pdf chatbot"
```

2. **Install dependencies:**
```bash
pip install langchain langchain-community langchain-openai
pip install chromadb tiktoken
pip install pypdf  # or PyPDF2
pip install jupyter ipykernel
```

Or using UV:
```bash
uv pip install langchain langchain-community langchain-openai chromadb tiktoken pypdf jupyter ipykernel
```

3. **Set up OpenAI API key:**

Create a `.env` file in the project root (or use parent directory's .env):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Quick Start

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook "Ask A Book Questions.ipynb"
```

2. **Follow the notebook steps:**
   - Upload or specify your PDF file path
   - Run the cells to load and process the PDF
   - Ask questions in the interactive interface

## 📖 How It Works

### Step 1: Document Loading

```python
from langchain.document_loaders import PyPDFLoader

# Load PDF document
loader = PyPDFLoader("path/to/your/document.pdf")
documents = loader.load()
```

### Step 2: Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each chunk
    chunk_overlap=200,      # Overlap between chunks
    length_function=len
)

chunks = text_splitter.split_documents(documents)
```

**Why split documents?**
- LLMs have token limits
- Smaller chunks improve retrieval accuracy
- Reduces processing time and costs

### Step 3: Create Embeddings

```python
from langchain.embeddings import OpenAIEmbeddings

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"  # OpenAI's embedding model
)
```

**About Embeddings:**
- Convert text into numerical vectors (1536 dimensions)
- Capture semantic meaning
- Enable similarity search

### Step 4: Store in Vector Database

```python
from langchain.vectorstores import Chroma

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Optional: persist to disk
)
```

### Step 5: Create Retrieval Chain

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0  # Deterministic responses
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # How to combine retrieved docs
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Number of relevant chunks
    ),
    return_source_documents=True  # Include sources in response
)
```

### Step 6: Ask Questions

```python
# Query the system
question = "What is the main topic of the document?"
result = qa_chain({"query": question})

print("Answer:", result['result'])
print("Sources:", result['source_documents'])
```

## 📊 Configuration Options

### Text Splitting Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `chunk_size` | 1000 | Characters per chunk | 500-2000 depending on content |
| `chunk_overlap` | 200 | Overlap between chunks | 10-20% of chunk_size |
| `separator` | "\n\n" | Split on paragraphs | Use "\n" for denser text |

### Retrieval Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `k` | 4 | Number of chunks to retrieve | 3-5 for focused, 5-10 for comprehensive |
| `search_type` | similarity | Type of search | Use "mmr" for diversity |
| `score_threshold` | None | Minimum similarity score | 0.7-0.8 for high precision |

### LLM Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `temperature` | 0 | Creativity (0-1) | 0 for factual, 0.7 for creative |
| `max_tokens` | 256 | Response length | Adjust based on needs |
| `model_name` | gpt-3.5-turbo | OpenAI model | gpt-4 for better quality |

## 🎯 Use Cases

### 1. Research Assistant
- Query academic papers
- Extract key findings
- Summarize sections

### 2. Legal Document Review
- Search contracts
- Find specific clauses
- Compare terms

### 3. Technical Documentation
- Query API documentation
- Find code examples
- Troubleshoot issues

### 4. Educational Content
- Study from textbooks
- Generate quiz questions
- Explain complex topics

## 🔧 Customization

### Using Different PDF Loaders

**PyMuPDF (faster, better extraction):**
```python
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("document.pdf")
documents = loader.load()
```

**Unstructured (handles complex layouts):**
```python
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("document.pdf")
documents = loader.load()
```

### Adding Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Add memory to maintain context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Now it remembers previous questions
result1 = qa_chain({"question": "What is RAG?"})
result2 = qa_chain({"question": "Can you explain that in simpler terms?"})
```

### Custom Prompt Template

```python
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: Let me help you with that."""

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)
```

## 📈 Performance Tips

### 1. Optimize Chunk Size
- **Smaller chunks (500-800)**: Better for precise answers
- **Larger chunks (1500-2000)**: Better for context-heavy questions

### 2. Use Filters
```python
# Filter by metadata
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "specific_document.pdf"}
    }
)
```

### 3. Batch Processing
```python
# Process multiple PDFs
import glob

pdf_files = glob.glob("documents/*.pdf")
all_documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    all_documents.extend(docs)

# Create vectorstore from all documents
vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings
)
```

### 4. Caching
```python
from langchain.cache import InMemoryCache
import langchain

# Enable caching to save on API calls
langchain.llm_cache = InMemoryCache()
```

## 🐛 Troubleshooting

### Issue: PDF Not Loading

**Solution:**
```python
# Try different loader
from langchain.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("document.pdf")
documents = loader.load()
```

### Issue: Poor Answer Quality

**Solutions:**
1. Increase `k` (retrieve more chunks)
2. Use `gpt-4` instead of `gpt-3.5-turbo`
3. Adjust chunk size and overlap
4. Add custom prompt template

### Issue: API Rate Limits

**Solutions:**
1. Add delays between requests
2. Use local embeddings (HuggingFace)
3. Implement exponential backoff

```python
from langchain.embeddings import HuggingFaceEmbeddings

# Use local embeddings (no API calls)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Issue: High Costs

**Solutions:**
1. Use `gpt-3.5-turbo` instead of `gpt-4`
2. Reduce `max_tokens` in responses
3. Implement caching
4. Use local models with Ollama

## 💰 Cost Estimation

### OpenAI Pricing (as of Nov 2025)

**Embeddings (text-embedding-ada-002):**
- $0.0001 per 1K tokens
- Average book (~100K words) ≈ 130K tokens = $0.013

**GPT-3.5-Turbo:**
- Input: $0.0015 per 1K tokens
- Output: $0.002 per 1K tokens
- Average query (4 chunks + question) ≈ 1.5K tokens input, 500 tokens output = $0.0035

**GPT-4:**
- Input: $0.03 per 1K tokens  
- Output: $0.06 per 1K tokens
- Average query = $0.075

**Example: Processing 10 PDFs (1000 pages total)**
- Embeddings: ~$1-2
- Queries (100 with GPT-3.5): ~$0.35
- **Total: ~$1.50-2.50**

## 📚 Next Steps

After mastering this basic implementation, explore:

1. **[Hybrid Search RAG](../Hybrid%20Search%20Rag/)** - Combine keyword and semantic search
2. **[Vision RAG](../Hybrid%20Search%20Rag/Vision_RAG.ipynb)** - Add image understanding
3. **[LangChain Multimodal](../multimodal%20rag/)** - Process tables and images
4. **[LightRAG](../LightRAG-main/)** - Graph-based retrieval

## 🔗 Resources

### Documentation
- [LangChain PDF Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
- [Chroma Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### Tutorials
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Building Q&A Systems](https://python.langchain.com/docs/use_cases/question_answering/)

### Papers
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)

## 📄 Example Notebook Structure

The `Ask A Book Questions.ipynb` notebook includes:

1. **Setup & Installation** - Install required packages
2. **API Configuration** - Set up OpenAI API key
3. **PDF Loading** - Load and inspect the PDF
4. **Text Processing** - Split into chunks
5. **Embedding Creation** - Generate embeddings
6. **Vector Store** - Create Chroma database
7. **QA Chain Setup** - Configure retrieval chain
8. **Interactive Q&A** - Query interface
9. **Evaluation** - Test with sample questions
10. **Cleanup** - Optional database persistence

## 🎓 Learning Outcomes

After completing this project, you will understand:

- ✅ How RAG works fundamentally
- ✅ Vector embeddings and similarity search
- ✅ Document chunking strategies
- ✅ LangChain framework basics
- ✅ Vector database operations
- ✅ LLM prompt engineering
- ✅ Cost optimization techniques

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- Additional PDF loaders
- Better error handling
- UI improvements
- Performance optimizations
- More examples

## 📝 License

MIT License - see main repository LICENSE file.

---

**⭐ Ready to build more advanced RAG systems? Check out the other projects in this repository!**

[← Back to Main README](../README.md)

Last Updated: November 6, 2025
