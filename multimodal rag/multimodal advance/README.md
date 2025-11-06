# LangChain Multimodal RAG with Table & Image Extraction

Production-ready multimodal RAG system using the Unstructured library for advanced PDF parsing with automatic table detection, image extraction, and OCR capabilities.

## 📋 Overview

This project demonstrates a state-of-the-art multimodal RAG system that can:
- Extract and parse text from PDFs with high accuracy
- Detect and extract table structures
- Extract and encode images from documents
- Perform OCR on images containing text
- Generate separate summaries for text, tables, and images
- Implement multi-vector retrieval for optimal relevance

**Sample Document**: `attention.pdf` (Attention Is All You Need - Transformer paper)

## ✨ Key Features

- **High-Resolution PDF Parsing**: Uses `hi_res` strategy for maximum accuracy
- **Table Structure Inference**: Automatically detects and preserves table structure
- **Image Extraction**: Extracts images and converts to base64 for API usage
- **OCR Integration**: Tesseract OCR for text in images
- **Separate Summarization**: Different strategies for text, tables, and images
- **Multi-Vector Retrieval**: Stores summaries as vectors, retrieves original content
- **Dual LLM Support**: Groq (fast, free) for text/tables, OpenAI for images
- **Production Ready**: Full error handling and optimization

## 🏗️ Architecture

```
┌──────────────────┐
│     PDF File     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│   Unstructured Partition     │
│   (hi_res strategy)          │
│   - Poppler (PDF→Images)     │
│   - Tesseract (OCR)          │
│   - Table Transformer        │
└────────┬─────────────────────┘
         │
    ┌────┴────────┐
    │             │
    ▼             ▼
┌────────┐   ┌──────────────┐
│  Text  │   │ Tables + Imgs│
│ Chunks │   │              │
└───┬────┘   └──────┬───────┘
    │               │
    ▼               ▼
┌────────┐   ┌──────────────┐
│ Groq   │   │ GPT-4 Vision │
│Summary │   │   Summary    │
└───┬────┘   └──────┬───────┘
    │               │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Multi-Vector  │
    │   ChromaDB    │
    │               │
    │ - Summary Vec │
    │ - Original Doc│
    └───────┬───────┘
            │
            ▼
      ┌──────────┐
      │  Query   │
      └─────┬────┘
            │
            ▼
    ┌───────────────┐
    │   Retrieval   │
    │ (summaries)   │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Return      │
    │ (original)    │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ GPT-4 Vision  │
    │    Answer     │
    └───────────────┘
```

## 🚀 Getting Started

### Prerequisites

**Software Requirements:**
- Python 3.11 or higher (required for `unstructured` library)
- UV package manager (recommended) or pip
- **Poppler** (PDF rendering)
- **Tesseract OCR** (text extraction from images)

**API Keys Required:**
- OpenAI (for image summarization with GPT-4 Vision)
- Groq (for text/table summarization - fast and free)

### System Setup

#### Windows

1. **Install Poppler:**
```powershell
winget install oschwartz10612.Poppler --accept-source-agreements --accept-package-agreements
```

2. **Install Tesseract OCR:**
```powershell
winget install UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
```

3. **Refresh PATH:**
```powershell
# Restart terminal or run:
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### Linux

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libmagic-dev
```

#### macOS

```bash
brew install poppler tesseract libmagic
```

### Python Dependencies

1. **Navigate to project directory:**
```bash
cd "multimodal rag/multimodal advance"
```

2. **Create virtual environment:**
```bash
# Using UV (recommended)
uv venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Or using standard Python
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Core RAG libraries
uv pip install langchain langchain-community langchain-openai langchain-groq
uv pip install chromadb tiktoken python-dotenv

# Unstructured with PDF support
uv pip install "unstructured[pdf]==0.15.13"

# PDF and image processing
uv pip install "pdfminer.six==20221105"
uv pip install pillow pdf2image pypdf

# Deep learning for table detection
uv pip install layoutparser torch torchvision
uv pip install opencv-python-headless

# Additional utilities
uv pip install pandas tabulate
```

**Note**: Use `unstructured==0.15.13` for Python 3.11/3.12 compatibility.

### Configuration

Create `.env` file in project root:

```env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here  # Optional, for Gemini
COHERE_API_KEY=your_cohere_key_here  # Optional, for reranking
```

### Quick Start

1. **Launch Jupyter:**
```bash
jupyter notebook "langchain_multimodal (1).ipynb"
```

2. **Place your PDF:**
   - Put your PDF in the project directory or `content/` folder
   - Update `file_path` in cell 8

3. **Run the notebook cells sequentially**

## 📖 How It Works

### Step 1: PDF Partitioning with Unstructured

```python
from unstructured.partition.pdf import partition_pdf
import os

# Configure system paths for Windows
poppler_path = r"C:\Users\...\poppler\Library\bin"
tesseract_path = r"C:\Program Files\Tesseract-OCR"
os.environ["PATH"] = poppler_path + os.pathsep + tesseract_path + os.pathsep + os.environ["PATH"]

# Partition PDF with hi_res strategy
chunks = partition_pdf(
    filename="attention.pdf",
    infer_table_structure=True,            # Detect tables
    strategy="hi_res",                      # High accuracy mode
    extract_image_block_types=["Image"],   # Extract images
    extract_image_block_to_payload=True,   # Base64 encode
    chunking_strategy="by_title",          # Smart chunking
    max_characters=10000,
    combine_text_under_n_chars=2000,
)
```

**What happens internally:**
1. PDF is converted to images using Poppler
2. Layout analysis identifies text blocks, tables, images
3. Table Transformer model infers table structure
4. Tesseract OCR extracts text from images
5. Content is chunked by semantic sections

### Step 2: Separate Elements by Type

```python
# Categorize extracted elements
tables = []
texts = []
images = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)
        
        # Extract images from composite elements
        for el in chunk.metadata.orig_elements:
            if "Image" in str(type(el)):
                images.append(el.metadata.image_base64)
```

### Step 3: Generate Summaries

**For Text and Tables (using Groq):**
```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Fast, free LLM for text/tables
model = ChatGroq(
    temperature=0.5, 
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Table or text chunk: {element}
""")

summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Batch process for speed
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
table_summaries = summarize_chain.batch(
    [table.metadata.text_as_html for table in tables],
    {"max_concurrency": 3}
)
```

**For Images (using GPT-4 Vision):**
```python
from langchain_openai import ChatOpenAI

prompt_template = """Describe the image in detail. For context,
the image is part of a research paper explaining the transformers
architecture. Be specific about graphs, such as bar plots."""

messages = [
    ("user", [
        {"type": "text", "text": prompt_template},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image}"},
        },
    ])
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

image_summaries = chain.batch(images)
```

### Step 4: Create Multi-Vector Store

```python
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Vector store for summaries
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=OpenAIEmbeddings()
)

# Storage for original documents
store = InMemoryStore()

# Multi-vector retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id",
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) 
    for i, summary in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables and images similarly
```

**Why Multi-Vector Retrieval?**
- Summaries are indexed (better for search)
- Original documents are retrieved (better for context)
- Best of both worlds: search efficiency + full context

### Step 5: Query with Context

```python
def parse_docs(docs):
    """Separate base64 images from text"""
    b64_images = []
    texts = []
    for doc in docs:
        try:
            b64decode(doc)
            b64_images.append(doc)
        except:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def build_prompt(kwargs):
    """Build multimodal prompt"""
    context = kwargs["context"]
    question = kwargs["question"]
    
    # Combine text context
    text_context = ""
    for text_element in context["texts"]:
        text_context += text_element.text
    
    # Create prompt
    prompt_content = [{
        "type": "text",
        "text": f"""Answer based on the following context:
        Context: {text_context}
        Question: {question}"""
    }]
    
    # Add images to prompt
    for image in context["images"]:
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })
    
    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content)
    ])

# Create RAG chain
chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Query
response = chain.invoke("What is the attention mechanism?")
```

## ⚙️ Configuration Options

### Chunking Strategy

```python
# By title (semantic sections)
chunking_strategy="by_title"

# Basic (fixed size)
chunking_strategy="basic"

# No chunking
chunking_strategy=None
```

### Table Detection

```python
# Enable table structure inference
infer_table_structure=True  # Recommended

# Disable for faster processing
infer_table_structure=False
```

### Image Extraction

```python
# Extract all images
extract_image_block_types=["Image"]

# Extract tables as images too
extract_image_block_types=["Image", "Table"]

# No image extraction
extract_image_block_types=[]
```

### Model Selection

```python
# For text/tables - choose based on speed/quality
model = ChatGroq(model="llama-3.1-8b-instant")    # Fast, good quality
model = ChatGroq(model="llama-3.1-70b-versatile") # Slower, better quality
model = ChatGroq(model="mixtral-8x7b-32768")      # Large context window

# For images
model = ChatOpenAI(model="gpt-4o-mini")           # Cost-effective
model = ChatOpenAI(model="gpt-4o")                 # Best quality
```

## 📊 Performance & Costs

### Processing Time

| Document Size | Tables | Images | Processing Time | Notes |
|--------------|---------|---------|-----------------|-------|
| 10 pages | 2 | 5 | ~30 seconds | With hi_res |
| 50 pages | 10 | 20 | ~3 minutes | Parallel processing |
| 100 pages | 20 | 50 | ~8 minutes | May need batching |

**Optimization Tips:**
- Use `strategy="fast"` for text-only documents
- Reduce `max_characters` for faster chunking
- Increase `max_concurrency` for summarization
- Process large PDFs in batches

### Cost Estimation

**Per 100-page Research Paper:**

| Component | Cost | Notes |
|-----------|------|-------|
| PDF Parsing | Free | Local processing |
| Text Embeddings | ~$0.05 | OpenAI ada-002 |
| Text Summaries | Free | Groq (LLaMA) |
| Image Summaries (10 images) | ~$0.10 | GPT-4o-mini |
| Queries (10) | ~$0.02 | GPT-4o-mini |
| **Total** | **~$0.17** | |

**Cost Reduction:**
- Use Gemini for image summaries (free tier available)
- Use local embeddings (HuggingFace)
- Cache summaries for repeated processing

## 🎯 Use Cases

### 1. Research Paper Analysis
```python
# Extract methodology, results, figures
query = "What was the main innovation in this paper?"
```

### 2. Financial Reports
```python
# Extract tables, charts, financial data
query = "What were the Q4 revenue figures?"
```

### 3. Technical Documentation
```python
# Extract diagrams, code snippets, procedures
query = "Show me the system architecture diagram"
```

### 4. Legal Documents
```python
# Extract clauses, tables, signatures
query = "What are the termination conditions?"
```

## 🐛 Troubleshooting

### Issue: `PDFInfoNotInstalledError`

**Cause**: Poppler not found in PATH

**Solution:**
```python
# Add Poppler to PATH in notebook
import os
poppler_path = r"C:\...\poppler\Library\bin"
os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
```

### Issue: `TesseractNotFoundError`

**Cause**: Tesseract not installed or not in PATH

**Solution:**
```bash
# Windows
winget install UB-Mannheim.TesseractOCR

# Then add to PATH or specify in code
```

### Issue: `ImportError` with unstructured

**Cause**: Version incompatibility

**Solution:**
```bash
# Use compatible version
uv pip install --force-reinstall "unstructured[pdf]==0.15.13"
uv pip install "pdfminer.six==20221105"
```

### Issue: Out of memory with large PDFs

**Solution:**
```python
# Process in pages
import fitz
doc = fitz.open("large.pdf")

for page_num in range(0, len(doc), 10):  # Batch of 10
    pages = doc[page_num:page_num+10]
    # Process batch
    ...
```

### Issue: Poor table extraction

**Solution:**
```python
# Try different strategies
strategy="hi_res"  # Better accuracy
infer_table_structure=True  # Enable table detection

# Or extract as images
extract_image_block_types=["Image", "Table"]
```

## 📚 Advanced Features

### Custom Summarization Prompts

```python
# Customize for domain-specific content
medical_prompt = """Summarize this medical content focusing on:
- Diagnoses and findings
- Treatment recommendations  
- Patient outcomes
Content: {element}
"""

legal_prompt = """Summarize this legal content focusing on:
- Key obligations
- Deadlines and dates
- Parties involved
Content: {element}
"""
```

### Metadata Filtering

```python
# Add metadata during chunking
chunks = partition_pdf(
    filename="document.pdf",
    metadata_filename="research_paper.pdf",
    ...
)

# Filter during retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "research_paper.pdf"}
    }
)
```

### Async Processing

```python
import asyncio

async def process_pdf_async(pdf_path):
    """Process PDF asynchronously"""
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(
        None, partition_pdf, pdf_path
    )
    return chunks

# Process multiple PDFs concurrently
pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = await asyncio.gather(*[
    process_pdf_async(pdf) for pdf in pdfs
])
```

## 🔬 Comparison with Other Methods

| Feature | This Project | Vision RAG | Basic RAG |
|---------|--------------|------------|-----------|
| Table Detection | ✅ Structured | ❌ | ❌ |
| Image Understanding | ✅ Base64 + OCR | ✅ CLIP | ❌ |
| Text Quality | ✅ High (Unstructured) | ⚠️ Medium | ⚠️ Medium |
| Setup Complexity | ⚠️ High | ⚠️ High | ✅ Low |
| Processing Speed | ⚠️ Slow (hi_res) | ⚠️ Medium | ✅ Fast |
| Accuracy | ✅ Excellent | ✅ Good | ⚠️ Basic |
| Cost | ⚠️ Medium | ⚠️ Medium | ✅ Low |

## 📈 Best Practices

### 1. Choose Right Strategy

```python
# For text-heavy documents
strategy="fast"  # Faster, less accurate

# For documents with tables/complex layouts
strategy="hi_res"  # Slower, more accurate
```

### 2. Optimize Batch Processing

```python
# Process summaries in batches
text_summaries = summarize_chain.batch(
    texts,
    {"max_concurrency": 5}  # Adjust based on API limits
)
```

### 3. Cache Results

```python
import diskcache

cache = diskcache.Cache('./cache')

@cache.memoize()
def get_pdf_chunks(pdf_path):
    return partition_pdf(filename=pdf_path, ...)

# Subsequent calls use cache
chunks = get_pdf_chunks("attention.pdf")  # Fast!
```

### 4. Monitor Costs

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.invoke("question")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Total Cost: ${cb.total_cost}")
```

## 🎓 Learning Outcomes

After completing this project, you will understand:

- ✅ Advanced PDF parsing with Unstructured library
- ✅ Table structure inference and extraction
- ✅ Multi-modal content processing
- ✅ Multi-vector retrieval strategies
- ✅ System dependencies management (Poppler, Tesseract)
- ✅ Production-ready error handling
- ✅ Cost optimization strategies
- ✅ Batch processing and async operations

## 📚 Next Steps

1. **Add Reranking**: Implement Cohere reranking for better results
2. **Graph Integration**: Combine with knowledge graphs (see [LightRAG](../../LightRAG-main/))
3. **Fine-tune Models**: Train custom table detection models
4. **Deploy API**: Create FastAPI endpoint for production use

## 🔗 Resources

### Documentation
- [Unstructured Docs](https://docs.unstructured.io/)
- [LangChain Multi-Vector](https://python.langchain.com/docs/how_to/multi_vector/)
- [Poppler Utils](https://poppler.freedesktop.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LayoutLM](https://arxiv.org/abs/1912.13318) - Document understanding
- [Table Transformer](https://arxiv.org/abs/2110.00061) - Table detection

### Tutorials
- [Building Multimodal RAG](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [Unstructured Tutorial](https://docs.unstructured.io/open-source/core-functionality/partitioning)

---

**⭐ This is the most advanced RAG implementation in the repository! Perfect for production use cases.**

[← Back to Main README](../../README.md)

Last Updated: November 6, 2025
