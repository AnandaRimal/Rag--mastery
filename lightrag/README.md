# LightRAG - Graph-Enhanced RAG with Knowledge Graphs

Advanced Retrieval-Augmented Generation system that combines traditional vector search with knowledge graph reasoning for superior contextual understanding.

## 📋 Overview

LightRAG is a next-generation RAG system that builds and queries knowledge graphs from your documents. Unlike traditional RAG which only stores text chunks, LightRAG:

- **Extracts Entities & Relationships** from documents
- **Builds Knowledge Graphs** automatically
- **Performs Graph-Based Reasoning** during retrieval
- **Supports Multiple Query Modes** (naive, local, global, hybrid)
- **Enables Community Detection** for hierarchical understanding

**What makes it special**: It understands not just what documents say, but how concepts relate to each other.

## ✨ Key Features

- **Automatic Knowledge Graph Construction**: Extracts entities and relationships using LLMs
- **Graph Storage Options**: NetworkX (local), Neo4j (production), MongoDB
- **4 Query Modes**:
  - `naive`: Simple vector search
  - `local`: Entity-centric local graph search
  - `global`: Community-based global understanding
  - `hybrid`: Best of both local + global
- **Multiple LLM Support**: OpenAI, Ollama, Gemini, Azure OpenAI
- **Streaming Responses**: Real-time answer generation
- **Async/Await Support**: Production-ready async operations
- **Embedding Flexibility**: BM25, BGE, OpenAI, custom embeddings

## 🏗️ Architecture

```
┌──────────────┐
│  Documents   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│   LLM Processing         │
│   - Entity Extraction    │
│   - Relation Extraction  │
│   - Community Detection  │
└──────┬───────────────────┘
       │
   ┌───┴────┐
   │        │
   ▼        ▼
┌─────┐  ┌──────────────┐
│Vec  │  │Knowledge     │
│Store│  │Graph         │
│     │  │- Entities    │
│     │  │- Relations   │
│     │  │- Communities │
└──┬──┘  └──────┬───────┘
   │            │
   └─────┬──────┘
         │
         ▼
  ┌──────────────┐
  │   Query      │
  └──────┬───────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌──────┐   ┌──────┐
│Local │   │Global│
│Graph │   │Comm. │
└───┬──┘   └───┬──┘
    │          │
    └────┬─────┘
         │
         ▼
   ┌──────────┐
   │ Answer   │
   │+ Graph   │
   │Context   │
   └──────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Ollama (for local LLMs) or API keys for cloud LLMs
- Optional: Neo4j for production graph storage

### Installation

1. **Install LightRAG:**
```bash
pip install lightrag-hku
```

2. **Install dependencies:**
```bash
pip install aiohttp
pip install numpy networkx
pip install nano-vectordb  # For vector storage
pip install ollama  # If using Ollama

# Optional: For Neo4j support
pip install neo4j

# Optional: For advanced embeddings
pip install sentence-transformers
```

3. **Setup Ollama (Local LLMs):**
```bash
# Install Ollama from https://ollama.ai

# Pull models
ollama pull qwen2:latest      # Main LLM
ollama pull bge-m3:latest     # Embeddings
```

### Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./demo",
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 8192},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3:latest",
                host="http://localhost:11434"
            ),
        ),
    )
    
    # Initialize storage
    await rag.initialize_storages()
    
    # Insert documents
    with open("document.txt", "r") as f:
        await rag.ainsert(f.read())
    
    # Query (hybrid mode - best results)
    response = await rag.aquery(
        "What are the main concepts?",
        param=QueryParam(mode="hybrid")
    )
    print(response)

asyncio.run(main())
```

## 📖 Detailed Usage

### Query Modes Explained

#### 1. Naive Mode
Traditional vector similarity search only.

```python
# Fast but limited context
response = await rag.aquery(
    "What is quantum computing?",
    param=QueryParam(mode="naive")
)
```

**When to use**: Simple fact retrieval, speed is critical.

#### 2. Local Mode
Entity-centric search using local graph neighborhoods.

```python
# Explores relationships around specific entities
response = await rag.aquery(
    "How does PayPal relate to Peter Thiel?",
    param=QueryParam(mode="local")
)
```

**When to use**: Questions about specific entities and their connections.

**How it works**:
1. Extract entities from query
2. Find entities in graph
3. Explore 1-2 hop neighborhoods
4. Aggregate connected information

#### 3. Global Mode
Community-based search for high-level understanding.

```python
# Uses community summaries for broad topics
response = await rag.aquery(
    "What are the main themes in the book?",
    param=QueryParam(mode="global")
)
```

**When to use**: Broad questions, summarization, thematic queries.

**How it works**:
1. Documents grouped into communities (clustering)
2. Each community gets AI-generated summary
3. Query searches community summaries
4. Returns hierarchical understanding

#### 4. Hybrid Mode (Recommended)
Combines local + global for comprehensive answers.

```python
# Best of both worlds
response = await rag.aquery(
    "Explain the PayPal Mafia's impact on Silicon Valley",
    param=QueryParam(mode="hybrid")
)
```

**When to use**: Complex questions requiring both detail and context.

### Streaming Responses

```python
import inspect

response = await rag.aquery(
    "Question here",
    param=QueryParam(mode="hybrid", stream=True)
)

if inspect.isasyncgen(response):
    async for chunk in response:
        print(chunk, end="", flush=True)
else:
    print(response)
```

### Working with Different LLMs

#### OpenAI

```python
from lightrag.llm.openai import openai_complete_if_cache, openai_embedding
import os

os.environ["OPENAI_API_KEY"] = "your_key_here"

rag = LightRAG(
    working_dir="./openai_demo",
    llm_model_func=openai_complete_if_cache,
    llm_model_name="gpt-4o-mini",
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: openai_embedding(
            texts,
            model="text-embedding-3-small"
        ),
    ),
)
```

#### Gemini

```python
from lightrag.llm.gemini import gemini_complete, gemini_embed
import os

os.environ["GOOGLE_API_KEY"] = "your_key_here"

rag = LightRAG(
    working_dir="./gemini_demo",
    llm_model_func=gemini_complete,
    llm_model_name="gemini-1.5-flash",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: gemini_embed(
            texts,
            model="models/text-embedding-004"
        ),
    ),
)
```

#### Azure OpenAI

```python
from lightrag.llm.azure_openai import azure_openai_complete, azure_openai_embedding
import os

os.environ["AZURE_OPENAI_API_KEY"] = "your_key_here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-endpoint.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"

rag = LightRAG(
    working_dir="./azure_demo",
    llm_model_func=azure_openai_complete,
    llm_model_name="gpt-4o-mini",
    llm_model_kwargs={
        "deployment_name": "your-deployment-name"
    },
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: azure_openai_embedding(
            texts,
            deployment_name="your-embedding-deployment"
        ),
    ),
)
```

### Advanced Configuration

#### Custom Entity Extraction

```python
rag = LightRAG(
    working_dir="./custom",
    llm_model_func=ollama_model_complete,
    # Entity extraction settings
    entity_extract_max_gleaning=1,  # Multi-pass extraction
    entity_summary_to_max_tokens=500,
    # Graph settings
    graph_storage="NetworkXStorage",  # or "Neo4JStorage"
    # Chunking
    chunk_size=1200,
    chunk_overlap_size=100,
)
```

#### Neo4j Graph Storage

```python
from lightrag.kg.neo4j_impl import Neo4JStorage

rag = LightRAG(
    working_dir="./neo4j_demo",
    llm_model_func=ollama_model_complete,
    graph_storage="Neo4JStorage",
    graph_storage_kwargs={
        "url": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "your_password",
    },
)
```

#### MongoDB Graph Storage

```python
from lightrag.kg.mongodb_impl import MongoDBStorage

rag = LightRAG(
    working_dir="./mongo_demo",
    llm_model_func=ollama_model_complete,
    graph_storage="MongoDBStorage",
    graph_storage_kwargs={
        "connection_string": "mongodb://localhost:27017",
        "database_name": "lightrag",
    },
)
```

## 🎯 Example Use Cases

### 1. Research Paper Analysis

```python
# Insert research papers
papers = ["paper1.txt", "paper2.txt", "paper3.txt"]
for paper in papers:
    with open(paper) as f:
        await rag.ainsert(f.read())

# Query with global mode for themes
response = await rag.aquery(
    "What are the common methodologies across these papers?",
    param=QueryParam(mode="global")
)

# Query with local mode for specific relationships
response = await rag.aquery(
    "How does the attention mechanism relate to transformers?",
    param=QueryParam(mode="local")
)
```

### 2. Book Understanding

```python
# Insert book (the dickens example)
with open("book.txt") as f:
    await rag.ainsert(f.read())

# Global queries for themes
await rag.aquery(
    "What are the main themes in the book?",
    param=QueryParam(mode="global")
)

# Local queries for character relationships
await rag.aquery(
    "How is Pip related to Magwitch?",
    param=QueryParam(mode="local")
)
```

### 3. Corporate Knowledge Base

```python
# Insert company docs
docs = ["handbook.txt", "policies.txt", "procedures.txt"]
for doc in docs:
    with open(doc) as f:
        await rag.ainsert(f.read())

# Hybrid mode for complex questions
response = await rag.aquery(
    "What is the process for requesting time off and how does it relate to project deadlines?",
    param=QueryParam(mode="hybrid")
)
```

### 4. Technical Documentation

```python
# Insert API docs, tutorials, guides
await rag.ainsert(open("api_reference.txt").read())
await rag.ainsert(open("user_guide.txt").read())

# Local mode for specific API relationships
response = await rag.aquery(
    "What authentication methods work with the /users endpoint?",
    param=QueryParam(mode="local")
)
```

## 📊 Performance & Costs

### Query Mode Comparison

| Mode | Speed | Context | Cost | Best For |
|------|-------|---------|------|----------|
| Naive | ⚡ Fastest | ⭐ Limited | 💰 Lowest | Simple retrieval |
| Local | ⚡⚡ Fast | ⭐⭐⭐ Good | 💰💰 Medium | Specific entities |
| Global | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | 💰💰 Medium | Broad topics |
| Hybrid | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | 💰💰💰 Highest | Complex questions |

### Processing Costs (with Ollama - Local)

**FREE** - All processing runs locally!

- LLM calls: FREE (Ollama)
- Embeddings: FREE (local BGE model)
- Storage: FREE (local files/NetworkX)

### Processing Costs (with Cloud LLMs)

**Per 100-page book:**

| Component | OpenAI (gpt-4o-mini) | Gemini (flash) | Notes |
|-----------|---------------------|----------------|-------|
| Entity extraction | ~$1.50 | ~$0.30 | Most expensive |
| Relation extraction | ~$1.00 | ~$0.20 | |
| Embeddings | ~$0.05 | FREE | |
| Community summaries | ~$0.50 | ~$0.10 | One-time |
| Queries (10) | ~$0.10 | ~$0.02 | Per query |
| **Total** | **~$3.15** | **~$0.62** | |

**Cost Optimization:**
- Use Ollama for free local processing
- Use Gemini Flash (60x cheaper than GPT-4)
- Cache entity/relation extractions
- Reuse knowledge graphs across queries

## 🔬 Comparison with Other RAG Methods

| Feature | LightRAG | Traditional RAG | GraphRAG (Microsoft) |
|---------|----------|-----------------|---------------------|
| Graph Construction | ✅ Automatic | ❌ | ✅ Manual config |
| Entity Extraction | ✅ LLM-based | ❌ | ✅ LLM-based |
| Global Understanding | ✅ Communities | ❌ | ✅ Communities |
| Local Reasoning | ✅ Graph walks | ⚠️ Vector only | ✅ Graph queries |
| Setup Complexity | ✅ Simple | ✅ Simple | ⚠️ Complex |
| Query Speed | ⚠️ Slower | ✅ Fast | ⚠️ Slower |
| Context Quality | ✅ Excellent | ⚠️ Good | ✅ Excellent |
| Cost | ⚠️ Higher | ✅ Lower | ⚠️ Highest |

**When to use LightRAG:**
- ✅ Need to understand relationships between concepts
- ✅ Questions span multiple documents
- ✅ Want both detail and high-level understanding
- ✅ Can afford slightly higher latency for better answers

**When to use Traditional RAG:**
- ✅ Simple fact retrieval
- ✅ Speed is critical
- ✅ Budget constrained
- ✅ Documents are independent

## 🔧 Troubleshooting

### Issue: "Ollama connection refused"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify models are pulled
ollama list
```

### Issue: "Embedding dimension mismatch"

**Solution:**
```python
# Check actual embedding dimension
test_embedding = ollama_embed(
    ["test"],
    embed_model="bge-m3:latest"
)
print(f"Dimension: {len(test_embedding[0])}")

# Update in LightRAG init
embedding_func=EmbeddingFunc(
    embedding_dim=1024,  # Match actual dimension
    ...
)
```

### Issue: "Out of memory during graph construction"

**Solution:**
```python
# Process documents in batches
async def insert_batch(rag, texts, batch_size=5):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            await rag.ainsert(text)
        # Allow garbage collection
        await asyncio.sleep(1)
```

### Issue: "Neo4j connection failed"

**Solution:**
```bash
# Start Neo4j
neo4j start

# Check connection
curl http://localhost:7474

# Verify credentials in config
```

### Issue: "Slow entity extraction"

**Solution:**
```python
# Reduce gleaning passes
entity_extract_max_gleaning=0  # Single pass

# Use faster model
llm_model_name="llama3.2:latest"  # Smaller, faster

# Increase timeout
llm_model_kwargs={"timeout": 600}
```

## 📚 Advanced Features

### Graph Visualization

```python
# Generate HTML visualization
from examples.graph_visual_with_html import visualize_graph

visualize_graph(
    working_dir="./dickens",
    output_file="graph.html"
)
```

### Neo4j Visualization

```python
# Use Neo4j Browser for interactive exploration
# 1. Setup Neo4j storage (see above)
# 2. Open Neo4j Browser: http://localhost:7474
# 3. Run Cypher queries:

# View all entities
MATCH (n) RETURN n LIMIT 100

# View entity relationships
MATCH (a)-[r]->(b) 
WHERE a.name = "Peter Thiel"
RETURN a, r, b

# Community detection
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
```

### Custom Knowledge Graph Insertion

```python
from examples.insert_custom_kg import insert_custom_knowledge

# Insert pre-defined entities and relations
await insert_custom_knowledge(
    rag,
    entities=[
        {"name": "Python", "type": "Language"},
        {"name": "Django", "type": "Framework"},
    ],
    relations=[
        {"source": "Django", "target": "Python", "type": "WRITTEN_IN"},
    ]
)
```

### Query Generation

```python
from examples.generate_query import generate_queries

# Generate test queries from documents
queries = generate_queries(
    rag,
    num_queries=10,
    focus="technical concepts"
)

for query in queries:
    response = await rag.aquery(query, param=QueryParam(mode="hybrid"))
    print(f"Q: {query}\nA: {response}\n")
```

### Reranking

```python
from examples.rerank_example import rerank_results

# Rerank retrieved chunks before generation
response = await rerank_results(
    rag,
    query="What is machine learning?",
    top_k=10,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

## 🎓 Learning Outcomes

After working with LightRAG, you'll understand:

- ✅ Knowledge graph construction from unstructured text
- ✅ Entity and relationship extraction using LLMs
- ✅ Community detection algorithms (Louvain)
- ✅ Graph-based retrieval strategies
- ✅ Hybrid search architectures
- ✅ Async Python programming patterns
- ✅ Production graph database integration (Neo4j)
- ✅ Cost optimization for graph-based RAG

## 📈 Best Practices

### 1. Choose Right Query Mode

```python
# For specific entity questions → local
await rag.aquery("How is X related to Y?", param=QueryParam(mode="local"))

# For broad themes → global
await rag.aquery("What are the main topics?", param=QueryParam(mode="global"))

# For complex multi-part questions → hybrid
await rag.aquery("Explain X and its impact on Y", param=QueryParam(mode="hybrid"))
```

### 2. Optimize Graph Construction

```python
# Balance accuracy vs speed
rag = LightRAG(
    entity_extract_max_gleaning=1,  # 0=fast, 2=accurate
    chunk_size=1200,                 # Larger=more context
    chunk_overlap_size=100,          # Prevent info loss
)
```

### 3. Use Appropriate Storage

```python
# Development → NetworkX
graph_storage="NetworkXStorage"

# Production (small) → MongoDB
graph_storage="MongoDBStorage"

# Production (large) → Neo4j
graph_storage="Neo4JStorage"
```

### 4. Cache Aggressively

```python
# Enable caching
rag = LightRAG(
    enable_llm_cache=True,  # Cache LLM responses
)

# Reuse knowledge graphs
# Don't rebuild if documents haven't changed
```

## 📚 Resources

### Documentation
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [NetworkX Docs](https://networkx.org/documentation/)
- [Neo4j Graph DB](https://neo4j.com/docs/)

### Papers
- [Graph RAG](https://arxiv.org/abs/2404.16130) - Microsoft's approach
- [Knowledge Graphs](https://arxiv.org/abs/2003.02320) - Survey paper
- [Community Detection](https://arxiv.org/abs/0803.0476) - Louvain algorithm

### Related Projects
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LangChain Graph](https://python.langchain.com/docs/use_cases/graph/)
- [LlamaIndex KG](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/)

---

**🌟 LightRAG represents the cutting edge of RAG technology - combining the simplicity of vector search with the reasoning power of knowledge graphs!**

[← Back to Main README](../README.md)

Last Updated: November 6, 2025
