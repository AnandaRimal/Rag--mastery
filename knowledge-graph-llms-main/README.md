# Knowledge Graph Generator with LangChain & LLMs

Interactive Streamlit application for automatically extracting entities, relationships, and building beautiful knowledge graphs from any text using LangChain's graph transformers and OpenAI GPT-4o.

![Knowledge Graph Visualization](https://github.com/user-attachments/assets/4fef9158-8dd8-432d-bb8a-b53953a82c6c)

👉 **Video Tutorial**: [Watch on YouTube](https://www.youtube.com/watch?v=O-T_6KOXML4)

## 📋 Overview

Transform unstructured text into interactive, explorable knowledge graphs with automatic entity and relationship extraction. Perfect for:
- Research paper analysis
- Document understanding
- Knowledge base visualization
- Concept mapping
- Information discovery

**What makes it special**: One-click conversion from text to interactive graph with physics-based visualization.

## ✨ Key Features

- **Dual Input Modes**: Upload `.txt` files or paste text directly
- **Automatic Extraction**: GPT-4o powered entity and relationship detection
- **Interactive Visualization**: Drag, zoom, filter nodes and edges
- **Physics-Based Layout**: ForceAtlas2 algorithm for natural graph positioning
- **Customizable Display**: Filter nodes/edges, adjust physics parameters
- **Web-Based UI**: Clean Streamlit interface, no installation required

## 🏗️ Architecture

```
┌──────────────┐
│  Text Input  │
│  (txt/paste) │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   LangChain      │
│   Document       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   GPT-4o         │
│   Entity Extract │
│   - Entities     │
│   - Relations    │
│   - Types        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Graph          │
│   Transformer    │
│   - Nodes        │
│   - Edges        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   PyVis          │
│   Visualization  │
│   - Force Layout │
│   - Interactive  │
└──────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (required for GPT-4o)
- **Basic knowledge**: Python, command line

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/knowledge-graph-llms
cd knowledge-graph-llms-main
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
langchain>=0.1.0
langchain-experimental>=0.0.45
langchain-openai>=0.1.0
python-dotenv>=1.0.0
pyvis>=0.3.2
streamlit>=1.32.0
```

3. **Configure OpenAI API:**

Create `.env` file:
```env
OPENAI_API_KEY=sk-your-api-key-here
```

### Quick Start

1. **Launch the app:**
```bash
streamlit run app.py
```

2. **Access the interface:**
   - Opens automatically at `http://localhost:8501`
   - Or manually navigate to the URL

3. **Create your first graph:**
   - Choose input method (Upload or Paste)
   - Add your text
   - Click "Generate Knowledge Graph"
   - Explore the interactive visualization!

## 📖 Detailed Usage

### Input Methods

#### Method 1: Upload Text File

```python
# Sidebar: Select "Upload txt"
# Click "Browse files"
# Choose .txt file from your computer
# Click "Generate Knowledge Graph"
```

**Example text file (`sample.txt`):**
```
Marie Curie was a Polish physicist and chemist who conducted 
pioneering research on radioactivity. She was the first woman 
to win a Nobel Prize, and remains the only person to win Nobel 
Prizes in two different sciences—Physics and Chemistry. She 
worked at the University of Paris alongside her husband Pierre Curie.
```

#### Method 2: Direct Text Input

```python
# Sidebar: Select "Input text"
# Paste or type text in the text area
# Click "Generate Knowledge Graph"
```

### Understanding the Output

**Nodes (Entities):**
- **Color-coded by type** (Person, Organization, Location, etc.)
- **Hover for details** (entity type, properties)
- **Drag to reposition**

**Edges (Relationships):**
- **Labeled with relationship type**
- **Directional arrows** show relationship flow
- **Hover for additional info**

### Interactive Features

```python
# Navigation
- Drag: Click and drag background
- Zoom: Mouse wheel
- Pan: Click node and drag

# Filtering (top-left menu)
- Filter by node type
- Hide/show specific relationships
- Search for entities

# Physics Controls
- Pause/resume layout
- Adjust spring constants
- Modify repulsion forces
```

## ⚙️ Configuration

### Customizing Graph Layout

Edit `generate_knowledge_graph.py`:

```python
net.set_options("""
{
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -100,    # Node repulsion
            "centralGravity": 0.01,           # Center pull
            "springLength": 200,              # Edge length
            "springConstant": 0.08            # Edge strength
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
    }
}
""")
```

**Parameter Guide:**

| Parameter | Default | Effect | When to Adjust |
|-----------|---------|--------|----------------|
| `gravitationalConstant` | -100 | Node repulsion | Nodes too close → more negative |
| `centralGravity` | 0.01 | Center attraction | Spread out → increase |
| `springLength` | 200 | Edge length | Edges too long/short → adjust |
| `springConstant` | 0.08 | Edge rigidity | Layout unstable → decrease |

### Changing LLM Model

```python
# In generate_knowledge_graph.py

# Current: GPT-4o (best quality)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Alternative: GPT-4o-mini (faster, cheaper)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Alternative: GPT-3.5-turbo (cheapest)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
```

### Customizing Appearance

```python
# In generate_knowledge_graph.py - Network() initialization

net = Network(
    height="1200px",        # Graph height
    width="100%",           # Graph width
    directed=True,          # Show arrow directions
    bgcolor="#222222",      # Dark background
    font_color="white",     # Text color
    filter_menu=True,       # Enable filtering
    cdn_resources='remote'  # Load vis.js from CDN
)
```

## 📊 Use Cases & Examples

### 1. Academic Research - Paper Analysis

**Input:**
```
The transformer architecture, introduced in "Attention is All You Need",
relies on self-attention mechanisms. It was developed by researchers at
Google Brain and has revolutionized NLP. Key components include multi-head
attention, positional encoding, and feed-forward networks. Vaswani et al.
demonstrated its effectiveness on machine translation tasks.
```

**Extracted Graph:**
- **Entities**: Transformer, Attention Mechanism, Google Brain, Vaswani, NLP, Machine Translation
- **Relations**: 
  - Transformer → DEVELOPED_BY → Google Brain
  - Transformer → USES → Attention Mechanism
  - Transformer → APPLICATION → Machine Translation

### 2. Business Intelligence - Company Relationships

**Input:**
```
Microsoft acquired GitHub in 2018 for $7.5 billion. GitHub is a platform
for software developers built on Git version control. Microsoft CEO Satya
Nadella emphasized the importance of the developer community. GitHub
maintains its independent operation under Microsoft's ownership.
```

**Extracted Graph:**
- **Entities**: Microsoft, GitHub, Satya Nadella, Git, Developers
- **Relations**:
  - Microsoft → ACQUIRED → GitHub
  - GitHub → BUILT_ON → Git
  - Satya Nadella → CEO_OF → Microsoft
  - GitHub → SERVES → Developers

### 3. Historical Analysis - Events & People

**Input:**
```
The Manhattan Project was a research program during World War II that
produced the first nuclear weapons. Led by physicist J. Robert Oppenheimer,
the project was based in Los Alamos, New Mexico. It involved collaboration
between the United States, United Kingdom, and Canada.
```

**Extracted Graph:**
- **Entities**: Manhattan Project, J. Robert Oppenheimer, Los Alamos, World War II, Nuclear Weapons
- **Relations**:
  - Manhattan Project → LED_BY → Oppenheimer
  - Manhattan Project → LOCATED_IN → Los Alamos
  - Manhattan Project → OCCURRED_DURING → World War II
  - Manhattan Project → PRODUCED → Nuclear Weapons

### 4. Technical Documentation - API Relationships

**Input:**
```
The REST API uses HTTP methods like GET, POST, PUT, and DELETE. Authentication
is handled via OAuth 2.0 tokens. Responses are returned in JSON format. The
API rate limit is 1000 requests per hour per user. Webhooks can be configured
for real-time notifications.
```

**Extracted Graph:**
- **Entities**: REST API, HTTP, OAuth 2.0, JSON, Webhooks
- **Relations**:
  - REST API → USES → HTTP
  - REST API → AUTHENTICATION → OAuth 2.0
  - REST API → RETURNS → JSON
  - REST API → SUPPORTS → Webhooks

## 📚 Advanced Features

### Programmatic Graph Generation

Use `generate_knowledge_graph.py` directly:

```python
from generate_knowledge_graph import generate_knowledge_graph

# Generate graph from text
text = "Your text here..."
net = generate_knowledge_graph(text)

# Graph saved automatically to knowledge_graph.html
```

### Async Graph Extraction

```python
import asyncio
from generate_knowledge_graph import extract_graph_data

async def process_documents(texts):
    """Process multiple documents concurrently"""
    tasks = [extract_graph_data(text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results

# Use it
texts = ["Document 1...", "Document 2...", "Document 3..."]
graph_docs = asyncio.run(process_documents(texts))
```

### Custom Node Filtering

```python
def filter_nodes_by_type(graph_documents, allowed_types):
    """Keep only specific entity types"""
    filtered_nodes = [
        node for node in graph_documents[0].nodes
        if node.type in allowed_types
    ]
    # Rebuild graph with filtered nodes
    # ...
    
# Example: Only show Person and Organization
allowed_types = ["Person", "Organization"]
filtered = filter_nodes_by_type(graph_docs, allowed_types)
```

### Export Graph Data

```python
def export_to_networkx(graph_documents):
    """Convert to NetworkX for analysis"""
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_documents[0].nodes:
        G.add_node(node.id, type=node.type)
    
    # Add edges
    for rel in graph_documents[0].relationships:
        G.add_edge(rel.source.id, rel.target.id, type=rel.type)
    
    return G

# Use with NetworkX algorithms
G = export_to_networkx(graph_docs)
centrality = nx.betweenness_centrality(G)
communities = nx.community.louvain_communities(G)
```

## 📊 Performance & Costs

### Processing Time

| Input Size | Entities | Relations | Time | Notes |
|-----------|----------|-----------|------|-------|
| 100 words | ~5-10 | ~5-15 | ~3s | Simple text |
| 500 words | ~20-30 | ~30-50 | ~8s | Article |
| 2000 words | ~50-100 | ~100-200 | ~25s | Research paper |

**Optimization Tips:**
- Break large documents into sections
- Process sections in parallel
- Cache results for repeated processing

### Cost Estimation

**OpenAI API Costs (GPT-4o):**

| Input Size | Tokens | Cost per Run | Notes |
|-----------|--------|--------------|-------|
| 100 words | ~300 | ~$0.01 | Short paragraph |
| 500 words | ~1500 | ~$0.03 | Article |
| 2000 words | ~6000 | ~$0.12 | Research paper |

**Cost Reduction Strategies:**
- Use `gpt-4o-mini` (60% cheaper)
- Use `gpt-3.5-turbo` (90% cheaper)
- Cache extracted graphs
- Batch process multiple documents

**Monthly Usage Examples:**

| Scenario | Docs/Month | Size | Cost |
|----------|-----------|------|------|
| Personal research | 50 | 500 words | ~$1.50 |
| Content analysis | 200 | 1000 words | ~$12 |
| Enterprise KG | 1000 | 2000 words | ~$120 |

## 🐛 Troubleshooting

### Issue: "OpenAI API key not found"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify key format
cat .env
# Should show: OPENAI_API_KEY=sk-...

# Restart the app
streamlit run app.py
```

### Issue: Graph doesn't display

**Solution:**
```python
# Check browser console for errors (F12)
# Verify knowledge_graph.html was created
ls -la knowledge_graph.html

# Try opening HTML directly in browser
open knowledge_graph.html
```

### Issue: Poor entity extraction

**Solution:**
```python
# Switch to GPT-4o for better quality
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Provide more context in text
# Bad: "He worked there."
# Good: "Albert Einstein worked at Princeton University."

# Increase temperature slightly for creativity
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
```

### Issue: Graph layout is messy

**Solution:**
```python
# Increase repulsion between nodes
"gravitationalConstant": -200  # More negative

# Increase edge length
"springLength": 300

# Let physics settle longer
"minVelocity": 0.5  # Lower value
```

### Issue: Missing relationships

**Solution:**
```python
# Use more explicit text
# Implicit: "Einstein. Special Relativity."
# Explicit: "Einstein developed Special Relativity in 1905."

# Try GPT-4o instead of 3.5
# Check if entities were detected
print(graph_documents[0].nodes)
```

## 🔬 Comparison with Other Tools

| Feature | This Project | Neo4j | GraphDB | Obsidian Graph |
|---------|-------------|-------|---------|----------------|
| Automatic Extraction | ✅ LLM-powered | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| Interactive Viz | ✅ PyVis | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| Setup | ✅ Simple | ⚠️ Complex | ⚠️ Complex | ⚠️ Medium |
| Cost | ⚠️ API calls | ✅ Free (Community) | 💰 License | ✅ Free |
| Web UI | ✅ Streamlit | ✅ Browser | ✅ Browser | ❌ Desktop |
| Programming API | ✅ Python | ✅ Cypher | ✅ SPARQL | ❌ |
| Scale | ⚠️ Small-medium | ✅ Large | ✅ Large | ⚠️ Small |

**When to use this project:**
- ✅ Quick exploration of text content
- ✅ Prototyping knowledge extraction
- ✅ Educational/research purposes
- ✅ Small to medium documents

**When to use alternatives:**
- Neo4j: Large-scale production systems, complex queries
- GraphDB: Semantic web, ontologies, SPARQL
- Obsidian: Personal knowledge management, note-taking

## 🎓 Learning Outcomes

After completing this project, you'll understand:

- ✅ LangChain graph transformers and LLM integration
- ✅ Entity and relationship extraction with GPT models
- ✅ Graph visualization with PyVis
- ✅ Force-directed graph layouts (ForceAtlas2)
- ✅ Streamlit web application development
- ✅ Async Python programming
- ✅ Knowledge graph data structures
- ✅ Interactive visualization techniques

## 📈 Best Practices

### 1. Input Text Quality

```python
# ❌ Poor: Vague, ambiguous
"He went there and did it."

# ✅ Good: Specific, explicit
"Albert Einstein worked at Princeton University from 1933 to 1955."
```

### 2. Document Size

```python
# ❌ Too large
text = entire_book  # 100,000+ words

# ✅ Optimal
chapter = book.split_chapters()[0]  # 2000-5000 words

# Or process in batches
for chunk in chunked_text(book, 2000):
    graph = generate_knowledge_graph(chunk)
```

### 3. Caching Results

```python
import pickle

# Save extracted graph
with open('graph_cache.pkl', 'wb') as f:
    pickle.dump(graph_documents, f)

# Load cached graph
with open('graph_cache.pkl', 'rb') as f:
    graph_documents = pickle.load(f)
```

### 4. Error Handling

```python
try:
    graph = generate_knowledge_graph(text)
except Exception as e:
    print(f"Error: {e}")
    # Fallback: try with simpler model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    graph = generate_knowledge_graph(text)
```

## 🔗 Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LangChain Graph Transformers](https://python.langchain.com/docs/use_cases/graph/)
- [PyVis Documentation](https://pyvis.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Papers
- [Knowledge Graphs](https://arxiv.org/abs/2003.02320) - Survey
- [Graph Neural Networks](https://arxiv.org/abs/1901.00596)
- [ForceAtlas2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679) - Layout algorithm

### Tutorials
- [YouTube Tutorial](https://www.youtube.com/watch?v=O-T_6KOXML4) - Original video
- [LangChain Knowledge Graphs](https://python.langchain.com/docs/use_cases/graph/quickstart)
- [Building KGs with LLMs](https://medium.com/@knowledge-graphs-llms)

## 📚 Next Steps

1. **Deploy to Cloud**: Host on Streamlit Cloud, Heroku, or AWS
2. **Add Neo4j Backend**: Scale to production graph database
3. **Implement CRUD**: Add, edit, delete nodes/edges manually
4. **Advanced Analytics**: PageRank, community detection, path finding
5. **Multi-Document**: Merge graphs from multiple documents

## 📝 License

MIT License - Free for personal and commercial use.

---

**🌐 Transform any text into an interactive knowledge graph in seconds!**

[← Back to Main README](../README.md)

Last Updated: November 6, 2025
