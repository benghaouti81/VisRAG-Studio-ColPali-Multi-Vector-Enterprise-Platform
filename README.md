# VisRAG Studio: ColPali Multi-Vector Enterprise Platform

An enterprise-grade, full-stack visual retrieval and multimodal generation platform implementing the state-of-the-art architectures 
- **ColPali**: *Efficient Document Retrieval with Vision Language Models* 
- **VisRAG**: *Vision-based Retrieval-Augmented Generation on Multi-modality Documents* 
- **LightRAG**: *Simple and Fast Retrieval-Augmented Generation* 

---

## 1. Research Foundations & Mathematics

### A. ColPali Multi-Vector Late Interaction 
Traditional OCR-based RAG parses documents into unstructured text strings, destroying tables, multi-column layouts, font hierarchies, charts, and equations. ColPali replaces OCR by treating the entire document page as an image and projecting visual patches into token-level vector representations.

For a document page $D$ represented by $N_d$ patch embeddings $D = \{d_1, d_2, \dots, d_{N_d}\} \in \mathbb{R}^{N_d \times K}$ and a user query $Q$ with $N_q$ token embeddings $Q = \{q_1, q_2, \dots, q_{N_q}\} \in \mathbb{R}^{N_q \times K}$, the Late Interaction MaxSim score is computed as:

$$\text{Score}(Q, D) = \sum_{i=1}^{N_q} \max_{j=1}^{N_d} \left( q_i \cdot d_j^T \right)$$

This allows every query word to match the exact visual patch on the page that best satisfies it, generating an attention heatmap overlay across the document canvas.

### B. VisRAG-Gen Multimodal Grounding 
Retrieved top-K page images (at 150-300 DPI) are directly provided as visual inputs to the Gemini multimodal model (`gemini-3.8-flash`). The model outputs grounded factual answers tagged with exact visual bounding boxes:

$$B = [y_{min}, x_{min}, y_{max}, x_{max}] \in [0.0, 1.0]^4$$

allowing the user to inspect the exact table cells, diagrams, or headers cited.

### C. LightRAG Dual-Level Entity Graph 
A directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ tracks cross-page references, entity hierarchies, and relational context to resolve multi-hop queries across complex reports.

---

## 2. Quickstart: Run on Ubuntu VPS (Docker Compose)

### Prerequisites
- Ubuntu 22.04+ LTS (or any Linux host)
- Docker Engine & Docker Compose (`docker compose version >= 2.20`)
- Optional: NVIDIA Container Toolkit for CUDA acceleration

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/your-org/visrag-colpali-enterprise.git
cd visrag-colpali-enterprise

# Copy environment template
cp .env.example .env

# Set your Gemini API Key in .env
nano .env
# GEMINI_API_KEY=your_gemini_api_key_here
# DEVICE=cpu   # or DEVICE=cuda if GPU available
```

### Step 2: Build and Launch Containers
```bash
docker compose up --build -d
```

### Step 3: Verify Deployment
Check running services:
```bash
docker compose ps
```

Expected output:
| Container Name | Service | Status | Ports |
|---|---|---|---|
| `visrag-colpali-backend` | FastAPI Engine | Up (healthy) | `0.0.0.0:8000->8000` |
| `visrag-colpali-frontend` | React Nginx | Up | `0.0.0.0:3000->3000` |
| `visrag-qdrant` | Vector Database | Up | `0.0.0.0:6333->6333` |

Open `http://<your-server-ip>:3000` in your web browser.

---

## 3. API Specification & Endpoints

### 1. Document Indexing
```bash
curl -X POST "http://localhost:8000/api/documents/index" \
  -F "file=@sample_report.pdf" \
  -F "dpi=150"
```

### 2. ColPali Visual Search (MaxSim)
```bash
curl -X POST "http://localhost:8000/api/search/visual" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the nDCG@5 score of ColPali on the ViDoRe benchmark?",
    "top_k": 3
  }'
```

### 3. VisRAG Grounded Chat Generation
```bash
curl -X POST "http://localhost:8000/api/generate/grounded-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare ColPali against BM25 and BGE-M3 in Table 1.",
    "top_k": 3,
    "include_graph_path": true
  }'
```

---

## 4. Architecture Directory Layout
```
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py              # FastAPI router & Gemini grounding pipeline
│   ├── retriever.py         # ColPali engine, patch embeddings & MaxSim
│   └── graph_engine.py      # LightRAG entity-relationship graph (NetworkX)
├── src/
│   ├── App.tsx              # Split-screen responsive workspace
│   ├── components/
│   │   ├── DocumentCanvas.tsx # Visual PDF canvas with heatmaps & bounding boxes
│   │   ├── ChatInterface.tsx  # Conversational grounding & citations
│   │   ├── ColPaliInspector.tsx # Tensor & Late-Interaction MaxSim breakdown
│   │   ├── GraphExplorer.tsx  # LightRAG interactive entity visualizer
│   │   └── DocumentManager.tsx # PDF dropzone & benchmark library
│   └── types.ts             # TypeScript domain definitions
├── docker-compose.yml       # Production microservice orchestrator
├── server.ts                # Express full-stack bridge with @google/genai
└── package.json
```
