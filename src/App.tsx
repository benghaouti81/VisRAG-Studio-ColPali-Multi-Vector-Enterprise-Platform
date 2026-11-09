import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { DocumentCanvas } from "./components/DocumentCanvas";
import { ChatInterface } from "./components/ChatInterface";
import { GraphExplorer } from "./components/GraphExplorer";
import { ColPaliInspector } from "./components/ColPaliInspector";
import { DocumentManager } from "./components/DocumentManager";
import { ArchitectureModal } from "./components/ArchitectureModal";
import {
  DocumentMeta,
  DocumentPage,
  BoundingBox,
  TokenScore,
  TokenHeatmap,
  ChatMessage,
  Citation,
  GraphData,
  SystemHealth,
} from "./types";

export default function App() {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("arxiv_2407_01449");
  const [currentPageNumber, setCurrentPageNumber] = useState<number>(1);
  const [activeBoundingBox, setActiveBoundingBox] = useState<BoundingBox | null>(null);
  const [activeHeatmap, setActiveHeatmap] = useState<number[][] | null>(null);
  const [tokenHeatmaps, setTokenHeatmaps] = useState<TokenHeatmap[]>([]);
  const [tokenScores, setTokenScores] = useState<TokenScore[]>([]);
  const [totalScore, setTotalScore] = useState<number>(0);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [], total_nodes: 0, total_edges: 0 });
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);

  const [activeView, setActiveView] = useState<"workspace" | "graph" | "inspector">("workspace");
  const [isArchitectureOpen, setIsArchitectureOpen] = useState<boolean>(false);
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Chat message history with research welcome
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "msg_welcome",
      role: "assistant",
      content:
        "Welcome to **VisRAG Studio**. The system has indexed the seminal research papers on visual document retrieval:\n- **ColPali** (*arXiv:2407.01449*): Multi-vector vision retrieval bypassing OCR.\n- **VisRAG** (*arXiv:2410.10594*): Vision-based grounded generation with Gemini 3.8 Flash.\n- **LightRAG** (*arXiv:2410.05779*): Dual-level entity graph augmentation.\n\nTry clicking any benchmark question below or ask anything about the document layout, tables, or formulas. Each answer includes interactive citation badges that jump directly to the target bounding box on the canvas.",
      timestamp: "Just now",
      model_used: "VisRAG-Gen (Gemini 3.8 Flash)",
    },
  ]);

  // Fetch initial documents, graph, and system health
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [docsRes, healthRes, graphRes] = await Promise.all([
          fetch("/api/documents"),
          fetch("/api/health"),
          fetch("/api/graph"),
        ]);

        if (docsRes.ok) {
          const docsData = await docsRes.json();
          setDocuments(docsData.documents || []);
        }

        if (healthRes.ok) {
          const healthData = await healthRes.json();
          setSystemHealth(healthData);
        }

        if (graphRes.ok) {
          const gData = await graphRes.json();
          setGraphData(gData);
        }
      } catch (err) {
        console.error("Error loading initial data:", err);
      }
    };

    fetchData();
  }, []);

  // Compute active document and current page object
  const currentDoc = documents.find((d) => d.id === selectedDocId);
  const totalPages = currentDoc?.total_pages || 1;

  // Construct DocumentPage representation
  const currentPageData: DocumentPage | null = currentDoc
    ? {
        page_number: currentPageNumber,
        doc_id: currentDoc.id,
        title: currentDoc.title,
        section:
          currentDoc.pages_meta.find((p) => p.page_number === currentPageNumber)?.section ||
          `Page ${currentPageNumber}`,
        width: 800,
        height: 1100,
        patch_grid: [16, 16],
        key_regions:
          selectedDocId === "arxiv_2407_01449" && currentPageNumber === 3
            ? [
                {
                  label: "Table 1: ViDoRe Benchmark Results",
                  category: "table",
                  bbox: { ymin: 0.18, xmin: 0.08, ymax: 0.62, xmax: 0.92, label: "Table 1 (81.3% nDCG@5)", category: "table", confidence: 0.99 },
                  text_content: "ViDoRe Retrieval Benchmark (nDCG@5): ColPali achieves 81.3% overall vs BM25 52.4% and BGE-M3 64.1%.",
                  tokens: ["table", "vidore", "benchmark", "results", "colpali", "81.3", "bm25", "52.4", "ndcg@5"],
                },
                {
                  label: "Evaluation Findings Analysis",
                  category: "metric",
                  bbox: { ymin: 0.68, xmin: 0.1, ymax: 0.92, xmax: 0.9, label: "Analysis Findings", category: "metric", confidence: 0.95 },
                  text_content: "ColPali achieves notable gains on TabFqa (+31.2%) and InfoVQA (+26.4%).",
                  tokens: ["gains", "tabfqa", "infovqa", "ndcg"],
                },
              ]
            : selectedDocId === "arxiv_2407_01449" && currentPageNumber === 2
            ? [
                {
                  label: "Section 3: Mathematical Formulation",
                  category: "header",
                  bbox: { ymin: 0.06, xmin: 0.1, ymax: 0.14, xmax: 0.9, label: "Section 3 Heading", category: "header", confidence: 0.97 },
                  text_content: "3. Multi-Vector Late Interaction for Document Images",
                  tokens: ["section", "multi-vector", "late", "interaction"],
                },
                {
                  label: "Equation 1: MaxSim Operator",
                  category: "equation",
                  bbox: { ymin: 0.22, xmin: 0.15, ymax: 0.42, xmax: 0.85, label: "Equation 1: MaxSim Formula", category: "equation", confidence: 0.99 },
                  text_content: "Score(Q, D) = sum_{q in Q} max_{d in D} (q . d^T)",
                  tokens: ["equation", "score", "maxsim", "operator", "sum", "dot", "product"],
                },
              ]
            : [
                {
                  label: "Main Document Section",
                  category: "header",
                  bbox: { ymin: 0.05, xmin: 0.1, ymax: 0.18, xmax: 0.9, label: "Title Section", category: "header", confidence: 0.98 },
                  text_content: currentDoc.title,
                  tokens: currentDoc.title.toLowerCase().split(/\s+/),
                },
              ],
      }
    : null;

  // Handle user query submission
  const handleSendMessage = async (queryText: string) => {
    setIsLoading(true);

    // Add user message to thread
    const userMsg: ChatMessage = {
      id: `msg_user_${Date.now()}`,
      role: "user",
      content: queryText,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      // 1. First trigger visual search to get full ColPali late-interaction tensor and heatmap
      const searchRes = await fetch("/api/search/visual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, top_k: 3 }),
      });

      let topSearch = null;
      if (searchRes.ok) {
        const searchData = await searchRes.json();
        if (searchData.results && searchData.results.length > 0) {
          topSearch = searchData.results[0];
          setActiveHeatmap(topSearch.heatmap);
          setTokenHeatmaps(topSearch.token_heatmaps || []);
          setTokenScores(topSearch.token_scores || []);
          setTotalScore(topSearch.score);

          // Auto-navigate canvas to top retrieved page
          setSelectedDocId(topSearch.doc_id);
          setCurrentPageNumber(topSearch.page_number);
        }
      }

      // 2. Call VisRAG Grounded Chat Generation pipeline
      const chatRes = await fetch("/api/generate/grounded-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: queryText,
          doc_id: topSearch ? topSearch.doc_id : selectedDocId,
          top_k: 3,
        }),
      });

      if (chatRes.ok) {
        const chatData = await chatRes.json();

        // If citations returned, set primary bounding box
        if (chatData.citations && chatData.citations.length > 0) {
          const primaryCite = chatData.citations[0];
          setActiveBoundingBox(primaryCite.bbox);
          setCurrentPageNumber(primaryCite.page_number);
        }

        const assistantMsg: ChatMessage = {
          id: `msg_asst_${Date.now()}`,
          role: "assistant",
          content: chatData.answer,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          citations: chatData.citations,
          retrieved_pages: chatData.retrieved_pages,
          model_used: chatData.model_used,
          token_breakdown: topSearch ? topSearch.token_scores : undefined,
        };

        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (err) {
      console.error("Error in VisRAG pipeline:", err);
      const errorMsg: ChatMessage = {
        id: `msg_err_${Date.now()}`,
        role: "assistant",
        content: "An error occurred while executing the ColPali Late Interaction search.",
        timestamp: "Now",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Jump to citation when badge is clicked in chat
  const handleJumpToCitation = (citation: Citation) => {
    setSelectedDocId(citation.doc_id);
    setCurrentPageNumber(citation.page_number);
    setActiveBoundingBox(citation.bbox);
    setActiveView("workspace");
  };

  // Select token heatmap on canvas from token breakdown
  const handleSelectTokenHeatmap = (tokenStr: string) => {
    setActiveView("workspace");
  };

  // Jump to entity page from graph explorer
  const handleJumpToEntityPage = (pageNumber: number, bbox?: BoundingBox) => {
    setCurrentPageNumber(pageNumber);
    if (bbox) {
      setActiveBoundingBox(bbox);
    }
    setActiveView("workspace");
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Top Enterprise Navigation */}
      <Navbar
        documents={documents}
        selectedDocId={selectedDocId}
        onSelectDoc={(id) => {
          setSelectedDocId(id);
          setCurrentPageNumber(1);
          setActiveBoundingBox(null);
        }}
        systemHealth={systemHealth}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        onOpenUpload={() => setIsUploadOpen(true)}
        activeView={activeView}
        setActiveView={setActiveView}
      />

      {/* Main Workspace Body */}
      <main className="flex-1 overflow-hidden p-2 sm:p-3">
        {activeView === "workspace" && (
          <div className="h-full grid grid-cols-1 lg:grid-cols-12 gap-3 overflow-hidden">
            {/* Left 7 Columns: Interactive Document Canvas */}
            <div className="h-full lg:col-span-7 overflow-hidden">
              <DocumentCanvas
                page={currentPageData}
                totalPages={totalPages}
                currentPageNumber={currentPageNumber}
                onPageChange={(p) => {
                  setCurrentPageNumber(p);
                  setActiveBoundingBox(null);
                }}
                activeBoundingBox={activeBoundingBox}
                onSelectBoundingBox={(bbox) => setActiveBoundingBox(bbox)}
                heatmap={activeHeatmap}
                tokenHeatmaps={tokenHeatmaps}
                tokenScores={tokenScores}
                onSelectToken={() => {}}
              />
            </div>

            {/* Right 5 Columns: VisRAG Grounded Chat Interface */}
            <div className="h-full lg:col-span-5 overflow-hidden">
              <ChatInterface
                messages={messages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                onJumpToCitation={handleJumpToCitation}
                onSelectTokenHeatmap={handleSelectTokenHeatmap}
              />
            </div>
          </div>
        )}

        {/* LightRAG Knowledge Graph View */}
        {activeView === "graph" && (
          <div className="h-full">
            <GraphExplorer
              graphData={graphData}
              onJumpToEntityPage={handleJumpToEntityPage}
            />
          </div>
        )}

        {/* ColPali Late Interaction Tensor Inspector View */}
        {activeView === "inspector" && (
          <div className="h-full">
            <ColPaliInspector
              tokenScores={tokenScores}
              tokenHeatmaps={tokenHeatmaps}
              totalScore={totalScore}
              docTitle={currentDoc?.title || "Document"}
              pageNumber={currentPageNumber}
              onSelectTokenForCanvas={() => setActiveView("workspace")}
            />
          </div>
        )}
      </main>

      {/* Architecture & Math Foundations Modal */}
      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />

      {/* Document Manager & Ingestion Modal */}
      <DocumentManager
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        documents={documents}
        onSelectDoc={(id) => {
          setSelectedDocId(id);
          setCurrentPageNumber(1);
          setActiveBoundingBox(null);
        }}
        onUploadSuccess={(newDoc) => {
          setDocuments((prev) => [newDoc, ...prev]);
          setSelectedDocId(newDoc.id);
          setCurrentPageNumber(1);
        }}
      />
    </div>
  );
}
