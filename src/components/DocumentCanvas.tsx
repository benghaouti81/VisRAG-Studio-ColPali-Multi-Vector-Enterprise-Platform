import React, { useState, useRef, useEffect } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Eye,
  EyeOff,
  Grid,
  ChevronLeft,
  ChevronRight,
  Flame,
  Layers,
  Sparkles,
  Download,
  Focus
} from "lucide-react";
import { DocumentPage, BoundingBox, TokenScore, TokenHeatmap, KeyRegion } from "../types";

interface DocumentCanvasProps {
  page: DocumentPage | null;
  totalPages: number;
  currentPageNumber: number;
  onPageChange: (newPage: number) => void;
  activeBoundingBox: BoundingBox | null;
  onSelectBoundingBox: (bbox: BoundingBox) => void;
  heatmap: number[][] | null;
  tokenHeatmaps?: TokenHeatmap[];
  tokenScores?: TokenScore[];
  highlightQuery?: string;
  focusedTokenIndex?: number | null;
  onSelectToken?: (idx: number | null) => void;
}

export const DocumentCanvas: React.FC<DocumentCanvasProps> = ({
  page,
  totalPages,
  currentPageNumber,
  onPageChange,
  activeBoundingBox,
  onSelectBoundingBox,
  heatmap,
  tokenHeatmaps = [],
  tokenScores = [],
  highlightQuery = "",
  focusedTokenIndex = null,
  onSelectToken,
}) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showPatchGrid, setShowPatchGrid] = useState<boolean>(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);
  const [selectedTokenName, setSelectedTokenName] = useState<string>("all");

  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const boundingBoxRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to active bounding box when triggered from chat citation
  useEffect(() => {
    if (activeBoundingBox && boundingBoxRef.current && canvasContainerRef.current) {
      boundingBoxRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [activeBoundingBox]);

  // Determine active heatmap to display
  let currentHeatmap: number[][] | null = heatmap;
  if (selectedTokenName !== "all" && tokenHeatmaps.length > 0) {
    const found = tokenHeatmaps.find((th) => th.token.toLowerCase() === selectedTokenName.toLowerCase());
    if (found) {
      currentHeatmap = found.heatmap;
    }
  }

  // Get color for bounding box category
  const getCategoryStyles = (category: string, isActive: boolean) => {
    if (isActive) {
      return {
        border: "border-cyan-400 ring-4 ring-cyan-400/30 bg-cyan-500/15 shadow-lg shadow-cyan-500/30",
        badge: "bg-cyan-500 text-slate-950 font-bold",
      };
    }
    switch (category) {
      case "table":
        return {
          border: "border-blue-500/80 bg-blue-500/10 hover:border-blue-400 hover:bg-blue-500/20",
          badge: "bg-blue-600 text-white font-medium",
        };
      case "chart":
        return {
          border: "border-amber-500/80 bg-amber-500/10 hover:border-amber-400 hover:bg-amber-500/20",
          badge: "bg-amber-600 text-white font-medium",
        };
      case "equation":
        return {
          border: "border-purple-500/80 bg-purple-500/10 hover:border-purple-400 hover:bg-purple-500/20",
          badge: "bg-purple-600 text-white font-medium",
        };
      case "metric":
        return {
          border: "border-emerald-500/80 bg-emerald-500/10 hover:border-emerald-400 hover:bg-emerald-500/20",
          badge: "bg-emerald-600 text-white font-medium",
        };
      default:
        return {
          border: "border-indigo-500/70 bg-indigo-500/10 hover:border-indigo-400 hover:bg-indigo-500/20",
          badge: "bg-indigo-600 text-white font-medium",
        };
    }
  };

  // Convert normalized [0, 1] score to attention heatmap RGBA color
  const getHeatmapColor = (val: number, opacity: number) => {
    // Jet/Turbo-like colormap: low = transparent blue/cyan, mid = yellow/amber, high = red/magenta
    const clamped = Math.max(0, Math.min(1, val));
    if (clamped < 0.1) return "transparent";

    let r = 0;
    let g = 0;
    let b = 0;

    if (clamped < 0.35) {
      // Blue to Cyan
      const t = clamped / 0.35;
      r = 0;
      g = Math.round(180 * t);
      b = 255;
    } else if (clamped < 0.7) {
      // Cyan to Yellow
      const t = (clamped - 0.35) / 0.35;
      r = Math.round(255 * t);
      g = 255;
      b = Math.round(255 * (1 - t));
    } else {
      // Yellow to Crimson Red
      const t = (clamped - 0.7) / 0.3;
      r = 255;
      g = Math.round(255 * (1 - t * 0.85));
      b = Math.round(60 * t);
    }

    return `rgba(${r}, ${g}, ${b}, ${(opacity * (0.3 + 0.7 * clamped)).toFixed(2)})`;
  };

  if (!page) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-slate-400">
        <p>No document page selected</p>
      </div>
    );
  }

  const gridRows = page.patch_grid[0] || 16;
  const gridCols = page.patch_grid[1] || 16;

  return (
    <div className="h-full flex flex-col bg-slate-950 border border-slate-800/80 rounded-xl overflow-hidden shadow-2xl">
      {/* Canvas Top Toolbar */}
      <div className="bg-slate-900/90 border-b border-slate-800 px-3 py-2 flex flex-wrap items-center justify-between gap-2 z-20">
        {/* Left: Page Navigation */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onPageChange(Math.max(1, currentPageNumber - 1))}
            disabled={currentPageNumber <= 1}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:pointer-events-none text-slate-300 transition"
            title="Previous Page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <span className="text-xs font-mono font-medium text-slate-200 px-2 py-0.5 bg-slate-950 border border-slate-800 rounded">
            Page {currentPageNumber} of {totalPages}
          </span>

          <button
            onClick={() => onPageChange(Math.min(totalPages, currentPageNumber + 1))}
            disabled={currentPageNumber >= totalPages}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:pointer-events-none text-slate-300 transition"
            title="Next Page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <span className="text-xs text-slate-400 font-medium pl-2 hidden md:inline truncate max-w-[200px]">
            {page.section}
          </span>
        </div>

        {/* Center: Token Heatmap Inspector Selector */}
        {tokenScores && tokenScores.length > 0 && (
          <div className="flex items-center gap-1.5 bg-slate-950 px-2 py-1 rounded-lg border border-slate-800 text-xs">
            <Flame className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span className="text-slate-400 text-[11px] font-mono">Token:</span>
            <select
              value={selectedTokenName}
              onChange={(e) => setSelectedTokenName(e.target.value)}
              className="bg-transparent text-cyan-300 text-xs font-mono font-semibold focus:outline-none cursor-pointer max-w-[130px] truncate"
            >
              <option value="all" className="bg-slate-900 text-slate-200">
                MaxSim (All Tokens)
              </option>
              {tokenScores.map((ts, idx) => (
                <option key={idx} value={ts.token} className="bg-slate-900 text-cyan-300">
                  "{ts.token}" ({ts.max_score.toFixed(3)})
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Right: Layer Toggles & Zoom */}
        <div className="flex items-center gap-2">
          {/* Heatmap Toggle */}
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-2 py-1 text-xs rounded-md flex items-center gap-1 transition ${
              showHeatmap
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/50"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle ColPali Attention Heatmap"
          >
            <Flame className="w-3.5 h-3.5 text-amber-400" />
            <span className="hidden sm:inline">Heatmap</span>
          </button>

          {/* Bounding Box Toggle */}
          <button
            onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
            className={`px-2 py-1 text-xs rounded-md flex items-center gap-1 transition ${
              showBoundingBoxes
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/50"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle VisRAG Grounding Bounding Boxes"
          >
            <Focus className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Grounding</span>
          </button>

          {/* Patch Grid Toggle */}
          <button
            onClick={() => setShowPatchGrid(!showPatchGrid)}
            className={`px-2 py-1 text-xs rounded-md flex items-center gap-1 transition ${
              showPatchGrid
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/50"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle 16x16 ColPali Vision Patch Grid"
          >
            <Grid className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Patches</span>
          </button>

          {/* Zoom Controls */}
          <div className="flex items-center bg-slate-800 rounded-md p-0.5 text-slate-300">
            <button
              onClick={() => setZoom((z) => Math.max(0.6, z - 0.15))}
              className="p-1 hover:text-white transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1.0)}
              className="px-1 text-[11px] font-mono hover:text-white"
              title="Reset Zoom"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              onClick={() => setZoom((z) => Math.min(2.0, z + 0.15))}
              className="p-1 hover:text-white transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Document Viewport */}
      <div
        ref={canvasContainerRef}
        className="flex-1 overflow-auto bg-slate-900/70 p-4 sm:p-6 flex items-start justify-center relative scrollbar-thin scrollbar-thumb-slate-700"
      >
        {/* Rendered Document Sheet */}
        <div
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: "top center",
            width: "740px",
            minHeight: "1020px",
          }}
          className="relative bg-white text-slate-900 rounded-lg shadow-2xl border border-slate-300 overflow-hidden transition-transform duration-150 select-text"
        >
          {/* ==============================================================
              SCIENTIFIC DOCUMENT BODY (Faithful Representation of arXiv Papers)
             ============================================================== */}
          <div className="p-8 sm:p-10 font-serif leading-relaxed text-[13px] text-slate-800 space-y-4">
            {page.doc_id === "arxiv_2407_01449" && page.page_number === 1 && (
              <>
                <div className="text-center border-b border-slate-300 pb-5 mb-5 font-sans">
                  <h1 className="text-xl sm:text-2xl font-extrabold text-slate-950 tracking-tight leading-tight">
                    ColPali: Efficient Document Retrieval with Vision Language Models
                  </h1>
                  <p className="text-xs text-slate-600 mt-2 font-medium">
                    Manuel Faysse* · Hugues Sibille* · Tony Wu · Bilel Omrani · Gautier Viaud · Céline Hudelot · Pierre Colombo
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    CentraleSupélec, Université Paris-Saclay · Hugging Face · arXiv:2407.01449v1
                  </p>
                </div>

                <div className="bg-slate-50 p-4 rounded-md border border-slate-200">
                  <h2 className="font-sans font-bold text-xs uppercase tracking-wider text-slate-700 mb-1.5">
                    Abstract
                  </h2>
                  <p className="text-justify text-slate-700 leading-normal text-xs">
                    Document retrieval is an essential pillar of Retrieval-Augmented Generation (RAG). However, current
                    industrial pipelines suffer from a fundamental bottleneck: they rely on Optical Character
                    Recognition (OCR) and layout analysis tools that strip away essential visual information, including
                    fonts, colors, spatial layouts, tables, and charts. In this paper, we introduce <strong>ColPali</strong>,
                    a novel model architecture and training strategy that adapts Vision-Language Models (PaliGemma-3B)
                    to directly index page images via multi-vector late-interaction representations.
                  </p>
                </div>

                {/* Figure 1 Visual Mock */}
                <div className="my-6 border border-slate-200 bg-gradient-to-b from-slate-50 to-slate-100 p-4 rounded-lg shadow-sm font-sans">
                  <div className="text-xs font-bold text-slate-800 mb-2 flex items-center justify-between">
                    <span>Figure 1: Paradigm Comparison — Standard OCR-RAG vs. ColPali Visual Late Interaction</span>
                    <span className="text-[10px] text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded font-mono">arXiv:2407.01449</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 bg-red-50 border border-red-200 rounded text-red-900">
                      <div className="font-bold mb-1 text-red-700">❌ Traditional OCR Pipeline:</div>
                      <p className="text-[11px] leading-tight">
                        PDF ➔ OCR Engine ➔ Noisy Raw Text ➔ Chunking ➔ Single Dense Embedding.
                        <br />
                        <span className="text-red-600 font-medium">Layout, charts, and table alignment completely lost.</span>
                      </p>
                    </div>
                    <div className="p-3 bg-emerald-50 border border-emerald-200 rounded text-emerald-900">
                      <div className="font-bold mb-1 text-emerald-700">✅ ColPali Architecture:</div>
                      <p className="text-[11px] leading-tight">
                        PDF ➔ High-Res Page Image ➔ Vision Encoder ➔ 256/1024 Patch Multi-Vectors.
                        <br />
                        <span className="text-emerald-700 font-medium">Late Interaction MaxSim preserves complete 2D visual fidelity.</span>
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6 text-justify text-[12px] leading-normal">
                  <div>
                    <h3 className="font-sans font-bold text-xs uppercase text-slate-900 mb-1">1. Introduction</h3>
                    <p>
                      Modern enterprise documents are intrinsically multimodal: financial reports contain dense balance
                      sheets, scientific papers convey insights via vector graphs, and legal contracts encode clauses
                      using indented hierarchy. Existing dense retrievers (e.g. BGE, Contriever) compress entire
                      paragraphs into a single vector, losing localized specificity.
                    </p>
                  </div>
                  <div>
                    <h3 className="font-sans font-bold text-xs uppercase text-slate-900 mb-1">2. Related Work</h3>
                    <p>
                      ColBERT demonstrated that late interaction scoring achieves state-of-the-art retrieval accuracy by
                      deferring token interaction to query time. We extend this foundational insight into the visual
                      domain by replacing language tokens with image patch tokens produced by a Vision Transformer.
                    </p>
                  </div>
                </div>
              </>
            )}

            {page.doc_id === "arxiv_2407_01449" && page.page_number === 2 && (
              <>
                <div className="border-b border-slate-300 pb-3 mb-4 font-sans">
                  <h2 className="text-base font-bold text-slate-900">
                    3. Multi-Vector Late Interaction Formulation
                  </h2>
                  <p className="text-xs text-slate-500">Methodology & Mathematical Foundations</p>
                </div>

                <p className="text-justify text-xs leading-relaxed">
                  Given a document page image I ∈ ℝ^(H × W × 3), we process it using the vision
                  encoder backbone f_θ. The image is partitioned into non-overlapping spatial patches of size
                  p × p, producing a grid of visual tokens D ∈ ℝ^(N_d × d).
                </p>

                {/* Equation 1 Callout */}
                <div className="my-6 p-5 bg-indigo-50/70 border-2 border-indigo-200 rounded-lg text-center font-serif shadow-sm">
                  <div className="text-[11px] font-sans font-bold text-indigo-900 uppercase tracking-wider mb-2">
                    Equation (1): ColPali Late Interaction MaxSim Operator
                  </div>
                  <div className="text-lg font-bold text-slate-900 my-2 tracking-wide font-mono bg-white py-3 px-4 rounded border border-indigo-100 inline-block shadow-inner">
                    Score(Q, D) = ∑<sub>q ∈ Q</sub> max<sub>d ∈ D</sub> ( q · d<sup>T</sup> )
                  </div>
                  <p className="text-xs text-slate-600 mt-2 font-sans max-w-lg mx-auto">
                    Where Q ∈ ℝ^(N_q × d) is the set of query token representations, and D ∈
                    ℝ^(N_d × d) is the multi-vector page patch representation.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-6 text-justify text-[12px] leading-normal font-sans">
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded">
                    <h4 className="font-bold text-xs text-slate-800 mb-1">Patch Tokenization Details</h4>
                    <p className="text-[11px] text-slate-600">
                      We project PaliGemma visual tokens into a normalized 128-dimensional embedding subspace. With a
                      patch size of 14x14 pixels, a standard page yields 1,024 patches (or 256 patches in low-latency
                      mode), allowing fast parallel tensor dot-products.
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 border border-slate-200 rounded">
                    <h4 className="font-bold text-xs text-slate-800 mb-1">Interpretability & Attention Heatmaps</h4>
                    <p className="text-[11px] text-slate-600">
                      Because the score decomposes into per-token maximum similarity patch indices $j^*(i) = \arg\max_j (q_i \cdot d_j^T)$,
                      the system naturally yields an interpretable visual attention heatmap mapping each query word to its exact physical page patch.
                    </p>
                  </div>
                </div>
              </>
            )}

            {page.doc_id === "arxiv_2407_01449" && page.page_number === 3 && (
              <>
                <div className="border-b border-slate-300 pb-3 mb-4 font-sans">
                  <h2 className="text-base font-bold text-slate-900">
                    4. ViDoRe Benchmark Results & Baseline Comparisons
                  </h2>
                  <p className="text-xs text-slate-500">Evaluation on Visual Document Retrieval (arXiv:2407.01449)</p>
                </div>

                <p className="text-justify text-xs leading-relaxed">
                  We evaluate ColPali on the newly proposed <strong>ViDoRe</strong> (Visual Document Retrieval) benchmark,
                  encompassing 10 diverse tasks including TabFqa, DocVQA, InfoVQA, and synthetic visual documents.
                </p>

                {/* Table 1 Benchmark Results */}
                <div className="my-5 border border-slate-300 rounded-lg overflow-hidden shadow-sm font-sans text-xs">
                  <div className="bg-slate-100 px-4 py-2 border-b border-slate-300 font-bold text-slate-900 flex justify-between items-center">
                    <span>Table 1: ViDoRe Comprehensive Benchmark Evaluation (nDCG@5 Metric)</span>
                    <span className="text-[10px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-mono font-semibold">
                      ColPali SOTA: 81.3%
                    </span>
                  </div>
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-[11px] text-slate-600 font-semibold">
                        <th className="p-2.5">Retriever Model</th>
                        <th className="p-2.5">Modality</th>
                        <th className="p-2.5">TabFqa</th>
                        <th className="p-2.5">InfoVQA</th>
                        <th className="p-2.5">DocVQA</th>
                        <th className="p-2.5 bg-indigo-50 font-bold text-indigo-900">ViDoRe Average</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-slate-800">
                      <tr className="hover:bg-slate-50/80">
                        <td className="p-2.5 font-medium">BM25 (Standard OCR)</td>
                        <td className="p-2.5 text-slate-500">Text OCR</td>
                        <td className="p-2.5">44.8%</td>
                        <td className="p-2.5">38.2%</td>
                        <td className="p-2.5">51.0%</td>
                        <td className="p-2.5 bg-indigo-50/50 font-mono">52.4%</td>
                      </tr>
                      <tr className="hover:bg-slate-50/80">
                        <td className="p-2.5 font-medium">BGE-M3 (Dense Retriever)</td>
                        <td className="p-2.5 text-slate-500">Dense Text</td>
                        <td className="p-2.5">57.3%</td>
                        <td className="p-2.5">51.4%</td>
                        <td className="p-2.5">62.8%</td>
                        <td className="p-2.5 bg-indigo-50/50 font-mono">64.1%</td>
                      </tr>
                      <tr className="hover:bg-slate-50/80">
                        <td className="p-2.5 font-medium">Unstructured OCR + text-emb-3-large</td>
                        <td className="p-2.5 text-slate-500">Chunked Text</td>
                        <td className="p-2.5">59.1%</td>
                        <td className="p-2.5">54.0%</td>
                        <td className="p-2.5">64.7%</td>
                        <td className="p-2.5 bg-indigo-50/50 font-mono">65.8%</td>
                      </tr>
                      <tr className="bg-emerald-50/60 font-bold text-emerald-950 border-t-2 border-emerald-400">
                        <td className="p-2.5 text-emerald-800 flex items-center gap-1">
                          <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                          ColPali (Vision Multi-Vector)
                        </td>
                        <td className="p-2.5 text-emerald-700">Raw Images</td>
                        <td className="p-2.5 text-emerald-800">76.0%</td>
                        <td className="p-2.5 text-emerald-800">77.8%</td>
                        <td className="p-2.5 text-emerald-800">79.2%</td>
                        <td className="p-2.5 bg-emerald-100 font-mono text-emerald-900 text-sm">81.3%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="bg-slate-50 p-3.5 rounded border border-slate-200 text-xs text-justify">
                  <strong>Key Finding:</strong> ColPali surpasses traditional OCR pipelines by +28.9% absolute points
                  overall and demonstrates staggering gains on tabular data (+31.2% on TabFqa) where OCR string
                  flattening systematically breaks column-row relationships.
                </div>
              </>
            )}

            {page.doc_id === "arxiv_2410_10594" && (
              <>
                <div className="text-center border-b border-slate-300 pb-5 mb-5 font-sans">
                  <h1 className="text-xl sm:text-2xl font-extrabold text-slate-950 tracking-tight leading-tight">
                    VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents
                  </h1>
                  <p className="text-xs text-slate-600 mt-2 font-medium">
                    Shi Yu · Chaoyue Tang · Bowen Yu · Mengfei Lan · Shengding Hu · Zhiyuan Liu · Maosong Sun
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    Tsinghua University · ShengShu AI · arXiv:2410.10594
                  </p>
                </div>

                <div className="bg-slate-50 p-4 rounded-md border border-slate-200">
                  <h2 className="font-sans font-bold text-xs uppercase tracking-wider text-slate-700 mb-1.5">
                    Abstract & Motivation
                  </h2>
                  <p className="text-justify text-slate-700 leading-normal text-xs">
                    Current document RAG architectures typically convert PDFs into pure text, introducing severe
                    information loss. We present <strong>VisRAG</strong>, an end-to-end vision-based RAG framework that
                    maximizes visual fidelity across both retrieval (VisRAG-Ret) and generation (VisRAG-Gen). VisRAG
                    directly passes high-resolution page images to the multimodal LLM, establishing tight visual
                    grounding with bounding box citations.
                  </p>
                </div>

                {/* Figure 2 Callout */}
                <div className="my-6 border border-slate-200 bg-slate-50 p-4 rounded-lg font-sans">
                  <div className="text-xs font-bold text-slate-800 mb-2">
                    Figure 2: The Two Pillars of VisRAG
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 bg-white border border-slate-200 rounded shadow-sm">
                      <div className="font-bold text-indigo-700 mb-1">1. VisRAG-Ret (Visual Retrieval)</div>
                      <p className="text-[11px] text-slate-600">
                        Multi-vector ColPali / VLM index retrieving top-K page images directly from high-res scans.
                      </p>
                    </div>
                    <div className="p-3 bg-white border border-slate-200 rounded shadow-sm">
                      <div className="font-bold text-cyan-700 mb-1">2. VisRAG-Gen (Multimodal Generation)</div>
                      <p className="text-[11px] text-slate-600">
                        Feeds retrieved page images to Gemini 3.8 Flash, predicting exact bounding boxes [ymin, xmin, ymax, xmax].
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}

            {page.doc_id.startsWith("custom_") && (
              <div className="space-y-4">
                <div className="border-b border-slate-200 pb-3">
                  <h2 className="text-xl font-bold text-slate-900">{page.title}</h2>
                  <p className="text-xs text-slate-500 font-mono mt-1">
                    Custom Uploaded Document · Page {page.page_number} · 256 Visual Patches
                  </p>
                </div>
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-700">
                  <p className="font-bold mb-1">Indexed Content Summary:</p>
                  <p>{page.key_regions[0]?.text_content || "Document page content indexed via ColPali vision patches."}</p>
                </div>
              </div>
            )}
          </div>

          {/* ==============================================================
              LAYER 1: 16x16 / 32x32 VISION PATCH GRID OVERLAY
             ============================================================== */}
          {showPatchGrid && (
            <div className="absolute inset-0 pointer-events-none z-10 grid grid-cols-16 grid-rows-16 border border-indigo-400/30">
              {Array.from({ length: gridRows * gridCols }).map((_, idx) => {
                const r = Math.floor(idx / gridCols);
                const c = idx % gridCols;
                return (
                  <div
                    key={idx}
                    className="border border-indigo-300/20 flex items-center justify-center text-[7px] text-indigo-400/40 font-mono"
                  >
                    {r},{c}
                  </div>
                );
              })}
            </div>
          )}

          {/* ==============================================================
              LAYER 2: COLPALI ATTENTION HEATMAP OVERLAY (MaxSim)
             ============================================================== */}
          {showHeatmap && currentHeatmap && (
            <div
              className="absolute inset-0 pointer-events-none z-20 transition-opacity duration-300 grid grid-cols-16 grid-rows-16"
              style={{ opacity: heatmapOpacity }}
            >
              {currentHeatmap.map((row, rIdx) =>
                row.map((score, cIdx) => (
                  <div
                    key={`${rIdx}-${cIdx}`}
                    style={{
                      backgroundColor: getHeatmapColor(score, heatmapOpacity),
                      filter: "blur(4px)",
                    }}
                    className="w-full h-full"
                    title={`Patch (${rIdx}, ${cIdx}) - Score: ${score}`}
                  />
                ))
              )}
            </div>
          )}

          {/* ==============================================================
              LAYER 3: VISUAL GROUNDING BOUNDING BOXES
             ============================================================== */}
          {showBoundingBoxes && (
            <div className="absolute inset-0 pointer-events-auto z-30">
              {page.key_regions.map((region, idx) => {
                const bbox = region.bbox;
                const isActive =
                  activeBoundingBox?.label === bbox.label ||
                  (activeBoundingBox?.ymin === bbox.ymin && activeBoundingBox?.xmin === bbox.xmin);

                const styles = getCategoryStyles(region.category, isActive);

                return (
                  <div
                    key={idx}
                    ref={isActive ? boundingBoxRef : undefined}
                    onClick={() => onSelectBoundingBox(bbox)}
                    style={{
                      top: `${bbox.ymin * 100}%`,
                      left: `${bbox.xmin * 100}%`,
                      width: `${(bbox.xmax - bbox.xmin) * 100}%`,
                      height: `${(bbox.ymax - bbox.ymin) * 100}%`,
                    }}
                    className={`absolute border-2 rounded cursor-pointer transition-all duration-200 group ${styles.border} ${
                      isActive ? "animate-pulse" : ""
                    }`}
                  >
                    {/* Bounding Box Label Tag */}
                    <div
                      className={`absolute -top-3.5 left-2 px-1.5 py-0.5 rounded text-[10px] uppercase font-mono tracking-wider flex items-center gap-1 shadow-md ${styles.badge}`}
                    >
                      <span>{region.category}</span>
                      <span>·</span>
                      <span className="truncate max-w-[140px]">{region.label}</span>
                    </div>

                    {/* Hover Card on Bounding Box */}
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 absolute bottom-full mb-1 left-0 bg-slate-950/95 text-slate-100 p-2.5 rounded-lg border border-slate-700 shadow-2xl text-xs w-64 pointer-events-none z-50">
                      <div className="font-bold text-cyan-400 mb-0.5 flex justify-between items-center">
                        <span className="truncate">{region.label}</span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          {Math.round(bbox.confidence * 100)}% Conf
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-300 leading-snug line-clamp-3">
                        {region.text_content}
                      </p>
                      <div className="mt-1.5 pt-1.5 border-t border-slate-800 text-[9px] text-slate-400 font-mono">
                        Coords: [{bbox.ymin}, {bbox.xmin}, {bbox.ymax}, {bbox.xmax}]
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Canvas Bottom Status Bar */}
      <div className="bg-slate-900 border-t border-slate-800 px-4 py-2 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400 z-20">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span className="font-mono text-slate-300">
              {gridRows * gridCols} Multi-Vectors
            </span>
          </span>

          {showHeatmap && (
            <div className="flex items-center gap-2 pl-3 border-l border-slate-800">
              <span className="text-[11px] text-slate-400">Opacity:</span>
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                className="w-16 sm:w-24 accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded"
              />
              <span className="font-mono text-[11px] text-amber-300">
                {Math.round(heatmapOpacity * 100)}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="hidden sm:inline text-slate-500">
            Click citation badges in chat to jump & highlight targets
          </span>
          <span className="bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-slate-300">
            DPI: 150 High-Res
          </span>
        </div>
      </div>
    </div>
  );
};
