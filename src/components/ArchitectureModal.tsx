import React from "react";
import {
  X,
  Layers,
  Sparkles,
  Binary,
  Network,
  CheckCircle2,
  ExternalLink,
  BookOpen
} from "lucide-react";

interface ArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full p-6 space-y-6 shadow-2xl relative max-h-[90vh] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 text-slate-300 text-xs">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Layers className="w-5 h-5" />
            </span>
            <h2 className="text-lg font-bold text-white tracking-tight">
              VisRAG, ColPali & LightRAG Architecture
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Theoretical Foundations, Mathematical Formulations & Enterprise Engineering
          </p>
        </div>

        {/* Section 1: ColPali Multi-Vector Late Interaction */}
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <h3 className="font-bold text-white flex items-center gap-2">
              <Binary className="w-4 h-4 text-cyan-400" />
              1. ColPali: Multi-Vector Visual Late Interaction (arXiv:2407.01449)
            </h3>
            <span className="bg-indigo-950 text-indigo-300 font-mono text-[10px] px-2 py-0.5 rounded border border-indigo-800">
              arXiv:2407.01449
            </span>
          </div>

          <p className="text-slate-300 leading-relaxed text-justify">
            Traditional OCR-based RAG parses documents into unstructured text strings, destroying tables, multi-column
            layouts, font hierarchies, charts, and equations. ColPali replaces OCR by treating the entire document page
            as an image and projecting visual patches into token-level vector representations via PaliGemma-3B / SigLIP.
          </p>

          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 font-mono text-center text-sm font-bold text-cyan-300">
            Score(Q, D) = ∑<sub>q ∈ Q</sub> max<sub>d ∈ D</sub> ( q · d<sup>T</sup> )
          </div>

          <div className="grid grid-cols-2 gap-3 text-[11px] pt-1">
            <div className="p-2.5 bg-slate-900/60 rounded border border-slate-800">
              <span className="font-bold text-slate-200 block mb-1">Visual Patch Grid</span>
              <p className="text-slate-400">
                16x16 or 32x32 patches per page, mapped to a 128-dimensional embedding subspace.
              </p>
            </div>
            <div className="p-2.5 bg-slate-900/60 rounded border border-slate-800">
              <span className="font-bold text-slate-200 block mb-1">ViDoRe SOTA Performance</span>
              <p className="text-slate-400">
                81.3% nDCG@5 overall, outperforming BM25 (52.4%) by +28.9% and dense BGE-M3 (64.1%).
              </p>
            </div>
          </div>
        </div>

        {/* Section 2: VisRAG Generative Grounding */}
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <h3 className="font-bold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              2. VisRAG: Vision-based Grounded Generation (arXiv:2410.10594)
            </h3>
            <span className="bg-indigo-950 text-indigo-300 font-mono text-[10px] px-2 py-0.5 rounded border border-indigo-800">
              arXiv:2410.10594
            </span>
          </div>

          <p className="text-slate-300 leading-relaxed text-justify">
            Instead of feeding text chunks to the LLM, the generator pipeline (VisRAG-Gen) directly provides the
            retrieved high-resolution page images to the multimodal Vision-Language Model (Gemini 3.8 Flash). The model
            cites exact visual bounding boxes [ymin, xmin, ymax, xmax] normalized over the page.
          </p>

          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-[11px] font-mono text-amber-300">
            BoundingBox = [ymin: 0.18, xmin: 0.08, ymax: 0.62, xmax: 0.92] ➔ Table 1 Highlight
          </div>
        </div>

        {/* Section 3: LightRAG Dual-Level Entity Graph */}
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <h3 className="font-bold text-white flex items-center gap-2">
              <Network className="w-4 h-4 text-indigo-400" />
              3. LightRAG: Cross-Page Relational Knowledge Graph (arXiv:2410.05779)
            </h3>
            <span className="bg-indigo-950 text-indigo-300 font-mono text-[10px] px-2 py-0.5 rounded border border-indigo-800">
              arXiv:2410.05779
            </span>
          </div>

          <p className="text-slate-300 leading-relaxed text-justify">
            Maintains an entity-relationship graph connecting sections, entities (models, datasets, metrics), and
            tables across document pages. Enables answering complex multi-hop questions (e.g. tracking how an equation
            defined on Page 2 produced the empirical gains in Table 1 on Page 3).
          </p>
        </div>

        {/* Section 4: Production Deployment Architecture */}
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
          <h4 className="font-bold text-white text-xs">
            4. Microservices Docker Topology
          </h4>
          <div className="font-mono text-[11px] text-slate-400 space-y-1 bg-slate-900 p-3 rounded-lg border border-slate-800">
            <div>• <strong>FastAPI Backend:</strong> Port 8000 (PyTorch, ColPaliEngine, NetworkX, PyMuPDF)</div>
            <div>• <strong>React Frontend:</strong> Port 3000 (Vite + Tailwind CSS + Interactive Canvas)</div>
            <div>• <strong>Qdrant Vector DB:</strong> Port 6333 (Multi-Vector MaxSim Comparator)</div>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg transition active:scale-95"
          >
            Close Explainer
          </button>
        </div>
      </div>
    </div>
  );
};
