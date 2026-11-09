import React from "react";
import {
  Layers,
  Cpu,
  Sparkles,
  BookOpen,
  Network,
  Upload,
  BarChart3,
  ExternalLink,
  ChevronDown,
  Info
} from "lucide-react";
import { DocumentMeta, SystemHealth } from "../types";

interface NavbarProps {
  documents: DocumentMeta[];
  selectedDocId: string;
  onSelectDoc: (id: string) => void;
  systemHealth: SystemHealth | null;
  onOpenArchitecture: () => void;
  onOpenUpload: () => void;
  activeView: "workspace" | "graph" | "inspector";
  setActiveView: (view: "workspace" | "graph" | "inspector") => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  documents,
  selectedDocId,
  onSelectDoc,
  systemHealth,
  onOpenArchitecture,
  onOpenUpload,
  activeView,
  setActiveView,
}) => {
  const currentDoc = documents.find((d) => d.id === selectedDocId);

  return (
    <header className="bg-slate-950 border-b border-slate-800 text-slate-100 px-4 py-2.5 select-none sticky top-0 z-40">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Left: Branding & Core Architecture Pills */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-500 via-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/20">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">
                  VisRAG <span className="text-cyan-400 font-mono text-sm">Studio</span>
                </span>
                <span className="bg-indigo-950/80 text-indigo-300 text-[10px] font-semibold px-2 py-0.5 rounded-full border border-indigo-700/50">
                  ColPali Multi-Vector
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-none">
                Late Interaction MaxSim · VisRAG-Gen · LightRAG Graph
              </p>
            </div>
          </div>

          <div className="hidden xl:flex items-center gap-1.5 pl-3 border-l border-slate-800 text-[11px]">
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">
              <Cpu className="w-3 h-3 text-cyan-400" />
              {systemHealth?.device.toUpperCase() || "CPU"} · 256 Patches
            </span>
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
              <Sparkles className="w-3 h-3 text-amber-400" />
              Gemini 3.8 Flash
            </span>
            <span className="hidden 2xl:flex items-center gap-1 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              MaxSim Active
            </span>
          </div>
        </div>

        {/* Center: Document Selector */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-700/80 rounded-lg px-2.5 py-1 text-xs text-slate-200 hover:border-slate-600 transition shadow-inner">
              <BookOpen className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <select
                value={selectedDocId}
                onChange={(e) => onSelectDoc(e.target.value)}
                className="bg-transparent text-slate-200 text-xs focus:outline-none cursor-pointer pr-4 max-w-[220px] sm:max-w-[320px] truncate"
              >
                {documents.map((doc) => (
                  <option key={doc.id} value={doc.id} className="bg-slate-900 text-slate-100">
                    {doc.arxiv_id ? `[arXiv:${doc.arxiv_id}] ` : ""}{doc.title}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3 h-3 text-slate-400 shrink-0 pointer-events-none -ml-3" />
            </div>
          </div>

          <button
            onClick={onOpenUpload}
            className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-2.5 py-1.5 rounded-lg font-medium transition shadow-sm active:scale-95"
            title="Ingest new PDF or image document"
          >
            <Upload className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Ingest PDF</span>
          </button>
        </div>

        {/* Right: View Toggles & Math Explainer */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-medium">
            <button
              onClick={() => setActiveView("workspace")}
              className={`px-2.5 py-1 rounded-md transition flex items-center gap-1.5 ${
                activeView === "workspace"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Canvas & Chat</span>
            </button>

            <button
              onClick={() => setActiveView("graph")}
              className={`px-2.5 py-1 rounded-md transition flex items-center gap-1.5 ${
                activeView === "graph"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Network className="w-3.5 h-3.5" />
              <span>LightRAG Graph</span>
            </button>

            <button
              onClick={() => setActiveView("inspector")}
              className={`px-2.5 py-1 rounded-md transition flex items-center gap-1.5 ${
                activeView === "inspector"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span className="hidden md:inline">Late-Interaction Tensor</span>
              <span className="md:hidden">Tensor</span>
            </button>
          </div>

          <button
            onClick={onOpenArchitecture}
            className="p-1.5 text-slate-400 hover:text-cyan-400 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition"
            title="ColPali & VisRAG Mathematical Foundations"
          >
            <Info className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
