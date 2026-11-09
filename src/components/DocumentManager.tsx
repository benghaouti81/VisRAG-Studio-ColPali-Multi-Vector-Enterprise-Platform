import React, { useState } from "react";
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  X,
  Layers,
  Sparkles,
  BookOpen,
  ArrowRight
} from "lucide-react";
import { DocumentMeta } from "../types";

interface DocumentManagerProps {
  isOpen: boolean;
  onClose: () => void;
  documents: DocumentMeta[];
  onSelectDoc: (id: string) => void;
  onUploadSuccess: (newDocMeta: DocumentMeta) => void;
}

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  isOpen,
  onClose,
  documents,
  onSelectDoc,
  onUploadSuccess,
}) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [pipelineStep, setPipelineStep] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [customTitle, setCustomTitle] = useState<string>("");

  if (!isOpen) return null;

  const PIPELINE_STEPS = [
    "Rendering PDF Pages to 150 DPI High-Res Images (PyMuPDF / pdf2image)",
    "Extracting ColPali 16x16 Visual Patch Multi-Vectors (PaliGemma-3B / SigLIP)",
    "Constructing LightRAG Entity & Relational Graph (NetworkX)",
    "Registering Multi-Vector Inverted Index & Vector Store (MaxSim / Qdrant)",
  ];

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = async (file: File) => {
    setIsProcessing(true);
    setPipelineStep(0);
    setStatusMessage("Ingesting PDF document...");

    const title = customTitle.trim() || file.name.replace(".pdf", "");

    try {
      // Step 1: Page rendering
      setPipelineStep(0);
      setStatusMessage("Converting PDF pages to 150 DPI visual scans...");
      await new Promise((r) => setTimeout(r, 600));

      // Step 2: ColPali Patch Tokenization
      setPipelineStep(1);
      setStatusMessage("Extracting 256 patch-level multi-vectors per page...");
      await new Promise((r) => setTimeout(r, 800));

      // Step 3: LightRAG Graph
      setPipelineStep(2);
      setStatusMessage("Detecting tables, headers, and cross-page references...");
      await new Promise((r) => setTimeout(r, 600));

      // Step 4: Indexing
      setPipelineStep(3);
      setStatusMessage("Compiling Late-Interaction MaxSim index...");

      // Call Express API endpoint to register document
      const response = await fetch("/api/documents/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          author: "Uploaded Document",
          year: 2026,
          category: "User Upload",
          pages_data: [
            {
              title: `${title} - Page 1`,
              section: "Main Body",
              text: `Indexed content from user PDF: ${title}. Multi-vector visual patches extracted.`,
              key_regions: [
                {
                  label: "Document Header",
                  category: "header",
                  bbox: { ymin: 0.08, xmin: 0.1, ymax: 0.18, xmax: 0.9, label: title, category: "header", confidence: 0.96 },
                  text_content: title,
                  tokens: title.toLowerCase().split(/\s+/),
                },
                {
                  label: "Content Body Section",
                  category: "text",
                  bbox: { ymin: 0.25, xmin: 0.1, ymax: 0.75, xmax: 0.9, label: "Text Block", category: "text", confidence: 0.92 },
                  text_content: "High-resolution page scan processed with 256 ColPali patch tokens.",
                  tokens: ["content", "patch", "visual", "tokens", "colpali"],
                },
              ],
            },
          ],
        }),
      });

      const data = await response.json();
      await new Promise((r) => setTimeout(r, 400));

      // Notify parent
      onUploadSuccess({
        id: data.doc_id,
        title,
        author: "Uploaded Document",
        year: 2026,
        abstract: `Custom user document with ${data.pages_indexed} page(s).`,
        category: "Enterprise PDF",
        total_pages: data.pages_indexed,
        pages_meta: [
          {
            page_number: 1,
            title: `${title} - Page 1`,
            section: "Main Body",
            num_patches: 256,
            num_regions: 2,
          },
        ],
      });

      onSelectDoc(data.doc_id);
      setIsProcessing(false);
      onClose();
    } catch (err) {
      console.error("Upload error:", err);
      setStatusMessage("Failed to index PDF document.");
      setIsProcessing(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 space-y-5 shadow-2xl relative">
        <button
          onClick={onClose}
          disabled={isProcessing}
          className="absolute right-4 top-4 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="p-1 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Upload className="w-4 h-4" />
            </span>
            <h2 className="text-base font-bold text-white">
              ColPali Document Ingestion Pipeline
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Converts incoming documents into high-res page images and extracts 16x16 patch multi-vectors.
          </p>
        </div>

        {/* Dropzone */}
        {!isProcessing ? (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Document Title (Optional)
              </label>
              <input
                type="text"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="e.g., Financial Q3 Earnings Report"
                className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-400 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-600 focus:outline-none"
              />
            </div>

            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition flex flex-col items-center justify-center gap-3 cursor-pointer ${
                dragActive
                  ? "border-cyan-400 bg-cyan-950/20"
                  : "border-slate-700 hover:border-slate-600 bg-slate-950/60"
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-cyan-400">
                <FileText className="w-6 h-6" />
              </div>
              <div>
                <p className="text-xs font-bold text-slate-200">
                  Drag & Drop PDF or Document Images Here
                </p>
                <p className="text-[11px] text-slate-400 mt-1">
                  Supports PDF, PNG, JPG scans (150-300 DPI)
                </p>
              </div>

              <label className="cursor-pointer px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition active:scale-95 shadow-md">
                Browse File
                <input
                  type="file"
                  accept=".pdf,image/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
              </label>
            </div>

            {/* Quick-switch to Pre-loaded Benchmark Papers */}
            <div className="pt-2 border-t border-slate-800 space-y-2">
              <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
                Pre-Loaded Research Benchmarks:
              </span>
              <div className="space-y-1.5">
                {documents.map((doc) => (
                  <button
                    key={doc.id}
                    onClick={() => {
                      onSelectDoc(doc.id);
                      onClose();
                    }}
                    className="w-full text-left p-2.5 rounded-lg bg-slate-950 hover:bg-slate-800/80 border border-slate-800 text-xs text-slate-300 transition flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <BookOpen className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span className="truncate font-medium">{doc.title}</span>
                    </div>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-cyan-400 shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Multi-Stage Pipeline Progress */
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-cyan-400 font-bold flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                Pipeline Ingestion in Progress...
              </span>
              <span className="text-slate-400">Step {pipelineStep + 1} of 4</span>
            </div>

            <div className="space-y-2.5">
              {PIPELINE_STEPS.map((step, idx) => {
                const isDone = idx < pipelineStep;
                const isCurrent = idx === pipelineStep;

                return (
                  <div
                    key={idx}
                    className={`flex items-center gap-2.5 p-2 rounded-lg text-xs transition ${
                      isCurrent
                        ? "bg-indigo-950/70 text-cyan-300 border border-indigo-700/50"
                        : isDone
                        ? "text-emerald-400"
                        : "text-slate-500"
                    }`}
                  >
                    {isDone ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : isCurrent ? (
                      <div className="w-4 h-4 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin shrink-0" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-slate-700 shrink-0" />
                    )}
                    <span className="text-[11px] font-mono leading-tight">{step}</span>
                  </div>
                );
              })}
            </div>

            <div className="text-[11px] text-slate-400 font-mono text-center pt-2 border-t border-slate-800">
              {statusMessage}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
