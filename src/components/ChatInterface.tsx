import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  Bot,
  User,
  ExternalLink,
  Target,
  Flame,
  ArrowRight,
  RefreshCw,
  HelpCircle,
  Hash,
  CheckCircle2,
  Table,
  LineChart,
  Binary
} from "lucide-react";
import { ChatMessage, Citation, BoundingBox, TokenScore } from "../types";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (query: string) => Promise<void>;
  isLoading: boolean;
  onJumpToCitation: (citation: Citation) => void;
  onSelectTokenHeatmap?: (token: string) => void;
}

const PRESET_QUERIES = [
  {
    label: "ViDoRe Table 1 Benchmark",
    query: "What is the nDCG@5 score of ColPali on the ViDoRe benchmark compared to BM25 and BGE-M3?",
    icon: Table,
  },
  {
    label: "Equation 1: MaxSim Operator",
    query: "Explain Equation 1: How does the ColPali Late-Interaction MaxSim operator work mathematically?",
    icon: Binary,
  },
  {
    label: "VisRAG Multimodal Grounding",
    query: "Why does VisRAG replace raw OCR text with high-resolution page images in the generator?",
    icon: LineChart,
  },
];

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  onJumpToCitation,
  onSelectTokenHeatmap,
}) => {
  const [inputQuery, setInputQuery] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isLoading) return;
    const q = inputQuery;
    setInputQuery("");
    onSendMessage(q);
  };

  const handlePresetClick = (q: string) => {
    if (isLoading) return;
    setInputQuery("");
    onSendMessage(q);
  };

  return (
    <div className="h-full flex flex-col bg-slate-950 border border-slate-800/80 rounded-xl overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-3 flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-white uppercase tracking-wider">
              VisRAG Grounded Synthesis
            </h2>
            <p className="text-[11px] text-slate-400">
              Late Interaction Retrieval + Gemini Multimodal Vision Grounding
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 bg-slate-950 px-2 py-1 rounded border border-slate-800 text-[11px] font-mono text-cyan-400">
          <Target className="w-3.5 h-3.5" />
          <span>Bounding Box Citing</span>
        </div>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-700">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shadow-inner">
              <Bot className="w-6 h-6" />
            </div>
            <div className="max-w-md">
              <h3 className="text-sm font-bold text-slate-200">
                Enterprise VisRAG & ColPali Research Assistant
              </h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Ask questions over the loaded research papers or your uploaded PDFs. Responses are directly grounded
                using ColPali patch multi-vectors with exact bounding box citations.
              </p>
            </div>

            {/* Quick Test Presets */}
            <div className="w-full max-w-md space-y-2 pt-2">
              <p className="text-[11px] font-mono text-slate-500 uppercase tracking-wider text-left">
                Recommended Benchmark Queries:
              </p>
              {PRESET_QUERIES.map((preset, idx) => {
                const Icon = preset.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => handlePresetClick(preset.query)}
                    className="w-full text-left p-2.5 rounded-lg bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-indigo-500/50 text-xs text-slate-300 transition flex items-center justify-between group active:scale-[0.99]"
                  >
                    <div className="flex items-center gap-2.5 truncate">
                      <Icon className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span className="truncate">{preset.label}</span>
                    </div>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-cyan-400 transition shrink-0" />
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col space-y-2 ${
                msg.role === "user" ? "items-end" : "items-start"
              }`}
            >
              {/* Message Header */}
              <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono px-1">
                {msg.role === "user" ? (
                  <>
                    <span>You</span>
                    <User className="w-3 h-3 text-slate-300" />
                  </>
                ) : (
                  <>
                    <Bot className="w-3 h-3 text-indigo-400" />
                    <span className="text-indigo-300 font-semibold">
                      {msg.model_used || "VisRAG-Gen (Gemini 3.8 Flash)"}
                    </span>
                    <span>·</span>
                    <span className="text-slate-500">{msg.timestamp}</span>
                  </>
                )}
              </div>

              {/* Message Bubble */}
              <div
                className={`max-w-[92%] p-3.5 rounded-xl text-xs leading-relaxed ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white shadow-md rounded-tr-none"
                    : "bg-slate-900 border border-slate-800 text-slate-200 shadow-md rounded-tl-none space-y-3"
                }`}
              >
                {/* Formatted Text Content */}
                <div className="whitespace-pre-wrap">{msg.content}</div>

                {/* Query Token MaxSim Scores Bar */}
                {msg.token_breakdown && msg.token_breakdown.length > 0 && (
                  <div className="pt-2 border-t border-slate-800">
                    <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono mb-1.5">
                      <span className="flex items-center gap-1">
                        <Flame className="w-3 h-3 text-amber-400" />
                        ColPali Query Token MaxSim Scores:
                      </span>
                      <span className="text-[9px] text-slate-500">Click token to filter heatmap</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {msg.token_breakdown.map((t, idx) => (
                        <button
                          key={idx}
                          onClick={() => onSelectTokenHeatmap && onSelectTokenHeatmap(t.token)}
                          className="px-2 py-0.5 rounded bg-slate-950 hover:bg-indigo-950 border border-slate-800 hover:border-indigo-600/60 text-[10px] font-mono text-cyan-300 transition flex items-center gap-1"
                          title={`Query Token "${t.token}" matched patch (${t.patch_coord.row}, ${t.patch_coord.col}) with dot-product ${t.max_score.toFixed(3)}`}
                        >
                          <span className="text-slate-400">"{t.token}"</span>
                          <span className="font-bold text-amber-400">{t.max_score.toFixed(2)}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Citation Badges (Interactive Grounding Links) */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="pt-2 border-t border-slate-800/80 space-y-1.5">
                    <p className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
                      <Target className="w-3 h-3 text-cyan-400" />
                      Visual Document Citations (Click to Jump & Highlight):
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {msg.citations.map((cite, cIdx) => (
                        <button
                          key={cIdx}
                          onClick={() => onJumpToCitation(cite)}
                          className="px-2.5 py-1.5 rounded-lg bg-indigo-950/70 hover:bg-indigo-900 border border-indigo-700/60 hover:border-cyan-400 text-slate-100 text-[11px] font-medium transition flex items-center gap-2 group shadow-sm active:scale-95"
                          title="Click to view target bounding box on the Document Canvas"
                        >
                          <span className="px-1.5 py-0.2 bg-cyan-500 text-slate-950 text-[10px] font-mono font-bold rounded">
                            p.{cite.page_number}
                          </span>
                          <span className="truncate max-w-[130px] font-semibold text-slate-200">
                            {cite.bbox.label}
                          </span>
                          <span className="text-[10px] font-mono text-amber-300">
                            MaxSim: {cite.maxsim_score.toFixed(3)}
                          </span>
                          <ExternalLink className="w-3 h-3 text-cyan-400 opacity-60 group-hover:opacity-100 transition" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex flex-col space-y-2 items-start">
            <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono px-1">
              <Bot className="w-3 h-3 text-indigo-400" />
              <span className="text-indigo-300 font-semibold">VisRAG-Gen Pipeline</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl rounded-tl-none text-xs text-slate-300 flex items-center gap-3 shadow-md">
              <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin" />
              <div>
                <p className="font-semibold text-slate-200">
                  Late Interaction Scoring & Multimodal Grounding...
                </p>
                <p className="text-[11px] text-slate-400 font-mono">
                  Tokenizing query ➔ Computing MaxSim over 256 patches ➔ Gemini 3.8 Flash Vision Grounding
                </p>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box Form */}
      <form onSubmit={handleSubmit} className="p-3 bg-slate-900/95 border-t border-slate-800">
        <div className="relative flex items-center">
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Ask anything about the visual layout, tables, or formulas..."
            disabled={isLoading}
            className="w-full bg-slate-950 border border-slate-700 focus:border-cyan-400 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-400/50 pr-12 transition shadow-inner"
          />
          <button
            type="submit"
            disabled={!inputQuery.trim() || isLoading}
            className="absolute right-1.5 p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:pointer-events-none text-white transition active:scale-95"
            title="Send query"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono mt-2 px-1">
          <span>ColPali Late Interaction: Score(Q, D) = ∑ max(q · dᵀ)</span>
          <span className="text-cyan-400 font-medium">VisRAG Dual-Stage</span>
        </div>
      </form>
    </div>
  );
};
