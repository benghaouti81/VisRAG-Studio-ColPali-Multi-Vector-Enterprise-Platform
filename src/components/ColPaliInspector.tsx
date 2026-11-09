import React, { useState } from "react";
import {
  Layers,
  Cpu,
  Flame,
  ArrowRight,
  Sparkles,
  Binary,
  CheckCircle2,
  Info,
  Maximize2
} from "lucide-react";
import { TokenScore, TokenHeatmap } from "../types";

interface ColPaliInspectorProps {
  tokenScores: TokenScore[];
  tokenHeatmaps: TokenHeatmap[];
  totalScore: number;
  docTitle: string;
  pageNumber: number;
  onSelectTokenForCanvas?: (token: string) => void;
}

export const ColPaliInspector: React.FC<ColPaliInspectorProps> = ({
  tokenScores,
  tokenHeatmaps,
  totalScore,
  docTitle,
  pageNumber,
  onSelectTokenForCanvas,
}) => {
  const [activeTokenIdx, setActiveTokenIdx] = useState<number>(0);

  const activeToken = tokenScores[activeTokenIdx] || tokenScores[0];
  const activeHeatmap = tokenHeatmaps[activeTokenIdx]?.heatmap || [];

  return (
    <div className="h-full flex flex-col bg-slate-950 border border-slate-800/80 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Header */}
      <div className="bg-slate-900 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Binary className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-white uppercase tracking-wider">
              ColPali Late Interaction Tensor Inspector
            </h2>
            <p className="text-[11px] text-slate-400">
              arXiv:2407.01449 · Mathematical Decomposition of MaxSim Operator
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-300 bg-slate-950 px-2.5 py-1 rounded border border-slate-800">
            Total MaxSim Score: <strong className="text-cyan-400">{totalScore.toFixed(4)}</strong>
          </span>
        </div>
      </div>

      {/* Main Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 scrollbar-thin scrollbar-thumb-slate-700 text-slate-300 text-xs">
        {/* Mathematical Equation Card */}
        <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
            <span className="text-indigo-400 font-bold uppercase">Late Interaction Operator (arXiv:2407.01449)</span>
            <span>Dimensions: Q ∈ ℝ^(N_q × 128), D ∈ ℝ^(256 × 128)</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80 font-mono text-center text-sm font-bold text-cyan-300">
            Score(Q, D) = ∑<sub>q ∈ Q</sub> max<sub>d ∈ D</sub> ( q · d<sup>T</sup> )
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Rather than compressing an entire PDF page into a single vector, ColPali preserves all 256 visual patch
            representations. Each query token performs an independent nearest-neighbor dot product over the document
            patches, retaining exact spatial relationships.
          </p>
        </div>

        {/* Token Breakdown Table */}
        <div className="space-y-2">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Query Token MaxSim Breakdown (Page {pageNumber})
          </h3>
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-950/80 border-b border-slate-800 text-[10px] text-slate-400 font-mono uppercase">
                  <th className="p-2.5">Token (q_i)</th>
                  <th className="p-2.5">Max Score (q_i · d*^T)</th>
                  <th className="p-2.5">Attended Patch (Row, Col)</th>
                  <th className="p-2.5">Normalized Bounding Box</th>
                  <th className="p-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {tokenScores.map((ts, idx) => {
                  const isSelected = activeTokenIdx === idx;
                  return (
                    <tr
                      key={idx}
                      onClick={() => setActiveTokenIdx(idx)}
                      className={`cursor-pointer transition ${
                        isSelected
                          ? "bg-indigo-950/60 text-cyan-300"
                          : "hover:bg-slate-850 text-slate-300"
                      }`}
                    >
                      <td className="p-2.5 font-bold text-slate-100 flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                        "{ts.token}"
                      </td>
                      <td className="p-2.5 font-semibold text-amber-300">
                        {ts.max_score.toFixed(4)}
                      </td>
                      <td className="p-2.5 text-slate-300">
                        Patch #{ts.best_patch_idx} (R:{ts.patch_coord.row}, C:{ts.patch_coord.col})
                      </td>
                      <td className="p-2.5 text-slate-400 text-[10px]">
                        [{ts.norm_coord.ymin}, {ts.norm_coord.xmin}, {ts.norm_coord.ymax}, {ts.norm_coord.xmax}]
                      </td>
                      <td className="p-2.5 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onSelectTokenForCanvas) onSelectTokenForCanvas(ts.token);
                          }}
                          className="px-2 py-0.5 rounded bg-slate-800 hover:bg-cyan-600 text-[10px] text-slate-200 hover:text-white transition"
                        >
                          View Heatmap
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Selected Token Patch Heatmap Grid */}
        {activeToken && activeHeatmap.length > 0 && (
          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-200 font-mono">
                Patch Similarity Heatmap for Token:{" "}
                <span className="text-cyan-400">"{activeToken.token}"</span>
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                Peak: Patch #{activeToken.best_patch_idx} ({activeToken.max_score.toFixed(4)})
              </span>
            </div>

            <div className="grid grid-cols-16 gap-0.5 p-2 bg-slate-950 border border-slate-800 rounded-lg max-w-md mx-auto aspect-square">
              {activeHeatmap.map((row, rIdx) =>
                row.map((val, cIdx) => {
                  const isPeak =
                    rIdx === activeToken.patch_coord.row && cIdx === activeToken.patch_coord.col;
                  // Color scale
                  const brightness = Math.round(val * 255);
                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      style={{
                        backgroundColor: isPeak
                          ? "#06b6d4"
                          : `rgba(99, 102, 241, ${val.toFixed(2)})`,
                      }}
                      className={`w-full h-full rounded-[1px] transition-all ${
                        isPeak ? "ring-2 ring-white scale-125 z-10 shadow-lg" : ""
                      }`}
                      title={`Patch (${rIdx}, ${cIdx}) - Dot Product: ${val.toFixed(3)}`}
                    />
                  );
                })
              )}
            </div>

            <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono pt-1">
              <span>0.0 (Low Dot Product)</span>
              <div className="h-1.5 w-32 bg-gradient-to-r from-indigo-950 via-indigo-500 to-cyan-400 rounded"></div>
              <span>1.0 (MaxSim Peak)</span>
            </div>
          </div>
        )}

        {/* Theoretical Advantage vs Dense OCR */}
        <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 space-y-2">
          <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            Why Late Interaction Solves Document RAG Bottlenecks:
          </h4>
          <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-400 leading-relaxed">
            <li>
              <strong>Preserves Multi-Topic Pages:</strong> Dense embeddings average together all sentences on a page, causing catastrophic forgetting. MaxSim queries each word independently.
            </li>
            <li>
              <strong>Robust to Complex Tables:</strong> Standard OCR reads tables left-to-right, scrambling headers and rows. ColPali's 2D patch grid retains relational column alignment.
            </li>
            <li>
              <strong>Direct Visual Grounding:</strong> The argmax patch index $j^* = \arg\max_j (q_i \cdot d_j^T)$ provides direct pixel coordinates without hallucinated token offsets.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
