import React, { useState, useMemo } from "react";
import {
  Network,
  Share2,
  Layers,
  Sparkles,
  Search,
  ExternalLink,
  Target,
  ArrowRight,
  Filter,
  CheckCircle2,
  Table,
  Binary
} from "lucide-react";
import { GraphData, GraphNode, GraphEdge, BoundingBox } from "../types";

interface GraphExplorerProps {
  graphData: GraphData;
  onJumpToEntityPage?: (pageNumber: number, bbox?: BoundingBox) => void;
}

export const GraphExplorer: React.FC<GraphExplorerProps> = ({
  graphData,
  onJumpToEntityPage,
}) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState<string>("");
  const [activeCategory, setActiveCategory] = useState<string>("all");

  // Filter nodes based on user search and category
  const filteredNodes = useMemo(() => {
    return graphData.nodes.filter((node) => {
      const matchesSearch =
        node.label.toLowerCase().includes(searchFilter.toLowerCase()) ||
        node.description.toLowerCase().includes(searchFilter.toLowerCase());
      const matchesCategory =
        activeCategory === "all" || node.category === activeCategory || node.nodeType === activeCategory;
      return matchesSearch && matchesCategory;
    });
  }, [graphData.nodes, searchFilter, activeCategory]);

  const selectedNode = graphData.nodes.find((n) => n.id === selectedNodeId) || graphData.nodes[0];

  // Connected edges to the selected node
  const relatedEdges = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.edges.filter(
      (e) => e.source === selectedNode.id || e.target === selectedNode.id
    );
  }, [selectedNode, graphData.edges]);

  const getNodeBadgeColor = (category: string) => {
    switch (category) {
      case "model":
        return "bg-cyan-500/20 text-cyan-300 border-cyan-500/40";
      case "dataset":
        return "bg-indigo-500/20 text-indigo-300 border-indigo-500/40";
      case "table":
        return "bg-blue-500/20 text-blue-300 border-blue-500/40";
      case "equation":
        return "bg-purple-500/20 text-purple-300 border-purple-500/40";
      case "section":
        return "bg-slate-700/40 text-slate-300 border-slate-600/40";
      default:
        return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    }
  };

  return (
    <div className="h-full flex flex-col bg-slate-950 border border-slate-800/80 rounded-xl overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="bg-slate-900 border-b border-slate-800 px-4 py-3 flex flex-wrap items-center justify-between gap-3 z-10">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Network className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-white uppercase tracking-wider">
              LightRAG Relational Knowledge Graph
            </h2>
            <p className="text-[11px] text-slate-400">
              arXiv:2410.05779 · Cross-Page Entity Tracking & Multi-Hop Reasoning
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
            {graphData.nodes.length} Entities · {graphData.edges.length} Relations
          </span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="bg-slate-900/60 border-b border-slate-800 px-4 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="relative flex-1 min-w-[180px] max-w-xs">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5 pointer-events-none" />
          <input
            type="text"
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            placeholder="Search entities or relations..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>

        <div className="flex items-center gap-1 overflow-x-auto text-[11px] font-mono">
          {["all", "model", "dataset", "table", "equation", "section"].map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-2 py-1 rounded capitalize transition ${
                activeCategory === cat
                  ? "bg-indigo-600 text-white font-semibold"
                  : "bg-slate-900 text-slate-400 hover:text-slate-200"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Split Graph Workspace */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-3 overflow-hidden">
        {/* Left 2 Cols: Interactive Graph Visualizer & Nodes Matrix */}
        <div className="md:col-span-2 overflow-y-auto p-4 space-y-4 border-r border-slate-800 bg-slate-950/60 scrollbar-thin scrollbar-thumb-slate-700">
          <div className="p-3 bg-slate-900/70 border border-slate-800 rounded-lg text-xs text-slate-300 flex items-center justify-between">
            <span className="font-mono text-[11px] text-slate-400">
              Select any entity node to inspect cross-page references & multi-hop links:
            </span>
            <span className="text-cyan-400 font-mono text-[11px]">
              {filteredNodes.length} Matches
            </span>
          </div>

          {/* Node Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {filteredNodes.map((node) => {
              const isSelected = selectedNode?.id === node.id;
              const badgeClass = getNodeBadgeColor(node.category);

              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? "bg-indigo-950/80 border-cyan-400 ring-2 ring-cyan-400/20 shadow-lg shadow-cyan-500/10"
                      : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span
                      className={`text-[10px] uppercase font-mono font-bold px-2 py-0.5 rounded border ${badgeClass}`}
                    >
                      {node.category}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      Page {node.pageNumber}
                    </span>
                  </div>

                  <h3 className="font-bold text-xs text-slate-100 truncate">{node.label}</h3>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug line-clamp-2">
                    {node.description}
                  </p>

                  {/* Properties preview */}
                  {node.properties && (
                    <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex flex-wrap gap-1">
                      {Object.entries(node.properties).map(([k, v]) => (
                        <span
                          key={k}
                          className="bg-slate-950 px-1.5 py-0.5 rounded text-[9px] font-mono text-slate-400 border border-slate-800"
                        >
                          <strong className="text-slate-300">{k}:</strong> {String(v)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Multi-Hop Reasoning Path Callout */}
          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2">
            <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              LightRAG Multi-Hop Graph Reasoning Path:
            </h4>
            <div className="flex flex-wrap items-center gap-2 p-2.5 bg-slate-950 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300">
              <span className="px-2 py-0.5 bg-cyan-950 text-cyan-300 rounded border border-cyan-800">
                ColPali Model (p.1)
              </span>
              <ArrowRight className="w-3 h-3 text-slate-500" />
              <span className="px-2 py-0.5 bg-purple-950 text-purple-300 rounded border border-purple-800">
                MaxSim Operator (p.2)
              </span>
              <ArrowRight className="w-3 h-3 text-slate-500" />
              <span className="px-2 py-0.5 bg-indigo-950 text-indigo-300 rounded border border-indigo-800">
                ViDoRe Benchmark (p.3)
              </span>
              <ArrowRight className="w-3 h-3 text-slate-500" />
              <span className="px-2 py-0.5 bg-emerald-950 text-emerald-300 rounded border border-emerald-800">
                Table 1 (81.3% nDCG@5)
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              By maintaining relational connections between model architecture on Page 1 and experimental results on
              Page 3, LightRAG resolves complex multi-hop queries without cross-document hallucinations.
            </p>
          </div>
        </div>

        {/* Right Col: Selected Entity Details & Relations */}
        <div className="overflow-y-auto p-4 space-y-4 bg-slate-900/90 scrollbar-thin scrollbar-thumb-slate-700 text-xs">
          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`text-[10px] font-mono uppercase font-bold px-2 py-0.5 rounded border ${getNodeBadgeColor(
                      selectedNode.category
                    )}`}
                  >
                    {selectedNode.category}
                  </span>
                  <span className="text-slate-400 font-mono text-[11px]">
                    Located on Page {selectedNode.pageNumber}
                  </span>
                </div>
                <h2 className="text-sm font-bold text-white mt-2">{selectedNode.label}</h2>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  {selectedNode.description}
                </p>
              </div>

              {/* Jump to Document Canvas Button */}
              {onJumpToEntityPage && (
                <button
                  onClick={() =>
                    onJumpToEntityPage(selectedNode.pageNumber, selectedNode.bbox)
                  }
                  className="w-full py-2 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition flex items-center justify-center gap-2 active:scale-95 shadow-sm"
                >
                  <Target className="w-4 h-4 text-cyan-300" />
                  <span>View on Document Canvas (Page {selectedNode.pageNumber})</span>
                </button>
              )}

              {/* Relationships Section */}
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <h4 className="text-[11px] font-bold text-slate-300 uppercase tracking-wider font-mono">
                  Relational Connections ({relatedEdges.length})
                </h4>

                <div className="space-y-2">
                  {relatedEdges.map((edge, idx) => {
                    const isOutgoing = edge.source === selectedNode.id;
                    const otherNodeId = isOutgoing ? edge.target : edge.source;
                    const otherNode = graphData.nodes.find((n) => n.id === otherNodeId);

                    return (
                      <div
                        key={idx}
                        className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-[11px] space-y-1"
                      >
                        <div className="flex items-center gap-1.5 font-mono text-cyan-400 font-semibold">
                          <span>{isOutgoing ? "➔" : "⬅"}</span>
                          <span>{edge.relation}</span>
                        </div>
                        <div className="text-slate-300 font-medium">
                          {otherNode?.label || otherNodeId}
                        </div>
                        {edge.description && (
                          <p className="text-[10px] text-slate-400">{edge.description}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Bounding Box Information */}
              {selectedNode.bbox && (
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">
                    Visual Bounding Box Coordinates:
                  </span>
                  <div className="font-mono text-xs text-amber-300">
                    [{selectedNode.bbox.ymin}, {selectedNode.bbox.xmin},{" "}
                    {selectedNode.bbox.ymax}, {selectedNode.bbox.xmax}]
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-slate-500">Select an entity from the list to view its connections.</p>
          )}
        </div>
      </div>
    </div>
  );
};
