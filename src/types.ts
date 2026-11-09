export type EntityCategory = "model" | "metric" | "dataset" | "table" | "method" | "section" | "equation" | "finding" | "header" | "text" | "chart";

export interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  label: string;
  category: "table" | "chart" | "header" | "text" | "metric" | "equation";
  confidence: number;
}

export interface KeyRegion {
  label: string;
  category: "table" | "chart" | "header" | "text" | "metric" | "equation";
  bbox: BoundingBox;
  text_content: string;
  tokens: string[];
}

export interface TokenScore {
  token: string;
  token_idx: number;
  max_score: number;
  best_patch_idx: number;
  patch_coord: { row: number; col: number };
  norm_coord: { ymin: number; xmin: number; ymax: number; xmax: number };
}

export interface TokenHeatmap {
  token: string;
  heatmap: number[][]; // [16, 16] normalized 0.0 - 1.0
}

export interface DocumentPage {
  page_number: number;
  doc_id: string;
  title: string;
  section: string;
  width: number;
  height: number;
  patch_grid: [number, number];
  key_regions: KeyRegion[];
  svg_render?: string;
  image_base64?: string;
}

export interface DocumentMeta {
  id: string;
  title: string;
  author: string;
  year: number;
  arxiv_id?: string;
  abstract: string;
  category: string;
  total_pages: number;
  pages_meta: {
    page_number: number;
    title: string;
    section: string;
    num_patches: number;
    num_regions: number;
  }[];
}

export interface SearchResultItem {
  doc_id: string;
  page_number: number;
  page_title: string;
  section: string;
  score: number;
  token_scores: TokenScore[];
  heatmap: number[][];
  token_heatmaps?: TokenHeatmap[];
  key_regions: KeyRegion[];
}

export interface Citation {
  page_number: number;
  doc_id: string;
  maxsim_score: number;
  bbox: BoundingBox;
  snippet: string;
  category: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  retrieved_pages?: SearchResultItem[];
  model_used?: string;
  token_breakdown?: TokenScore[];
}

export interface GraphNode {
  id: string;
  label: string;
  nodeType: "page" | "entity";
  category: EntityCategory;
  docId: string;
  pageNumber: number;
  bbox?: BoundingBox;
  description: string;
  properties?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  description: string;
  weight: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
}

export interface SystemHealth {
  status: string;
  framework: string;
  device: string;
  embedding_dim: number;
  patch_grid: string;
  total_documents: number;
  graph_nodes: number;
  graph_edges: number;
  gemini_api_configured: boolean;
  gemini_model: string;
}
