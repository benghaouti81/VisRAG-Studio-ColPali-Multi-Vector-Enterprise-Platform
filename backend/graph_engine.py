"""
LightRAG Entity & Relational Graph Augmentation Engine
Implements dual-level entity-relationship graph indexing from LightRAG (arXiv:2410.05779).
Enables multi-hop visual reasoning across document pages, tables, figures, and sections.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

logger = logging.getLogger("LightRAGGraphEngine")


class DocumentGraphEngine:
    """
    Entity-Relationship graph manager for cross-page multimodal reasoning.
    Builds a semantic graph connecting concepts, tables, equations, and citations.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        logger.info("Initialized LightRAG DocumentGraphEngine.")

    def add_page_node(self, doc_id: str, page_number: int, title: str = ""):
        """Adds a document page container node to the graph."""
        page_node_id = f"page::{doc_id}::p{page_number}"
        self.graph.add_node(
            page_node_id,
            node_type="page",
            doc_id=doc_id,
            page_number=page_number,
            label=f"Page {page_number}",
            description=title or f"Page {page_number} of document {doc_id}"
        )
        return page_node_id

    def add_entity(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,  # "model", "metric", "dataset", "table", "method", "author", "section"
        doc_id: str,
        page_number: int,
        bounding_box: Optional[Dict[str, float]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Adds a semantic entity node linked to its source page and visual bounding box.
        """
        full_node_id = f"entity::{entity_type}::{entity_id.lower().replace(' ', '_')}"
        props = properties or {}
        
        self.graph.add_node(
            full_node_id,
            node_type="entity",
            category=entity_type,
            name=entity_name,
            label=entity_name,
            doc_id=doc_id,
            page_number=page_number,
            bbox=bounding_box or {"ymin": 0.1, "xmin": 0.1, "ymax": 0.3, "xmax": 0.9},
            **props
        )

        # Automatically link entity to the page it resides on
        page_node_id = f"page::{doc_id}::p{page_number}"
        if not self.graph.has_node(page_node_id):
            self.add_page_node(doc_id, page_number)
            
        self.graph.add_edge(
            page_node_id,
            full_node_id,
            relation="contains_entity",
            weight=1.0
        )
        return full_node_id

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,  # "outperforms", "evaluated_on", "defines", "references", "compares_to"
        description: str = "",
        weight: float = 1.0
    ):
        """Adds a directed relationship between two entities or sections."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.debug(f"Skipping edge {source_id} -> {target_id}: one or both nodes missing.")
            return
            
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation_type,
            description=description,
            weight=weight
        )

    def query_multihop_path(
        self,
        start_entity_keyword: str,
        end_entity_keyword: str,
        max_hops: int = 4
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Finds multi-hop reasoning path between two entities across pages.
        """
        source_candidates = [
            n for n, d in self.graph.nodes(data=True)
            if start_entity_keyword.lower() in str(d.get("name", "")).lower()
        ]
        target_candidates = [
            n for n, d in self.graph.nodes(data=True)
            if end_entity_keyword.lower() in str(d.get("name", "")).lower()
        ]

        if not source_candidates or not target_candidates:
            return None

        # Look for shortest path
        try:
            shortest = None
            undirected = self.graph.to_undirected()
            for s in source_candidates:
                for t in target_candidates:
                    if s != t and nx.has_path(undirected, s, t):
                        path = nx.shortest_path(undirected, s, t)
                        if len(path) <= max_hops + 1:
                            if shortest is None or len(path) < len(shortest):
                                shortest = path
            
            if shortest:
                path_details = []
                for node_id in shortest:
                    data = self.graph.nodes[node_id]
                    path_details.append({
                        "node_id": node_id,
                        "label": data.get("label", node_id),
                        "type": data.get("node_type", "entity"),
                        "category": data.get("category", "concept"),
                        "page": data.get("page_number", 1),
                        "bbox": data.get("bbox", None)
                    })
                return path_details
        except Exception as e:
            logger.error(f"Error querying multi-hop path: {e}")
        return None

    def get_entity_subgraph(
        self,
        keywords: List[str],
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Extracts a localized entity-relationship subgraph relevant to a query.
        Returns serialized nodes and edges for visualization.
        """
        seed_nodes = set()
        for node_id, data in self.graph.nodes(data=True):
            node_name = str(data.get("name", "")).lower()
            node_desc = str(data.get("description", "")).lower()
            for kw in keywords:
                if kw.lower() in node_name or kw.lower() in node_desc:
                    seed_nodes.add(node_id)
                    break

        if not seed_nodes:
            # Fallback to returning the top connected nodes
            degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
            seed_nodes = {n for n, _ in degrees[:8]}

        # Expand neighborhood by depth
        subgraph_nodes = set(seed_nodes)
        undirected = self.graph.to_undirected()
        for seed in seed_nodes:
            if seed in undirected:
                lengths = nx.single_source_shortest_path_length(undirected, seed, cutoff=depth)
                subgraph_nodes.update(lengths.keys())

        sub_g = self.graph.subgraph(subgraph_nodes)

        nodes_list = []
        for n in sub_g.nodes():
            d = dict(self.graph.nodes[n])
            d["id"] = n
            nodes_list.append(d)

        edges_list = []
        for u, v, k, d in sub_g.edges(data=True, keys=True):
            edges_list.append({
                "source": u,
                "target": v,
                "key": k,
                "relation": d.get("relation", "relates_to"),
                "description": d.get("description", ""),
                "weight": d.get("weight", 1.0)
            })

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list)
        }

    def export_full_graph(self) -> Dict[str, Any]:
        """Exports the entire graph for UI visualization."""
        nodes = []
        for n, d in self.graph.nodes(data=True):
            item = dict(d)
            item["id"] = n
            nodes.append(item)

        edges = []
        for u, v, k, d in self.graph.edges(data=True, keys=True):
            edges.append({
                "source": u,
                "target": v,
                "key": k,
                "relation": d.get("relation", "related"),
                "description": d.get("description", ""),
                "weight": d.get("weight", 1.0)
            })

        return {"nodes": nodes, "edges": edges}
