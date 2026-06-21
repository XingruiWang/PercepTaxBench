#!/usr/bin/env python3
"""
Consolidate hierarchical objects in OpenImages dataset using BGE embeddings and clustering.
This script identifies overly specific subobjects and groups them into parent categories
using semantic similarity based on embeddings.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, OPTICS, KMeans
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalObjectConsolidator:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initialize the consolidator with BGE model for embeddings."""
        self.model = SentenceTransformer(model_name)
        self.consolidation_mapping = {}
        self.reverse_mapping = {}
        
    def load_objects(self, objects_file: str) -> List[str]:
        """Load object names from file."""
        with open(objects_file, 'r') as f:
            objects = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(objects)} objects from {objects_file}")
        return objects
    
    def identify_hierarchical_patterns(self, objects: List[str]) -> Dict[str, List[str]]:
        """Identify hierarchical patterns in object names."""
        patterns = defaultdict(list)
        
        for obj in objects:
            # Split by common separators
            parts = obj.replace('_', ' ').replace('-', ' ').split()
            
            # Look for hierarchical patterns
            if len(parts) >= 2:
                # Pattern: "parent child" or "parent child type"
                parent = parts[0]
                child = ' '.join(parts[1:])
                patterns[parent].append(obj)
                
                # Also consider "child parent" patterns
                if len(parts) >= 3:
                    child_parent = ' '.join(parts[:-1])
                    patterns[child_parent].append(obj)
        
        # Filter patterns with multiple objects
        hierarchical_groups = {k: v for k, v in patterns.items() if len(v) > 1}
        
        logger.info(f"Found {len(hierarchical_groups)} hierarchical patterns")
        for parent, children in list(hierarchical_groups.items())[:10]:
            logger.info(f"  {parent}: {children}")
        
        return hierarchical_groups
    
    def generate_embeddings(self, objects: List[str]) -> np.ndarray:
        """Generate BGE embeddings for all objects."""
        logger.info(f"Generating embeddings for {len(objects)} objects...")
        embeddings = self.model.encode(objects, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def cluster_objects(self, objects: List[str], embeddings: np.ndarray, 
                       min_cluster_size: int = 2, min_samples: int = 1, 
                       cluster_selection_epsilon: float = 0.3) -> Dict[int, List[str]]:
        """Cluster objects based on semantic similarity using K-means for target cluster count."""
        # Use cluster_selection_epsilon as the number of clusters (k)
        n_clusters = int(cluster_selection_epsilon)
        logger.info(f"Clustering objects with K-means, n_clusters={n_clusters}")
        
        # Normalize embeddings for cosine distance
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(embeddings, norm='l2')
        
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = clustering.fit_predict(normalized_embeddings)
        
        # Group objects by cluster
        clusters = defaultdict(list)
        for obj, label in zip(objects, cluster_labels):
            clusters[label].append(obj)
        
        # Remove noise cluster (-1)
        if -1 in clusters:
            noise_objects = clusters.pop(-1)
            logger.info(f"Found {len(noise_objects)} noise objects (not clustered)")
        
        logger.info(f"Created {len(clusters)} clusters")
        for cluster_id, cluster_objects in list(clusters.items())[:10]:
            logger.info(f"  Cluster {cluster_id}: {cluster_objects}")
        
        return dict(clusters)
    
    def select_representative_objects(self, clusters: Dict[int, List[str]], 
                                    embeddings: np.ndarray, objects: List[str]) -> Dict[str, str]:
        """Select representative objects for each cluster."""
        consolidation_mapping = {}
        object_to_idx = {obj: i for i, obj in enumerate(objects)}
        
        for cluster_id, cluster_objects in clusters.items():
            if len(cluster_objects) <= 1:
                continue
                
            # Get embeddings for this cluster
            cluster_indices = [object_to_idx[obj] for obj in cluster_objects]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find object closest to centroid
            similarities = cosine_similarity([centroid], cluster_embeddings)[0]
            best_idx = np.argmax(similarities)
            representative = cluster_objects[best_idx]
            
            # Map all objects in cluster to representative
            for obj in cluster_objects:
                if obj != representative:
                    consolidation_mapping[obj] = representative
                    self.reverse_mapping.setdefault(representative, []).append(obj)
        
        logger.info(f"Created consolidation mapping for {len(consolidation_mapping)} objects")
        return consolidation_mapping
    
    def apply_consolidation(self, objects: List[str], consolidation_mapping: Dict[str, str]) -> List[str]:
        """Apply consolidation mapping to create final object list."""
        consolidated_objects = set()
        
        for obj in objects:
            if obj in consolidation_mapping:
                # Use consolidated version
                consolidated_objects.add(consolidation_mapping[obj])
            else:
                # Keep original
                consolidated_objects.add(obj)
        
        final_objects = sorted(list(consolidated_objects))
        logger.info(f"Consolidated {len(objects)} objects to {len(final_objects)} objects")
        return final_objects
    
    def save_results(self, original_objects: List[str], consolidated_objects: List[str],
                    consolidation_mapping: Dict[str, str], output_dir: str):
        """Save consolidation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save consolidated object list
        with open(output_path / "consolidated_objects.txt", 'w') as f:
            for obj in consolidated_objects:
                f.write(f"{obj}\n")
        
        # Save consolidation mapping
        with open(output_path / "consolidation_mapping.json", 'w') as f:
            json.dump(consolidation_mapping, f, indent=2)
        
        # Save reverse mapping
        with open(output_path / "reverse_mapping.json", 'w') as f:
            json.dump(self.reverse_mapping, f, indent=2)
        
        # Save consolidation report
        report = {
            "original_count": len(original_objects),
            "consolidated_count": len(consolidated_objects),
            "reduction_ratio": len(consolidated_objects) / len(original_objects),
            "consolidated_objects": consolidated_objects,
            "consolidation_mapping": consolidation_mapping,
            "reverse_mapping": self.reverse_mapping
        }
        
        with open(output_path / "consolidation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        logger.info(f"Reduction ratio: {report['reduction_ratio']:.3f}")
    
    def consolidate(self, objects_file: str, output_dir: str, 
                   min_cluster_size: int = 2, min_samples: int = 1, 
                   cluster_selection_epsilon: float = 0.3) -> None:
        """Main consolidation pipeline."""
        # Load objects
        objects = self.load_objects(objects_file)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(objects)
        
        # Cluster objects
        clusters = self.cluster_objects(objects, embeddings, min_cluster_size, min_samples, cluster_selection_epsilon)
        
        # Select representative objects
        consolidation_mapping = self.select_representative_objects(clusters, embeddings, objects)
        
        # Apply consolidation
        consolidated_objects = self.apply_consolidation(objects, consolidation_mapping)
        
        # Save results
        self.save_results(objects, consolidated_objects, consolidation_mapping, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Consolidate hierarchical objects using BGE embeddings and HDBSCAN")
    parser.add_argument("--objects_file", required=True, help="Path to objects list file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--min_cluster_size", type=int, default=2, help="HDBSCAN min_cluster_size parameter")
    parser.add_argument("--min_samples", type=int, default=1, help="HDBSCAN min_samples parameter")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.3, help="HDBSCAN cluster_selection_epsilon parameter")
    
    args = parser.parse_args()
    
    consolidator = HierarchicalObjectConsolidator()
    consolidator.consolidate(args.objects_file, args.output_dir, args.min_cluster_size, args.min_samples, args.cluster_selection_epsilon)

if __name__ == "__main__":
    main()
