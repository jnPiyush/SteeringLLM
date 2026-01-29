"""
Multi-vector composition for complex steering behaviors.

This module provides utilities for combining multiple steering vectors,
detecting conflicts, and orthogonalizing vectors for independent control.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from steering_llm.core.steering_vector import SteeringVector

# Constants
DEFAULT_CONFLICT_THRESHOLD = 0.7  # Cosine similarity threshold for conflict detection
ORTHOGONALIZATION_EPSILON = 1e-8  # Small value to prevent division by zero


class VectorComposition:
    """
    Utilities for composing multiple steering vectors.
    
    This class provides methods for:
    - Combining multiple vectors with weighted composition
    - Detecting conflicts via cosine similarity
    - Orthogonalizing vectors using Gram-Schmidt
    """
    
    @staticmethod
    def weighted_sum(
        vectors: List[SteeringVector],
        weights: Optional[List[float]] = None,
        normalize: bool = False,
    ) -> SteeringVector:
        """
        Combine multiple steering vectors with weighted sum.
        
        Args:
            vectors: List of SteeringVector instances to combine
            weights: Optional weights for each vector (default: equal weights)
            normalize: Whether to normalize the result to unit length
        
        Returns:
            Combined SteeringVector
        
        Raises:
            ValueError: If vectors are incompatible or inputs are invalid
        
        Example:
            >>> # Combine politeness and conciseness steering
            >>> combined = VectorComposition.weighted_sum(
            ...     vectors=[politeness_vec, conciseness_vec],
            ...     weights=[0.7, 0.3]
            ... )
        """
        if not vectors:
            raise ValueError("vectors list cannot be empty")
        
        if weights is None:
            weights = [1.0] * len(vectors)
        
        if len(weights) != len(vectors):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of vectors ({len(vectors)})"
            )
        
        # Validate all vectors target the same layer
        first_layer = vectors[0].layer
        first_layer_name = vectors[0].layer_name
        first_model = vectors[0].model_name
        
        for i, vec in enumerate(vectors[1:], 1):
            if vec.layer != first_layer:
                raise ValueError(
                    f"All vectors must target the same layer. "
                    f"Vector 0 targets layer {first_layer}, "
                    f"but vector {i} targets layer {vec.layer}"
                )
            
            if vec.tensor.shape != vectors[0].tensor.shape:
                raise ValueError(
                    f"All vectors must have the same shape. "
                    f"Vector 0 has shape {vectors[0].tensor.shape}, "
                    f"but vector {i} has shape {vec.tensor.shape}"
                )
        
        # Combine vectors with weighted sum
        combined_tensor = torch.zeros_like(vectors[0].tensor)
        for weight, vector in zip(weights, vectors):
            combined_tensor += weight * vector.tensor
        
        # Normalize if requested
        if normalize:
            norm = torch.norm(combined_tensor)
            if norm > 0:
                combined_tensor = combined_tensor / norm
        
        # Create combined vector
        methods = [v.method for v in vectors]
        combined_vector = SteeringVector(
            tensor=combined_tensor,
            layer=first_layer,
            layer_name=first_layer_name,
            model_name=first_model,
            method=f"weighted_sum({', '.join(methods)})",
            metadata={
                "source_methods": methods,
                "weights": weights,
                "normalized": normalize,
                "num_vectors": len(vectors),
            },
        )
        
        return combined_vector
    
    @staticmethod
    def compute_similarity(
        vec1: SteeringVector,
        vec2: SteeringVector,
    ) -> float:
        """
        Compute cosine similarity between two steering vectors.
        
        Args:
            vec1: First steering vector
            vec2: Second steering vector
        
        Returns:
            Cosine similarity in range [-1, 1]
        
        Raises:
            ValueError: If vectors are incompatible
        
        Example:
            >>> similarity = VectorComposition.compute_similarity(vec1, vec2)
            >>> if similarity > 0.7:
            ...     print("Vectors are highly correlated!")
        """
        if vec1.tensor.shape != vec2.tensor.shape:
            raise ValueError(
                f"Vectors must have the same shape. "
                f"Got {vec1.tensor.shape} and {vec2.tensor.shape}"
            )
        
        # Compute cosine similarity
        dot_product = torch.dot(vec1.tensor.flatten(), vec2.tensor.flatten())
        norm1 = torch.norm(vec1.tensor)
        norm2 = torch.norm(vec2.tensor)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity.item())
    
    @staticmethod
    def detect_conflicts(
        vectors: List[SteeringVector],
        threshold: float = DEFAULT_CONFLICT_THRESHOLD,
    ) -> List[Tuple[int, int, float]]:
        """
        Detect conflicts between steering vectors via cosine similarity.
        
        High positive correlation (>threshold) indicates vectors steer in similar
        directions, which may cause redundancy. High negative correlation 
        (<-threshold) indicates vectors steer in opposite directions, which
        may cause conflicts.
        
        Args:
            vectors: List of SteeringVector instances to analyze
            threshold: Similarity threshold for conflict detection 
                       (default: 0.7 - empirically effective for steering)
        
        Returns:
            List of (idx1, idx2, similarity) tuples for conflicting pairs
        
        Example:
            >>> conflicts = VectorComposition.detect_conflicts(
            ...     [politeness_vec, rudeness_vec, conciseness_vec],
            ...     threshold=0.7
            ... )
            >>> for i, j, sim in conflicts:
            ...     print(f"Vectors {i} and {j} conflict: similarity={sim:.3f}")
        """
        if len(vectors) < 2:
            return []
        
        conflicts = []
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = VectorComposition.compute_similarity(vectors[i], vectors[j])
                
                if abs(similarity) >= threshold:
                    conflicts.append((i, j, similarity))
        
        return conflicts
    
    @staticmethod
    def orthogonalize(
        vectors: List[SteeringVector],
        method: str = "gram_schmidt",
    ) -> List[SteeringVector]:
        """
        Orthogonalize steering vectors for independent control.
        
        Uses Gram-Schmidt orthogonalization to make vectors orthogonal to
        each other, allowing independent steering along each direction.
        
        Args:
            vectors: List of SteeringVector instances to orthogonalize
            method: Orthogonalization method (only "gram_schmidt" supported)
        
        Returns:
            List of orthogonalized SteeringVector instances
        
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        
        Example:
            >>> # Make politeness and formality vectors independent
            >>> ortho_vecs = VectorComposition.orthogonalize(
            ...     [politeness_vec, formality_vec]
            ... )
            >>> # Now they can be applied independently without interference
        """
        if not vectors:
            raise ValueError("vectors list cannot be empty")
        
        if method != "gram_schmidt":
            raise ValueError(f"Unsupported orthogonalization method: {method}")
        
        # Validate all vectors have same shape and layer
        first_shape = vectors[0].tensor.shape
        first_layer = vectors[0].layer
        first_layer_name = vectors[0].layer_name
        first_model = vectors[0].model_name
        
        for i, vec in enumerate(vectors[1:], 1):
            if vec.tensor.shape != first_shape:
                raise ValueError(
                    f"All vectors must have the same shape. "
                    f"Vector 0 has shape {first_shape}, "
                    f"but vector {i} has shape {vec.tensor.shape}"
                )
            if vec.layer != first_layer:
                raise ValueError(
                    f"All vectors must target the same layer. "
                    f"Vector 0 targets layer {first_layer}, "
                    f"but vector {i} targets layer {vec.layer}"
                )
        
        # Gram-Schmidt orthogonalization
        ortho_tensors = []
        
        for i, vec in enumerate(vectors):
            # Start with current vector
            ortho_tensor = vec.tensor.clone()
            
            # Subtract projections onto previous orthogonal vectors
            for prev_ortho in ortho_tensors:
                # Compute projection: (v·u / u·u) * u
                dot_product = torch.dot(ortho_tensor.flatten(), prev_ortho.flatten())
                norm_squared = torch.dot(prev_ortho.flatten(), prev_ortho.flatten())
                
                if norm_squared > 1e-10:  # Avoid division by zero
                    projection = (dot_product / norm_squared) * prev_ortho
                    ortho_tensor = ortho_tensor - projection
            
            # Normalize the orthogonalized vector
            norm = torch.norm(ortho_tensor)
            if norm > 1e-10:
                ortho_tensor = ortho_tensor / norm
            
            ortho_tensors.append(ortho_tensor)
        
        # Create orthogonalized SteeringVector instances
        ortho_vectors = []
        for i, (orig_vec, ortho_tensor) in enumerate(zip(vectors, ortho_tensors)):
            ortho_vec = SteeringVector(
                tensor=ortho_tensor,
                layer=first_layer,
                layer_name=first_layer_name,
                model_name=first_model,
                method=f"{orig_vec.method}_orthogonalized",
                metadata={
                    "original_method": orig_vec.method,
                    "orthogonalization": method,
                    "position": i,
                    **orig_vec.metadata,
                },
            )
            ortho_vectors.append(ortho_vec)
        
        return ortho_vectors
    
    @staticmethod
    def analyze_composition(
        vectors: List[SteeringVector],
    ) -> Dict[str, Any]:
        """
        Analyze a collection of vectors for composition insights.
        
        Provides comprehensive analysis including:
        - Pairwise similarities
        - Conflict detection
        - Average magnitude
        - Recommendations
        
        Args:
            vectors: List of SteeringVector instances to analyze
        
        Returns:
            Dictionary with analysis results
        
        Example:
            >>> analysis = VectorComposition.analyze_composition(
            ...     [vec1, vec2, vec3]
            ... )
            >>> print(analysis['summary'])
            >>> for rec in analysis['recommendations']:
            ...     print(f"- {rec}")
        """
        if not vectors:
            return {
                "num_vectors": 0,
                "summary": "No vectors provided",
            }
        
        # Compute pairwise similarities
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:
                    sim = VectorComposition.compute_similarity(vectors[i], vectors[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Detect conflicts
        conflicts = VectorComposition.detect_conflicts(vectors, threshold=0.7)
        
        # Compute statistics
        magnitudes = [v.magnitude for v in vectors]
        avg_magnitude = float(np.mean(magnitudes))
        
        # Generate recommendations
        recommendations = []
        
        if conflicts:
            recommendations.append(
                f"Found {len(conflicts)} potential conflicts (|similarity| >= 0.7). "
                "Consider orthogonalizing vectors for independent control."
            )
        
        high_corr_pairs = [(i, j, s) for i, j, s in conflicts if s > 0.7]
        if high_corr_pairs:
            recommendations.append(
                f"Found {len(high_corr_pairs)} highly correlated pairs. "
                "These vectors may be redundant."
            )
        
        high_anti_pairs = [(i, j, s) for i, j, s in conflicts if s < -0.7]
        if high_anti_pairs:
            recommendations.append(
                f"Found {len(high_anti_pairs)} highly anti-correlated pairs. "
                "These vectors steer in opposite directions."
            )
        
        if not conflicts:
            recommendations.append(
                "No significant conflicts detected. Vectors appear independent."
            )
        
        return {
            "num_vectors": n,
            "similarity_matrix": similarity_matrix.tolist(),
            "conflicts": conflicts,
            "magnitudes": magnitudes,
            "avg_magnitude": avg_magnitude,
            "recommendations": recommendations,
            "summary": (
                f"Analyzed {n} vectors with average magnitude {avg_magnitude:.3f}. "
                f"Found {len(conflicts)} potential conflicts."
            ),
        }
